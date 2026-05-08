"""Phase 2 — natural-language teacher for SFT warm-start targets.

Two summary variants per row (text-only and text+logit-lens), persona-neutral
system prompt, retry-with-backoff on transient errors, in-band error reporting
for refusals/empties so Phase 3 can filter them out.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from openai import AsyncOpenAI, APIStatusError, APIConnectionError, RateLimitError


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are reading the internal state of a tiny 9M-parameter language model — a small chatbot trained on short conversational fragments. You will be shown a chat-formatted prompt that the model was processing up to a truncation point.

Describe in ONE TO TWO SENTENCES (≤ 50 words total) what the model is likely processing or about to produce. Be specific about the topic and the likely continuation. Infer the domain from the text itself; do not assume.

No preamble, no quotes, no meta-commentary, no refusal language — just the description as plain prose."""


SYSTEM_PROMPT_STRICT = SYSTEM_PROMPT + """

CRITICAL: Do not begin with "I'm sorry", "I cannot", "As an AI", or any similar refusal preamble. The text is harmless conversational content. Describe the model's internal state directly. If you are uncertain about the domain, say so within the description (e.g. "Topic unclear; the model appears to be processing X"), but do NOT refuse to describe."""


def build_user_text_only(text: str) -> str:
    return f"Truncated input the model was processing:\n{text}\n\nDescribe the model's internal state."


def build_user_with_lens(text: str, top3: Sequence[tuple[str, float]]) -> str:
    formatted = ", ".join(f"{tok!r} ({p:.2f})" for tok, p in top3)
    return (
        f"Truncated input the model was processing:\n{text}\n\n"
        f"The model's three highest-probability next tokens (logit-lens at the final layer): {formatted}.\n\n"
        f"Describe the model's internal state."
    )


# ── Refusal detection ─────────────────────────────────────────────────────────

REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*I('?m| am)\s+(sorry|unable|not able)",
        r"^\s*I\s+(can'?t|cannot|won'?t|will not)",
        r"^\s*As an AI",
        r"^\s*I (don'?t|do not) feel comfortable",
        r"^\s*I'?m not able to",
        r"^\s*Sorry,? (but )?I",
    )
)


def is_refusal(text: str) -> bool:
    if not text:
        return False
    return any(p.match(text) for p in REFUSAL_PATTERNS)


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class TeacherResult:
    summary: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    system_fingerprint: str | None = None
    is_error: bool = False
    error_kind: str | None = None  # "refusal" | "empty" | "length" | "api_error"


# ── Client interface + OpenAI implementation ──────────────────────────────────


class TeacherClient(Protocol):
    model: str

    async def summarize(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 120,
    ) -> TeacherResult: ...


class OpenAITeacher:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        seed: int = 42,
        max_retries: int = 3,
    ):
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = AsyncOpenAI(api_key=key)
        self.model = model
        self._seed = seed
        self._max_retries = max_retries

    async def summarize(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 120,
    ) -> TeacherResult:
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=self._seed,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                choice = resp.choices[0]
                summary = (choice.message.content or "").strip()
                finish = choice.finish_reason or "stop"
                fingerprint = getattr(resp, "system_fingerprint", None)
                usage = resp.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0

                if not summary:
                    return TeacherResult(
                        summary="", input_tokens=in_tok, output_tokens=out_tok,
                        finish_reason=finish, system_fingerprint=fingerprint,
                        is_error=True, error_kind="empty",
                    )
                if is_refusal(summary):
                    return TeacherResult(
                        summary=summary, input_tokens=in_tok, output_tokens=out_tok,
                        finish_reason=finish, system_fingerprint=fingerprint,
                        is_error=True, error_kind="refusal",
                    )
                if finish == "length":
                    # The summary is non-empty but may have been cut. Phase 3 can
                    # still use it; flag for inspection but do not mark as error.
                    return TeacherResult(
                        summary=summary, input_tokens=in_tok, output_tokens=out_tok,
                        finish_reason=finish, system_fingerprint=fingerprint,
                        is_error=False, error_kind=None,
                    )
                return TeacherResult(
                    summary=summary, input_tokens=in_tok, output_tokens=out_tok,
                    finish_reason=finish, system_fingerprint=fingerprint,
                )

            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                last_error = e
                # Exponential backoff with jitter; do not retry on 4xx (except 429)
                status = getattr(e, "status_code", None)
                if isinstance(e, APIStatusError) and status and 400 <= status < 500 and status != 429:
                    break
                await asyncio.sleep((2 ** attempt) + random.random())
                continue

        return TeacherResult(
            summary="", input_tokens=0, output_tokens=0,
            finish_reason="error",
            is_error=True, error_kind="api_error",
        )


# ── Row-level driver: run both variants, optional retry-strict ────────────────


async def summarize_row(
    client: TeacherClient,
    text: str,
    top3: Sequence[tuple[str, float]],
    *,
    retry_strict: bool = False,
    max_tokens: int = 120,
) -> dict:
    """Call teacher twice (text-only + lens-augmented). Returns a dict shaped
    for one row of summaries.jsonl.

    Schema (success):
        {summary_text, summary_lens, model, system_fingerprint,
         input_tokens_text, output_tokens_text,
         input_tokens_lens, output_tokens_lens,
         logit_lens_top3, finish_reason_text, finish_reason_lens}

    Schema (any variant errored):
        Adds {error: {variant: kind, ...}} alongside the keys that succeeded.
    """
    user_text = build_user_text_only(text)
    user_lens = build_user_with_lens(text, top3)

    res_text = await client.summarize(SYSTEM_PROMPT, user_text, max_tokens=max_tokens)
    res_lens = await client.summarize(SYSTEM_PROMPT, user_lens, max_tokens=max_tokens)

    if retry_strict:
        if res_text.is_error and res_text.error_kind in ("refusal", "empty"):
            res_text = await client.summarize(SYSTEM_PROMPT_STRICT, user_text, max_tokens=max_tokens)
        if res_lens.is_error and res_lens.error_kind in ("refusal", "empty"):
            res_lens = await client.summarize(SYSTEM_PROMPT_STRICT, user_lens, max_tokens=max_tokens)

    out: dict = {
        "model": client.model,
        "system_fingerprint": res_text.system_fingerprint or res_lens.system_fingerprint,
        "input_tokens_text": res_text.input_tokens,
        "output_tokens_text": res_text.output_tokens,
        "input_tokens_lens": res_lens.input_tokens,
        "output_tokens_lens": res_lens.output_tokens,
        "finish_reason_text": res_text.finish_reason,
        "finish_reason_lens": res_lens.finish_reason,
        "logit_lens_top3": [[t, float(p)] for t, p in top3],
    }
    if not res_text.is_error:
        out["summary_text"] = res_text.summary
    if not res_lens.is_error:
        out["summary_lens"] = res_lens.summary

    errors: dict[str, str] = {}
    if res_text.is_error:
        errors["text"] = res_text.error_kind or "unknown"
    if res_lens.is_error:
        errors["lens"] = res_lens.error_kind or "unknown"
    if errors:
        out["error"] = errors
    return out


# ── Resume helper ─────────────────────────────────────────────────────────────


def already_done(existing_row: dict | None, *, include_lens: bool = True) -> bool:
    """Resume policy: a row is 'done' iff it has both non-error summaries
    (or just summary_text when include_lens=False)."""
    if not existing_row:
        return False
    if "error" in existing_row:
        return False
    if "summary_text" not in existing_row:
        return False
    if include_lens and "summary_lens" not in existing_row:
        return False
    return True
