"""Phase 2 unit tests — mocked client, no API spend."""

from __future__ import annotations

import pytest

from nla.teacher import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_STRICT,
    TeacherResult,
    already_done,
    build_user_text_only,
    build_user_with_lens,
    is_refusal,
    summarize_row,
)


# ── Mock client ───────────────────────────────────────────────────────────────


class MockTeacher:
    """Replay scripted responses. Each call pops from `responses`; if a queue
    is keyed by user-prompt substring, that one wins."""

    def __init__(self, responses, model: str = "mock-model"):
        self.model = model
        self.calls: list[tuple[str, str]] = []
        # responses can be a flat list, or dict[match-substring -> list]
        self.responses = responses

    async def summarize(self, system, user, *, temperature=0.0, max_tokens=120):
        self.calls.append((system, user))
        if isinstance(self.responses, dict):
            for needle, queue in self.responses.items():
                if needle in user and needle in system:
                    return queue.pop(0)
            for needle, queue in self.responses.items():
                if needle in user or needle in system:
                    return queue.pop(0)
            raise AssertionError(f"No mock response matched system={system[:30]!r}, user={user[:30]!r}")
        return self.responses.pop(0)


def _ok(text: str = "Test summary.") -> TeacherResult:
    return TeacherResult(
        summary=text, input_tokens=100, output_tokens=10,
        finish_reason="stop", system_fingerprint="fp_test",
    )


def _refusal(text: str = "I'm sorry, I can't help with that.") -> TeacherResult:
    return TeacherResult(
        summary=text, input_tokens=100, output_tokens=10,
        finish_reason="stop", system_fingerprint="fp_test",
        is_error=True, error_kind="refusal",
    )


def _empty() -> TeacherResult:
    return TeacherResult(
        summary="", input_tokens=100, output_tokens=0,
        finish_reason="stop", system_fingerprint="fp_test",
        is_error=True, error_kind="empty",
    )


# ── Prompt-builder tests ──────────────────────────────────────────────────────


def test_user_prompt_text_only_contains_text():
    p = build_user_text_only("hello world")
    assert "hello world" in p
    assert "logit-lens" not in p
    assert "Describe the model" in p


def test_user_prompt_with_lens_includes_top3():
    p = build_user_with_lens("hello world", [("foo", 0.5), ("bar", 0.3), ("baz", 0.1)])
    assert "hello world" in p
    assert "logit-lens" in p
    assert "'foo'" in p and "(0.50)" in p
    assert "'bar'" in p and "(0.30)" in p
    assert "'baz'" in p and "(0.10)" in p


def test_system_prompt_persona_neutral():
    """Lock-in for plan decision 4: system prompt must NOT name fish/Guppy/aquarium."""
    lower = SYSTEM_PROMPT.lower()
    for forbidden in ("fish", "guppy", "aquarium", "tank", "water"):
        assert forbidden not in lower, f"persona-leak: {forbidden!r} found in SYSTEM_PROMPT"


def test_strict_prompt_extends_base():
    assert SYSTEM_PROMPT_STRICT.startswith(SYSTEM_PROMPT)
    assert "CRITICAL" in SYSTEM_PROMPT_STRICT
    assert "refusal preamble" in SYSTEM_PROMPT_STRICT


# ── is_refusal tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I'm sorry, but I can't help.", True),
        ("I cannot provide that.", True),
        ("I am not able to describe.", True),
        ("As an AI language model, I", True),
        ("Sorry, but I need more context.", True),
        ("Topic unclear; the model appears to be processing X.", False),
        ("Sorry to hear that, the model is greeting.", False),  # Sorry not at start of refusal pattern
        ("The model is processing a greeting.", False),
        ("", False),
    ],
)
def test_is_refusal_patterns(text, expected):
    assert is_refusal(text) == expected


# ── summarize_row dual-variant ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_summarize_row_calls_both_variants():
    client = MockTeacher([_ok("text-summary"), _ok("lens-summary")])
    out = await summarize_row(client, "hi", [("a", 0.9), ("b", 0.05), ("c", 0.01)])
    assert out["summary_text"] == "text-summary"
    assert out["summary_lens"] == "lens-summary"
    assert out["model"] == "mock-model"
    assert out["logit_lens_top3"] == [["a", 0.9], ["b", 0.05], ["c", 0.01]]
    assert "error" not in out
    # Two calls, both with the persona-neutral system prompt
    assert len(client.calls) == 2
    assert all(c[0] == SYSTEM_PROMPT for c in client.calls)
    assert "logit-lens" not in client.calls[0][1]
    assert "logit-lens" in client.calls[1][1]


@pytest.mark.asyncio
async def test_summarize_row_marks_refusal_without_retry():
    """Default: no retry_strict. Refusal is marked, other variant succeeds."""
    client = MockTeacher([_refusal(), _ok("lens-ok")])
    out = await summarize_row(client, "hi", [("a", 0.9)])
    assert "summary_text" not in out
    assert out["summary_lens"] == "lens-ok"
    assert out["error"] == {"text": "refusal"}


@pytest.mark.asyncio
async def test_summarize_row_retry_strict_recovers():
    """retry_strict=True: refusal triggers a second call with the strict prompt."""
    client = MockTeacher([_refusal(), _ok("lens-ok"), _ok("text-strict-ok")])
    out = await summarize_row(client, "hi", [("a", 0.9)], retry_strict=True)
    assert out["summary_text"] == "text-strict-ok"
    assert out["summary_lens"] == "lens-ok"
    assert "error" not in out
    # Calls: text(base, refusal), lens(base, ok), text(strict, ok)
    assert len(client.calls) == 3
    assert client.calls[0][0] == SYSTEM_PROMPT
    assert client.calls[1][0] == SYSTEM_PROMPT
    assert client.calls[2][0] == SYSTEM_PROMPT_STRICT


@pytest.mark.asyncio
async def test_summarize_row_retry_strict_still_fails():
    """retry_strict=True: if the strict retry also refuses, the row reports error."""
    client = MockTeacher([_refusal(), _ok("lens-ok"), _refusal("Sorry, I cannot.")])
    out = await summarize_row(client, "hi", [("a", 0.9)], retry_strict=True)
    assert "summary_text" not in out
    assert out["summary_lens"] == "lens-ok"
    assert out["error"] == {"text": "refusal"}


@pytest.mark.asyncio
async def test_summarize_row_empty_treated_as_error():
    client = MockTeacher([_empty(), _ok("lens-ok")])
    out = await summarize_row(client, "hi", [("a", 0.9)])
    assert "summary_text" not in out
    assert out["error"] == {"text": "empty"}


# ── Resume helper ─────────────────────────────────────────────────────────────


def test_already_done_full_row():
    row = {"row": 0, "summary_text": "x", "summary_lens": "y"}
    assert already_done(row) is True


def test_already_done_partial_row_recompute():
    row = {"row": 5, "summary_text": "x"}
    assert already_done(row) is False  # missing summary_lens → recompute


def test_already_done_error_row_recompute():
    row = {"row": 7, "summary_text": "x", "summary_lens": "y", "error": {"text": "refusal"}}
    assert already_done(row) is False  # error present → recompute


def test_already_done_text_only_mode():
    row = {"row": 9, "summary_text": "x"}
    assert already_done(row, include_lens=False) is True


def test_already_done_missing():
    assert already_done(None) is False
    assert already_done({}) is False
