"""Diagnose whether AV's generated summary tracks input content.

Loads a trained AV checkpoint, decodes 10 held-out rows, prints:
  - input text the substrate was processing
  - teacher summary (target)
  - AV's generated summary
  - input text bag-of-words overlap with AV's summary

If AV's summaries vary with the input — and especially if they share content
words — h_l3 carries useful signal. If AV produces near-identical generic
text regardless of input, h_l3 lacks the signal AV needs and the whole
Phase 3 → Phase 4 plan is moot.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nla.qwen import load_qwen
from nla.av import AV
from nla.data_phase3 import _build_av_prompt_ids, load_phase3_inputs
from nla.splits import make_or_load_split


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z]+", s.lower())) - {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "of", "in", "on", "at", "to", "for",
        "with", "by", "as", "and", "or", "but", "not", "if", "then", "else",
        "this", "that", "these", "those", "it", "its", "model", "likely",
        "processing", "about", "possibly", "may", "might",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to AV ckpt (best_av_*.pt or step_*.pt)")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("loading qwen + AV ...")
    base, tok, act_id = load_qwen(use_4bit=True, device_map="auto")
    av = AV(base, act_id).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state", {})
    missing, unexpected = av.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} keys from {args.ckpt} "
          f"(step={ckpt.get('step')}, metric={ckpt.get('metric_name', 'n/a')}={ckpt.get('metric_value', 'n/a')})")

    corpus, summaries, h = load_phase3_inputs()
    train_idx, eval_idx = make_or_load_split(len(corpus), seed=args.seed)
    print(f"corpus={len(corpus)}  eval={len(eval_idx)}")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(eval_idx, size=min(args.n, len(eval_idx)), replace=False).tolist()

    prompt_ids = torch.tensor(_build_av_prompt_ids(tok), dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(prompt_ids)

    print(f"\n{'=' * 80}")
    print("AV diagnostic — does the generated summary track input content?")
    print(f"{'=' * 80}\n")

    av.eval()
    overlaps = []
    for k, row_id in enumerate(sample_idx):
        row = corpus[row_id]
        teacher = summaries[row_id]["summary_text"]
        input_text = row["text"]
        category = row.get("category", "?")

        h_b = torch.from_numpy(h[row_id].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            full = av.generate(prompt_ids, attn, h_b,
                               max_new_tokens=args.max_new_tokens,
                               eos_token_id=tok.eos_token_id,
                               pad_token_id=tok.pad_token_id)
        new_ids = full[0, prompt_ids.shape[1]:].tolist()
        av_text = tok.decode(new_ids, skip_special_tokens=False)
        for stop in ("<|im_end|>", tok.eos_token):
            if stop and stop in av_text:
                av_text = av_text.split(stop, 1)[0]
        av_text = av_text.strip()

        # bag-of-words overlap (excluding common AV/teacher boilerplate)
        in_toks = tokens(input_text)
        av_toks = tokens(av_text)
        teacher_toks = tokens(teacher)
        in_av = in_toks & av_toks
        in_teacher = in_toks & teacher_toks
        overlaps.append((len(in_av), len(in_teacher), len(in_toks)))

        print(f"--- row {row_id}  category={category} -----------------------")
        print(f"  INPUT:    {input_text!r}")
        print(f"  TEACHER:  {teacher}")
        print(f"  AV:       {av_text}")
        print(f"  overlap(input ∩ av)={len(in_av)}  (input ∩ teacher)={len(in_teacher)}  |input|={len(in_toks)}")
        print(f"  shared(av): {sorted(in_av)}")
        print()

    av_avg = sum(o[0] for o in overlaps) / len(overlaps)
    teacher_avg = sum(o[1] for o in overlaps) / len(overlaps)
    print(f"\n=== summary ===")
    print(f"  avg(input ∩ av)     = {av_avg:.2f} content words shared per row")
    print(f"  avg(input ∩ teacher)= {teacher_avg:.2f} content words shared per row")
    print(f"  if av_avg << teacher_avg: AV is not capturing input content via h_l3")


if __name__ == "__main__":
    main()
