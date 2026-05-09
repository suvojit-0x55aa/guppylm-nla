"""Linear probe: does any substrate layer carry semantic content?

For each available layer in data/activations.npz, train a multinomial logistic
regression from h_layer → category (read from corpus.jsonl). Report held-out
accuracy and compare to chance baseline (max class freq).

If no layer beats chance, the substrate model is too small / wrong-architecture
for any AV-style verbalization to work — Phase 3 is moot. If h_l3 (our current
choice) is the worst layer, layer sweep is justified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nla.splits import make_or_load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/corpus.jsonl")
    p.add_argument("--activations", default="data/activations.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=200)
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    print("loading corpus + activations ...")
    corpus = []
    with open(args.corpus) as f:
        for line in f:
            corpus.append(json.loads(line))
    cats = [r.get("category", "?") for r in corpus]
    counts = Counter(cats)
    print(f"  rows = {len(corpus)}")
    print(f"  categories = {len(counts)}  (top 5: {counts.most_common(5)})")
    chance = max(counts.values()) / len(corpus)
    print(f"  chance baseline (most-common class) = {chance:.3f}")

    le = LabelEncoder()
    y = le.fit_transform(cats)

    train_idx, eval_idx = make_or_load_split(len(corpus), seed=args.seed)
    print(f"  train = {len(train_idx)}, eval = {len(eval_idx)}")

    npz = np.load(args.activations)
    print(f"  layer keys = {list(npz.keys())}")

    print()
    print(f"{'layer':<10} | {'eval_acc':>10} | {'chance':>10} | {'lift':>8} | {'top5_cm':>20}")
    print("-" * 70)
    rng = np.random.default_rng(args.seed)
    results = []
    for key in npz.keys():
        X = npz[key]
        if X.ndim != 2 or X.shape[0] != len(corpus):
            continue
        X_tr, X_ev = X[train_idx], X[eval_idx]
        y_tr, y_ev = y[train_idx], y[eval_idx]
        t0 = time.time()
        clf = LogisticRegression(max_iter=args.max_iter, n_jobs=-1,
                                 random_state=args.seed)
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        eval_acc = clf.score(X_ev, y_ev)
        elapsed = time.time() - t0
        # Confusion: top-5 most-confused class pairs (cheap)
        preds = clf.predict(X_ev)
        miscls = Counter()
        for yt, yp in zip(y_ev, preds):
            if yt != yp:
                miscls[(le.classes_[yt], le.classes_[yp])] += 1
        top_cm = ",".join(f"{a}->{b}({n})" for (a, b), n in miscls.most_common(2))
        lift = (eval_acc - chance) / (1.0 - chance)            # 0 = chance, 1 = perfect
        print(f"{key:<10} | {eval_acc:>10.3f} | {chance:>10.3f} | {lift:>+8.3f} | {top_cm:>20.20}")
        results.append((key, eval_acc, train_acc, lift, elapsed))

    print()
    print(f"{'best':<10} | {'eval_acc':>10} | {'train_acc':>10} | {'lift':>8} | {'time':>6}")
    print("-" * 60)
    for key, e, t, lf, secs in sorted(results, key=lambda r: -r[1]):
        flag = " ✓ ABOVE CHANCE" if e > chance + 0.05 else (" ⚠ AT CHANCE" if abs(e - chance) < 0.05 else "")
        print(f"{key:<10} | {e:>10.3f} | {t:>10.3f} | {lf:>+8.3f} | {secs:>5.1f}s{flag}")


if __name__ == "__main__":
    main()
