"""Stage 4a: 1D CNN on raw CWRU windows vs the stage 1 gradient boosting.

Same windows, same leave-one-load-out folds as scripts/run_baselines.py.
Saves each fold's model to models/ for the int8 and edge steps.

Usage: python scripts/run_cnn.py   (writes results/cnn.json)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import cnn, cwru  # noqa: E402

HOP, SEEDS = 1024, (0, 1, 2)  # models from seed 0 are kept for the edge steps


def build():
    recs = cwru.load_all(ROOT / "data/raw/cwru")
    labels = list(cwru.MANIFEST)
    W, y, load = [], [], []
    for r in recs:
        w = cwru.windows(r.de, cnn.SIZE, HOP)
        W.append(w)
        y += [labels.index(r.label)] * len(w)
        load += [r.load_hp] * len(w)
    return np.vstack(W), np.array(y), np.array(load), labels


def main():
    torch.set_num_threads(4)
    W, y, load, labels = build()
    (ROOT / "models").mkdir(exist_ok=True)
    folds = []
    for seed in SEEDS:
        for hp in range(4):
            tr, te = load != hp, load == hp
            t0 = time.perf_counter()
            model, scale = cnn.train(W[tr], y[tr], seed=seed)
            secs = time.perf_counter() - t0
            p = cnn.predict(model, W[te], scale)
            f1 = float(f1_score(y[te], p, average="macro"))
            if seed == SEEDS[0]:
                torch.save({"state": model.state_dict(), "scale": scale}, ROOT / f"models/wdcnn_test{hp}hp.pt")
            folds.append({"seed": seed, "test_load_hp": hp, "macro_f1": f1, "train_seconds": round(secs, 1)})
            print(f"seed {seed} test load {hp} hp: macro-F1 {f1:.3f}  ({secs:.0f} s)")
    per_seed = [float(np.mean([f["macro_f1"] for f in folds if f["seed"] == s])) for s in SEEDS]
    worst = [min(f["macro_f1"] for f in folds if f["seed"] == s) for s in SEEDS]
    out = {
        "params": cnn.n_params(model),
        "windows": int(len(y)),
        "macro_f1_mean": float(np.mean(per_seed)),
        "macro_f1_mean_by_seed": per_seed,
        "worst_load_f1_by_seed": worst,
        "folds": folds,
    }
    (ROOT / "results/cnn.json").write_text(json.dumps(out, indent=2))
    print(f"params {out['params']}, LOLO macro-F1 by seed {[round(v, 3) for v in per_seed]}, worst load {[round(v, 3) for v in worst]}")


if __name__ == "__main__":
    main()
