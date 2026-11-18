"""Stage 3: vibration vs motor current vs both, on Paderborn real-damage bearings.

3 classes (healthy, outer ring, inner ring), 5 bearings each. Evaluation is
leave-bearings-out: fold i tests on the i-th healthy, outer and inner bearing
and trains on the other twelve, so no physical bearing is on both sides.
A naive random window split is reported next to it to show the leak.

Usage: python scripts/run_paderborn.py   (needs paderborn_features.py; writes results/paderborn.json)
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import paderborn as P  # noqa: E402

CLASSES = list(P.BEARINGS)
SEED = 0


def load():
    d = np.load(ROOT / "data/processed/paderborn.npz")
    y = np.array([CLASSES.index(v) for v in d["label"]])
    rec = np.char.add(np.char.add(d["bearing"], "_"), np.char.add(d["condition"], np.char.add("_", d["index"].astype(str))))
    vn, cn = list(d["vib_names"]), list(d["cur_names"])
    lines = [i for i, n in enumerate(vn) if n[:4] in ("BPFI", "BPFO", "BSF_", "FTF_")]
    sidebands = [i for i, n in enumerate(cn) if "_sb_" in n]
    sets = {
        "vibration_time": d["Xv"][:, :9],
        "vibration_defect_lines": d["Xv"][:, lines],
        "vibration": d["Xv"],
        "current_sidebands": d["Xc"][:, sidebands],
        "current": d["Xc"],
        "vibration+current": np.hstack([d["Xv"], d["Xc"]]),
        "defect_lines+sidebands": np.hstack([d["Xv"][:, lines], d["Xc"][:, sidebands]]),
    }
    return sets, y, d["bearing"], d["condition"], rec


MODELS = {
    "gbm": lambda: HistGradientBoostingClassifier(random_state=SEED),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
}


def vote(pred, rec, y):
    """Majority vote over the 15 windows of each recording."""
    out_p, out_y = [], []
    for r in np.unique(rec):
        m = rec == r
        out_p.append(np.bincount(pred[m], minlength=3).argmax())
        out_y.append(y[m][0])
    return np.array(out_y), np.array(out_p)


def main():
    sets, y, bearing, cond, rec = load()
    folds = [{codes[i] for codes in P.BEARINGS.values()} for i in range(5)]
    out = {"windows": int(len(y)), "classes": CLASSES, "folds": [sorted(f) for f in folds], "results": {}}
    for (name, X), mname in ((kv, m) for kv in sets.items() for m in MODELS):
        model = MODELS[mname]
        win, recf, by_cond, cm = [], [], {c: [] for c in P.CONDITIONS}, np.zeros((3, 3), int)
        for test_set in folds:
            te = np.isin(bearing, list(test_set))
            p = model().fit(X[~te], y[~te]).predict(X[te])
            win.append(f1_score(y[te], p, average="macro"))
            ry, rp = vote(p, rec[te], y[te])
            recf.append(f1_score(ry, rp, average="macro"))
            cm += confusion_matrix(y[te], p, labels=[0, 1, 2])
            for c in P.CONDITIONS:
                m = cond[te] == c
                by_cond[c].append(f1_score(y[te][m], p[m], average="macro"))
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
        naive = f1_score(y[te], model().fit(X[tr], y[tr]).predict(X[te]), average="macro")
        out["results"][f"{name}/{mname}"] = {
            "n_features": int(X.shape[1]),
            "naive_window_f1": round(float(naive), 4),
            "bearing_out_window_f1": round(float(np.mean(win)), 4),
            "bearing_out_window_f1_by_fold": [round(float(v), 4) for v in win],
            "bearing_out_recording_f1": round(float(np.mean(recf)), 4),
            "bearing_out_f1_by_condition": {c: round(float(np.mean(v)), 4) for c, v in by_cond.items()},
            "confusion_rows_true_cols_pred": cm.tolist(),
        }
        r = out["results"][f"{name}/{mname}"]
        print(f"{name + '/' + mname:30s} naive {r['naive_window_f1']:.3f}  bearing-out window {r['bearing_out_window_f1']:.3f} "
              f"recording {r['bearing_out_recording_f1']:.3f}")
    (ROOT / "results/paderborn.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
