"""Baselines and DSP features on CWRU, under two evaluation splits.

naive  : windows shuffled and split 70/30. Overlapping windows from one
         recording land on both sides, which is how most published CWRU
         notebooks are evaluated.
lolo   : leave one motor load out. Train on three loads, test on the fourth,
         repeated for each load. No recording appears on both sides.

Usage: python scripts/run_baselines.py   (writes results/baselines.json)
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # joblib looks for wmic, which Windows 11 dropped
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import cwru, features  # noqa: E402

SIZE, HOP, SEED = 2048, 1024, 0


def build():
    recs = cwru.load_all(ROOT / "data/raw/cwru")
    labels = list(cwru.MANIFEST)
    X_t, X_e, y, load, rec = [], [], [], [], []
    for r in recs:
        w = cwru.windows(r.de, SIZE, HOP)
        X_t.append(features.time_features(w))
        X_e.append(features.envelope_features(w, r.fs, r.rpm))
        y += [labels.index(r.label)] * len(w)
        load += [r.load_hp] * len(w)
        rec += [r.file_id] * len(w)
    return (np.vstack(X_t), np.vstack(X_e), np.array(y), np.array(load), np.array(rec), labels)


def rms_alarm(rms_tr, y_tr, rms_te, y_te, labels):
    """ISO-style alarm: healthy mean + 3 sigma of window RMS on the train side."""
    h = rms_tr[y_tr == 0]
    thr = h.mean() + 3 * h.std()
    alarm = rms_te > thr
    det = {labels[c]: float(alarm[y_te == c].mean()) for c in range(1, len(labels))}
    return {
        "threshold_g": float(thr),
        "false_alarm_rate": float(alarm[y_te == 0].mean()),
        "detection_rate_mean": float(np.mean(list(det.values()))),
        "detection_rate_by_class": det,
    }


def models():
    return {
        "rf_time": (lambda: RandomForestClassifier(300, random_state=SEED, n_jobs=-1), "t"),
        "gbm_time": (lambda: HistGradientBoostingClassifier(random_state=SEED), "t"),
        "gbm_env": (lambda: HistGradientBoostingClassifier(random_state=SEED), "e"),
        "gbm_time_env": (lambda: HistGradientBoostingClassifier(random_state=SEED), "te"),
    }


def pick(Xt, Xe, which):
    return np.hstack([m for k, m in (("t", Xt), ("e", Xe)) if k in which])


def main():
    Xt, Xe, y, load, rec, labels = build()
    out = {"windows": int(len(y)), "window_samples": SIZE, "hop": HOP, "classes": labels}

    # naive split
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=SEED, stratify=y)
    naive = {"rms_alarm": rms_alarm(Xt[idx_tr, 0], y[idx_tr], Xt[idx_te, 0], y[idx_te], labels)}
    for name, (mk, which) in models().items():
        X = pick(Xt, Xe, which)
        p = mk().fit(X[idx_tr], y[idx_tr]).predict(X[idx_te])
        naive[name] = {"macro_f1": float(f1_score(y[idx_te], p, average="macro"))}
    out["naive"] = naive

    # leave one load out
    lolo = {"rms_alarm": [], **{n: [] for n in models()}}
    for hp in range(4):
        tr, te = load != hp, load == hp
        assert not set(rec[tr]) & set(rec[te])
        lolo["rms_alarm"].append(rms_alarm(Xt[tr, 0], y[tr], Xt[te, 0], y[te], labels))
        for name, (mk, which) in models().items():
            X = pick(Xt, Xe, which)
            p = mk().fit(X[tr], y[tr]).predict(X[te])
            lolo[name].append(float(f1_score(y[te], p, average="macro")))
    out["lolo"] = {
        "rms_alarm": {
            "false_alarm_rate_mean": float(np.mean([f["false_alarm_rate"] for f in lolo["rms_alarm"]])),
            "detection_rate_mean": float(np.mean([f["detection_rate_mean"] for f in lolo["rms_alarm"]])),
            "folds": lolo["rms_alarm"],
        },
        **{n: {"macro_f1_mean": float(np.mean(v)), "macro_f1_by_test_load": v} for n, v in lolo.items() if n != "rms_alarm"},
    }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/baselines.json").write_text(json.dumps(out, indent=2))
    print(f"windows={len(y)}")
    print(f"RMS alarm  naive: FA={naive['rms_alarm']['false_alarm_rate']:.3f} det={naive['rms_alarm']['detection_rate_mean']:.3f}")
    print(f"RMS alarm  lolo : FA={out['lolo']['rms_alarm']['false_alarm_rate_mean']:.3f} det={out['lolo']['rms_alarm']['detection_rate_mean']:.3f}")
    for n in models():
        print(f"{n:13s} naive F1={naive[n]['macro_f1']:.3f}  lolo F1={out['lolo'][n]['macro_f1_mean']:.3f}  by load={[round(v,3) for v in out['lolo'][n]['macro_f1_by_test_load']]}")


if __name__ == "__main__":
    main()
