"""Stage 2: early warning on the IMS run-to-failure tests.

For every bearing in every test, each detector learns from that bearing's
first REF_HOURS of monitored running (fit on the first two thirds, threshold
on the last third) and then watches the rest of the test.

* bearings that failed: lead time, the hours between the alarm latching
  on (coming on and staying on) and the end of the test
* bearings that survived: alarm onsets per day. All four bearings sit on one
  shaft, so once a bearing is failing its neighbours' sensors pick it up too.
  Onsets are counted until the failing bearing's alarm latches, by the same
  detector; after that an alarm on a neighbour is reporting a real fault.

Usage: python scripts/run_early_warning.py   (needs scripts/ims_features.py first;
writes results/early_warning.json)
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isense import ims, warning  # noqa: E402

REF_HOURS = 48.0
PERSIST = 3  # snapshots, 30 minutes at the usual 10-minute interval
ADAPT_HOURS = 72.0  # time constant of the drift-following threshold (stage 2b), fixed before running


def detectors():
    return [warning.RMSDetector(), warning.IForestDetector(), warning.AutoencoderDetector()]


def load(test: str):
    d = np.load(ROOT / f"data/processed/ims_{test}.npz")
    return d["hours"], d["X"], list(d["names"])


def run_test(test: ims.Test, ref_hours: float, persist: int, keep_scores=False):
    hours, X, names = load(test.name)
    assert names[0] == "rms"
    step_h = float(np.median(np.diff(hours)))
    n_ref = int(round(ref_hours / step_h))  # monitored time, not calendar time
    fit, cal = np.arange(n_ref * 2 // 3), np.arange(n_ref * 2 // 3, n_ref)
    rows, scores = [], {}
    raw = {}  # detector -> bearing -> score
    for det in detectors():
        raw[det.name] = {b + 1: det.fit(X[fit, b, :]).score(X[:, b, :]) for b in range(X.shape[1])}

    alarm_sets = {}
    for name, by_b in raw.items():
        alarm_sets[name] = {}
        for b, s in by_b.items():
            alarm_sets[name][b], thr = warning.alarms(s, cal, n_ref, persist)
            if keep_scores:
                scores[(name, b)] = (s, thr)
    # stage 2b variants, built from the same scores
    alarm_sets["rms_adaptive"] = {
        b: warning.adaptive_alarms(s, cal, n_ref, persist, tau=ADAPT_HOURS / step_h) for b, s in raw["rms"].items()
    }
    alarm_sets["vote_rms_iforest"] = {b: alarm_sets["rms"][b] & alarm_sets["iforest"][b] for b in raw["rms"]}

    for name, al in alarm_sets.items():
        latches = [warning.latch(al[b]) for b in test.failed]
        stop = min((i for i in latches if i is not None), default=len(hours))
        for bearing, a in al.items():
            r = {"test": test.name, "bearing": bearing, "detector": name, "failed": bearing in test.failed}
            if r["failed"]:
                i = warning.latch(a)
                r.update(mode=test.failed[bearing], lead_time_h=warning.lead_time_h(hours, a),
                         latched_at_h=float(hours[i]) if i is not None else None,
                         test_hours=float(hours[-1]))
            else:
                r.update(warning.false_alarms(a, n_ref, stop, step_h))
            rows.append(r)
    return rows, hours, scores


def run(ref_hours=REF_HOURS, persist=PERSIST):
    rows = []
    for test in ims.TESTS.values():
        if (ROOT / f"data/processed/ims_{test.name}.npz").exists():
            rows += run_test(test, ref_hours, persist)[0]
    return rows


def summary(rows):
    out = {}
    for name in dict.fromkeys(r["detector"] for r in rows):
        f = [r for r in rows if r["detector"] == name and r["failed"]]
        h = [r for r in rows if r["detector"] == name and not r["failed"]]
        days = sum(r["days"] for r in h)
        n = sum(r["onsets"] for r in h)
        out[name] = {
            "lead_time_h": {f"{r['test']}/b{r['bearing']}": round(r["lead_time_h"], 1) for r in f},
            "false_alarms": n,
            "healthy_bearing_days": round(days, 1),
            "false_alarms_per_week": round(7 * n / days, 2) if days else None,
        }
    return out


def main():
    rows = run()
    out = {"ref_hours": REF_HOURS, "persist": PERSIST, "summary": summary(rows), "rows": rows}
    out["sensitivity"] = {
        f"ref{rh:g}h_persist{p}": summary(run(rh, p))
        for rh in (24, 48, 96) for p in (1, 3, 6) if (rh, p) != (REF_HOURS, PERSIST)
    }
    global ADAPT_HOURS
    fixed, out["adapt_sweep"] = ADAPT_HOURS, {}
    for tau in (24.0, 72.0, 168.0):
        ADAPT_HOURS = tau
        out["adapt_sweep"][f"tau{tau:g}h"] = summary(run())["rms_adaptive"]
    ADAPT_HOURS = fixed
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/early_warning.json").write_text(json.dumps(out, indent=2))
    for name, s in out["summary"].items():
        print(f"{name:12s} lead(h)={s['lead_time_h']}  FA={s['false_alarms']} over "
              f"{s['healthy_bearing_days']} bearing-days = {s['false_alarms_per_week']}/week")


if __name__ == "__main__":
    main()
