"""Early warning: learn what healthy looks like, alarm when it stops looking like that.

Every detector here is fitted only on a bearing's own first hours of running,
which is what a plant has after installing or overhauling a machine: no fault
examples, just a stretch of normal operation. All detectors get the same alarm
rule so the comparison is about the score, not the tuning:

* the detector is fitted on the first two thirds of the reference period
* threshold = mean + 3 sigma of its score on the last third, which it has
  not seen. Setting it on the training snapshots instead makes the
  autoencoder alarm almost all the time, because it reconstructs data it was
  trained on better than any new healthy data.
* an alarm needs `persist` consecutive snapshots over the threshold, so a
  single noisy snapshot cannot raise one
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class Standardise:
    def fit(self, X):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


class RMSDetector:
    """The plant baseline: overall vibration level. Expects column 0 to be RMS."""

    name = "rms"

    def fit(self, X):
        return self

    def score(self, X):
        return X[:, 0]


class IForestDetector:
    name = "iforest"

    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, X):
        self.z = Standardise().fit(X)
        self.m = IsolationForest(n_estimators=300, random_state=self.seed).fit(self.z(X))
        return self

    def score(self, X):
        return -self.m.score_samples(self.z(X))


class AutoencoderDetector:
    """Small MLP autoencoder; the score is reconstruction error.

    With a few hundred reference snapshots and ~30 features, the bottleneck is
    kept narrow (4) so the network cannot simply copy its input.
    """

    name = "autoencoder"

    def __init__(self, seed=0, epochs=600, bottleneck=4):
        self.seed, self.epochs, self.bottleneck = seed, epochs, bottleneck

    def fit(self, X):
        import torch

        torch.manual_seed(self.seed)
        self.z = Standardise().fit(X)
        d = X.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, self.bottleneck), torch.nn.Tanh(),
            torch.nn.Linear(self.bottleneck, 16), torch.nn.Tanh(),
            torch.nn.Linear(16, d),
        )
        x = torch.tensor(self.z(X), dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = ((self.net(x) - x) ** 2).mean()
            loss.backward()
            opt.step()
        return self

    def score(self, X):
        import torch

        with torch.no_grad():
            x = torch.tensor(self.z(X), dtype=torch.float32)
            return ((self.net(x) - x) ** 2).mean(dim=1).numpy()


def threshold(ref_scores: np.ndarray) -> float:
    return float(ref_scores.mean() + 3 * ref_scores.std())


def persistent(over: np.ndarray, persist: int) -> np.ndarray:
    """True at i when over[i-persist+1 .. i] are all True."""
    run = np.zeros(len(over), dtype=int)
    for i, o in enumerate(over):
        run[i] = run[i - 1] + 1 if o and i else int(o)
    return run >= persist


def onsets(alarm: np.ndarray) -> np.ndarray:
    """Indices where the alarm switches on."""
    prev = np.concatenate([[False], alarm[:-1]])
    return np.flatnonzero(alarm & ~prev)


def latch(alarm: np.ndarray, min_share: float = 0.9) -> int | None:
    """First index where the alarm comes on and then stays on for at least
    min_share of the remaining snapshots. A real degradation alarm latches;
    a false one clears again."""
    tail_share = np.cumsum(alarm[::-1])[::-1] / np.arange(len(alarm), 0, -1)
    hits = np.flatnonzero(alarm & (tail_share >= min_share))
    return int(hits[0]) if len(hits) else None


def alarms(scores: np.ndarray, cal: np.ndarray, start: int, persist: int) -> tuple[np.ndarray, float]:
    """Alarm state per snapshot. The threshold comes from the calibration
    snapshots, which the detector was not fitted on; snapshots before `start`
    (the reference period) never alarm."""
    thr = threshold(scores[cal])
    a = persistent(scores > thr, persist)
    a[:start] = False
    return a, thr


def lead_time_h(hours: np.ndarray, alarm: np.ndarray) -> float:
    i = latch(alarm)
    return float(hours[-1] - hours[i]) if i is not None else 0.0


def false_alarms(alarm: np.ndarray, start: int, stop: int, step_h: float) -> dict:
    """Alarm onsets on a bearing that did not fail, between start and stop.

    Time is counted in snapshots watched, not calendar time: test 1 stops for
    six days in its first week."""
    a = alarm[start:stop]
    days = len(a) * step_h / 24
    n = len(onsets(a))
    return {"onsets": n, "days": days, "share_in_alarm": float(a.mean()) if len(a) else float("nan")}
