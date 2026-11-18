"""A small 1D CNN on raw vibration windows, sized for a microcontroller.

The layout follows WDCNN (Zhang et al., Sensors 2017): a wide first kernel
with a large stride acts as a learned filter bank and cuts the sequence
length early, so the later layers are cheap. About 22k parameters, so the
int8 weights fit in ~22 KB of flash.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

SIZE = 2048


class WDCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()

        def block(cin, cout, k, stride=1):
            return [nn.Conv1d(cin, cout, k, stride=stride), nn.BatchNorm1d(cout), nn.ReLU()]

        self.features = nn.Sequential(
            *block(1, 16, 64, stride=8), nn.MaxPool1d(2),   # 2048 -> 249 -> 124
            *block(16, 32, 3), nn.MaxPool1d(2),             # 122 -> 61
            *block(32, 64, 3), nn.MaxPool1d(2),             # 59 -> 29
            *block(64, 64, 3),                              # 27
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def prepare(w: np.ndarray, scale: float) -> torch.Tensor:
    """(n, SIZE) windows -> (n, 1, SIZE) float tensor, mean removed, divided by a
    scale learned on the training side."""
    x = (w - w.mean(axis=1, keepdims=True)) / scale
    return torch.tensor(x[:, None, :], dtype=torch.float32)


def train(w: np.ndarray, y: np.ndarray, seed: int = 0, epochs: int = 30, batch: int = 64) -> tuple[WDCNN, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scale = float(w.std())
    x, t = prepare(w, scale), torch.tensor(y, dtype=torch.long)
    model = WDCNN(int(y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)), max(1, len(x) // batch)):
            opt.zero_grad()
            loss_fn(model(x[idx]), t[idx]).backward()
            opt.step()
        sched.step()
    return model.eval(), scale


@torch.no_grad()
def predict(model: WDCNN, w: np.ndarray, scale: float) -> np.ndarray:
    return model(prepare(w, scale)).argmax(1).numpy()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
