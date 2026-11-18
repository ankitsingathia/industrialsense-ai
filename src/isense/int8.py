"""Integer-only inference for the WDCNN, the reference for the C kernel.

ONNX Runtime's int8 model runs on a PC. A Cortex-M4 has no ONNX Runtime, so
the network is quantised here by hand, with exactly the arithmetic the C code
in firmware/ performs. The C kernel is then tested against this file, output
for output.

Scheme (the common TFLite-style one):
* batch norm folded into each conv
* weights: int8, symmetric, one scale per output channel
* activations: int8, symmetric, one scale per layer, from the largest value
  seen on calibration windows
* accumulate in int32; requantise with a float32 multiplier, then round
  half up and clamp. The M4 has a single-precision FPU, so a float multiply
  per output is cheap and avoids fixed-point multiplier tricks.

Rounding is floor(v + 0.5) everywhere, in float32, so NumPy and C agree to
the bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cnn import WDCNN

F32 = np.float32


def _round(v: np.ndarray) -> np.ndarray:
    return np.floor(v.astype(F32) + F32(0.5))


@dataclass
class Conv:
    w: np.ndarray  # int8 (cout, cin, k)
    b: np.ndarray  # int32 (cout,)
    m: np.ndarray  # float32 (cout,) requantisation multiplier
    stride: int
    pool: bool


@dataclass
class QModel:
    in_scale: F32  # float input -> int8: q = round(x * in_scale)
    convs: list
    head_w: np.ndarray  # int8 (n_classes, 64)
    head_m: np.ndarray  # float32 (n_classes,): logit = acc * head_m + head_b
    head_b: np.ndarray  # float32 (n_classes,)
    pool_len: int


def _fold(conv: torch.nn.Conv1d, bn: torch.nn.BatchNorm1d):
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().numpy()
    w = conv.weight.detach().numpy() * g[:, None, None]
    b = (conv.bias.detach().numpy() - bn.running_mean.detach().numpy()) * g + bn.bias.detach().numpy()
    return w.astype(np.float64), b.astype(np.float64)


def _float_layers(model: WDCNN):
    f = list(model.features)
    convs = [m for m in f if isinstance(m, torch.nn.Conv1d)]
    bns = [m for m in f if isinstance(m, torch.nn.BatchNorm1d)]
    return [(*_fold(c, b), c.stride[0], i < 3) for i, (c, b) in enumerate(zip(convs, bns))]


def _conv_f(x, w, b, stride):
    """float conv1d, x (n, cin, L) -> (n, cout, L')."""
    k = w.shape[2]
    lout = (x.shape[2] - k) // stride + 1
    idx = np.arange(k)[None, :] + stride * np.arange(lout)[:, None]  # (lout, k)
    patches = x[:, :, idx]  # (n, cin, lout, k)
    return np.einsum("nclk,ock->nol", patches, w, optimize=True) + b[None, :, None]


def _pool(x):
    lout = x.shape[2] // 2
    return np.maximum(x[:, :, 0 : 2 * lout : 2], x[:, :, 1 : 2 * lout : 2])


def quantise(model: WDCNN, calib: np.ndarray) -> QModel:
    """calib: (n, 1, 2048) float windows, already mean-removed and scaled."""
    layers = _float_layers(model)
    # activation ranges from the float (BN-folded) network
    a = calib.astype(np.float64)
    ranges = [np.abs(a).max()]
    for w, b, stride, pool in layers:
        a = np.maximum(_conv_f(a, w, b, stride), 0)
        if pool:
            a = _pool(a)
        ranges.append(a.max())
    s_act = [r / 127.0 for r in ranges]  # real value of one int8 step, per layer boundary

    convs = []
    for i, (w, b, stride, pool) in enumerate(layers):
        s_w = np.abs(w).max(axis=(1, 2)) / 127.0
        s_w[s_w == 0] = 1.0
        wq = np.clip(np.round(w / s_w[:, None, None]), -127, 127).astype(np.int8)
        bq = np.round(b / (s_act[i] * s_w)).astype(np.int32)
        m = (s_act[i] * s_w / s_act[i + 1]).astype(F32)
        convs.append(Conv(wq, bq, m, stride, pool))

    hw = model.head.weight.detach().numpy().astype(np.float64)
    hb = model.head.bias.detach().numpy()
    pool_len = 27
    s_h = np.abs(hw).max(axis=1) / 127.0
    hq = np.clip(np.round(hw / s_h[:, None]), -127, 127).astype(np.int8)
    head_m = (s_act[-1] * s_h / pool_len).astype(F32)
    return QModel(F32(1.0 / s_act[0]), convs, hq, head_m, hb.astype(F32), pool_len)


def quantise_input(x: np.ndarray, q: QModel) -> np.ndarray:
    return np.clip(_round(x.astype(F32) * q.in_scale), -127, 127).astype(np.int8)


def logits(x: np.ndarray, q: QModel) -> np.ndarray:
    """x: (n, 1, 2048) float32 -> (n, n_classes) float32, integer arithmetic inside."""
    a = quantise_input(x, q).astype(np.int64)
    for c in q.convs:
        k = c.w.shape[2]
        lout = (a.shape[2] - k) // c.stride + 1
        idx = np.arange(k)[None, :] + c.stride * np.arange(lout)[:, None]
        acc = np.einsum("nclk,ock->nol", a[:, :, idx], c.w.astype(np.int64), optimize=True) + c.b[None, :, None]
        acc = np.maximum(acc, 0).astype(F32)  # ReLU on the int32 accumulator
        a = np.minimum(_round(acc * c.m[None, :, None]), 127).astype(np.int64)
        if c.pool:
            a = _pool(a)
    s = a.sum(axis=2)  # (n, 64) int, average pool folded into head_m
    acc = (s @ q.head_w.astype(np.int64).T).astype(F32)
    return acc * q.head_m[None, :] + q.head_b[None, :]
