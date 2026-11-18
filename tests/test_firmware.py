"""The C kernel must compute exactly what the NumPy int8 reference computes.

Uses a randomly initialised network, so it needs no dataset and runs in CI
wherever a C compiler is installed.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from isense import cnn, int8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CC = os.environ.get("CC", "gcc")
needs_cc = pytest.mark.skipif(shutil.which(CC) is None, reason="no C compiler")


def _random_model(seed=0):
    torch.manual_seed(seed)
    m = cnn.WDCNN()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    return m.eval()


def test_int8_reference_tracks_the_float_network():
    rng = np.random.default_rng(0)
    m = _random_model()
    x = rng.standard_normal((64, 1, cnn.SIZE)).astype(np.float32)
    q = int8.quantise(m, x[:32])
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    got = int8.logits(x, q)
    assert np.corrcoef(ref.ravel(), got.ravel())[0, 1] > 0.99


@needs_cc
def test_c_kernel_is_bit_exact_with_the_numpy_reference(tmp_path):
    from run_firmware import CFLAGS, write_header

    rng = np.random.default_rng(1)
    m = _random_model(1)
    x = rng.standard_normal((40, 1, cnn.SIZE)).astype(np.float32)
    q = int8.quantise(m, x[:20])
    for f in ("wdcnn.c", "wdcnn.h", "host_main.c"):
        shutil.copy(ROOT / "firmware" / f, tmp_path)
    write_header(q, tmp_path / "wdcnn_weights.h")
    exe = tmp_path / "host"
    subprocess.run([CC, *CFLAGS, str(tmp_path / "wdcnn.c"), str(tmp_path / "host_main.c"), "-lm", "-o", str(exe)],
                   check=True)
    with open(tmp_path / "x.bin", "wb") as f:
        np.int32(len(x)).tofile(f)
        x.tofile(f)
    subprocess.run([str(exe), str(tmp_path / "x.bin"), str(tmp_path / "y.bin")], check=True)
    c = np.fromfile(tmp_path / "y.bin", dtype=np.float32).reshape(len(x), -1)
    assert np.array_equal(c.view(np.uint32), int8.logits(x, q).view(np.uint32))
