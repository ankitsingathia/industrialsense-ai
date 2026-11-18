"""Stage 4b: ONNX export and int8 quantisation of the stage 4a CNN.

For each leave-one-load-out fold (seed-0 model from scripts/run_cnn.py):
* export to ONNX (fp32)
* quantise to int8 with ONNX Runtime static quantisation (QDQ, per-channel
  weights), calibrated on 256 windows from the training loads only
* measure macro-F1 on the held-out load, file size, and single-window latency
  on this laptop's CPU with one thread

Usage: python scripts/run_edge.py   (needs run_cnn.py first; writes results/edge.json)
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from isense import cnn  # noqa: E402
from run_cnn import build  # noqa: E402

CALIB, RUNS = 256, 2000


class Reader(CalibrationDataReader):
    def __init__(self, x: np.ndarray):
        self.it = iter([{"x": x[i : i + 1]} for i in range(len(x))])

    def get_next(self):
        return next(self.it, None)


def session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, x: np.ndarray) -> np.ndarray:
    return np.concatenate([sess.run(None, {"x": x[i : i + 256]})[0] for i in range(0, len(x), 256)]).argmax(1)


def latency_ms(sess, x1: np.ndarray) -> float:
    for _ in range(50):
        sess.run(None, {"x": x1})
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"x": x1})
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def main():
    W, y, load, _ = build()
    out_dir = ROOT / "models"
    rng = np.random.default_rng(0)
    folds = []
    for hp in range(4):
        ck = torch.load(out_dir / f"wdcnn_test{hp}hp.pt")
        model = cnn.WDCNN()
        model.load_state_dict(ck["state"])
        model.eval()
        tr, te = load != hp, load == hp
        x_tr = cnn.prepare(W[tr], ck["scale"]).numpy()
        x_te = cnn.prepare(W[te], ck["scale"]).numpy()

        fp32 = out_dir / f"wdcnn_test{hp}hp.onnx"
        torch.onnx.export(model, torch.from_numpy(x_te[:1]), fp32, input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
        int8 = out_dir / f"wdcnn_test{hp}hp.int8.onnx"
        calib = x_tr[rng.choice(len(x_tr), CALIB, replace=False)]
        quantize_static(str(fp32), str(int8), Reader(calib), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)

        row = {"test_load_hp": hp}
        with torch.no_grad():
            p_torch = model(torch.from_numpy(x_te)).argmax(1).numpy()
        for name, path in (("fp32", fp32), ("int8", int8)):
            s = session(path)
            p = run(s, x_te)
            row[name] = {
                "macro_f1": float(f1_score(y[te], p, average="macro")),
                "agrees_with_pytorch": float((p == p_torch).mean()),
                "file_kb": round(path.stat().st_size / 1024, 1),
                "latency_ms": round(latency_ms(s, x_te[:1]), 3),
            }
        folds.append(row)
        print(hp, row)

    def mean(k, m):
        return float(np.mean([f[k][m] for f in folds]))

    out = {
        "cpu": platform.processor(),
        "threads": 1,
        "summary": {k: {m: round(mean(k, m), 4) for m in ("macro_f1", "file_kb", "latency_ms")} for k in ("fp32", "int8")},
        "folds": folds,
    }
    (ROOT / "results/edge.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
