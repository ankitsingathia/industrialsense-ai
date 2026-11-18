"""Case Western Reserve University bearing data: manifest, download and loading.

Only the 12 kHz drive-end set is used. Each fault was seeded by EDM into an
SKF 6205-2RS bearing and recorded at four motor loads (0-3 hp). The manifest
below is the full grid: 1 healthy + 9 fault classes, 4 loads each.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

URL = "https://engineering.case.edu/sites/default/files/{id}.mat"
FS_FAULT = 12_000  # Hz, stated by CWRU for the 12k drive-end fault files

# file id per (class, load hp)
MANIFEST: dict[str, tuple[int, int, int, int]] = {
    "normal": (97, 98, 99, 100),
    "IR007": (105, 106, 107, 108),
    "B007": (118, 119, 120, 121),
    "OR007": (130, 131, 132, 133),
    "IR014": (169, 170, 171, 172),
    "B014": (185, 186, 187, 188),
    "OR014": (197, 198, 199, 200),
    "IR021": (209, 210, 211, 212),
    "B021": (222, 223, 224, 225),
    "OR021": (234, 235, 236, 237),
}

# SKF 6205-2RS defect frequencies as multiples of shaft speed (CWRU data sheet)
DEFECT_ORDERS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3568, "FTF": 0.39828}


def fault_type(label: str) -> str:
    """Collapse size into type: IR007 -> IR, normal -> N."""
    return "N" if label == "normal" else label.rstrip("0123456789")


@dataclass(frozen=True)
class Recording:
    file_id: int
    label: str
    load_hp: int
    rpm: float
    fs: int
    de: np.ndarray  # drive-end accelerometer, in g


def download(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for ids in MANIFEST.values():
        for fid in ids:
            p = dest / f"{fid}.mat"
            if not p.exists() or p.stat().st_size == 0:
                urllib.request.urlretrieve(URL.format(id=fid), p)
            out.append(p)
    return out


def _key(m: dict, suffix: str) -> str | None:
    hits = [k for k in m if k.endswith(suffix) and not k.startswith("__")]
    return hits[0] if hits else None


FS_NORMAL_RAW = 48_000  # measured, not stated: see DECISIONS.md, "Sample rate of the normal files"

# 98.mat and 99.mat carry no RPM variable; values from the CWRU data table.
RPM_FALLBACK = {98: 1772.0, 99: 1750.0}


def load(path: Path, label: str, load_hp: int, fs: int = FS_FAULT) -> Recording:
    """Load one file, returning the drive-end signal at 12 kHz.

    The healthy-baseline files are recorded at 48 kHz although they are listed
    with the 12 kHz data, so they are decimated by 4 here. Without this, a
    classifier can separate healthy from faulty by sample rate alone.
    """
    m = scipy.io.loadmat(path)
    fid = int(path.stem)
    # A few files carry extra variables from other recordings; prefer the one
    # whose number matches the file name.
    de_key = f"X{fid:03d}_DE_time" if f"X{fid:03d}_DE_time" in m else _key(m, "_DE_time")
    if de_key is None:
        raise ValueError(f"{path.name}: no drive-end channel")
    rpm_key = f"X{fid:03d}RPM" if f"X{fid:03d}RPM" in m else _key(m, "RPM")
    rpm = float(np.ravel(m[rpm_key])[0]) if rpm_key else RPM_FALLBACK.get(fid, float("nan"))
    x = np.ravel(m[de_key]).astype(np.float64)
    if label == "normal":
        x = scipy.signal.resample_poly(x, 1, FS_NORMAL_RAW // FS_FAULT)
    return Recording(fid, label, load_hp, rpm, fs, x)


def load_all(root: Path) -> list[Recording]:
    recs = []
    for label, ids in MANIFEST.items():
        for hp, fid in enumerate(ids):
            recs.append(load(root / f"{fid}.mat", label, hp))
    return recs


def windows(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Cut a 1-D signal into overlapping windows, shape (n, size)."""
    if len(x) < size:
        return np.empty((0, size))
    n = 1 + (len(x) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]
