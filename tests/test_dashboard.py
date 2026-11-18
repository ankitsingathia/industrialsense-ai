import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from api import app  # noqa: E402

client = TestClient(app)


def test_lists_the_three_ims_tests():
    r = client.get("/api/tests")
    assert r.status_code == 200
    assert [t["test"] for t in r.json()] == ["test1", "test2", "test3"]


def test_test2_replay_matches_the_stage_2b_result():
    d = client.get("/api/tests/test2").json()
    assert len(d["hours"]) == 982  # 984 snapshots minus the 2 taken after the rig stopped
    b1 = d["bearings"][0]
    assert b1["failed"] and b1["latched_h"] == pytest.approx(89.0)
    assert all(len(b["rms"]) == len(d["hours"]) == len(b["alarm"]) for b in d["bearings"])
    first = next(e for e in b1["events"] if e["h"] == pytest.approx(89.0))
    assert first["why"][0]["feature"].startswith("BPFO")  # the outer-race line, on the outer-race failure


def test_unknown_test_is_a_404():
    assert client.get("/api/tests/test9").status_code == 404
