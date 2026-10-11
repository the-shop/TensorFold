"""The decode profiler's per-call records from both serve loops (the extension's and Python's): one record a call, its
stamps in order, misses and bytes equal to the counters, and the pool's state the same as an unprofiled run."""

from __future__ import annotations

import random

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.streaming import pool, profile  # noqa: E402

from test_pool_native import EXPERTS, TOPK, call, make, snapshot, write_checkpoint  # noqa: E402
from test_pool_native import hs  # noqa: E402,F401  # the module-scoped extension fixture


@pytest.fixture(autouse=True)
def _off():
    profile.configure(None)
    yield
    profile.configure(None)


def calls(seed: int = 4) -> list[tuple[str, int, list[int]]]:
    rng = random.Random(seed)
    out = [("window", 0, rng.sample(range(EXPERTS), 5)), ("window", 1, rng.sample(range(EXPERTS), 6))]
    for _ in range(12):
        out.append(("rows", rng.randrange(3), [rng.randrange(EXPERTS) for _ in range(TOPK * rng.randint(1, 3))]))
    return out


def run(hs, folder, native: bool, profiled: bool, tmp_path):
    if profiled:
        profile.configure(tmp_path / "p.jsonl")
    s = make(hs, folder, slots=2 * EXPERTS + 3, native=native)
    for kind, layer, ids in calls():
        call(hs, s, kind, layer, ids)
    state = snapshot(hs, s)
    got = profile.ACTIVE.drain() if profiled else None
    s.close()
    profile.configure(None)
    return state, got


@pytest.mark.parametrize("native", [True, False])
def test_records_match_the_counters_and_leave_the_pool_unchanged(hs, tmp_path, native):  # noqa: F811
    folder = tmp_path / "ckpt"
    write_checkpoint(folder)
    plain, _ = run(hs, folder, native, False, tmp_path)
    state, (records, gpu, lost) = run(hs, folder, native, True, tmp_path)
    assert state == plain                                   # slots, table, LRU, free list, window and counters
    assert lost == {"calls_dropped": 0, "gpu_dropped": 0}
    assert gpu == []                                        # no GPU marks here: the host signals for the test
    assert [(r["kind"], r["layer"]) for r in records] == [(pool._KINDS[k], l) for k, l, _ in calls()]
    hits, misses, nbytes = state["counters"]
    rows = [r for r in records if r["kind"] == 0]
    assert sum(r["misses"] for r in rows) == misses
    assert sum(r["unique"] for r in rows) == hits + misses
    assert sum(r["bytes"] for r in records) == nbytes       # window fills count here, never as decode misses (R2)
    for r in records:
        assert 0 < r["t_submit"] <= r["t_sig"] <= r["t_ready"]
        if r["misses"]:
            assert r["t_sig"] <= r["t_read0"] <= r["t_read1"] <= r["t_ready"]
        else:
            assert r["t_read0"] == r["t_read1"] == 0


@pytest.mark.parametrize("native", [True, False])
def test_off_records_nothing(hs, tmp_path, native):  # noqa: F811
    folder = tmp_path / "ckpt"
    write_checkpoint(folder)
    s = make(hs, folder, slots=2 * EXPERTS + 3, native=native)
    call(hs, s, "rows", 0, [1, 2, 3, 4])
    assert s.profile_drain() == ([], 0)
    s.close()
