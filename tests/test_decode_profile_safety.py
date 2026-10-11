"""The decode profiler never fails a reply or a model load (a write that fails, a C++ clock off Python's), caps its
file, and its report tolerates a cut last line, attributes prompt rows-path calls to the prompt and flags rounds whose
calls are out of layer order. No GPU needed."""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from tensorfold.streaming import profile

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _off():
    profile.configure(None)
    yield
    profile.configure(None)


def tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeStreamer:
    def __init__(self, hs=None):
        self.hs, self.on = hs, None

    def profile_on(self):
        self.on = True

    def profile_off(self):
        self.on = False

    def profile_drain(self):
        return [], 0


# -- MEDIUM-1: a write failure logs once and turns the profiler off -------------------------------------------------

def _unwritable(tmp_path: Path, how: str) -> Path:
    if how == "missing-dir":
        return tmp_path / "no" / "such" / "dir" / "p.jsonl"
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    return folder / "p.jsonl"


@pytest.mark.parametrize("how", ["missing-dir", "read-only-dir"])
def test_a_write_failure_turns_the_profiler_off_once(tmp_path, capsys, how):
    if how == "read-only-dir" and os.geteuid() == 0:
        pytest.skip("root writes anywhere")
    rec = profile.configure(_unwritable(tmp_path, how))
    streamer = FakeStreamer()
    rec.attach(streamer)
    assert streamer.on is True
    rec.flush(job="a")                                           # no exception
    rec.flush(job="b")
    assert profile.ACTIVE is None and rec.disabled and streamer.on is False
    assert capsys.readouterr().out.count("decode profile off") == 1
    rec.attach(FakeStreamer())                                   # a pool built later stays off
    assert len(rec.streamers) == 1
    if how == "read-only-dir":
        (tmp_path / "ro").chmod(0o700)


def test_a_full_disk_mid_session_turns_the_profiler_off(tmp_path, monkeypatch, capsys):
    path = tmp_path / "p.jsonl"
    rec = profile.configure(path)
    rec.flush(job="a")
    assert path.read_text().count("\n") == 1

    def full(text):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(rec, "_write", full)
    rec.flush(job="b")
    assert profile.ACTIVE is None and "No space left" in rec.disabled
    assert "decode profile off" in capsys.readouterr().out


@pytest.mark.parametrize("how", ["missing-dir", "flush-raises"])
def test_the_server_replies_200_when_the_profile_cannot_be_written(tmp_path, monkeypatch, capsys, how):
    from test_think_call import script_app, served_post

    rec = profile.configure(_unwritable(tmp_path, "missing-dir") if how == "missing-dir" else tmp_path / "p.jsonl")
    if how == "flush-raises":
        def broken(**meta):
            raise ValueError("a bug in the profiler")

        monkeypatch.setattr(rec, "flush", broken)
    app = script_app()
    try:
        for _ in range(2):
            status, body = served_post(app, {"messages": [{"role": "user", "content": "List files"}]})
            assert status == 200, body
            assert json.loads(body)["choices"][0]["message"]
    finally:
        app.close()
    assert profile.ACTIVE is None and rec.disabled
    assert capsys.readouterr().out.count("decode profile off") == 1


# -- LOW-4: a clock check that fails warns and turns the profiler off, never aborting the load ------------------------

def test_a_clock_off_pythons_turns_the_profiler_off_without_raising(capsys):
    class FarClock:
        gpu = None

        def clock_ns(self):
            return profile.now() + 10 * profile.CLOCK_TOLERANCE_NS

        def gpu_profile(self, on):
            self.gpu = on

    rec = profile.configure("/dev/null")
    hs = FarClock()
    streamer = FakeStreamer(hs)
    rec.attach(streamer)                                         # no AssertionError
    assert profile.ACTIVE is None and "steady_clock" in rec.disabled
    assert streamer.on is None and hs.gpu in (None, False)
    assert "decode profile off" in capsys.readouterr().out


def test_a_close_clock_attaches():
    class Clock:
        gpu = None

        def clock_ns(self):
            return profile.now()

        def gpu_profile(self, on):
            self.gpu = on

    rec = profile.configure("/dev/null")
    hs = Clock()
    streamer = FakeStreamer(hs)
    rec.attach(streamer)
    assert profile.ACTIVE is rec and streamer.on is True and hs.gpu is True and rec.clock_offset_ns is not None


# -- LOW-2: the file is capped and rotated -------------------------------------------------------------------------

def test_the_file_rotates_past_its_cap(tmp_path):
    path = tmp_path / "p.jsonl"
    rec = profile.configure(path, max_mb=2000 / 2**20)          # 2000 bytes
    for i in range(40):
        rec.flush(job=f"job-{i}", pad="x" * 200)
    assert path.stat().st_size <= 2000 and (tmp_path / "p.jsonl.1").stat().st_size <= 2000
    jobs = [json.loads(line)["job"] for f in ("p.jsonl.1", "p.jsonl")
            for line in (tmp_path / f).read_text().splitlines()]
    assert jobs == [f"job-{i}" for i in range(40 - len(jobs), 40)]   # the newest requests, whole and in order


def test_the_cap_comes_from_the_environment(tmp_path):
    rec = profile.from_env({"TENSORFOLD_DECODE_PROFILE": str(tmp_path / "p.jsonl"),
                            "TENSORFOLD_DECODE_PROFILE_MAX_MB": "3"})
    assert rec.max_bytes == 3 << 20
    rec = profile.from_env({"TENSORFOLD_DECODE_PROFILE": str(tmp_path / "p.jsonl")})
    assert rec.max_bytes == profile.MAX_MB << 20


# -- the report: a cut last line, prompt rows-path calls, call order --------------------------------------------------

def _call(kind: int, layer: int, t: int, misses: int = 0) -> dict:
    return {"kind": kind, "layer": layer, "rows": 1, "unique": 10, "misses": misses, "bytes": misses * 100,
            "signal": 0, "t_submit": t, "t_sig": t + 10, "t_read0": t + 12 if misses else 0,
            "t_read1": t + 20 if misses else 0, "t_ready": t + 30}


def _round(t0: int, layers: int = 2, misses: int = 1, order: list[int] | None = None) -> dict:
    calls = [_call(0, layer, t0 + 100 * (i + 1), misses) for i, layer in enumerate(order or range(layers))]
    return {"type": "round", "t0": t0, "runs": [{"layers": layers, "rows": 1, "q": [t0 + 1] * layers}],
            "calls": calls, "t_read": t0 + 100 * (layers + 1), "t_end": t0 + 100 * (layers + 1) + 50,
            "rows": 1, "keep": 1, "committed": 1, "depth": 0}


def _request(t0: int) -> list[dict]:
    chunk = {"type": "chunk", "t0": t0, "t_end": t0 + 900,
             "calls": [_call(1, 0, t0 + 10, 7), _call(1, 1, t0 + 50, 7), _call(0, 0, t0 + 400, 5),
                       _call(0, 1, t0 + 500, 4)]}
    header = {"type": "request", "n": 1, "rounds": 2, "chunks": 1, "orphan_calls": [_call(0, 0, 1, 2)], "events": []}
    return [header, chunk, _round(t0 + 1000), _round(t0 + 2000)]


def test_prompt_rows_calls_count_as_prompt_never_as_decode(tmp_path):
    report = tool("decode_profile_report")
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in _request(10_000)))
    req = report.load(path)[0]
    assert report.misses_split(req) == {"decode": 4, "prompt": 9, "outside": 2}
    view = report.prefill_view(req, 100)
    assert "| 0 | 0.0 | 2 | 2 | 23 | 9 |" in view                 # 2 window + 2 rows calls, 14 + 9 experts read
    assert "decode rounds 4 + prompt chunks 9 + outside any span 2 = 15" in view
    rows = report.summarize(req["rounds"], {1})["rows"]
    assert sum(r["f"]["misses"] for r in rows) == 4 and all(r["f"]["ok"] for r in rows)


def test_a_cut_last_line_is_skipped_with_a_warning(tmp_path, capsys):
    report = tool("decode_profile_report")
    path = tmp_path / "p.jsonl"
    lines = [json.dumps(r) for r in _request(10_000)]
    path.write_text("\n".join(lines) + "\n" + lines[2][:40])       # kill -9 during the next flush
    reqs = report.load(path)
    assert len(reqs) == 1 and len(reqs[0]["rounds"]) == 2
    assert "malformed" in capsys.readouterr().err
    assert report.main([str(path), "--attention", "1"]) == 0


def test_a_header_glued_to_a_cut_line_is_kept(tmp_path, capsys):
    report = tool("decode_profile_report")
    first, second = _request(10_000), _request(50_000)
    second[0]["n"] = 2
    text = "\n".join(json.dumps(r) for r in first[:3]) + "\n" + json.dumps(first[3])[:30]
    text += "\n".join(json.dumps(r) for r in second) + "\n"           # the restarted server appends after the cut
    path = tmp_path / "p.jsonl"
    path.write_text(text)
    reqs = report.load(path)
    assert [r["header"]["n"] for r in reqs] == [1, 2] and len(reqs[1]["rounds"]) == 2
    assert "kept the request record" in capsys.readouterr().err


def test_rotated_files_are_read_oldest_first(tmp_path):
    report = tool("decode_profile_report")
    old, new = _request(10_000), _request(50_000)
    new[0]["n"] = 2
    (tmp_path / "p.jsonl.1").write_text("".join(json.dumps(r) + "\n" for r in old))
    (tmp_path / "p.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new))
    assert [r["header"]["n"] for r in report.load(tmp_path / "p.jsonl.1", tmp_path / "p.jsonl")] == [1, 2]


def test_rounds_with_calls_out_of_layer_order_or_negative_buckets_are_flagged():
    report = tool("decode_profile_report")
    assert report.round_buckets(_round(0, 3), {1}, None, None)["f"]["ok"]
    swapped = report.round_buckets(_round(0, 3, order=[0, 2, 1]), {1}, None, None)["f"]
    assert not swapped["ok"] and "order" in swapped["why"]
    late = _round(0, 3)
    late["t_read"] = late["calls"][-1]["t_ready"] - 5                 # tokens read before the last layer's release
    got = report.round_buckets(late, {1}, None, None)["f"]
    assert not got["ok"] and "negative" in got["why"]


def test_mode0_names_its_counters_rows_path():
    health = tool("decode_health_delta")
    out = health.delta({"hits": 0, "misses": 0, "bytes_read": 0}, {"hits": 10, "misses": 20, "bytes_read": 5000},
                       tokens=4, rounds=4, ms_round=10.0, expert_bytes=100)
    assert out["rows_MiB"] == 2000 / 2**20 and out["window_GiB"] == 3000 / 2**30
    assert not any(k.startswith("decode") for k in out)
    assert "16 rows or fewer" in health.__doc__
