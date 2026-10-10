"""A failed or cut-short snapshot write leaves no ``.partial.safetensors``: errors and startup both remove them."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402
from tensorfold.server.checkpoints import CheckpointEntry, spill_conversation  # noqa: E402

MODEL = "/models/qwen|mlx=1"


class Layer:
    def __init__(self) -> None:
        self.state = mx.arange(64).reshape(8, 8)
        self.offset = 8


def _disk_full(monkeypatch):
    """``mx.save_safetensors`` that writes some bytes, then fails as a full disk does."""

    def fail(path, arrays, metadata=None):
        with open(path, "wb") as handle:
            handle.write(b"\0" * 4096)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ps.mx, "save_safetensors", fail)


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_a_failed_write_removes_its_partial_and_raises(tmp_path, monkeypatch):
    _disk_full(monkeypatch)
    with pytest.raises(OSError):
        ps.save_snapshot(tmp_path, MODEL, [1, 2, 3], [Layer()])
    assert list(tmp_path.iterdir()) == []


def test_a_failed_spill_leaves_the_directory_as_it_was(tmp_path, monkeypatch):
    ps.save_snapshot(tmp_path, MODEL, [1, 2], [Layer()])
    before = sorted(p.name for p in tmp_path.iterdir())
    _disk_full(monkeypatch)
    entry = CheckpointEntry(tokens=[1, 2, 3, 4], cache=[Layer()], last_prompt=[1, 2, 3, 4], nbytes=512)
    assert spill_conversation(entry, tmp_path, MODEL, limit_bytes=1 << 30) is False
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_each_process_writes_its_own_partial(tmp_path, monkeypatch):
    written = []
    save = ps.mx.save_safetensors

    def record(path, arrays, metadata=None):
        written.append(os.path.basename(path))
        save(path, arrays, metadata=metadata)

    monkeypatch.setattr(ps.mx, "save_safetensors", record)
    target = ps.save_snapshot(tmp_path, MODEL, [4, 5], [Layer()])
    other = ps.save_snapshot(tmp_path, MODEL, [4, 6], [Layer()])
    assert [name.split("-")[0] for name in written] == [f"{target.stem}.{os.getpid()}", f"{other.stem}.{os.getpid()}"]
    assert all(name.endswith(".partial.safetensors") for name in written)
    assert len({name.split(".")[1] for name in written}) == 2           # each write its own partial, threads too
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([target.name, other.name])


def test_a_threads_partial_counts_as_its_process(tmp_path):
    dead = tmp_path / f"{'a' * 32}.{_dead_pid()}-7.partial.safetensors"
    live = tmp_path / f"{'b' * 32}.{os.getpid()}-3.partial.safetensors"
    for path in (dead, live):
        path.write_bytes(b"\0" * 100)
    assert ps.remove_stale_partials(tmp_path) == 100
    assert [p.name for p in tmp_path.iterdir()] == [live.name]


def test_startup_removes_partials_of_processes_that_are_gone(tmp_path):
    kept = ps.save_snapshot(tmp_path, MODEL, [7, 8], [Layer()])
    dead = tmp_path / f"{'a' * 32}.{_dead_pid()}.partial.safetensors"
    live = tmp_path / f"{'b' * 32}.{os.getpid()}.partial.safetensors"
    old = tmp_path / f"{'c' * 32}.partial.safetensors"           # 0.1.0's name, no pid: removed once it is stale
    fresh = tmp_path / f"{'d' * 32}.partial.safetensors"
    for path in (dead, live, old, fresh):
        path.write_bytes(b"\0" * 1000)
    stale = time.time() - ps.UNNAMED_PARTIAL_SECONDS - 60
    os.utime(old, (stale, stale))
    assert ps.remove_stale_partials(tmp_path) == 2000
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([kept.name, live.name, fresh.name])


def test_removing_partials_tolerates_a_missing_directory(tmp_path):
    assert ps.remove_stale_partials(tmp_path / "absent") == 0


def test_a_starting_server_clears_both_snapshot_directories(tmp_path, capsys):
    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.scheduler import Scheduler
    from tests.lane_fakes import FakeEngine

    blocks, sessions = tmp_path / "prefix-snapshots", tmp_path / "session-snapshots"
    pid = _dead_pid()
    for directory in (blocks, sessions):
        directory.mkdir()
        (directory / f"{'e' * 32}.{pid}.partial.safetensors").write_bytes(b"\0" * 2048)
    Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=CheckpointStore(2, copier=lambda c: c),
              snapshot_dir=blocks, session_dir=sessions, model_id=MODEL)
    assert list(blocks.iterdir()) == [] and list(sessions.iterdir()) == []
    assert capsys.readouterr().out.count("removed 1 unfinished snapshot write(s), 0.0 MiB") == 2


def test_a_snapshot_pruned_between_its_check_and_touch_is_written_again(tmp_path, monkeypatch):
    target = ps.save_snapshot(tmp_path, MODEL, [3, 4], [Layer()])
    real = ps.os.utime

    def pruned(path, *args, **kwargs):
        if str(path) == str(target):
            target.unlink(missing_ok=True)            # another thread's prune wins the race
            raise FileNotFoundError(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(ps.os, "utime", pruned)
    assert ps.save_snapshot(tmp_path, MODEL, [3, 4], [Layer()]) == target and target.exists()
    target.unlink()
    payload = ps.encode_snapshot(MODEL, [3, 4], [Layer()])
    target.write_bytes(b"x")
    assert ps.write_encoded(tmp_path, MODEL, [3, 4], payload) == target
    assert ps.verify_snapshot(target) == "ok"


def test_one_failed_conversation_save_does_not_cost_the_others(tmp_path, monkeypatch, capsys):
    from tensorfold.server.checkpoints import CheckpointStore, save_conversations

    store = CheckpointStore(4, copier=lambda c: c, sizer=lambda c: 1)
    store.insert([1, 2, 3], [Layer()], last_prompt=[1, 2, 3, 4])
    store.insert([5, 6], [Layer()], last_prompt=[5, 6, 7])
    real = ps.save_snapshot

    def flaky(directory, model_id, tokens, cache, **kwargs):
        if list(tokens) == [1, 2, 3]:
            raise OSError(5, "Input/output error")
        return real(directory, model_id, tokens, cache, **kwargs)

    monkeypatch.setattr(ps, "save_snapshot", flaky)
    assert save_conversations(store, tmp_path, MODEL, keep=2) == 1
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert "conversation save failed tokens=3" in capsys.readouterr().out


def test_a_day_old_partial_goes_even_when_its_pid_is_running(tmp_path):
    reused = tmp_path / f"{'a' * 32}.1-0.partial.safetensors"                 # pid 1: running, never ours
    live = tmp_path / f"{'b' * 32}.{os.getpid()}-9.partial.safetensors"
    for path in (reused, live):
        path.write_bytes(b"\0" * 10)
    day = time.time() - ps.STALE_PARTIAL_SECONDS - 60
    os.utime(reused, (day, day))
    assert ps.remove_stale_partials(tmp_path) == 10
    assert [p.name for p in tmp_path.iterdir()] == [live.name]


def test_an_empty_partial_is_removed_and_counted(tmp_path, capsys):
    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.scheduler import Scheduler
    from tests.lane_fakes import FakeEngine

    sessions = tmp_path / "session-snapshots"
    sessions.mkdir()
    empty = sessions / f"{'e' * 32}.{_dead_pid()}-0.partial.safetensors"
    empty.write_bytes(b"")
    Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=CheckpointStore(2, copier=lambda c: c),
              session_dir=sessions, model_id=MODEL)
    assert not empty.exists() and "removed 1 unfinished snapshot write(s), 0.0 MiB" in capsys.readouterr().out


def test_sizes_under_a_gib_are_logged_in_mib():
    from tensorfold.server.checkpoints import _size

    assert _size(5 * 1024**2) == "5.0 MiB" and _size(3 * 1024**3) == "3.00 GiB"
