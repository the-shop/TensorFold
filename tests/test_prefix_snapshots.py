"""Prefix snapshots: host-side state survives the disk round trip; warming never crosses models."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402


class Layer:
    def __init__(self) -> None:
        self.state_a = mx.arange(6).reshape(2, 3)
        self.slots = [None, mx.ones((2,))]
        self.history = np.array([[11, 12]], dtype=np.int64)
        self.offset = 7


def test_numpy_state_round_trips(tmp_path):
    ps.save_snapshot(tmp_path, "model-a|mlx=1", [1, 2, 3], [Layer()])
    [(tokens, cache)] = list(ps.load_snapshots(tmp_path, "model-a|mlx=1"))
    layer = cache[0]
    assert tokens == [1, 2, 3] and layer.offset == 7
    assert isinstance(layer.history, np.ndarray) and layer.history.tolist() == [[11, 12]]
    assert layer.slots[0] is None and mx.array_equal(layer.slots[1], mx.ones((2,)))


def test_warming_only_takes_blocks_of_the_same_model(tmp_path):
    ps.save_snapshot(tmp_path, "/models/qwen|mlx=1", [5, 6, 7], [Layer()])
    ps.save_snapshot(tmp_path, "/models/nemotron|mlx=1", [8, 9], [Layer()])
    assert ps.blocks_to_warm(tmp_path, "/models/qwen|mlx=2") == [[5, 6, 7]]
    assert ps.blocks_to_warm(tmp_path, "/models/other|mlx=1") == []


def test_keeping_the_newest_counts_one_model_only(tmp_path):
    for i in range(3):
        ps.save_snapshot(tmp_path, "/models/qwen|mlx=1", [i, 1], [Layer()], keep=2)
    ps.save_snapshot(tmp_path, "/models/nemotron|mlx=1", [9], [Layer()], keep=1)
    models = sorted(ps.read_metadata(p)["model"] for p in tmp_path.glob("*.safetensors"))
    assert models == ["/models/nemotron|mlx=1", "/models/qwen|mlx=1", "/models/qwen|mlx=1"]


def test_a_drafter_slot_is_left_out_of_a_saved_snapshot(tmp_path):
    from mlx_lm.models.cache import KVCache

    from tensorfold.families.qwen3_5.dflash_head import DraftSlot

    kv = KVCache()
    kv.update_and_fetch(mx.ones((1, 2, 4, 4)), mx.ones((1, 2, 4, 4)))
    path = ps.save_snapshot(tmp_path, "model", [1, 2, 3, 4], [kv, DraftSlot(object())])   # its drafter: no array
    tokens, cache = ps.load_snapshot(path, "model")
    assert tokens == [1, 2, 3, 4] and [type(item) for item in cache] == [KVCache]
    assert bool(mx.array_equal(cache[0].state[0], kv.state[0]).item())


def test_a_snapshot_is_synced_before_its_rename_and_the_directory_after(tmp_path, monkeypatch):
    import os

    events = []
    real_fsync, real_rename = os.fsync, type(tmp_path).rename

    def fsync(fd):
        import stat

        mode = os.fstat(fd).st_mode
        events.append(("fsync", "dir" if stat.S_ISDIR(mode) else "file"))
        real_fsync(fd)

    def rename(self, target):
        events.append(("rename", "file"))
        return real_rename(self, target)

    monkeypatch.setattr(ps.os, "fsync", fsync)
    monkeypatch.setattr(ps, "full_sync", fsync)
    monkeypatch.setattr(type(tmp_path), "rename", rename)
    ps.save_snapshot(tmp_path, "model-a|mlx=1", [1, 2, 3], [Layer()])
    assert events == [("fsync", "file"), ("rename", "file"), ("fsync", "dir")]


def test_a_writer_killed_mid_write_leaves_only_a_partial_that_readers_skip_and_startup_removes(tmp_path):
    import subprocess
    import sys

    script = (
        "import os, signal, sys\n"
        "import mlx.core as mx\n"
        "from tensorfold.engine import prefix_snapshots as ps\n"
        "def killed(path, arrays, metadata=None):\n"
        "    with open(path, 'wb') as handle:\n"
        "        handle.write(b'\\0' * 4096)\n"
        "        handle.flush()\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "from mlx_lm.models.cache import KVCache\n"
        "kv = KVCache()\n"
        "kv.update_and_fetch(mx.ones((1, 2, 4, 4)), mx.ones((1, 2, 4, 4)))\n"
        "ps.save_snapshot(ps.Path(sys.argv[1]), 'model-a|mlx=1', [1, 2, 3], [kv])\n"
        "ps.mx.save_safetensors = killed\n"
        "ps.save_snapshot(ps.Path(sys.argv[1]), 'model-a|mlx=1', [1, 2, 3, 4], [kv])\n"
    )
    import os

    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    done = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env)
    assert done.returncode == -9
    partials = list(tmp_path.glob("*.partial.safetensors"))
    assert len(partials) == 1
    assert [tokens for tokens, _ in ps.load_snapshots(tmp_path, "model-a|mlx=1")] == [[1, 2, 3]]
    assert [tokens for _, tokens in ps.DiskBlocks(tmp_path, "model-a|mlx=1").blocks()] == [[1, 2, 3]]
    assert ps.remove_stale_partials(tmp_path) == 4096 and not partials[0].exists()


def test_a_file_is_flushed_with_f_fullfsync_where_the_platform_has_it(tmp_path, monkeypatch):
    import fcntl
    import os

    calls = []
    real = fcntl.fcntl
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, op, *a: calls.append(op) or (0 if op == 51 else real(fd, op, *a)))
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append("fsync"))
    path = tmp_path / "f"
    path.write_bytes(b"x")
    ps.sync_file(path)
    assert calls == [51]

    def unsupported(fd, op, *a):
        raise OSError(45, "Operation not supported")

    calls.clear()
    monkeypatch.setattr(fcntl, "fcntl", unsupported)
    ps.sync_file(path)
    assert calls == ["fsync"]
