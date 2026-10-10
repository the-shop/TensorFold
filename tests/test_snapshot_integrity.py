"""A stored prefix is loaded only while its bytes match the checksum sealed into it: corruption is refused and deleted."""

from __future__ import annotations

import json
import struct

import pytest

mx = pytest.importorskip("mlx.core")

from tensorfold.engine import prefix_snapshots as ps  # noqa: E402

MODEL = "/models/qwen|mlx=1"


class Layer:
    def __init__(self) -> None:
        self.keys = mx.arange(4096, dtype=mx.float32).reshape(64, 64)
        self.offset = 64


def _saved(tmp_path, tokens=(1, 2, 3)):
    return ps.save_snapshot(tmp_path, MODEL, list(tokens), [Layer()])


def _data_start(path):
    with open(path, "rb") as handle:
        return 8 + struct.unpack("<Q", handle.read(8))[0]


def _flip(path, at):
    raw = bytearray(path.read_bytes())
    raw[at] ^= 0x40
    path.write_bytes(bytes(raw))


def test_a_saved_snapshot_carries_a_sealed_checksum_that_verifies(tmp_path):
    path = _saved(tmp_path)
    value = ps.read_metadata(path)[ps.CHECKSUM]
    assert value.startswith("crc32:") and "?" not in value
    assert ps.verify_snapshot(path) == "ok"
    tokens, cache = ps.load_snapshot(path, MODEL)
    assert tokens == [1, 2, 3] and bool(mx.array_equal(cache[0].keys, Layer().keys).item())


def test_a_flipped_tensor_byte_is_refused_and_the_file_deleted(tmp_path, capsys):
    path = _saved(tmp_path)
    _flip(path, _data_start(path) + 5000)                  # header intact: mx.load alone would accept it
    assert ps.verify_snapshot(path) == "corrupt"
    assert ps.load_snapshot(path, MODEL) is None
    assert not path.exists()
    assert "do not match its checksum; deleted" in capsys.readouterr().out


def test_a_truncated_snapshot_is_refused_and_deleted(tmp_path):
    path = _saved(tmp_path)
    raw = path.read_bytes()
    path.write_bytes(raw[:len(raw) - 100])
    assert ps.load_snapshot(path, MODEL) is None and not path.exists()


def test_edited_token_metadata_is_refused(tmp_path):
    path = _saved(tmp_path)
    raw = bytearray(path.read_bytes())
    size = struct.unpack("<Q", raw[:8])[0]
    header = raw[8:8 + size].decode()
    edited = header.replace('"[1, 2, 3]"', '"[1, 2, 4]"')
    assert edited != header and len(edited) == len(header)
    path.write_bytes(bytes(raw[:8]) + edited.encode() + bytes(raw[8 + size:]))
    assert ps.load_snapshot(path, MODEL) is None and not path.exists()


def test_garbage_is_refused_and_deleted(tmp_path):
    path = tmp_path / f"{'f' * 32}.safetensors"
    path.write_bytes(b"\x10\x00\x00\x00\x00\x00\x00\x00not json at all!" + b"\0" * 64)
    assert ps.load_snapshot(path, MODEL) is None and not path.exists()


def test_a_snapshot_without_a_checksum_is_never_loaded_nor_deleted(tmp_path, capsys):
    path = tmp_path / f"{ps.snapshot_key(MODEL, [1, 2])}.safetensors"
    arrays, meta = ps.snapshot_arrays(MODEL, [1, 2], [Layer()])
    del meta[ps.CHECKSUM]                                  # as 0.1.0 wrote them
    mx.save_safetensors(str(path), arrays, metadata=meta)
    assert ps.verify_snapshot(path) == "unsealed"
    assert ps.load_snapshot(path, MODEL) is None and path.exists()
    assert "no checksum" in capsys.readouterr().out
    assert ps.DiskBlocks(tmp_path, MODEL).blocks() == []   # not even offered to a prompt
    assert list(ps.load_snapshots(tmp_path, MODEL)) == []


def test_a_corrupt_block_found_on_demand_costs_a_prefill_and_is_gone_after(tmp_path):
    good = _saved(tmp_path, (1, 2))
    bad = _saved(tmp_path, (1, 2, 3))
    _flip(bad, _data_start(bad) + 64)
    blocks = ps.DiskBlocks(tmp_path, MODEL)
    path, tokens = blocks.best([1, 2, 3, 4], 0)
    assert tokens == [1, 2, 3] and ps.load_snapshot(path, MODEL) is None
    assert blocks.best([1, 2, 3, 4], 0) == (good, [1, 2])      # the next prompt resumes from the good one


def test_the_seal_rejects_a_header_without_one_unsealed_marker(tmp_path):
    path = tmp_path / "x.safetensors"
    mx.save_safetensors(str(path), {"a": mx.ones((2,))}, metadata={"format": "1"})
    with open(path, "r+b") as handle, pytest.raises(ValueError):
        ps.seal(handle)


def test_the_checksum_covers_every_metadata_key_but_itself(tmp_path):
    path = _saved(tmp_path)
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    meta = header["__metadata__"]
    assert set(meta) == {"format", "model", "tokens", "layers", "saved", ps.CHECKSUM}


def test_an_unreadable_snapshot_is_kept_not_deleted(tmp_path, capsys):
    import os

    path = _saved(tmp_path)
    os.chmod(path, 0)
    try:
        if os.access(path, os.R_OK):
            pytest.skip("running with privileges that read any file")
        with pytest.raises(PermissionError):
            ps.verify_snapshot(path)
        assert ps.load_snapshot(path, MODEL) is None
        assert path.exists() and "left in place" in capsys.readouterr().out
    finally:
        os.chmod(path, 0o644)
    assert ps.load_snapshot(path, MODEL) is not None            # readable again: loads


def test_an_io_error_mid_read_keeps_the_file(tmp_path, monkeypatch):
    import builtins
    import errno

    path = _saved(tmp_path)
    real_open = builtins.open

    class Failing:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def read(self, n=-1):
            raise OSError(errno.EIO, "Input/output error")

        def seek(self, at):
            return self.handle.seek(at)

    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: Failing(real_open(p, *a, **k)) if str(p) == str(path)
                        else real_open(p, *a, **k))
    assert ps.load_snapshot(path, MODEL) is None
    monkeypatch.undo()
    assert path.exists() and ps.verify_snapshot(path) == "ok"


@pytest.mark.parametrize("damage", ["junk", "cut-1000", "cut-tail", "huge-length", "not-a-table"])
def test_a_broken_file_is_deleted_and_logged_when_the_index_meets_it(tmp_path, capsys, damage):
    path = _saved(tmp_path, (1, 2, 3))
    raw = path.read_bytes()
    broken = {"junk": b"this is not a safetensors file",                    # 29 bytes of junk
              "cut-1000": raw[:1000], "cut-tail": raw[:-100],
              "huge-length": struct.pack("<Q", 1 << 62) + raw[8:],
              "not-a-table": struct.pack("<Q", 2) + b"[]"}[damage]
    path.write_bytes(broken)
    assert ps.DiskBlocks(tmp_path, MODEL).blocks() == []
    assert not path.exists() and "deleted corrupt snapshot" in capsys.readouterr().out


def test_read_metadata_never_allocates_a_garbage_header_length(tmp_path):
    path = tmp_path / "x.safetensors"
    path.write_bytes(b"\xff" * 29)
    with pytest.raises(ps.CorruptSnapshot):
        ps.read_metadata(path)


def test_startup_loading_deletes_a_broken_file_too(tmp_path, capsys):
    good = _saved(tmp_path, (1, 2))
    bad = _saved(tmp_path, (1, 2, 3))
    bad.write_bytes(bad.read_bytes()[:1000])
    assert [tokens for tokens, _ in ps.load_snapshots(tmp_path, MODEL)] == [[1, 2]]
    assert good.exists() and not bad.exists()


def test_a_refused_block_makes_way_for_the_next_longest_intact_one(tmp_path):
    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.scheduler import Scheduler
    from tests.lane_fakes import FakeEngine

    blocks, sessions = tmp_path / "prefix-snapshots", tmp_path / "session-snapshots"
    ps.save_snapshot(blocks, MODEL, list(range(20)), [Layer()])                 # the system block
    ps.save_snapshot(sessions, MODEL, list(range(40)), [Layer()])               # intact, shorter
    bad = ps.save_snapshot(sessions, MODEL, list(range(60)), [Layer()])         # the longest: corrupt
    _flip(bad, _data_start(bad) + 128)
    unread = ps.save_snapshot(sessions, MODEL, list(range(50)), [Layer()])     # unreadable: kept, skipped
    import os

    os.chmod(unread, 0)
    try:
        store = CheckpointStore(4, copier=lambda c: c)
        scheduler = Scheduler(FakeEngine(), lanes=1, eos_ids=frozenset(), checkpoints=store, snapshot_dir=blocks,
                              session_dir=sessions, model_id=MODEL)
        scheduler._read_disk_block(list(range(80)))
        assert [len(e.tokens) for e in store._entries] == [40]
        assert not bad.exists() and unread.exists()
    finally:
        os.chmod(unread, 0o644)
