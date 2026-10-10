"""Snapshot identities include prompt kernels, their modes, and library versions."""

from importlib import metadata
from types import SimpleNamespace

from tensorfold import cli, families
from tensorfold.families import qwen4_exp
from tensorfold.server import app


def test_snapshot_identity_changes_with_mlx_lm_version(monkeypatch, tmp_path):
    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["model_id"])
        raise Captured

    monkeypatch.setattr(app, "ChatApp", capture)
    monkeypatch.setattr(families, "kernel_version", lambda *args: "fixed-kernels")
    family = SimpleNamespace(title="fake", model_type="fake", package=SimpleNamespace(load=lambda *a, **k: (None, None)))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--prompt-cache-gib", "0"])
    for version in ("0.31.3", "0.31.4"):
        monkeypatch.setattr(metadata, "version", lambda name, value=version: value)
        try:
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
        except Captured:
            pass
    assert len(captured) == 2 and captured[0] != captured[1]
    assert "mlx_lm=0.31.3" in captured[0] and "mlx_lm=0.31.4" in captured[1]


def test_flash_prompt_attention_mode_changes_the_family_key(monkeypatch):
    family = families.families()["qwen4_exp"]
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=False))])
    dense = families.kernel_version(family, model)
    model.layers[0].self_attn.kernel_select = True
    sparse = families.kernel_version(family, model)
    assert dense != sparse
    monkeypatch.setenv("TF_FLASH_FUSED", "0")
    assert families.kernel_version(family, model) == sparse
    assert callable(qwen4_exp.kernel_version)


def test_flash_key_still_tracks_kernel_sources(monkeypatch):
    family = families.families()["qwen4_exp"]
    before = families.kernel_version(family, None)
    monkeypatch.setattr(qwen4_exp, "KERNEL_VERSION", "test-revision")
    assert families.kernel_version(family, None) != before


def test_flash_key_carries_the_prefill_key():
    family = families.families()["qwen4_exp"]
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=True))],
                            prefill_key="flash-prefill=fast;architecture=x;matmul=custom")
    fast = families.kernel_version(family, model)
    model.prefill_key = "flash-prefill=mlx"
    assert families.kernel_version(family, model) != fast and "prompt_attention=1" in fast


def _model_ids(monkeypatch, tmp_path, rounds):
    """The snapshot key ``serve`` builds for ``tmp_path`` once per ``rounds`` callback (run before each build)."""

    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["model_id"])
        raise Captured

    monkeypatch.setattr(app, "ChatApp", capture)
    monkeypatch.setattr(families, "kernel_version", lambda *args: "fixed-kernels")
    family = SimpleNamespace(title="fake", model_type="fake", package=SimpleNamespace(load=lambda *a, **k: (None, None)))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--prompt-cache-gib", "0"])
    for before in rounds:
        before()
        try:
            cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
        except Captured:
            pass
    return captured


def _shard(path, body: bytes):
    import json
    import struct

    header = json.dumps({"w": {"dtype": "U8", "shape": [len(body)], "data_offsets": [0, len(body)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + body)


def test_new_weights_at_the_same_path_change_the_snapshot_key(monkeypatch, tmp_path):
    from tensorfold.engine import prefix_snapshots as ps

    shard = tmp_path / "model-00001-of-00001.safetensors"
    _shard(shard, bytes(range(256)) * 64)
    (tmp_path / "config.json").write_text('{"model_type": "fake"}')

    def rewrite():
        _shard(shard, bytes(range(256)) * 63 + bytes(255 - b for b in range(256)))      # same size, new bytes
        ps._WEIGHTS.clear()

    first, same, rewritten = _model_ids(monkeypatch, tmp_path, [lambda: None, ps._WEIGHTS.clear, rewrite])
    assert first == same and first != rewritten
    assert first.split("|weights=")[0] == rewritten.split("|weights=")[0]   # the rest of the key is untouched
    assert "|tensorfold=" in first and "|mlx_lm=" in first and "|kernels=fixed-kernels" in first


def test_the_weights_fingerprint_tracks_config_and_index_but_not_mtime(tmp_path):
    import os

    from tensorfold.engine import prefix_snapshots as ps

    shard = tmp_path / "model-00001-of-00002.safetensors"
    _shard(shard, b"\1" * 100_000)
    (tmp_path / "config.json").write_text('{"a": 1}')
    base = ps.weights_fingerprint(tmp_path)
    os.utime(shard, (1_000_000, 1_000_000))
    ps._WEIGHTS.clear()
    assert ps.weights_fingerprint(tmp_path) == base                        # touched, not changed
    (tmp_path / "config.json").write_text('{"a": 2}')
    assert ps.weights_fingerprint(tmp_path) == base                        # read once a process
    ps._WEIGHTS.clear()
    changed_config = ps.weights_fingerprint(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    ps._WEIGHTS.clear()
    assert len({base, changed_config, ps.weights_fingerprint(tmp_path)}) == 3
    _shard(tmp_path / "model-00002-of-00002.safetensors", b"\2" * 10)
    ps._WEIGHTS.clear()
    with_shard = ps.weights_fingerprint(tmp_path)
    (tmp_path / f"{'a' * 32}.1-0.partial.safetensors").write_bytes(b"\0" * 64)    # not a weight shard
    ps._WEIGHTS.clear()
    assert ps.weights_fingerprint(tmp_path) == with_shard


def test_a_weights_change_turns_a_stored_prefix_into_a_miss(tmp_path):
    import pytest

    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import prefix_snapshots as ps

    weights = tmp_path / "model"
    weights.mkdir()
    _shard(weights / "model.safetensors", b"\1" * 4096)
    blocks = tmp_path / "blocks"
    key = f"{weights}|mlx=1|weights={ps.weights_fingerprint(weights)}"

    class Layer:
        def __init__(self):
            self.keys = mx.ones((2, 2))

    ps.save_snapshot(blocks, key, [1, 2, 3], [Layer()])
    assert ps.DiskBlocks(blocks, key).best([1, 2, 3, 4], 0) is not None
    _shard(weights / "model.safetensors", b"\2" * 4096)
    ps._WEIGHTS.clear()
    new_key = f"{weights}|mlx=1|weights={ps.weights_fingerprint(weights)}"
    assert new_key != key
    assert ps.DiskBlocks(blocks, new_key).best([1, 2, 3, 4], 0) is None
    assert list(ps.load_snapshots(blocks, new_key)) == []


def test_replacing_one_tensor_anywhere_in_a_large_shard_changes_the_fingerprint(tmp_path):
    import json
    import struct

    from tensorfold.engine import prefix_snapshots as ps

    names = [f"t{i}" for i in range(40)]
    each = 1 << 20                                  # 40 MiB of tensors: most bytes fall between the shard spans
    header = json.dumps({n: {"dtype": "U8", "shape": [each], "data_offsets": [i * each, (i + 1) * each]}
                         for i, n in enumerate(names)}).encode()
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\1" * (each * len(names)))
    base = ps.weights_fingerprint(tmp_path)
    start = 8 + len(header)
    for i in (3, 18, 31):                           # tensors none of the 16 shard spans lands on
        raw = bytearray(shard.read_bytes())
        raw[start + i * each:start + (i + 1) * each] = b"\2" * each
        shard.write_bytes(bytes(raw))
        ps._WEIGHTS.clear()
        assert ps.weights_fingerprint(tmp_path) != base
        raw[start + i * each:start + (i + 1) * each] = b"\1" * each
        shard.write_bytes(bytes(raw))
        ps._WEIGHTS.clear()
        assert ps.weights_fingerprint(tmp_path) == base
