"""Flash Next's snapshot key: kept across bit-identical code, moved by anything that changes a prompt cache's bits."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tensorfold import cli, families
from tensorfold.engine import numerics
from tensorfold.engine.prefill_plan import PrefillPlan

FAMILY = families.families()["qwen4_exp"]


def _key(**changes) -> str:
    parts = dict(mlx="0.32.2", mlx_lm="0.31.3", kernels="qwen4_exp-v1-n1|prompt_attention=1|flash-prefill=fast",
                 plan="grid2048+msg256:1:2.3", weights="w0", canary_value="520:aaaa")
    parts.update(changes)
    return numerics.model_id(Path("/models/q8"), **parts)


def test_every_pinned_part_moves_the_key():
    base = _key()
    for change in (dict(mlx="0.33.0"), dict(mlx_lm="0.31.4"), dict(plan="grid4096+msg256:1:2.3"),
                   dict(weights="w1"), dict(canary_value="520:bbbb"),
                   dict(kernels="qwen4_exp-v1-n2|prompt_attention=1|flash-prefill=fast"),
                   dict(kernels="qwen4_exp-v1-n1|prompt_attention=0|flash-prefill=fast"),
                   dict(kernels="qwen4_exp-v1-n1|prompt_attention=1|flash-prefill=mlx")):
        assert _key(**change) != base, change
    assert base.split("|")[0] == str(Path("/models/q8").resolve())          # blocks still group by model path
    assert "tensorfold=" not in base and f"|numerics=e{numerics.ENGINE_NUMERICS}|" in base


def test_strict_mode_pins_the_exact_code(monkeypatch):
    monkeypatch.setenv("TF_SNAPSHOT_KEY", "strict")
    assert numerics.strict_requested()
    strict = _key(strict=True)
    assert f"|code={numerics.code_fingerprint()}|" in strict and strict != _key()


def test_the_family_key_is_declared_not_hashed(monkeypatch):
    model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(kernel_select=True))],
                            prefill_key="flash-prefill=mlx")
    key = families.kernel_version(FAMILY, model)
    assert key.startswith("qwen4_exp-v1-n1|prompt_attention=1|flash-prefill=mlx")
    # the source is not read: a family without the declaration still keys on it
    monkeypatch.setattr(families, "kernel_source_version", lambda family: pytest.fail("source hashed"))
    assert families.kernel_version(FAMILY, model) == key
    monkeypatch.setattr(FAMILY.package, "NUMERICS_VERSION", 2)
    assert families.kernel_version(FAMILY, model) != key


def test_a_family_without_the_declaration_keeps_its_source_key():
    legacy = SimpleNamespace(package=SimpleNamespace(KERNEL_VERSION="v1"), module="tensorfold.families.qwen4_exp",
                             model_type="qwen4_exp")
    assert numerics.declared(legacy) is None
    assert families.kernel_version(legacy, None) == families.kernel_source_version(legacy)


def test_the_canary_cuts_two_full_chunks_and_a_short_tail():
    plan = PrefillPlan(2048, (7,), 256, (7, 9))
    tokens = numerics.canary_tokens(plan, 1000)
    assert tokens == numerics.canary_tokens(plan, 1000)                       # fixed
    assert plan.chunks(tokens).starts == [0, 256, 512] and len(tokens) == 512 + numerics.CANARY_TAIL
    assert len(tokens) - 512 <= 16                                            # Flash Next's fused rows
    assert all(0 <= t < 1000 for t in tokens)
    assert sum(t == 7 for t in tokens) == 2 and tokens[256:258] == [7, 9] and tokens[512:514] == [7, 9]
    grid = PrefillPlan(2048)
    assert len(numerics.canary_tokens(grid, 1000)) == numerics.CANARY_GRID


class _Engine:
    """An engine whose 'prefill' leaves a cache holding the tokens: its digest is a function of them."""

    def __init__(self, scale=1):
        self.scale = scale
        self.calls = 0

    def prefill_prefix(self, tokens):
        mx = pytest.importorskip("mlx.core")
        self.calls += 1
        layer = SimpleNamespace(keys=mx.array(tokens, dtype=mx.float32) * self.scale, offset=len(tokens))
        return [layer]


def test_the_canary_digest_follows_the_cache_bits():
    plan = PrefillPlan(2048, (7,), 256, (7, 9))
    same = numerics.canary_digest(_Engine(), plan, 1000)
    assert same == numerics.canary_digest(_Engine(), plan, 1000)
    assert same != numerics.canary_digest(_Engine(scale=1.0000001), plan, 1000)
    assert same.startswith(f"{512 + numerics.CANARY_TAIL}:")


@pytest.fixture
def served(monkeypatch, tmp_path):
    """``cli._declared_model_id`` on a stub model, counting canary runs."""

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen4_exp"}')
    engine = _Engine()
    monkeypatch.setattr("tensorfold.engine.lane_engine.LaneEngine", lambda model, prefill_plan=None: engine)
    model = SimpleNamespace(layers=[], args=SimpleNamespace(vocab_size=1000), prefill_key="flash-prefill=fast")
    plan = PrefillPlan(2048, (7,), 256, (7, 9))
    blocks = tmp_path / "blocks"

    def build(directories=(blocks, None), plan=plan):
        return cli._declared_model_id(FAMILY, model, None, plan, model_dir, "0.31.3", directories)

    return SimpleNamespace(build=build, engine=engine, blocks=blocks, model_dir=model_dir)


def test_the_canary_runs_once_per_build_and_environment(monkeypatch, served):
    first = served.build()
    assert served.engine.calls == 1 and (served.blocks / numerics.CANARY_FILE).is_file()
    assert served.build() == first and served.engine.calls == 1               # cached: a restart pays nothing
    monkeypatch.setattr("tensorfold.__version__", "9.9.9")
    monkeypatch.setattr(cli, "__version__", "9.9.9")
    assert served.build() == first and served.engine.calls == 1               # a release is not a key
    monkeypatch.setenv("TF_FLASH_SOMETHING", "1")                             # the environment moved: run again
    assert served.build() == first and served.engine.calls == 2
    monkeypatch.setattr(numerics, "_CODE", ["another-build"])                 # the code moved: run again
    assert served.build() == first and served.engine.calls == 3
    entries = json.loads((served.blocks / numerics.CANARY_FILE).read_text())
    assert len(entries) == 3 and {e["canary"] for e in entries.values()} == {first.split("|canary=")[1].split("|")[0]}


def test_an_mlx_upgrade_moves_the_key(monkeypatch, served):
    import mlx.core as mx

    first = served.build()
    monkeypatch.setattr(mx, "__version__", "99.0.0")
    assert served.build() != first and "|mlx=99.0.0|" in served.build()


def test_another_prompt_plan_moves_the_key(served):
    assert served.build() != served.build(plan=PrefillPlan(4096, (7,), 256, (7, 9)))


def test_serve_keys_a_declared_family_without_the_release(monkeypatch, tmp_path):
    from importlib import metadata

    from tensorfold.server import app

    captured = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.append(kwargs["model_id"])
        raise Captured

    monkeypatch.setattr(app, "ChatApp", capture)
    monkeypatch.setattr(families, "kernel_version", lambda *args: "qwen4_exp-v1-n1")
    monkeypatch.setattr(metadata, "version", lambda name: "0.31.3")
    package = SimpleNamespace(load=lambda *a, **k: (None, None), NUMERICS_VERSION=1, KERNEL_VERSION="v1")
    family = SimpleNamespace(title="fake", model_type="fake", package=package)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--prompt-cache-gib", "0",
                                          "--snapshot-dir", str(tmp_path / "blocks")])
    with pytest.raises(Captured):
        cli._serve_mlx(args, family, tmp_path, 0, [], 1 << 30)
    key = captured[0]
    assert "|tensorfold=" not in key and "|kernels=qwen4_exp-v1-n1|" in key and "|canary=" in key
    assert f"|numerics=e{numerics.ENGINE_NUMERICS}|" in key and "|weights=" in key


def test_a_moved_canary_moves_the_key(monkeypatch, served):
    first = served.build()
    served.engine.scale = 3                                                   # the next build computes other bits
    monkeypatch.setattr(numerics, "_CODE", ["changed-arithmetic"])
    assert served.build() != first


def test_without_stored_snapshots_no_canary_runs(served):
    key = served.build(directories=(None, None))
    assert served.engine.calls == 0 and "|canary=unused|" in key


def test_a_failed_canary_keys_on_the_exact_build(monkeypatch, served):
    def broken(tokens):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(served.engine, "prefill_prefix", broken)
    key = served.build()
    assert f"|canary=code-{numerics.code_fingerprint()}|" in key
    assert not (served.blocks / numerics.CANARY_FILE).exists()


class StoredLayer:
    def __init__(self, keys):
        self.keys = keys


def test_snapshots_of_other_numerics_are_never_read(monkeypatch, served):
    mx = pytest.importorskip("mlx.core")
    from tensorfold.engine import prefix_snapshots as ps

    ours = served.build()
    served.engine.scale = 3
    monkeypatch.setattr(numerics, "_CODE", ["changed-arithmetic"])
    theirs = served.build()

    ps.save_snapshot(served.blocks, ours, [1, 2, 3], [StoredLayer(mx.ones((2, 2)))])
    assert ps.DiskBlocks(served.blocks, ours).best([1, 2, 3, 4], 0) is not None
    assert ps.DiskBlocks(served.blocks, theirs).best([1, 2, 3, 4], 0) is None
    assert list(ps.load_snapshots(served.blocks, theirs)) == []
    assert len(list(ps.load_snapshots(served.blocks, ours))) == 1


# -- the package itself, edited: the real code paths on a tiny model (GPU) ----------------------------------------
def _edited_tree(tmp_path: Path, edits: dict[str, list[tuple[str, str]]]) -> Path:
    import tensorfold

    root = tmp_path / "src"
    shutil.copytree(Path(tensorfold.__file__).resolve().parent, root / "tensorfold",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for rel, swaps in edits.items():
        path = root / "tensorfold" / rel
        text = path.read_text()
        for old, new in swaps:
            assert text.count(old) >= 1, (rel, old)
            text = text.replace(old, new)
        path.write_text(text)
    return root


needs_gpu = pytest.mark.skipif(not __import__("mlx.core").core.metal.is_available(), reason="needs a Metal GPU")


@needs_gpu
def test_comment_and_refactor_edits_keep_the_key_and_bits(tmp_path):
    from tests.test_numerics_golden import tiny_report

    base = tiny_report()
    edited = tiny_report(_edited_tree(tmp_path, {
        # a comment inside a Metal kernel's source, and a Python comment and docstring
        "kernels/qwen/flash_next/v1/experts.py": [("(fp32, bf16-rounded) to WTS", "(fp32, rounded to bf16) to WTS")],
        "families/qwen4_exp/model.py": [('"""Qwen', '"""(Edited docstring.) Qwen'),
                                        ("\nimport", "\n# a comment\nimport")],
        # a refactor: a local renamed
        "families/qwen4_exp/model_layers.py": [("        outs = []\n", "        parts_out = outs = []\n")],
    }))
    assert edited["code"] != base["code"]                     # the strict fingerprint saw the edit (canary re-runs)
    assert edited["kernels"] == base["kernels"] and edited["declared"] == base["declared"]
    assert edited["digests"] == base["digests"]               # same bits: same canary, same key


@needs_gpu
def test_an_arithmetic_change_moves_the_canary_and_fails_the_golden(tmp_path):
    from tests.test_numerics_golden import tiny_report

    base = tiny_report()
    edited = tiny_report(_edited_tree(tmp_path, {
        "kernels/qwen/flash_next/v1/gdn.py": [("metal::rsqrt(ss + 1e-6f)", "metal::rsqrt(ss + 1e-3f)")],
    }))
    assert edited["declared"] == base["declared"]              # nobody bumped NUMERICS_VERSION
    moved = {k for k in base["digests"] if edited["digests"][k] != base["digests"][k]}
    assert {"q8-canary", "q4-canary"} <= moved                 # the canary (so the runtime key) moved by itself
    golden = json.loads((Path(__file__).parent / "fixtures" / "numerics_golden.json").read_text())
    recorded = golden.get(base["declared"], {}).get(base["environment"])
    if recorded is not None:
        assert recorded == base["digests"] and recorded != edited["digests"]    # and the golden test fails
