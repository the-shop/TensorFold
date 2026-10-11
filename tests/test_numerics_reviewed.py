"""Every code edit a prompt cache's bits can depend on is a checked decision about prompt snapshots.

Prompt snapshots of Flash Next key on its declared numerics (``NUMERICS_VERSION``, ``ENGINE_NUMERICS``) and a
start-up canary, not on its source, so a refactor keeps them. The canary cannot reach every path a real prompt
takes, so an edit that moves bits only on those paths would keep the key. This test makes each edit a decision: it
hashes the code with comments, docstrings, and the comments and blank space inside multi-line kernel sources
dropped (the AST, not the text), and fails when that hash differs from the one recorded as reviewed for the
current declared numerics.

The files hashed are the family and Flash Next kernel packages, the SSD expert streaming code (Python and the
hostsync C++), and every tensorfold module the golden run (tests/numerics_tiny.py) imports apart from the server,
CLI and updater: engine, snapshots, dense kernels. The sorted list is stored in the fixture, so a file joining or
leaving it shows in review; on a Metal GPU the test reruns the golden and fails when the imported set changed.

When it fails, either:

* the edit can move a prompt cache's bits: bump ``NUMERICS_VERSION`` (families/qwen4_exp/__init__.py) or
  ``ENGINE_NUMERICS`` (engine/numerics.py), record the golden (``TF_RECORD_NUMERICS_GOLDEN=1 pytest
  tests/test_numerics_golden.py``) and then this hash; or
* it cannot: tests/test_numerics_golden.py must pass unchanged, then record the hash with
  ``TF_RECORD_NUMERICS_REVIEWED=1 pytest tests/test_numerics_reviewed.py`` (which reruns the golden comparison
  first and refuses to record when it fails) and commit tests/fixtures/numerics_reviewed.json.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tensorfold import families
from tensorfold.engine import numerics

TESTS = Path(__file__).resolve().parent
REVIEWED = TESTS / "fixtures" / "numerics_reviewed.json"
GOLDEN = TESTS / "fixtures" / "numerics_golden.json"
PACKAGES = ("families/qwen4_exp", "kernels/qwen/flash_next/v1")
SOURCES = ("streaming/*.py", "streaming/hostsync/*.py", "streaming/hostsync/*.cpp")
_C_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

needs_gpu = pytest.mark.skipif(not __import__("mlx.core").core.metal.is_available(), reason="needs a Metal GPU")


def _drop_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:] or [ast.Pass()]
    return body


def _source_text(text: str) -> str:
    """A multi-line string (a kernel's source) without C comments, runs of blanks or empty lines."""

    lines = (" ".join(line.split()) for line in _C_COMMENT.sub(" ", text).splitlines())
    return "\n".join(line for line in lines if line)


def _dump(node: object) -> str:
    """A canonical form of a node: its type and non-empty fields, positions aside (stable across Python versions that
    add empty fields)."""

    if isinstance(node, ast.AST):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = _drop_docstring(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" in node.value:
            return f"Constant({_source_text(node.value)!r})"
        fields = []
        for name in node._fields:
            value = getattr(node, name, None)
            if value is None or value == []:
                continue
            fields.append(f"{name}={_dump(value)}")
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return f"[{', '.join(_dump(item) for item in node)}]"
    return repr(node)


def _root(root: Path | None) -> Path:
    import tensorfold

    return Path(root) if root is not None else Path(tensorfold.__file__).resolve().parent


def static_files(root: Path | None = None) -> set[str]:
    """The family's and Flash Next kernels' files and the SSD expert streaming code (relative paths)."""

    base = _root(root)
    found = {p for package in PACKAGES for p in (base / package).rglob("*") if p.is_file()}
    found |= {p for pattern in SOURCES for p in base.glob(pattern)}
    return {p.relative_to(base).as_posix() for p in found
            if "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}


def source_digest(root: Path | None = None, files: list[str] | None = None) -> str:
    """sha256 of ``files`` (default: the recorded list) as the interpreter sees them: Python by normalised AST, other
    files (C++, CUDA sources, tables) by bytes."""

    base = _root(root)
    if files is None:
        files = json.loads(REVIEWED.read_text())["files"]
    digest = hashlib.sha256()
    for rel in sorted(files):
        path = base / rel
        digest.update(rel.encode() + b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
        elif path.suffix == ".py":
            digest.update(_dump(ast.parse(path.read_text(), filename=rel)).encode())
        else:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def declared() -> str:
    return f"{numerics.declared(families.families()['qwen4_exp'])}|e{numerics.ENGINE_NUMERICS}"


def record_reviewed(src: Path | None, path: Path, golden_path: Path = GOLDEN, report: dict | None = None) -> None:
    """Record the code under ``src`` (a package tree's parent; None: the installed one) as reviewed, after its golden
    run matches the recorded digests unchanged."""

    from tests.test_numerics_golden import tiny_report, verdict

    report = tiny_report(src) if report is None else report
    golden = json.loads(golden_path.read_text()) if golden_path.is_file() else {}
    state, why = verdict(report, golden)
    assert state == "pass", (
        f"not recording the reviewed source digest: {why or 'no golden digests on this GPU'}. The golden test must "
        "pass unchanged first (or bump NUMERICS_VERSION and record the golden for the new version).")
    root = None if src is None else Path(src) / "tensorfold"
    files = sorted(static_files(root) | set(report["files"]))
    path.write_text(json.dumps({"numerics": report["declared"], "digest": source_digest(root, files),
                                "files": files}, indent=1, sort_keys=True) + "\n")


def test_the_code_is_reviewed_for_its_numerics():
    if os.environ.get("TF_RECORD_NUMERICS_REVIEWED") == "1":
        assert __import__("mlx.core").core.metal.is_available(), "recording needs a Metal GPU to run the golden"
        record_reviewed(None, REVIEWED)
        return
    reviewed = json.loads(REVIEWED.read_text())
    files, key = reviewed["files"], declared()
    how = ("Prompt snapshots key on the declared numerics and the canary, not on this code, so decide: if the edit "
           "can move any prompt cache's bits, bump NUMERICS_VERSION (families/qwen4_exp/__init__.py) or "
           "ENGINE_NUMERICS (engine/numerics.py) and record the golden for it; if it cannot, "
           "tests/test_numerics_golden.py must pass UNCHANGED (no re-record) before you update this hash. Record with "
           "TF_RECORD_NUMERICS_REVIEWED=1 pytest tests/test_numerics_reviewed.py and commit "
           "tests/fixtures/numerics_reviewed.json.")
    added = sorted(static_files() - set(files))
    assert not added, f"files joined the family, its kernels or the streaming sources: {added}. {how}"
    if __import__("mlx.core").core.metal.is_available():
        from tests.test_numerics_golden import tiny_report

        current = static_files() | set(tiny_report()["files"])
        joined, left = sorted(current - set(files)), sorted(set(files) - current)
        assert not (joined or left), f"the modules a prefill runs through changed: +{joined} -{left}. {how}"
    digest = source_digest(None, files)
    assert (reviewed["numerics"], reviewed["digest"]) == (key, digest), (
        f"code a prompt cache's bits can depend on changed (normalised source digest {digest} for {key}; reviewed: "
        f"{reviewed['digest']} for {reviewed['numerics']}). {how}")


def _tree(tmp_path: Path, rel: str, old: str, new: str) -> Path:
    """A copy of the installed package with one edit, at ``tmp_path``/tensorfold."""

    import tensorfold

    root = tmp_path / "tensorfold"
    shutil.copytree(Path(tensorfold.__file__).resolve().parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    path = root / rel
    text = path.read_text()
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new, 1))
    return root


GDN = "kernels/qwen/flash_next/v1/gdn.py"
INV = "      const float inv = metal::rsqrt(ss + 1e-6f) * (sg == 0"
KNORM = "        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)\n"


@pytest.mark.parametrize(("rel", "old", "new"), [
    ("families/qwen4_exp/model.py", '"""Qwen', '"""(Edited docstring.) Qwen'),
    ("families/qwen4_exp/model.py", "\nimport", "\n# a comment\nimport"),
    ("kernels/qwen/flash_next/v1/experts.py", "(fp32, bf16-rounded) to WTS", "(fp32, rounded to bf16) to WTS"),
    (GDN, INV, "      // a note\n" + INV),
    (GDN, INV, INV.replace("inv =", "inv /* a note */ =")),
    (GDN, INV, INV.replace("inv =", "inv  =")),
    (GDN, "      ss = simd_sum(ss);\n", "      ss = simd_sum(ss);   \n\n\n"),
    ("engine/prefix_snapshots.py", '"""One stored block', '"""(Edited.) One stored block'),
])
def test_comments_and_docstrings_keep_the_digest(tmp_path, rel, old, new):
    assert source_digest(_tree(tmp_path, rel, old, new)) == source_digest()


@pytest.mark.parametrize(("rel", "old", "new"), [
    ("kernels/qwen/flash_next/v1/prefill_mm.py", "MAX_SORTED_ROWS = 32768", "MAX_SORTED_ROWS = 600"),
    ("kernels/qwen/flash_next/v1/gdn.py", "metal::rsqrt(ss + 1e-6f) * (isq", "metal::rsqrt(ss + 1e-3f) * (isq"),
    ("families/qwen4_exp/model_layers.py", "        outs = []\n", "        parts_out = outs = []\n"),
    ("engine/prefill_plan.py", "            nxt = last + self.step\n", "            nxt = last + self.step - 16\n"),
    ("engine/prefix_snapshots.py", "            setattr(item, name, arrays[f\"{index}.{name}\"])\n",
     "            setattr(item, name, arrays[f\"{index}.{name}\"] * 1)\n"),
    ("kernels/qwen/dense/v1/lane_qmm.py", "        return y.reshape(*lead, N)\n",
     "        return (y * 1).reshape(*lead, N)\n"),
    ("streaming/hostsync/hostsync.cpp", "#include", "#include <cmath>\n#include"),
])
def test_code_edits_move_the_digest(tmp_path, rel, old, new):
    assert rel in json.loads(REVIEWED.read_text())["files"]
    assert source_digest(_tree(tmp_path, rel, old, new)) != source_digest()


@needs_gpu
def test_a_bit_change_cannot_be_recorded_without_a_bump(tmp_path):
    """The round-2 review's repro: an epsilon edit, then both recorders, with NUMERICS_VERSION unchanged."""

    from tests.test_numerics_golden import record, tiny_report, verdict

    _tree(tmp_path / "src", "families/qwen4_exp/model_layers.py", KNORM, KNORM.replace("1e-6)", "1e-5)"))
    report = tiny_report(tmp_path / "src")
    golden = tmp_path / "golden.json"
    shutil.copy(GOLDEN, golden)
    assert verdict(report, json.loads(golden.read_text()))[0] == "fail"
    with pytest.raises(AssertionError, match="refusing to re-record"):
        record(report, golden)
    assert golden.read_bytes() == GOLDEN.read_bytes()
    with pytest.raises(AssertionError, match="must pass unchanged"):
        record_reviewed(tmp_path / "src", tmp_path / "reviewed.json", golden, report)
    assert not (tmp_path / "reviewed.json").exists()
