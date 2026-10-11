"""Every code edit to Flash Next's family or kernels is a checked decision about prompt snapshots.

Prompt snapshots of Flash Next key on its declared numerics (``NUMERICS_VERSION``, ``ENGINE_NUMERICS``) and a
start-up canary, not on its source, so a refactor keeps them. The canary cannot reach every path a real prompt
takes, so an edit that moves bits only on those paths would keep the key. This test makes each edit a decision: it
hashes the family and kernel packages with comments, docstrings and the comments inside multi-line kernel sources
dropped (the AST, not the text), and fails when that hash differs from the one recorded as reviewed for the
current declared numerics. Then either:

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
import sys
from pathlib import Path

import pytest

from tensorfold import families
from tensorfold.engine import numerics

TESTS = Path(__file__).resolve().parent
REVIEWED = TESTS / "fixtures" / "numerics_reviewed.json"
GOLDEN = TESTS / "fixtures" / "numerics_golden.json"
PACKAGES = ("families/qwen4_exp", "kernels/qwen/flash_next/v1")
_C_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def _drop_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[1:] or [ast.Pass()]
    return body


def _source_text(text: str) -> str:
    """A multi-line string (a kernel's source) without C comments, trailing blanks or empty lines."""

    lines = (line.rstrip() for line in _C_COMMENT.sub("", text).splitlines())
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


def source_digest(root: Path | None = None) -> str:
    """sha256 of the family's and kernels' code as the interpreter sees it: Python files by normalised AST, other
    files (CUDA sources, tables) by bytes."""

    import tensorfold

    base = Path(root) if root is not None else Path(tensorfold.__file__).resolve().parent
    digest = hashlib.sha256()
    for package in PACKAGES:
        for path in sorted(p for p in (base / package).rglob("*") if p.is_file()):
            rel = path.relative_to(base)
            if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo"):
                continue
            digest.update(rel.as_posix().encode() + b"\0")
            if path.suffix == ".py":
                digest.update(_dump(ast.parse(path.read_text(), filename=str(rel))).encode())
            else:
                digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:16]


def declared() -> str:
    return f"{numerics.declared(families.families()['qwen4_exp'])}|e{numerics.ENGINE_NUMERICS}"


def _golden_passes() -> str | None:
    """None when the golden digests of this code match the recorded ones on this MLX and GPU, else why not."""

    try:
        import mlx.core as mx
    except ImportError:
        return "MLX is not installed"
    if not mx.metal.is_available():
        return "no Metal GPU to run the golden test on"
    from tests.test_numerics_golden import tiny_report

    report = tiny_report()
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}
    recorded = golden.get(report["declared"], {}).get(report["environment"])
    if recorded is None:
        return f"no golden digests recorded for {report['declared']} on {report['environment']}"
    if recorded != report["digests"]:
        moved = sorted(k for k in set(recorded) | set(report["digests"]) if recorded.get(k) != report["digests"].get(k))
        return f"the golden digests moved for {moved}"
    return None


def test_the_family_and_kernel_code_is_reviewed_for_its_numerics():
    digest, key = source_digest(), declared()
    reviewed = json.loads(REVIEWED.read_text()) if REVIEWED.is_file() else {}
    if os.environ.get("TF_RECORD_NUMERICS_REVIEWED") == "1":
        why = _golden_passes()
        assert why is None, f"not recording the reviewed source digest: {why}. The golden test must pass unchanged " \
                            "first (or bump NUMERICS_VERSION and record the golden for the new version)."
        reviewed[key] = digest
        REVIEWED.write_text(json.dumps(reviewed, indent=1, sort_keys=True) + "\n")
        return
    assert reviewed.get(key) == digest, (
        f"Flash Next's family or kernel code changed (normalised source digest {digest}; reviewed for {key}: "
        f"{reviewed.get(key)}). Prompt snapshots key on the declared numerics and the canary, not on this code, so "
        "decide: if the edit can move any prompt cache's bits, bump NUMERICS_VERSION (families/qwen4_exp/__init__.py) "
        "or ENGINE_NUMERICS (engine/numerics.py) and record the golden for it; if it cannot, "
        "tests/test_numerics_golden.py must pass UNCHANGED (no re-record) before you update this hash. Record with "
        "TF_RECORD_NUMERICS_REVIEWED=1 pytest tests/test_numerics_reviewed.py and commit "
        "tests/fixtures/numerics_reviewed.json.")


def _tree(tmp_path: Path, rel: str, old: str, new: str) -> Path:
    import shutil

    import tensorfold

    root = tmp_path / "tensorfold"
    shutil.copytree(Path(tensorfold.__file__).resolve().parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    path = root / rel
    text = path.read_text()
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new, 1))
    return root


@pytest.mark.parametrize(("rel", "old", "new"), [
    ("families/qwen4_exp/model.py", '"""Qwen', '"""(Edited docstring.) Qwen'),
    ("families/qwen4_exp/model.py", "\nimport", "\n# a comment\nimport"),
    ("kernels/qwen/flash_next/v1/experts.py", "(fp32, bf16-rounded) to WTS", "(fp32, rounded to bf16) to WTS"),
])
def test_comments_and_docstrings_keep_the_digest(tmp_path, rel, old, new):
    assert source_digest(_tree(tmp_path, rel, old, new)) == source_digest()


@pytest.mark.parametrize(("rel", "old", "new"), [
    ("kernels/qwen/flash_next/v1/prefill_mm.py", "MAX_SORTED_ROWS = 32768", "MAX_SORTED_ROWS = 600"),
    ("kernels/qwen/flash_next/v1/gdn.py", "metal::rsqrt(ss + 1e-6f) * (isq", "metal::rsqrt(ss + 1e-3f) * (isq"),
    ("families/qwen4_exp/model_layers.py", "        outs = []\n", "        parts_out = outs = []\n"),
])
def test_code_edits_move_the_digest(tmp_path, rel, old, new):
    assert source_digest(_tree(tmp_path, rel, old, new)) != source_digest()


def test_recording_refuses_when_the_golden_moved(monkeypatch, tmp_path):
    me = sys.modules[__name__]

    monkeypatch.setenv("TF_RECORD_NUMERICS_REVIEWED", "1")
    monkeypatch.setattr(me, "REVIEWED", tmp_path / "reviewed.json")
    monkeypatch.setattr(me, "_golden_passes", lambda: "the golden digests moved for ['q8-tail-1']")
    with pytest.raises(AssertionError, match="must pass unchanged"):
        me.test_the_family_and_kernel_code_is_reviewed_for_its_numerics()
    assert not (tmp_path / "reviewed.json").exists()
