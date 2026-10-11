"""Every code edit a prompt cache's bits can depend on is a checked decision about prompt snapshots.

Prompt snapshots of Flash Next key on its declared numerics (``NUMERICS_VERSION``, ``ENGINE_NUMERICS``) and a
start-up canary, not on its source, so a refactor keeps them. The canary cannot reach every path a real prompt
takes, so an edit that moves bits only on those paths would keep the key. This test makes each edit a decision: it
hashes the code with comments, docstrings, and the comments and blank space inside multi-line kernel sources
dropped (the AST, not the text), and fails when that hash differs from the one recorded as reviewed for the
current declared numerics.

The files hashed are every file of the tensorfold package except an explicit allow-list of code that cannot change
a prompt cache's bits (HTTP and admission code, argument parsing, the updater, docs), plus the golden's own
generator (tests/numerics_tiny.py and these two tests). The allow-list and the resulting file list are stored in
tests/fixtures/numerics_reviewed.json, so a change to either shows in review. On a Metal GPU the test also fails
when the golden run imports an allow-listed file.

What it catches is an accidental edit: one nobody meant to move bits. A deliberate re-record is possible, by one
command whose fixture diff is the decision. Reviewers must treat any diff to tests/fixtures/numerics_reviewed.json
or tests/fixtures/numerics_golden.json as a NUMERICS_VERSION decision. When the test fails, either:

* the edit can move a prompt cache's bits: bump ``NUMERICS_VERSION`` (families/qwen4_exp/__init__.py) or
  ``ENGINE_NUMERICS`` (engine/numerics.py), record the golden (``TF_RECORD_NUMERICS_GOLDEN=1 pytest
  tests/test_numerics_golden.py``) and then this hash; or
* it cannot: tests/test_numerics_golden.py must pass unchanged, then record the hash with
  ``TF_RECORD_NUMERICS_REVIEWED=1 pytest tests/test_numerics_reviewed.py`` (which reruns the golden comparison
  first and refuses to record when it fails) and commit tests/fixtures/numerics_reviewed.json.
"""

from __future__ import annotations

import ast
import fnmatch
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
ROOT = TESTS.parent
REVIEWED = TESTS / "fixtures" / "numerics_reviewed.json"
GOLDEN = TESTS / "fixtures" / "numerics_golden.json"
PACKAGE = "src/tensorfold"
GENERATORS = ("tests/numerics_tiny.py", "tests/test_numerics_golden.py", "tests/test_numerics_reviewed.py")
# used only when no fixture exists yet: the fixture's "allow" is the list in force
DEFAULT_ALLOW = {
    "__main__.py": "entry point",
    "cli.py": "argument parsing and serve wiring (an edit still reruns the canary through the code fingerprint)",
    "serve_options.py": "argument checks",
    "hub.py": "downloads",
    "update.py": "release checks",
    "tool_parameters.py": "tool-call schemas",
    "server/admission.py": "admission",
    "server/cancellation.py": "request cancellation",
    "server/errors.py": "HTTP errors",
    "server/http.py": "HTTP",
    "server/memory_budget.py": "memory admission",
    "server/messages.py": "chat message shapes",
    "server/prompt_memory.py": "prompt admission",
    "server/prompts.py": "chat templates (tokens in, not cache bits)",
    "server/request_options.py": "request parsing",
    "server/stacks.py": "debug stack dumps",
    "server/stopping.py": "stop strings",
    "server/text.py": "reply text",
    "server/tool_policy.py": "tool-call policy",
    "server/tools.py": "tool-call parsing",
    "*.md": "docs",
}
_SKIP_SUFFIXES = (".pyc", ".pyo", ".so", ".dylib")
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


def load_reviewed() -> dict:
    return json.loads(REVIEWED.read_text()) if REVIEWED.is_file() else {"allow": DEFAULT_ALLOW, "files": []}


def allowed(rel: str, allow: dict[str, str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in allow)


def package_files(root: Path | None = None) -> list[str]:
    """Every file of the package under ``root`` (a repository-like tree; None: this one), relative to the package."""

    base = (root or ROOT) / PACKAGE
    return sorted(p.relative_to(base).as_posix() for p in base.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix not in _SKIP_SUFFIXES)


def hashed_files(root: Path | None = None, allow: dict[str, str] | None = None) -> list[str]:
    """The files the digest covers, relative to the repository: the package apart from ``allow``, and the
    generators."""

    allow = load_reviewed()["allow"] if allow is None else allow
    base = root or ROOT
    files = [f"{PACKAGE}/{rel}" for rel in package_files(root) if not allowed(rel, allow)]
    return sorted(files + [g for g in GENERATORS if (base / g).is_file()])


def source_digest(root: Path | None = None, files: list[str] | None = None) -> str:
    """sha256 of ``files`` (default: those hashed under ``root``) as the interpreter sees them: Python by normalised
    AST, other files (C++, CUDA, tables) by bytes."""

    base = root or ROOT
    files = hashed_files(root) if files is None else files
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


def reviewed_mismatch(root: Path | None, version: str) -> str | None:
    """None when the code under ``root`` is the code recorded as reviewed for ``version``, else how it differs."""

    reviewed = load_reviewed()
    if reviewed.get("numerics") != version:
        return f"the reviewed code is recorded for {reviewed.get('numerics')}, not {version}"
    files = hashed_files(root, reviewed["allow"])
    if files != reviewed["files"]:
        return (f"the hashed files changed: +{sorted(set(files) - set(reviewed['files']))} "
                f"-{sorted(set(reviewed['files']) - set(files))}")
    digest = source_digest(root, files)
    if digest != reviewed["digest"]:
        return f"the code differs from the reviewed code (digest {digest}, reviewed {reviewed['digest']})"
    return None


def _imported(report: dict) -> set[str]:
    return {f"{PACKAGE}/{rel}" for rel in report["files"]}


def record_reviewed(root: Path | None, path: Path, golden_path: Path = GOLDEN, report: dict | None = None) -> None:
    """Record the code under ``root`` (a repository-like tree; None: this one) as reviewed, after its golden run
    matches the recorded digests unchanged."""

    from tests.test_numerics_golden import tiny_report, verdict

    if report is None:
        report = tiny_report() if root is None else tiny_report(root / "src", root / "tests" / "numerics_tiny.py")
    golden = json.loads(golden_path.read_text()) if golden_path.is_file() else {}
    state, why = verdict(report, golden)
    assert state == "pass", (
        f"not recording the reviewed source digest: {why or 'no golden digests on this GPU'}. The golden test must "
        "pass unchanged first (or bump NUMERICS_VERSION and record the golden for the new version).")
    allow = load_reviewed()["allow"]
    files = hashed_files(root, allow)
    unhashed = sorted(_imported(report) - set(files))
    assert not unhashed, f"the golden run imports allow-listed files {unhashed}: take them off the allow-list"
    path.write_text(json.dumps({"numerics": report["declared"], "digest": source_digest(root, files),
                                "allow": allow, "files": files}, indent=1, sort_keys=True) + "\n")


def test_the_code_is_reviewed_for_its_numerics():
    if os.environ.get("TF_RECORD_NUMERICS_REVIEWED") == "1":
        assert __import__("mlx.core").core.metal.is_available(), "recording needs a Metal GPU to run the golden"
        record_reviewed(None, REVIEWED)
        return
    how = ("Prompt snapshots key on the declared numerics and the canary, not on this code, so decide: if the edit "
           "can move any prompt cache's bits, bump NUMERICS_VERSION (families/qwen4_exp/__init__.py) or "
           "ENGINE_NUMERICS (engine/numerics.py) and record the golden for it; if it cannot, "
           "tests/test_numerics_golden.py must pass UNCHANGED (no re-record) before you update this hash. Record with "
           "TF_RECORD_NUMERICS_REVIEWED=1 pytest tests/test_numerics_reviewed.py and commit "
           "tests/fixtures/numerics_reviewed.json.")
    reviewed = load_reviewed()
    if __import__("mlx.core").core.metal.is_available():
        from tests.test_numerics_golden import tiny_report

        unhashed = sorted(_imported(tiny_report()) - set(hashed_files(None, reviewed["allow"])))
        assert not unhashed, f"the golden run imports allow-listed files {unhashed}: take them off the allow-list"
    why = reviewed_mismatch(None, declared())
    assert why is None, f"code a prompt cache's bits can depend on changed: {why}. {how}"


_READERS = {"get", "getenv", "setdefault", "pop"}


def _env_names(tree: ast.AST) -> list[tuple[int, str | None]]:
    """(line, name) of each environment read in a module; name None when it is not a literal or module constant."""

    constants = {t.id: n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
                 for t in n.targets if isinstance(t, ast.Name)}

    def environ(node: ast.AST) -> bool:
        if isinstance(node, ast.IfExp):                         # (os.environ if environ is None else environ)
            return environ(node.body) or environ(node.orelse)
        return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
            isinstance(node, ast.Name) and node.id in ("env", "environ"))

    def name(arg: ast.AST) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        return constants.get(arg.id) if isinstance(arg, ast.Name) else None

    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args and (
                (node.func.attr in _READERS and environ(node.func.value))
                or (node.func.attr == "getenv" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os")):
            found.append((node.lineno, name(node.args[0])))
        elif isinstance(node, ast.Subscript) and environ(node.value) and isinstance(node.ctx, ast.Load):
            found.append((node.lineno, name(node.slice)))
    return found


# reads whose name is computed, each checked by hand: cli sets a family's MLX_ENV (MLX_* names) before MLX starts
_COMPUTED_READS = {("cli.py", "setdefault")}


def test_every_environment_variable_the_package_reads_is_fingerprinted():
    outside = []
    for rel in package_files():
        if not rel.endswith(".py"):
            continue
        source = (ROOT / PACKAGE / rel).read_text()
        for line, name in _env_names(ast.parse(source)):
            if name is None:
                text = source.splitlines()[line - 1]
                if not any(rel == f and f".{call}(" in text for f, call in _COMPUTED_READS):
                    outside.append(f"{rel}:{line} (computed name)")
            elif not name.startswith(numerics.ENV_PREFIXES):
                outside.append(f"{rel}:{line} {name}")
    assert not outside, (f"environment reads outside numerics.ENV_PREFIXES {numerics.ENV_PREFIXES}: {outside}. The "
                         "canary cache would not see them: rename the variable or extend ENV_PREFIXES.")


def test_the_env_scan_sees_every_way_of_reading():
    source = ("import os\nLIMIT = 'QWEN_X'\nenv = os.environ\nos.environ.get('A_1')\nos.getenv('B_2')\n"
              "os.environ['C_3']\nenv.get(LIMIT)\nos.environ.get(key)\n(os.environ if e is None else e).get('D_4')\n")
    found = sorted(_env_names(ast.parse(source)))
    assert [n for _, n in found] == ["A_1", "B_2", "C_3", "QWEN_X", None, "D_4"]


def _tree(tmp_path: Path, edits: dict[str, str]) -> Path:
    """A repository-like copy (the package and the generators) at ``tmp_path`` with ``edits``: text appended to a file
    (a key ``path::before=<anchor>`` inserts it before the anchor's first occurrence instead)."""

    shutil.copytree(ROOT / PACKAGE, tmp_path / PACKAGE, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "tests").mkdir()
    for generator in GENERATORS:
        shutil.copy(ROOT / generator, tmp_path / generator)
    for key, text in edits.items():
        rel, _, anchor = key.partition("::before=")
        path = tmp_path / rel
        source = path.read_text()
        if anchor:
            assert anchor in source, (rel, anchor)
            source = source.replace(anchor, text + anchor, 1)
        else:
            source += text
        path.write_text(source)
    return tmp_path


def _digest(tmp_path: Path, edits: dict[str, str]) -> str:
    return source_digest(_tree(tmp_path, edits), load_reviewed()["files"])


KERNEL = "src/tensorfold/kernels/qwen/flash_next/v1/gdn.py"
PLAIN = '\n_PROBE = r"""\n  float a = 1.0f;\n  a = a * 2.0f;\n"""\n'


@pytest.mark.parametrize("edited", [
    {KERNEL: '\n_PROBE = r"""\n  float a = 1.0f;  // a note\n  a = a * 2.0f;\n"""\n'},
    {KERNEL: '\n_PROBE = r"""\n  float a /* a note */ = 1.0f;\n  a  =  a * 2.0f;   \n\n\n"""\n'},
    {KERNEL: PLAIN + "# a trailing comment\n"},
    {KERNEL: PLAIN + 'def _probe():\n    """A docstring."""\n    return 1\n'},
])
def test_comments_docstrings_and_blank_space_keep_the_digest(tmp_path, edited):
    plain = {KERNEL: PLAIN + ("def _probe():\n    return 1\n" if "_probe" in edited[KERNEL] else "")}
    assert _digest(tmp_path / "a", plain) == _digest(tmp_path / "b", edited)


@pytest.mark.parametrize("rel", [
    "src/tensorfold/engine/prefill_step.py", "src/tensorfold/engine/prefill_plan.py",
    "src/tensorfold/engine/prefix_snapshots.py", "src/tensorfold/engine/gpu_sampling.py",
    "src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py", "src/tensorfold/families/qwen4_exp/mtp.py",
    "src/tensorfold/streaming/pack.py", "src/tensorfold/server/scheduler.py", KERNEL,
    "tests/numerics_tiny.py", "tests/test_numerics_golden.py", "tests/test_numerics_reviewed.py",
])
def test_a_code_edit_anywhere_a_prefill_runs_moves_the_digest(tmp_path, rel):
    assert rel in load_reviewed()["files"]
    assert _digest(tmp_path, {rel: "\n_PROBE = 1\n"}) != source_digest()


def test_a_cpp_edit_moves_the_digest(tmp_path):
    rel = "src/tensorfold/streaming/hostsync/hostsync.cpp"
    assert rel in load_reviewed()["files"]
    assert _digest(tmp_path, {rel: "\nstatic int probe_ = 1;\n"}) != source_digest()


def test_allow_listed_files_do_not_move_it(tmp_path):
    assert "src/tensorfold/server/http.py" not in load_reviewed()["files"]
    assert _digest(tmp_path, {"src/tensorfold/server/http.py": "\n_PROBE = 1\n"}) == source_digest()


# 1-4 row tails (GDN's PIPE kernel) scaled: canary-blind (its tail has 8 rows), seen by the golden's tail cases
TAIL_EDIT = {KERNEL: ("\n_gdn_step = gdn_step\n\n\ndef gdn_step(projected, *args, **kwargs):\n"
                      "    if int(projected.shape[0]) <= PIPE_ROWS:\n        projected = projected * 1.0078125\n"
                      "    return _gdn_step(projected, *args, **kwargs)\n")}
# a new environment string, as a macOS or mlx-lm update would give
ENV_EDIT = {"tests/numerics_tiny.py::before=\nif __name__ == \"__main__\":":
            "\n_environment = environment\n\n\ndef environment() -> str:\n"
            "    return _environment() + \"|python=x\"\n\n"}


def _refused(tmp_path: Path, edits: dict[str, str]) -> tuple[dict, Path]:
    from tests.test_numerics_golden import tiny_report, verdict

    tree = _tree(tmp_path / "tree", edits)
    report = tiny_report(tree / "src", tree / "tests" / "numerics_tiny.py")
    golden = tmp_path / "golden.json"
    shutil.copy(GOLDEN, golden)
    assert verdict(report, json.loads(golden.read_text()))[0] == "fail"
    return report, golden


@needs_gpu
def test_a_bit_change_cannot_be_recorded_without_a_bump(tmp_path):
    """Round 2's repro: a bit change, then both recorders, NUMERICS_VERSION unchanged."""

    from tests.test_numerics_golden import record

    report, golden = _refused(tmp_path, TAIL_EDIT)
    with pytest.raises(AssertionError, match="refusing to re-record"):
        record(report, golden, tmp_path / "tree")
    assert golden.read_bytes() == GOLDEN.read_bytes()
    with pytest.raises(AssertionError, match="must pass unchanged"):
        record_reviewed(tmp_path / "tree", tmp_path / "reviewed.json", golden, report)
    assert not (tmp_path / "reviewed.json").exists()


@needs_gpu
def test_a_new_environment_cannot_re_record_a_bit_change(tmp_path):
    """Round 3's k12b repro: a canary-blind bit change plus a new environment string, then both recorders."""

    from tests.test_numerics_golden import record

    report, golden = _refused(tmp_path, {**TAIL_EDIT, **ENV_EDIT})
    assert report["environment"].endswith("|python=x")
    with pytest.raises(AssertionError, match="only for the code reviewed"):
        record(report, golden, tmp_path / "tree")
    assert golden.read_bytes() == GOLDEN.read_bytes()
    with pytest.raises(AssertionError, match="must pass unchanged"):
        record_reviewed(tmp_path / "tree", tmp_path / "reviewed.json", golden, report)
    assert not (tmp_path / "reviewed.json").exists()
