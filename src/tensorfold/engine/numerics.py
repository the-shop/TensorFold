"""Snapshot identity for families that declare their numerics: kept across code edits that keep the declared
numerics and the canary's bits, refused across a declared bump or a canary change.

A stored prompt state is reusable only by code that would compute the same bytes for the same tokens. Hashing the
source refuses that across every comment, refactor or release; trusting a version number alone trusts every
developer to bump it. A family that declares ``NUMERICS_VERSION`` gets a key of three parts instead:

* what is pinned verbatim: the model path and weights, MLX and mlx-lm versions, the prompt plan and prefill route
  (with the fused-tail matmul flags);
* what is declared: the family's ``NUMERICS_VERSION`` and this module's ``ENGINE_NUMERICS``, bumped whenever a change
  moves bits;
* what is measured: the digest of a short canary prompt (two plan chunks and an 8-row tail) prefilled on the loaded
  model through the production plan, then stored and restored as a snapshot, so an undeclared change on a path the
  canary reaches, or to how states are stored and restored, still moves the key.

The canary does not reach every path a real prompt takes (chunks of thousands of rows, tails of 1-4 rows, more than
DENSE_KEYS keys): an edit that moves bits only there keeps the key unless the declared part is bumped. Two tests
make that bump a checked decision rather than a memory: tests/test_numerics_golden.py pins a tiny model's prompt
caches on cases past each of those thresholds (and never re-records a declared version's digests), and
tests/test_numerics_reviewed.py fails on any code edit to the modules a prefill runs through (the family, its
kernels, the engine, snapshots, streaming) until the bump is made or the edit is recorded as reviewed (after the
golden passes).

The canary is recomputed whenever the installed code (every file of the package, comments included), the
environment (TF_*/MLX_* variables, GPU, OS, Python), the load options (SSD experts, PLE, drafts, kernels, lanes,
memory) or the declared part changes, and cached otherwise.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# Bump when shared engine code changes the bits of a prompt's cache (chunk feeding, resume, state copies).
ENGINE_NUMERICS = 1
CANARY_FILE = "numerics-canary.json"
CANARY_SEED = 0x7E50F01D
CANARY_TAIL = 8                     # a last chunk this short takes the fused small-row path
CANARY_GRID = 512                   # canary length when the plan cuts on its grid alone
_CACHE_ENTRIES = 32
_CODE: list[str] = []


def declared(family: Any) -> str | None:
    """The family's declared numerics (``qwen4_exp-v1-n1``), or None when it keys on its source instead."""

    version = getattr(family.package, "NUMERICS_VERSION", None)
    if version is None:
        return None
    kernels = getattr(family.package, "KERNEL_VERSION", "")
    return f"{family.model_type}-{kernels}-n{int(version)}" if kernels else f"{family.model_type}-n{int(version)}"


def code_fingerprint(root: Path | None = None) -> str:
    """sha256 of every file of the installed package (relative path and bytes), compiled caches aside.

    The installed package's is taken once and kept: call it before the family's modules are imported (``serve``
    does) so it names the code that runs; ``code_on_disk`` reads the files again."""

    if root is None and _CODE:
        return _CODE[0]
    value = _fingerprint(root)
    if root is None:
        _CODE.append(value)
    return value


def code_on_disk() -> str:
    """The installed package's fingerprint read now, not the one kept since start-up."""

    return _fingerprint(None)


def _fingerprint(root: Path | None) -> str:
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(base)
        if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo"):
            continue
        digest.update(rel.as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def environment_fingerprint() -> str:
    """What outside the code can move bits without showing in the key: variables, GPU, OS, Python, MLX build."""

    import mlx.core as mx

    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except Exception:  # noqa: BLE001 - no GPU: the CPU build is the device
        info = {}
    parts = {
        "env": sorted((k, v) for k, v in os.environ.items() if k.startswith(("TF_", "TENSORFOLD_", "MLX_"))),
        "gpu": [str(info.get("architecture", "")), str(info.get("device_name", ""))],
        "os": [platform.system(), platform.release(), platform.mac_ver()[0], platform.machine()],
        "python": sys.version,
        "mlx": [mx.__version__, str(getattr(mx, "__file__", ""))],
        "numpy": np.__version__,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def canary_tokens(plan: Any, vocab: int) -> list[int]:
    """A fixed prompt whose chunks under ``plan`` are two full ones and a short tail where the plan cuts at replies,
    else ``CANARY_GRID`` tokens on its grid. Its filler never holds a marker the plan cuts at."""

    rng = np.random.default_rng(CANARY_SEED)
    assistant = [int(t) for t in getattr(plan, "assistant", ())]
    markers = {*map(int, getattr(plan, "openers", ())), *assistant}
    vocab = int(vocab)

    def filler(n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            out += [int(t) for t in rng.integers(0, vocab, size=n) if int(t) not in markers]
        return out[:n]

    if not assistant:
        return filler(CANARY_GRID)
    span = int(plan.min_chunk)
    first = filler(span)
    second = assistant + filler(span - len(assistant))
    tail = assistant + filler(max(0, CANARY_TAIL - len(assistant)))
    return first + second + tail


def cache_digest(cache: list[Any]) -> str:
    """sha256 of exactly what a snapshot stores of a cache: each array's name, dtype, shape and bytes, and the
    layers' classes and plain fields."""

    import mlx.core as mx

    from tensorfold.engine.prefix_snapshots import snapshot_arrays

    arrays, meta = snapshot_arrays("canary", [], cache)
    digest = hashlib.sha256(meta["layers"].encode())
    for name in sorted(arrays):
        value = arrays[name]
        mx.eval(value)
        digest.update(f"\0{name}|{value.dtype}|{tuple(value.shape)}\0".encode())
        flat = value.reshape(-1)
        raw = np.array(flat.view(mx.uint8)) if flat.size else np.zeros(0, np.uint8)
        digest.update(raw.tobytes())
    return digest.hexdigest()


def restored(cache: list[Any], tokens: list[int]) -> list[Any]:
    """``cache`` as a later server reads it back: encoded as a snapshot file and loaded through ``load_snapshot`` (in a
    temporary directory, never the snapshot one), so an edit to storing or restoring states moves the digest too."""

    import tempfile

    from tensorfold.engine.prefix_snapshots import encode_snapshot, load_snapshot

    payload = encode_snapshot("canary", tokens, cache)
    with tempfile.TemporaryDirectory(prefix="tensorfold-canary-") as directory:
        path = Path(directory) / "canary.safetensors"
        path.write_bytes(payload)
        del payload
        loaded = load_snapshot(path, "canary")
    if loaded is None:
        raise RuntimeError("the canary's prompt state did not load back from its snapshot")
    return loaded[1]


def canary_digest(engine: Any, plan: Any, vocab: int) -> str:
    """Prefill the canary on ``engine`` (the production plan), store and restore the cache as a snapshot, and
    digest what was restored."""

    tokens = canary_tokens(plan, vocab)
    cache = engine.prefill_prefix(tokens)
    try:
        return f"{len(tokens)}:{cache_digest(restored(cache, tokens))[:16]}"
    finally:
        del cache
        release = getattr(engine, "release_rounds", None)
        if release is not None:
            release()


def _entry_ok(entry: Any) -> bool:
    return (isinstance(entry, dict) and isinstance(entry.get("canary"), str)
            and isinstance(entry.get("saved"), (int, float)) and not isinstance(entry.get("saved"), bool))


def _load_cache(path: Path) -> tuple[dict[str, Any], bool]:
    """The cache file's well-formed entries, and whether it held anything else (then it is rewritten)."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}, False
    except OSError:
        return {}, True
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):      # not JSON, not UTF-8, nested past the parser
        return {}, True
    if not isinstance(data, dict):
        return {}, True
    entries = {k: v for k, v in data.items() if isinstance(k, str) and _entry_ok(v)}
    return entries, len(entries) != len(data)


def _store_cache(path: Path, entries: dict[str, Any]) -> None:
    newest = sorted(((k, v) for k, v in entries.items() if _entry_ok(v)), key=lambda item: float(item[1]["saved"]),
                    reverse=True)
    data = dict(newest[:_CACHE_ENTRIES])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        partial.write_text(json.dumps(data, indent=1, sort_keys=True))
        partial.replace(path)
    except OSError as exc:
        print(f"[tensorfold] could not record the numerics canary in {path}: {exc}", flush=True)


def options_fingerprint(options: dict[str, Any] | None) -> str:
    """The load options that pick compute paths (SSD experts, PLE on SSD, drafts, lane kernels, lanes, memory), in a
    canonical form."""

    return json.dumps(options or {}, sort_keys=True, default=str, separators=(",", ":"))


def canary(identity: str, compute: Callable[[], str], cache_file: Path | None,
           options: dict[str, Any] | None = None) -> str:
    """The canary digest for ``identity`` (the declared key) on this code, environment and load options, computed on
    a miss. A cache file holding anything but well-formed entries is rewritten."""

    code = code_fingerprint()
    lookup = hashlib.sha256(f"{identity}\0{code}\0{environment_fingerprint()}\0{options_fingerprint(options)}"
                            .encode()).hexdigest()
    entries, corrupt = _load_cache(cache_file) if cache_file is not None else ({}, False)
    if corrupt:
        print(f"[tensorfold] the numerics canary cache {cache_file} held malformed entries; rewriting it", flush=True)
    known = entries.get(lookup)
    if known is not None:
        if corrupt:
            _store_cache(cache_file, entries)
        return known["canary"]
    started = time.perf_counter()
    value = compute()
    print(f"[tensorfold] numerics canary {value} for this build in {time.perf_counter() - started:.1f}s "
          "(prompt snapshots carry it: a build that computes other bits never reads them)", flush=True)
    if cache_file is not None:
        if code_on_disk() != code:
            print("[tensorfold] the installed code changed while this server started: its numerics canary is not "
                  "recorded", flush=True)
            return value
        entries, _ = _load_cache(cache_file)            # another server may have written meanwhile
        entries[lookup] = {"canary": value, "saved": time.time()}
        _store_cache(cache_file, entries)
    return value


def model_id(model_dir: Path, *, mlx: str, mlx_lm: str, kernels: str, plan: str, weights: str,
             canary_value: str, strict: bool = False) -> str:
    """The snapshot key of a declared-numerics family (the model path first: blocks group by it)."""

    key = (f"{Path(model_dir).resolve()}|mlx={mlx}|mlx_lm={mlx_lm}|kernels={kernels}|prefill={plan}"
           f"|numerics=e{ENGINE_NUMERICS}|canary={canary_value}")
    if strict:
        key += f"|code={code_fingerprint()}"
    return f"{key}|weights={weights}"


def strict_requested() -> bool:
    return os.environ.get("TF_SNAPSHOT_KEY", "").strip().lower() == "strict"


def model_vocab(model: Any, tokenizer: Any = None) -> int:
    """The embedding's row count: canary ids stay inside it."""

    for holder in (model, getattr(model, "model", None)):
        args = getattr(holder, "args", None)
        size = getattr(args, "vocab_size", None)
        if size:
            return int(size)
    size = getattr(tokenizer, "vocab_size", None)
    if size:
        return int(size)
    raise ValueError("cannot tell the model's vocabulary size for the numerics canary")


def cache_file_for(*directories: Path | None) -> Path | None:
    for directory in directories:
        if directory is not None:
            return Path(directory) / CANARY_FILE
    return None


__all__ = ["CANARY_FILE", "ENGINE_NUMERICS", "cache_digest", "cache_file_for", "canary", "canary_digest",
           "canary_tokens", "code_fingerprint", "code_on_disk", "declared", "environment_fingerprint", "model_id",
           "model_vocab", "options_fingerprint", "restored", "strict_requested"]
