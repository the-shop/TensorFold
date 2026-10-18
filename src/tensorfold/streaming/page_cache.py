"""Page-cache residency of checkpoint files (mincore) and a no-sudo eviction (mmap + msync MS_INVALIDATE).

    python tools/page_cache.py resident MODEL_DIR [--experts]
    python tools/page_cache.py evict MODEL_DIR

``resident`` prints each file's bytes in the page cache; ``--experts`` counts only the routed experts' byte ranges.
``evict`` drops every *.safetensors of MODEL_DIR from the page cache, then prints residency. Run it
before every cold SSD benchmark, with no server holding the model open: F_NOCACHE reads skip caching new pages but
still serve pages already resident, so a "cold" run otherwise reads whatever earlier runs left behind."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import sys
from pathlib import Path

import numpy as np

MS_INVALIDATE = 0x2           # <sys/mman.h> on macOS and Linux
PAGE = mmap.PAGESIZE

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
_FAILED = ctypes.c_void_p(-1).value


def _mapped(path: Path, fn) -> None:
    """fn(address, size) over a shared read-only mapping of the whole file (nothing for an empty file)."""

    n = os.path.getsize(path)
    if n == 0:
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, n, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, _FAILED):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            fn(addr, n)
        finally:
            _libc.munmap(addr, n)
    finally:
        os.close(fd)


def pages(path: str | Path) -> np.ndarray:
    """One 0/1 flag a page of ``path``: 1 when that page is in the page cache."""

    out = np.zeros(-(-os.path.getsize(path) // PAGE), dtype=np.uint8)

    def probe(addr: int, n: int) -> None:
        vec = ctypes.create_string_buffer(len(out))
        if _libc.mincore(addr, n, vec) != 0:
            raise OSError(ctypes.get_errno(), f"mincore {path}")
        out[:] = np.frombuffer(vec.raw, dtype=np.uint8) & 1

    _mapped(Path(path), probe)
    return out


def resident(path: str | Path, ranges: list[tuple[int, int]] | None = None) -> int:
    """Bytes of ``path`` in the page cache (whole pages), or of the pages touching ``ranges`` [(offset, nbytes)]."""

    flags = pages(path)
    if ranges is None:
        return int(flags.sum()) * PAGE
    hit = np.zeros_like(flags, dtype=bool)
    for offset, nbytes in ranges:
        if nbytes > 0:
            hit[offset // PAGE:(offset + nbytes - 1) // PAGE + 1] = True
    return int(flags[hit].sum()) * PAGE


def evict(path: str | Path) -> None:
    """Drop ``path``'s pages from the page cache: msync(MS_INVALIDATE) over a shared mapping (no sudo needed)."""

    def drop(addr: int, n: int) -> None:
        if _libc.msync(addr, n, MS_INVALIDATE) != 0:
            raise OSError(ctypes.get_errno(), f"msync {path}")

    _mapped(Path(path), drop)


def model_files(model_dir: Path) -> list[Path]:
    """The checkpoint shards of ``model_dir`` (regular files only: mmap refuses anything else)."""

    return sorted(p for p in Path(model_dir).glob("*.safetensors") if p.is_file())


def model_experts(model_dir: Path) -> tuple[dict, int]:
    """(routed expert tensor names by key, experts a layer) as the family serving ``model_dir`` streams them."""

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config") or config
    kind = str(config.get("model_type", ""))
    if kind.startswith("glm5_next"):
        from tensorfold.families.glm5_next.stream import expert_names as glm

        return glm(model_dir, int(text["num_hidden_layers"])), int(text["n_routed_experts"])
    if kind.startswith("qwen4_exp"):
        from tensorfold.families.qwen4_exp.stream import expert_names as qwen

        return qwen(int(text["num_hidden_layers"])), int(text["num_experts"])
    raise ValueError(f"no SSD expert streaming for model_type {kind!r}")


def expert_ranges(model_dir: Path) -> dict[Path, list[tuple[int, int]]]:
    """Each file's routed-expert byte ranges [(offset, nbytes)], as the expert pool reads them."""

    from tensorfold.streaming.pool import sources

    names, experts = model_experts(model_dir)
    found, _ = sources(model_dir, names)
    out: dict[Path, list[tuple[int, int]]] = {}
    for key, src in found.items():               # (layer, proj, part) a stack of every expert, else one expert
        out.setdefault(Path(src.path), []).append((src.offset, src.nbytes * (experts if len(key) == 3 else 1)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("resident", "evict"))
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--experts", action="store_true", help="count only the routed experts' byte ranges")
    args = ap.parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    files = model_files(model_dir)
    if not files:
        print(f"no *.safetensors in {model_dir}", file=sys.stderr)
        return 1
    ranges = expert_ranges(model_dir) if args.experts else {}     # reads the headers: before an eviction
    if args.action == "evict":
        for path in files:
            evict(path)
    total = size = 0
    for path in files:
        if args.experts and path not in ranges:
            continue
        span = ranges.get(path) if args.experts else None
        got = resident(path, span)
        n = sum(b for _, b in span) if span else os.path.getsize(path)
        total, size = total + got, size + n
        print(f"{path.name}\t{got}\t{n}")
    print(f"total\t{total}\t{size}\t{100 * total / max(size, 1):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
