"""A contiguous copy of the routed experts: each (layer, expert)'s parts in one aligned record, read with one pread.

The pack is ``experts.pack`` (records) beside ``experts.pack.json`` (the index). Record r = (layer position, expert)
holds every projection's weight, scales and biases, in the pool's order (projections sorted, then PARTS), back to
back, padded to ``alignment``. The index carries a hash of the source tensor table and of each source file's size,
inode and modification time, and a sample of the experts' bytes (the head and tail 4 KiB of every part of the first
and last expert of each layer); validate re-reads that sample from the checkpoint and from the pack. A rewritten,
copied or touched checkpoint is refused; a change that keeps every file's size, inode and mtime and avoids the
sampled bytes is not detected. Packs are opt-in: TENSORFOLD_EXPERT_PACK=1 (MODEL_DIR/experts.pack) or =PATH."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

VERSION = 2
SAMPLE = 4096                 # bytes from the head and from the tail of each sampled part
ALIGNMENT = 16384
PACK_NAME = "experts.pack"
ENV = "TENSORFOLD_EXPERT_PACK"
_OFF = {"", "0", "off", "no", "none", "false"}
_ON = {"1", "on", "yes", "true"}


class PackError(ValueError):
    """A pack that is missing, incomplete or built from another checkpoint than the sources it would serve."""


@dataclass
class Layout:
    """Where each part of an expert sits in its record, and how records are spaced."""

    layers: list[int]                     # the layers with routed experts, in record order
    experts: int
    order: list[tuple[str, str]]          # (proj, part), the pool's order
    part_nbytes: list[int]
    part_offsets: list[int]
    payload: int
    stride: int
    alignment: int

    def record(self, layer: int, expert: int) -> int:
        """Byte offset of (layer, expert)'s record in the pack."""

        return (self.layers.index(layer) * self.experts + expert) * self.stride

    @property
    def total(self) -> int:
        return len(self.layers) * self.experts * self.stride


def index_path(pack: Path) -> Path:
    return pack.with_name(pack.name + ".json")


def _source(found: dict, layer: int, proj: str, part: str, expert: int):
    one = found.get((layer, proj, part, expert))
    if one is not None:
        return one, one.offset
    stack = found[(layer, proj, part)]
    return stack, stack.offset + expert * stack.nbytes


def layout(found: dict, experts: int, alignment: int = ALIGNMENT) -> Layout:
    """The record layout of ``found``'s experts (keys (layer, proj, part) of stacks or (..., expert))."""

    from tensorfold.streaming.pool import PARTS

    check_alignment(alignment)
    projs = sorted({key[1] for key in found})
    layers = sorted({key[0] for key in found})
    order = [(proj, part) for proj in projs for part in PARTS]
    sizes = [_source(found, layers[0], proj, part, 0)[0].nbytes for proj, part in order]
    offsets, at = [], 0
    for n in sizes:
        offsets.append(at)
        at += n
    stride = -(-at // alignment) * alignment
    return Layout(layers, experts, order, sizes, offsets, at, stride, alignment)


def check_alignment(alignment: int) -> int:
    """Record alignment: a power of two of at least 4 KiB (uncached reads into page-aligned buffers)."""

    if not isinstance(alignment, int) or alignment < 4096 or alignment & (alignment - 1):
        raise PackError(f"pack alignment {alignment!r}: a power of two of at least 4096 bytes is needed")
    return alignment


def source_hash(found: dict) -> str:
    """A digest of where every expert tensor's bytes live (file name, offset, bytes) and each file's size, inode
    and modification time (a rewritten, replaced or copied checkpoint changes it)."""

    digest = hashlib.sha256()
    for key in sorted(found, key=lambda k: tuple(str(x) for x in k)):
        src = found[key]
        digest.update(f"{key}|{Path(src.path).name}|{src.offset}|{src.nbytes}\n".encode())
    for path in sorted({Path(src.path) for src in found.values()}):
        st = path.stat()
        digest.update(f"{path.name}|{st.st_size}|{st.st_ino}|{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _spans(nbytes: int) -> list[tuple[int, int]]:
    """(offset, bytes) of a part's sampled head and tail."""

    n = min(SAMPLE, nbytes)
    return [(0, n), (nbytes - n, n)]


def content_sample(found: dict, lay: Layout, read_pack: Callable[[int, int], bytes] | None = None) -> str:
    """sha256 over the head and tail of every part of each layer's first and last expert, read from the checkpoint
    (or, with ``read_pack(offset, nbytes)``, from the pack's records)."""

    digest = hashlib.sha256()
    fds: dict = {}
    try:
        for layer in lay.layers:
            for e in sorted({0, lay.experts - 1}):
                for (proj, part), nb, off in zip(lay.order, lay.part_nbytes, lay.part_offsets):
                    for at, n in _spans(nb):
                        if read_pack is not None:
                            digest.update(read_pack(lay.record(layer, e) + off + at, n))
                            continue
                        src, base = _source(found, layer, proj, part, e)
                        path = Path(src.path)
                        if path not in fds:
                            fds[path] = _uncached(path)
                        digest.update(_pread(fds[path], n, base + at))
    finally:
        for fd in fds.values():
            os.close(fd)
    return digest.hexdigest()


def _uncached(path: Path, flags: int = os.O_RDONLY) -> int:
    fd = os.open(path, flags, 0o644)
    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
    return fd


def _pread(fd: int, nbytes: int, offset: int) -> bytes:
    out = bytearray()
    while len(out) < nbytes:
        got = os.pread(fd, nbytes - len(out), offset + len(out))
        if not got:
            raise OSError(f"short read at {offset + len(out)}")
        out += got
    return bytes(out)


def build(found: dict, experts: int, out: Path, *, alignment: int = ALIGNMENT, buffer_bytes: int = 256 << 20,
          progress: Callable[[str], None] | None = None) -> Path:
    """Write the pack for ``found`` to ``out`` (and its index), reading the sources uncached in bounded chunks."""

    out = Path(out)
    lay = layout(found, experts, alignment)
    sample = content_sample(found, lay)                  # before the copy: a checkpoint changing under it is refused
    free = shutil.disk_usage(out.parent).free
    if free < lay.total + (1 << 30):
        raise PackError(f"the pack needs {lay.total / 2**30:.1f} GiB on {out.parent}, {free / 2**30:.1f} GiB is free")
    for stale in (out, index_path(out)):
        if stale.exists():
            stale.unlink()
    partial = out.with_name(out.name + ".partial")
    src_fds = {path: _uncached(path) for path in {Path(s.path) for s in found.values()}}
    dst = _uncached(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    group = max(1, min(experts, buffer_bytes // lay.stride))
    try:
        os.ftruncate(dst, lay.total)
        for li, layer in enumerate(lay.layers):
            for e0 in range(0, experts, group):
                g = min(group, experts - e0)
                buf = bytearray(g * lay.stride)
                view = memoryview(buf)
                for (proj, part), nb, off in zip(lay.order, lay.part_nbytes, lay.part_offsets):
                    stack = found.get((layer, proj, part))
                    if stack is not None:                  # g experts of a stack: one contiguous read
                        data = _pread(src_fds[Path(stack.path)], g * nb, stack.offset + e0 * nb)
                        for j in range(g):
                            view[j * lay.stride + off:j * lay.stride + off + nb] = data[j * nb:(j + 1) * nb]
                        continue
                    for j in range(g):
                        src, at = _source(found, layer, proj, part, e0 + j)
                        if src.nbytes != nb:
                            raise PackError(f"expert tensor of layer {layer} {proj}.{part}: {src.nbytes} bytes, "
                                            f"expected {nb}")
                        view[j * lay.stride + off:j * lay.stride + off + nb] = _pread(src_fds[Path(src.path)], nb, at)
                at, done = (li * experts + e0) * lay.stride, 0
                while done < len(buf):
                    done += os.pwrite(dst, view[done:], at + done)
            if progress is not None:
                progress(f"layer {layer}: {li + 1}/{len(lay.layers)} layers packed")
        os.fsync(dst)
    finally:
        os.close(dst)
        for fd in src_fds.values():
            os.close(fd)
    os.replace(partial, out)
    index = {"version": VERSION, "pack": out.name, "alignment": alignment, "layers": lay.layers, "experts": experts,
             "order": [list(o) for o in lay.order], "part_nbytes": lay.part_nbytes, "part_offsets": lay.part_offsets,
             "payload": lay.payload, "stride": lay.stride, "pack_nbytes": lay.total, "source_hash": source_hash(found),
             "content_sample": sample}
    tmp = index_path(out).with_name(index_path(out).name + ".partial")
    tmp.write_text(json.dumps(index, indent=1))
    os.replace(tmp, index_path(out))                       # the index last: its presence means the pack is whole
    return out


def validate(pack: Path, found: dict, experts: int) -> Layout:
    """The pack's layout when it matches ``found`` byte for byte in placement, else PackError."""

    pack = Path(pack)
    idx = index_path(pack)
    if not pack.is_file() or not idx.is_file():
        raise PackError(f"expert pack {pack}: the pack or its index {idx.name} is missing")
    try:
        index = json.loads(idx.read_text())
    except (OSError, ValueError) as error:
        raise PackError(f"expert pack {pack}: unreadable index ({error})") from None
    if index.get("version") != VERSION:
        raise PackError(f"expert pack {pack}: index version {index.get('version')}, expected {VERSION}")
    alignment = index.get("alignment")
    if not isinstance(alignment, int):
        raise PackError(f"expert pack {pack}: the index has no alignment")
    lay = layout(found, experts, alignment)
    want = {"layers": lay.layers, "experts": experts, "order": [list(o) for o in lay.order],
            "part_nbytes": lay.part_nbytes, "part_offsets": lay.part_offsets, "stride": lay.stride,
            "pack_nbytes": lay.total}
    for name, value in want.items():
        if index.get(name) != value:
            raise PackError(f"expert pack {pack}: {name} differs from the checkpoint's experts (rebuild the pack)")
    if index.get("source_hash") != source_hash(found):
        raise PackError(f"expert pack {pack}: built from other checkpoint files (stale, rebuild the pack)")
    if pack.stat().st_size != lay.total:
        raise PackError(f"expert pack {pack}: {pack.stat().st_size} bytes, the index says {lay.total}")
    sample = index.get("content_sample")
    if sample != content_sample(found, lay):
        raise PackError(f"expert pack {pack}: the checkpoint's expert bytes differ from the pack's (stale, rebuild)")
    fd = _uncached(pack)
    try:
        if sample != content_sample(found, lay, lambda at, n: _pread(fd, n, at)):
            raise PackError(f"expert pack {pack}: its records do not hold the checkpoint's bytes (corrupt, rebuild)")
    finally:
        os.close(fd)
    return lay


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
    raise PackError(f"no SSD expert streaming for model_type {kind!r}")


def choose(model_dir: Path, env: dict | None = None, pack: str | Path | bool | None = None) -> Path | None:
    """The pack to use, or None (the default: packs are opt-in). ``pack``: False is off, True is
    MODEL_DIR/experts.pack, a path is that pack; None follows TENSORFOLD_EXPERT_PACK (unset/0: off, 1:
    MODEL_DIR/experts.pack, else a path)."""

    if pack is False:
        return None
    if pack is True:
        return Path(model_dir) / PACK_NAME
    if pack is not None:
        return Path(pack).expanduser()
    env = os.environ if env is None else env
    value = (env.get(ENV) or "").strip()
    if value.lower() in _OFF:
        return None
    if value.lower() in _ON:
        return Path(model_dir) / PACK_NAME
    return Path(value).expanduser()


__all__ = ["ALIGNMENT", "ENV", "PACK_NAME", "Layout", "PackError", "build", "check_alignment", "choose",
           "content_sample", "index_path", "layout", "model_experts", "source_hash", "validate"]
