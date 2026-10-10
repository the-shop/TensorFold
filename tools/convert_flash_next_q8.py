"""Stream a BF16 Hugging Face Qwen3.8 Flash Next checkpoint into an MLX affine 8-bit TensorFold checkpoint.

Tensor by tensor (expert stacks and big tables in row chunks), never holding more than one chunk plus its
quantized output: reads with pread (no page cache), quantizes with mx.quantize under a hard MLX memory limit,
and writes each output tensor at its planned offset in the planned safetensors shard.

Layout (what tensorfold.families.qwen4_exp reads):
  model.language_model.X -> language_model.model.X; lm_head -> language_model.lm_head; mtp.X -> language_model.mtp.X
  mlp.experts.gate_up_proj [E, 2I, D] -> mlp.switch_mlp.{gate,up}_proj, mlp.experts.down_proj -> switch_mlp.down_proj
  conv1d weights [C, 1, K] -> [C, K, 1]; the vision tower is dropped.
Every Linear / Embedding / SwitchLinear of the text model and MTP head is 8-bit (groups of 64; the n-gram tables,
160 wide, groups of 32) except the routers (mlp.gate), which stay bf16 as the fused decode reads them.

usage: python convert_flash_next_q8.py SRC DST [--limit-gb 14] [--shard-gb 5] [--dry-run]
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

BITS = 8
GROUP = 64
PLE_GROUP = 32
CHUNK_BYTES = 512 << 20           # source bytes a chunk reads at most
_NP = {"BF16": np.uint16, "F32": np.float32, "I64": np.int64, "U32": np.uint32, "F16": np.float16}
_ITEM = {"BF16": 2, "F32": 4, "I64": 8, "U32": 4, "F16": 2}


def read_header(path: Path) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return 8 + n, h


def map_name(name: str) -> str | None:
    if name.startswith(("model.visual", "model.language_model.visual")):
        return None
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model."):]
    if name.startswith("lm_head."):
        return "language_model." + name
    if name.startswith("mtp."):
        return "language_model." + name
    raise ValueError(f"unexpected tensor {name}")


def quantized_modules(config: dict) -> set[str]:
    """Checkpoint module paths TensorFold quantizes (from its own module tree), routers excluded."""

    from tensorfold.families.qwen4_exp.model import Qwen4Exp
    from tensorfold.families.qwen4_exp.model_layers import Config
    from tensorfold.families.qwen4_exp.mtp import FlashMTP

    cfg = Config.from_dict(config)
    out = set()
    for prefix, module in (("language_model.", Qwen4Exp(cfg)), ("language_model.mtp.", FlashMTP(cfg))):
        for path, m in module.named_modules():
            if hasattr(m, "to_quantized") and not path.endswith("mlp.gate"):
                path = re.sub(r"ple_embedding\.shards\.(\d+)$", r"ple_embedding.ngram_embedding.shard_\1", path)
                out.add(prefix + path)
    return out


def plan(src: Path, qmods: set[str]) -> list[dict]:
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(index.values()))
    jobs: list[dict] = []
    for fname in files:
        base, header = read_header(src / fname)
        for name in sorted(header, key=lambda k: header[k]["data_offsets"][0]):
            entry = header[name]
            out = map_name(name)
            if out is None:
                continue
            shape, dtype = list(entry["shape"]), entry["dtype"]
            source = {"path": str(src / fname), "offset": base + entry["data_offsets"][0], "shape": shape,
                      "dtype": dtype}
            if out.endswith(".mlp.experts.gate_up_proj"):
                stem = out[:-len("experts.gate_up_proj")] + "switch_mlp."
                jobs.append({"kind": "gate_up", "src": source, "outs": [stem + "gate_proj", stem + "up_proj"],
                             "group": GROUP})
                continue
            if out.endswith(".mlp.experts.down_proj"):
                stem = out[:-len("experts.down_proj")] + "switch_mlp."
                jobs.append({"kind": "quant", "src": source, "outs": [stem + "down_proj"], "group": GROUP})
                continue
            if out.endswith(".weight") and out[:-len(".weight")] in qmods:
                group = PLE_GROUP if ".ngram_embedding.shard_" in out else GROUP
                if shape[-1] % group:
                    raise ValueError(f"{name}: {shape[-1]} columns do not split into groups of {group}")
                jobs.append({"kind": "quant", "src": source, "outs": [out[:-len(".weight")]], "group": group})
                continue
            if "conv1d.weight" in out and len(shape) == 3 and shape[1] == 1 and shape[2] != 1:
                jobs.append({"kind": "conv", "src": source, "outs": [out]})
                continue
            jobs.append({"kind": "copy", "src": source, "outs": [out]})
    return jobs


def outputs_of(job: dict) -> list[tuple[str, list[int], str]]:
    """(name, shape, dtype) of each tensor a job writes."""

    shape, dtype = job["src"]["shape"], job["src"]["dtype"]
    if job["kind"] == "copy":
        return [(job["outs"][0], shape, dtype)]
    if job["kind"] == "conv":
        return [(job["outs"][0], [shape[0], shape[2], shape[1]], dtype)]
    if dtype != "BF16":
        raise ValueError(f"{job['outs']}: quantizing needs a bf16 source, got {dtype}")
    g = job["group"]
    lead, cols = shape[:-1], shape[-1]
    if job["kind"] == "gate_up":
        lead = [shape[0], shape[1] // 2]
    res = []
    for name in job["outs"]:
        res += [(name + ".weight", lead + [cols * BITS // 32], "U32"),
                (name + ".scales", lead + [cols // g], "BF16"),
                (name + ".biases", lead + [cols // g], "BF16")]
    return res


def nbytes(shape: list[int], dtype: str) -> int:
    return int(np.prod(shape, dtype=np.int64)) * _ITEM[dtype]


def pread_into(fd: int, offset: int, buf: np.ndarray) -> None:
    view = memoryview(buf.reshape(-1).view(np.uint8))
    done, size = 0, view.nbytes
    while done < size:
        n = os.preadv(fd, [view[done:]], offset + done)
        if n <= 0:
            raise OSError(f"short read at {offset + done}")
        done += n


def pwrite_all(fd: int, offset: int, data: np.ndarray) -> None:
    view = memoryview(np.ascontiguousarray(data).reshape(-1).view(np.uint8))
    done = 0
    while done < view.nbytes:
        done += os.pwrite(fd, view[done:], offset + done)


def nocache(fd: int) -> None:
    fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)


def run_job(job: dict, fds: dict, place: dict, out_fd: int) -> None:
    import mlx.core as mx

    src = job["src"]
    fd = fds.get(src["path"])
    if fd is None:
        fd = fds[src["path"]] = os.open(src["path"], os.O_RDONLY)
        nocache(fd)
    shape, dtype = src["shape"], src["dtype"]
    if job["kind"] in ("copy", "conv"):
        buf = np.empty(shape, dtype=_NP[dtype])
        pread_into(fd, src["offset"], buf)
        if job["kind"] == "conv":
            buf = np.ascontiguousarray(np.moveaxis(buf, 2, 1))
        pwrite_all(out_fd, place[job["outs"][0]], buf)
        return
    g = job["group"]
    lead_rows = shape[0]
    row_bytes = nbytes(shape[1:], dtype) if len(shape) > 1 else 2
    step = max(1, CHUNK_BYTES // row_bytes)
    names = job["outs"]
    for r0 in range(0, lead_rows, step):
        r1 = min(lead_rows, r0 + step)
        buf = np.empty([r1 - r0] + shape[1:], dtype=np.uint16)
        pread_into(fd, src["offset"] + r0 * row_bytes, buf)
        x = mx.array(buf).view(mx.bfloat16)
        if job["kind"] == "gate_up":
            half = shape[1] // 2
            parts = [x[:, :half, :], x[:, half:, :]]
        else:
            parts = [x]
        for name, part in zip(names, parts):
            w, s, b = mx.quantize(part, group_size=g, bits=BITS)
            mx.eval(w, s, b)
            for suffix, arr in ((".weight", w), (".scales", s), (".biases", b)):
                host = np.array(arr.view(mx.uint32) if suffix == ".weight" else arr.view(mx.uint16))
                per_row = host.nbytes // (r1 - r0)
                pwrite_all(out_fd, place[name + suffix] + r0 * per_row, host)
            del w, s, b
        del x, parts, buf
        mx.clear_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--limit-gb", type=float, default=14.0)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-shard", type=int, default=None, help="write only this output shard (1-based)")
    args = ap.parse_args()

    import mlx.core as mx

    mx.set_memory_limit(int(args.limit_gb * 2**30))
    mx.set_cache_limit(1 << 30)
    config = json.loads((args.src / "config.json").read_text())
    qmods = quantized_modules(config)
    jobs = plan(args.src, qmods)

    # shards: consecutive jobs until shard-gb; a job's tensors never split across files
    shards: list[list[dict]] = [[]]
    size = 0
    limit = int(args.shard_gb * 1e9)
    for job in jobs:
        jb = sum(nbytes(s, d) for _, s, d in outputs_of(job))
        if shards[-1] and size + jb > limit:
            shards.append([])
            size = 0
        shards[-1].append(job)
        size += jb
    total = len(shards)
    weight_map, grand = {}, 0
    layouts = []
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        header, at = {}, 0
        for job in shard:
            for name, shape, dtype in outputs_of(job):
                n = nbytes(shape, dtype)
                header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + n]}
                weight_map[name] = fname
                at += n
        header["__metadata__"] = {"format": "mlx"}
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        layouts.append((fname, header, raw, at))
        grand += at
    qcount = sum(1 for j in jobs if j["kind"] in ("quant", "gate_up"))
    print(f"[convert] {len(jobs)} source tensors ({qcount} quantized), {len(weight_map)} output tensors, "
          f"{total} shards, {grand / 1e9:.1f} GB", flush=True)
    if args.dry_run:
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    for p in args.src.iterdir():
        if p.is_file() and not p.name.endswith(".ok") and not p.name.endswith(".safetensors") \
                and p.name not in ("model.safetensors.index.json", "config.json"):
            shutil.copy2(p, args.dst / p.name)
    quant = {"group_size": GROUP, "bits": BITS, "mode": "affine"}
    config["quantization"] = dict(quant)
    config["quantization_config"] = dict(quant)
    (args.dst / "config.json").write_text(json.dumps(config, indent=2))
    (args.dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": grand}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))

    fds: dict = {}
    t0 = time.time()
    done_bytes = 0
    for i, ((fname, header, raw, data_len), shard) in enumerate(zip(layouts, shards), 1):
        if args.only_shard and i != args.only_shard:
            continue
        path = args.dst / fname
        marker = args.dst / (fname + ".done")
        if marker.exists():
            done_bytes += data_len
            continue
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        nocache(fd)
        os.pwrite(fd, struct.pack("<Q", len(raw)) + raw, 0)
        base = 8 + len(raw)
        os.ftruncate(fd, base + data_len)
        place = {name: base + e["data_offsets"][0] for name, e in header.items() if name != "__metadata__"}
        for job in shard:
            run_job(job, fds, place, fd)
        os.fsync(fd)
        os.close(fd)
        marker.write_text("ok")
        done_bytes += data_len
        el = time.time() - t0
        print(f"[convert] {fname} {data_len / 1e9:.2f} GB  total {done_bytes / 1e9:.1f}/{grand / 1e9:.1f} GB "
              f"{el:.0f}s  peak mlx {mx.get_peak_memory() / 2**30:.2f} GiB", flush=True)
    for f in fds.values():
        os.close(f)
    print("[convert] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
