"""Mode C of the decode profiler: Flash Next decode parts on random weights of the q8 checkpoint's shapes (no model
data is read, no lock needed). Floors to set Mode A's buckets against:

  * GPU read bandwidth (a plain reduction, and an 8-bit lm_head matvec) - the measured GB/s for the floor;
  * the dense projections a decode forward reads, by class (36 GDN layers, 12 sparse-attention layers, 97
    hyper-connection pairs, shared experts, lm_head), through the engine's own ``project`` (lane_qmm on M5), rows 1..4;
  * routed experts: 10 x rows experts per layer through the resident expert kernels, 48 layers;
  * the hostsync mark round trip: 48 GPU->host->GPU marks a forward answered by the native pool server with no
    reads, against the same forward without marks (the per-mark floor of every decode forward);
  * one tiny mx.eval round trip (what Mode B's barriers cost each).

    python tools/bench_flash_decode_parts.py [--rows 1,2,3,4] [--reps 7] [--json OUT]"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

H, VOCAB, LAYERS, GDN_LAYERS, ATTN_LAYERS = 2560, 248320, 48, 36, 12
# (out, in) of each 8-bit g64 projection a layer reads, from the q8 checkpoint's headers
GDN = [(10240, H), (6144, H), (48, H), (48, H), (H, 6144)]
ATTN = [(12288, H), (512, H), (512, H), (H, 6144), (640, H)]
HC = [(320, H), (10240, 320)]                      # one hyper-connection: down, up (x2 a layer, +1 final mixer)
SHARED = [(640, H), (640, H), (H, 640)]
EXPERT = (640, H)                                  # gate and up; down is (H, 640)
TOPK, EXPERTS_HELD = 10, 64


def timed(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def linear(out_f: int, in_f: int) -> nn.QuantizedLinear:
    lin = nn.QuantizedLinear(in_f, out_f, bias=False, group_size=64, bits=8)
    mx.eval(lin.parameters())
    return lin


def nbytes(lins) -> int:
    return sum(l.weight.nbytes + l.scales.nbytes + l.biases.nbytes for l in lins)


def bandwidth(reps: int) -> dict:
    a = mx.random.randint(0, 1 << 30, (2**29,), dtype=mx.uint32)      # 2 GiB
    mx.eval(a)
    ms = timed(lambda: mx.eval(mx.sum(a)), reps)
    del a
    head = linear(VOCAB, H)
    from tensorfold.families.qwen4_exp.decode import project

    x = mx.random.normal((1, H)).astype(mx.bfloat16)
    ms_head = timed(lambda: mx.eval(project(x, head)), reps)
    return {"reduce_GBps": 2**31 / (ms * 1e-3) / 1e9, "lm_head_ms": ms_head,
            "lm_head_GBps": nbytes([head]) / (ms_head * 1e-3) / 1e9}


def dense(rows: list[int], reps: int) -> dict:
    from tensorfold.families.qwen4_exp.decode import project

    classes = {
        "gdn": [linear(*s) for _ in range(GDN_LAYERS) for s in GDN],
        "attn": [linear(*s) for _ in range(ATTN_LAYERS) for s in ATTN],
        "hc": [linear(*s) for _ in range(2 * LAYERS + 1) for s in HC],
        "shared": [linear(*s) for _ in range(LAYERS) for s in SHARED],
        "lm_head": [linear(VOCAB, H)],
    }
    out = {}
    for name, lins in classes.items():
        for r in rows:
            xs = {}

            def run():
                outs = []
                for lin in lins:
                    k = int(lin.weight.shape[1]) * 4
                    x = xs.setdefault(k, mx.random.normal((r, k)).astype(mx.bfloat16))
                    outs.append(project(x, lin))
                mx.eval(outs)

            ms = timed(run, reps)
            out[f"{name}_r{r}"] = {"ms": ms, "GiB": nbytes(lins) / 2**30, "GBps": nbytes(lins) / (ms * 1e-3) / 1e9}
        del lins
        mx.clear_cache()
    return out


def routed(rows: list[int], reps: int) -> dict:
    from types import SimpleNamespace

    from tensorfold.kernels.qwen.flash_next.v1 import experts

    def stack(o, i):
        w = mx.random.normal((EXPERTS_HELD, o, i)).astype(mx.bfloat16) * 0.02
        q, s, b = mx.quantize(w, group_size=64, bits=8)
        mx.eval(q, s, b)
        return SimpleNamespace(weight=q, scales=s, biases=b)

    held = 8                                   # 8 distinct layers' stacks (2.5 GiB) cycled over 48 layers: no cache reuse
    layers = [(stack(*EXPERT), stack(*EXPERT), stack(H, EXPERT[0])) for _ in range(held)]
    sh = [stack(640, H), stack(640, H), stack(H, 640)]
    shared = tuple(SimpleNamespace(weight=m.weight[0], scales=m.scales[0], biases=m.biases[0]) for m in sh)
    out = {}
    for r in rows:
        x = mx.random.normal((r, H)).astype(mx.bfloat16)
        rng = np.random.default_rng(r)
        logits = [mx.array(rng.normal(0, 1, (r, EXPERTS_HELD + 1)).astype(np.float32)) for _ in range(LAYERS)]

        def run():
            ys = []
            for li in range(LAYERS):
                g, u, d = layers[li % held]
                act, picks, _ = experts.expert_gateup(x, logits[li], TOPK, EXPERTS_HELD, g, u, shared=shared[:2])
                ys.append(experts.expert_down_y(act, picks, d, shared[2]))
            mx.eval(ys)

        ms = timed(run, reps)
        # distinct experts a row set reads (10 per row; rows rarely share all) - the bytes are the upper bound
        read = LAYERS * min(EXPERTS_HELD, TOPK * r) * (3 * EXPERT[0] * H * (1 + 4 / 64))
        out[f"experts_r{r}"] = {"ms": ms, "GBps_upper": read / (ms * 1e-3) / 1e9}
    return out


def marks(reps: int) -> dict:
    """48 hold() marks a forward, answered by the native pool server (every call a hit), against no marks."""

    from tensorfold.streaming import build, pool

    hs = build.load()
    folder = Path(tempfile.mkdtemp(prefix="tf-bench-marks-"))
    experts_n, top = 8, 1
    tensors, data, names = {}, b"", {}
    for layer in range(LAYERS):
        for proj in ("gate", "up", "down"):
            for part, (dtype, size) in {"weight": ("U32", 4), "scales": ("BF16", 2), "biases": ("BF16", 2)}.items():
                one = 16 * size
                name = f"language_model.model.layers.{layer}.mlp.switch_mlp.{proj}_proj.{part}"
                tensors[name] = {"dtype": dtype, "shape": [experts_n, 4, 4], "data_offsets": [len(data), len(data) + experts_n * one]}
                data += bytes(experts_n * one)
                names[(layer, proj, part)] = name
    header = json.dumps(tensors).encode()
    (folder / "model-00001-of-00001.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + data)
    found, shapes = pool.sources(folder, names)
    s = pool.Streamer(found, shapes, layers=LAYERS, experts=experts_n, top_k=top, slots=4 * experts_n * LAYERS,
                      box_rows=1, hostsync=hs, native=True)
    x = mx.ones((1, H), dtype=mx.bfloat16)
    mx.eval(x)

    def forward(with_marks: bool):
        h = x
        for layer in range(LAYERS):
            h = h * 1.0001
            if with_marks:
                h = s.hold(h, "rows", layer, 1)
            mx.async_eval(h)
        mx.eval(h)

    plain = timed(lambda: forward(False), reps)
    marked = timed(lambda: forward(True), reps)
    stats = s.stats()
    s.close()
    tiny = mx.ones((4,))
    mx.eval(tiny)
    ev = timed(lambda: [mx.eval(tiny + i) for i in range(100)], reps) / 100
    return {"forward_no_marks_ms": plain, "forward_48_marks_ms": marked, "per_mark_us": (marked - plain) / LAYERS * 1e3,
            "pool": stats, "eval_roundtrip_us": ev * 1e3}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--only", default="bandwidth,marks,dense,routed")
    args = ap.parse_args(argv)
    rows = [int(r) for r in args.rows.split(",")]
    result = {"device": mx.device_info() if hasattr(mx, "device_info") else {}}
    for part in args.only.split(","):
        fn = {"bandwidth": lambda: bandwidth(args.reps), "marks": lambda: marks(args.reps),
              "dense": lambda: dense(rows, args.reps), "routed": lambda: routed(rows, args.reps)}[part]
        result[part] = fn()
        print(json.dumps({part: result[part]}, default=str), flush=True)
        mx.clear_cache()
    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
