"""Pack a checkpoint's routed experts contiguously for the SSD expert pool: one aligned record per (layer, expert).

    python tools/pack_experts.py MODEL_DIR [--out PATH] [--buffer-mb 256]

Writes MODEL_DIR/experts.pack and experts.pack.json (the index, written last). The pool reads a missing expert
with one pread of its record when the pack validates against the checkpoint (TENSORFOLD_EXPERT_PACK=path picks
another location, =0 turns packs off). Needs as much free disk as the experts take."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from tensorfold.streaming import pack
from tensorfold.streaming.pool import sources


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="the pack file (default MODEL_DIR/experts.pack)")
    ap.add_argument("--buffer-mb", type=int, default=256, help="read buffer bound (default 256 MiB)")
    ap.add_argument("--alignment", type=int, default=pack.ALIGNMENT, help="record alignment in bytes (16 KiB)")
    ap.add_argument("--check", action="store_true", help="only validate an existing pack against the checkpoint")
    args = ap.parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    out = (args.out or model_dir / pack.PACK_NAME).expanduser().resolve()
    names, experts = pack.model_experts(model_dir)
    found, _ = sources(model_dir, names)
    if args.check:
        lay = pack.validate(out, found, experts)
        print(f"{out}: valid, {len(lay.layers)} layers x {experts} experts, {lay.stride} B records")
        return 0
    lay = pack.layout(found, experts, args.alignment)
    print(f"packing {len(lay.layers)} layers x {experts} experts: {lay.payload} B an expert in {lay.stride} B "
          f"records, {lay.total / 2**30:.1f} GiB -> {out}", flush=True)
    start = time.monotonic()
    pack.build(found, experts, out, alignment=args.alignment, buffer_bytes=args.buffer_mb << 20,
               progress=lambda line: print(f"  {line} ({time.monotonic() - start:.0f}s)", flush=True))
    pack.validate(out, found, experts)
    print(f"done in {time.monotonic() - start:.0f}s: {out} and {pack.index_path(out).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
