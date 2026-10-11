"""Decode profile (TENSORFOLD_DECODE_PROFILE JSONL) -> per-round time buckets, per request and phase.

    python tools/decode_profile_report.py [PROFILE.jsonl.1] PROFILE.jsonl [--attention 3,7,...]
                                          [--expert-bytes 5222400] [--json OUT]

Pass the rotated ``.1`` file first when TENSORFOLD_DECODE_PROFILE_MAX_MB rotated it. A malformed line (a flush cut by
kill -9 or a full disk) is skipped with a warning on stderr; a request header glued to its end is recovered.

Each decode round is cut at host stamps the round really passes (R1): before layer 0's signal, then per layer the
host's answer [signal seen -> ready] and the segment [ready(L-1) -> signal(L)], then the tail [ready(47) -> tokens
read] and the host work after the read. The cuts telescope, so the buckets always sum to the round's wall time; that
is an identity, not a check. A round is flagged instead when its calls are not exactly one per layer in layer order
(a lost or extra record), or a bucket comes out negative (stamps out of order). Inside a segment, the GPU start/end
of the command buffers that carry the marks split expert(L-1) GPU, Python-bound GPU idle and dense(L) GPU, where
those buffers have clean boundaries; other segments are split by the clean ones' mean fractions for that layer type.

Only calls inside a decode round count as decode. A prompt chunk of 16 rows or fewer (a short prompt, or the tail
chunk of a long one) takes the same rows path as decode (kind 0) and its misses count in /health "misses" too; those
calls fall inside the prompt chunk's span, so the prompt-chunk table shows them (rows calls) and the decode tables
do not. Decode SSD bytes are decode-round misses x the expert's bytes (R2: window fills are reported apart)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXPERT_BYTES = 5_222_400          # one routed expert of the q8 checkpoint: 9 parts (gate/up/down x weight/scales/biases)
NS = 1e-6                         # ns -> ms
SLACK_NS = 50_000                 # a buffer started within 50 us of Python's commit: the GPU waited for Python
BUCKETS = ("pre_host", "gpu_backlog0", "dense_gdn", "dense_attn", "expert_gpu", "ssd_stall", "host_wake",
           "py_idle", "tail", "conclude", "keep_rows", "settle_build", "after_other", "gap")
LABELS = {
    "pre_host": "pre-layer-0 host (plan, build, commit)",
    "gpu_backlog0": "GPU backlog before layer 0 (MTP chain + embed; A bound)",
    "dense_gdn": "dense GPU: GDN layers (hc x2, GDN, router, shared)",
    "dense_attn": "dense GPU: sparse-attn layers (hc x2, attn+indexer, router, shared)",
    "expert_gpu": "routed-expert GPU (48 layers)",
    "ssd_stall": "SSD miss stall (host reads)",
    "host_wake": "host answer w/o reads (wake, LRU, slot table, release)",
    "py_idle": "GPU idle, Python-bound (graph build behind)",
    "tail": "tail: mixer + lm_head + draw + early MTP + readback",
    "conclude": "host after read: conclude",
    "keep_rows": "host after read: keep_rows",
    "settle_build": "host after read: settle (MTP chain build)",
    "after_other": "host after read: other",
    "gap": "inter-round gap (scheduler, detokenize, HTTP)",
}


def _parse(line: str, where: str) -> dict | None:
    """One JSONL record; a malformed line is skipped with a warning, keeping a whole record glued to its end."""

    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    tail = line.rfind('{"type": ')
    rec = None
    if tail > 0:
        try:
            rec = json.loads(line[tail:])
        except json.JSONDecodeError:
            rec = None
    print(f"warning: {where}: skipped a malformed line ({len(line)} chars)"
          + (f", kept the {rec['type']} record at its end" if rec else ""), file=sys.stderr)
    return rec


def load(*paths: Path) -> list[dict]:
    """Requests: header plus its chunks and rounds, in file order (files in the order given)."""

    requests: list[dict] = []
    for path in paths:
        requests += _load(path)
    return requests


def _load(path: Path) -> list[dict]:
    requests: list[dict] = []
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        rec = _parse(line, f"{path}:{n}")
        if rec is None:
            continue
        if rec["type"] == "request":
            requests.append({"header": rec, "rounds": [], "chunks": []})
        elif requests:
            requests[-1]["rounds" if rec["type"] == "round" else "chunks"].append(rec)
    return requests


def main_run(rnd: dict) -> dict | None:
    runs = [r for r in rnd.get("runs", []) if r.get("q")]
    return max(runs, key=lambda r: r["layers"]) if runs else None


def round_buckets(rnd: dict, attention: set[int], ratios: dict | None, nxt: dict | None) -> dict:
    """One round's buckets (ns) and facts; ``ratios``: mean clean-segment fractions per layer type for the rest."""

    out = dict.fromkeys(BUCKETS, 0.0)
    calls = sorted((c for c in rnd.get("calls", []) if c["kind"] == 0), key=lambda c: c["t_submit"])
    run = main_run(rnd)
    layers = run["layers"] if run else 48
    facts = {"rows": rnd.get("rows"), "keep": rnd.get("keep"), "committed": rnd.get("committed", 0),
             "depth": rnd.get("depth"), "kind": rnd.get("kind"), "wall": rnd["t_end"] - rnd["t0"],
             "calls": len(calls), "misses": sum(c["misses"] for c in calls), "unique": sum(c["unique"] for c in calls),
             "clean": 0, "unclean": 0, "buffers": len({c.get("sig_buffer") for c in calls} |
                                                     {c.get("wait_buffer") for c in calls} - {None}),
             "ple_ms": (run or {}).get("ple_ns", 0) * NS, "build_ms": ((rnd.get("t_hidden") or rnd["t0"])
                                                                        - (rnd.get("t_plan") or rnd["t0"])) * NS,
             "mtp_runs": sum(1 for r in rnd.get("runs", []) if r is not run), "ok": True, "why": ""}
    if len(calls) != layers or "t_read" not in rnd:
        facts["ok"], facts["why"] = False, f"{len(calls)} calls for {layers} layers" if "t_read" in rnd else "no read"
        return {"b": out, "f": facts, "seg": []}
    order = [c.get("layer") for c in calls]
    if order != list(range(layers)):
        facts["ok"], facts["why"] = False, f"call layers {order[:8]}... not 0..{layers - 1} in order"
        return {"b": out, "f": facts, "seg": []}
    q = run["q"] if run else [None] * layers
    for c in calls:
        host = c["t_ready"] - c["t_sig"]
        ssd = (c["t_read1"] - c["t_read0"]) if c["misses"] else 0
        out["ssd_stall"] += ssd
        out["host_wake"] += host - ssd
    # before layer 0's signal
    c0 = calls[0]
    pre = c0["t_sig"] - rnd["t0"]
    start0 = c0.get("sig_start")
    if start0 and rnd["t0"] <= start0 <= c0["t_sig"]:
        dense0 = c0["t_sig"] - start0
        backlog = max(0, start0 - q[0]) if q[0] else 0
        backlog = min(backlog, pre - dense0)
        out["dense_attn" if 0 in attention else "dense_gdn"] += dense0
        out["gpu_backlog0"] += backlog
        out["pre_host"] += pre - dense0 - backlog
    else:
        out["pre_host"] += pre
    segs = []
    for layer in range(1, layers):
        prev, cur = calls[layer - 1], calls[layer]
        seg = cur["t_sig"] - prev["t_ready"]
        we, ss = prev.get("wait_end"), cur.get("sig_start")
        typ = "attn" if layer in attention else "gdn"
        clean = (we is not None and ss is not None and prev.get("wait_buffer") != cur.get("sig_buffer")
                 and prev["t_ready"] <= we <= cur["t_sig"] and ss >= prev["t_ready"] and ss <= cur["t_sig"])
        if clean:
            expert = we - prev["t_ready"]
            gap = max(0, ss - we)
            idle = gap if q[layer] is not None and q[layer] >= ss - SLACK_NS else 0
            dense = seg - expert - idle
            facts["clean"] += 1
            segs.append((typ, seg, expert, idle, dense))
        else:
            facts["unclean"] += 1
            r = (ratios or {}).get(typ)
            if r is None:
                expert, idle, dense = 0.0, 0.0, seg
            else:
                expert, idle = seg * r[0], seg * r[1]
                dense = seg - expert - idle
        out["expert_gpu"] += expert
        out["py_idle"] += idle
        out["dense_attn" if typ == "attn" else "dense_gdn"] += dense
    last = calls[-1]
    tail = rnd["t_read"] - last["t_ready"]
    we = last.get("wait_end")
    expert47 = (we - last["t_ready"]) if we is not None and last["t_ready"] <= we <= rnd["t_read"] else 0
    out["expert_gpu"] += expert47
    out["tail"] += tail - expert47
    t = rnd
    out["conclude"] += t.get("t_concluded", t["t_read"]) - t["t_read"]
    out["keep_rows"] += t.get("t_kept", t.get("t_concluded", t["t_read"])) - t.get("t_concluded", t["t_read"])
    out["settle_build"] += t.get("t_settled", t.get("t_kept", t["t_read"])) - t.get("t_kept", t["t_read"])
    out["after_other"] += t["t_end"] - t.get("t_settled", t.get("t_kept", t["t_read"]))
    if nxt is not None:
        out["gap"] += max(0, nxt["t0"] - rnd["t_end"])
    negative = [k for k, v in out.items() if v < 0]
    if negative:
        facts["ok"], facts["why"] = False, f"negative buckets {negative} (stamps out of order)"
    if "t_head" in rnd:                       # Mode B: the round's own evals give exact head / MTP times
        facts["b_head_ms"] = (rnd["t_head"] - (run["t_end"] if run else rnd["t_hidden"])) * NS
        facts["b_spec_ms"] = (rnd.get("t_spec", rnd["t_head"]) - rnd["t_head"]) * NS
        facts["b_chain_ms"] = (rnd.get("t_chain", rnd["t_settled"]) - rnd["t_settled"]) * NS
    return {"b": out, "f": facts, "seg": segs}


def clean_ratios(rounds: list[dict], attention: set[int]) -> dict:
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for rnd in rounds:
        for typ, seg, expert, idle, _ in round_buckets(rnd, attention, None, None)["seg"]:
            a = acc[typ]
            a[0] += seg
            a[1] += expert
            a[2] += idle
    return {typ: (a[1] / a[0], a[2] / a[0]) for typ, a in acc.items() if a[0] > 0}


def mode_b(rounds: list[dict], attention: set[int]) -> dict:
    """Mode B per-layer evals: mean ms per forward of each part (only rounds that ran with prof.sync)."""

    parts = defaultdict(float)
    n = 0
    for rnd in rounds:
        run = main_run(rnd)
        if not run or not run.get("b"):
            continue
        calls = sorted((c for c in rnd.get("calls", []) if c["kind"] == 0), key=lambda c: c["t_submit"])
        n += 1
        for layer, (t0, t1, t2, t3, t4) in enumerate(run["b"]):
            typ = "attn" if layer in attention else "gdn"
            parts[f"hc_attn_{typ}"] += t1 - t0
            parts[f"mixer_{typ}"] += t2 - t1
            parts[f"hc_mlp_{typ}"] += t3 - t2
            if layer < len(calls):
                c = calls[layer]
                parts[f"router_{typ}"] += c["t_sig"] - t3
                parts[f"host_{typ}"] += c["t_ready"] - c["t_sig"]
                parts[f"experts_{typ}"] += t4 - c["t_ready"]
            else:
                parts[f"moe_{typ}"] += t4 - t3
        parts["mixer_final"] += run.get("mixer_ns", 0)
        if "t_head" in rnd:
            parts["head_draw"] += rnd["t_head"] - run["t_end"]
            parts["early_spec"] += rnd.get("t_spec", rnd["t_head"]) - rnd["t_head"]
            parts["mtp_chain"] += rnd.get("t_chain", rnd["t_settled"]) - rnd["t_settled"]
    return {k: v / n * NS for k, v in parts.items()} if n else {}


def summarize(rounds: list[dict], attention: set[int]) -> dict:
    ratios = clean_ratios(rounds, attention)
    rows = []
    for i, rnd in enumerate(rounds):
        rows.append(round_buckets(rnd, attention, ratios, rounds[i + 1] if i + 1 < len(rounds) else None))
    return {"rows": rows, "ratios": ratios}


def table(rows: list[dict], title: str, expert_bytes: int) -> str:
    good = [r for r in rows if r["f"]["ok"]]
    if not good:
        return f"### {title}\n\n(no complete rounds; {len(rows)} rounds flagged)\n"
    n = len(good)
    tokens = sum(r["f"]["committed"] for r in good) or 1
    wall = sum(r["f"]["wall"] + r["b"]["gap"] for r in good)
    lines = [f"### {title}", "",
             f"rounds={n} (flagged {len(rows) - n}) tokens={tokens} tok/round={tokens / n:.2f} "
             f"rows/round={statistics.mean(r['f']['rows'] or 0 for r in good):.2f} "
             f"depth={statistics.mean(r['f']['depth'] or 0 for r in good):.2f} "
             f"ms/round(wall+gap)={wall / n * NS:.1f} tok/s={tokens / (wall * 1e-9):.1f} "
             f"clean segments={sum(r['f']['clean'] for r in good)}/"
             f"{sum(r['f']['clean'] + r['f']['unclean'] for r in good)} "
             f"mark buffers/round={statistics.mean(r['f']['buffers'] for r in good):.1f}", "",
             "| bucket | ms/round | ms/token | % |", "|---|---|---|---|"]
    for key in BUCKETS:
        v = sum(r["b"][key] for r in good)
        lines.append(f"| {LABELS[key]} | {v / n * NS:.2f} | {v / tokens * NS:.2f} | {100 * v / wall:.1f} |")
    misses = sum(r["f"]["misses"] for r in good)
    unique = sum(r["f"]["unique"] for r in good)
    ssd = sum(r["b"]["ssd_stall"] for r in good)
    mib = misses * expert_bytes / 2**20
    lines += [f"| **total** | {wall / n * NS:.2f} | {wall / tokens * NS:.2f} | 100 |", "",
              f"misses/round={misses / n:.1f} of unique {unique / n:.1f} (hit {100 * (1 - misses / max(unique, 1)):.1f}%) "
              f"SSD MiB/round={mib / n:.1f} MiB/token={mib / tokens:.1f} "
              f"read GB/s while stalled={(misses * expert_bytes) / max(ssd, 1):.2f} "
              f"PLE gather ms/round={statistics.mean(r['f']['ple_ms'] for r in good):.2f} "
              f"graph build ms/round={statistics.mean(r['f']['build_ms'] for r in good):.2f} "
              f"MTP forwards/round={statistics.mean(r['f']['mtp_runs'] for r in good):.2f}", ""]
    flagged = [r["f"]["why"] for r in rows if not r["f"]["ok"]]
    if flagged:
        lines += [f"flagged: {flagged[:5]}", ""]
    return "\n".join(lines)


def prefill_view(req: dict, expert_bytes: int) -> str:
    h = req["header"]
    chunks = req["chunks"]
    rounds = req["rounds"]
    lines = []
    if h.get("t_received"):
        fr = rounds[0]["t0"] if rounds else 0
        first_chunk = chunks[0]["t0"] if chunks else 0
        last_chunk = chunks[-1]["t_end"] if chunks else 0
        pieces = [("received -> started", h["t_started"] - h["t_received"])]
        if chunks:
            pieces += [("started -> first chunk", first_chunk - h["t_started"]),
                       ("prompt chunks", last_chunk - first_chunk),
                       ("last chunk -> prefilled", h["t_prefilled"] - last_chunk)]
        else:
            pieces += [("started -> prefilled", h["t_prefilled"] - h["t_started"])]
        if fr:
            pieces += [("prefilled -> first round", fr - h["t_prefilled"]),
                       ("first round", rounds[0]["t_end"] - fr)]
        pieces += [("TTFT (received -> first token)", h["t_first_token"] - h["t_received"])]
        lines += ["| TTFT phase | ms |", "|---|---|"] + [f"| {k} | {v * NS:.1f} |" for k, v in pieces] + [""]
    if chunks:
        # a chunk of <= 16 rows runs the decode rows path (kind 0): its misses are prompt reads, counted here
        lines += [("| chunk | wall ms | window calls | rows calls | experts read | rows misses | GiB | read ms | "
                   "host answer ms | rest (GPU + Python) ms |"), "|---|---|---|---|---|---|---|---|---|---|"]
        for i, ch in enumerate(chunks):
            calls = ch.get("calls", [])
            window = sum(1 for c in calls if c["kind"] == 1)
            rows_calls = [c for c in calls if c["kind"] == 0]
            reads = sum(c["misses"] for c in calls)
            nbytes = sum(c["bytes"] for c in calls)
            read_ms = sum(c["t_read1"] - c["t_read0"] for c in calls if c["misses"]) * NS
            host = sum(c["t_ready"] - c["t_sig"] for c in calls) * NS
            wall = (ch["t_end"] - ch["t0"]) * NS
            lines.append(f"| {i} | {wall:.1f} | {window} | {len(rows_calls)} | {reads} | "
                         f"{sum(c['misses'] for c in rows_calls)} | {nbytes / 2**30:.2f} | {read_ms:.1f} | "
                         f"{host:.1f} | {wall - host:.1f} |")
        lines.append("")
    split = misses_split(req)
    lines += [(f"rows-path misses (/health \"misses\"): decode rounds {split['decode']} + prompt chunks "
               f"{split['prompt']} + outside any span {split['outside']} = {sum(split.values())}"), ""]
    if h.get("events"):
        lines += ["events: " + ", ".join(f"{e['name']}={(e['t_end'] - e['t0']) * NS:.1f} ms" for e in h["events"]), ""]
    return "\n".join(lines)


def misses_split(req: dict) -> dict:
    """The request's rows-path (kind 0) misses by phase: decode rounds, prompt chunks and calls outside both (load-time
    checks land in request 1's orphans). Their sum is what /health's "misses" moved by over the request."""

    def rows(calls: list[dict]) -> int:
        return sum(c["misses"] for c in calls if c["kind"] == 0)

    return {"decode": sum(rows(r.get("calls", [])) for r in req["rounds"]),
            "prompt": sum(rows(c.get("calls", [])) for c in req["chunks"]),
            "outside": rows(req["header"].get("orphan_calls", []))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", type=Path, nargs="+",
                    help="profile files, oldest first (a rotated .1 before the live one)")
    ap.add_argument("--attention", default=",".join(str(i) for i in range(3, 48, 4)),
                    help="0-based sparse-attention layer ids (default: every 4th from 3, the q8 checkpoint)")
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    ap.add_argument("--labels", default="", help="comma list naming the requests in order (e.g. A-cold,B,C)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    attention = {int(x) for x in args.attention.split(",") if x}
    labels = [x for x in args.labels.split(",") if x]
    out_json = []
    for i, req in enumerate(load(*args.profile)):
        h = req["header"]
        name = labels[i] if i < len(labels) else f"request {h['n']}"
        print(f"## {name}: job={h.get('job')} prompt={h.get('prompt')} cached={h.get('cached')} tokens={h.get('tokens')} "
              f"sha={h.get('sha')} rounds={h.get('rounds')} chunks={h.get('chunks')} sync={h.get('sync')} "
              f"dropped calls={h.get('calls_dropped')} gpu={h.get('gpu_dropped')}\n")
        print(prefill_view(req, args.expert_bytes))
        rounds = req["rounds"]
        if not rounds:
            continue
        s = summarize(rounds, attention)
        print(table(s["rows"][:1], f"{name}: round 1 (prefill->decode switch)", args.expert_bytes))
        print(table(s["rows"][1:], f"{name}: rounds 2..{len(rounds)}", args.expert_bytes))
        b = mode_b(rounds, attention)
        if b:
            print("Mode B (eval barriers; use as ratios) mean ms per round: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in sorted(b.items())) + "\n")
        out_json.append({"name": name, "header": {k: v for k, v in h.items() if k != "orphan_calls"},
                         "misses": misses_split(req),
                         "ratios": s["ratios"], "modeB": b,
                         "rounds": [{"b": {k: v * NS for k, v in r["b"].items()}, "f": r["f"]} for r in s["rows"]]})
    if args.json:
        args.json.write_text(json.dumps(out_json, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
