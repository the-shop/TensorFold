"""Mode 0 of the decode profiler: no code in the server, only /health's expert counters around each request.

    python tools/decode_health_delta.py rows RESULTS.jsonl          # rows with ex0/ex1 (/health "experts" before and
                                                                    # after) and srvline (the server's done line)
    python tools/decode_health_delta.py live URL REQUEST.json [...]  # snapshot /health around each request itself

Decode SSD bytes are misses x one expert's bytes: only decode calls count misses, while bytes_read also counts the
prompt's window fills, reported apart (VERIFY R2)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

EXPERT_BYTES = 5_222_400


def delta(ex0: dict, ex1: dict, tokens: int, rounds: int, ms_round: float | None, expert_bytes: int) -> dict:
    d = {k: int(ex1.get(k, 0)) - int(ex0.get(k, 0)) for k in ("hits", "misses", "bytes_read", "preads")}
    decode_bytes = d["misses"] * expert_bytes
    out = {**{f"d_{k}": v for k, v in d.items()}, "tokens": tokens, "rounds": rounds,
           "decode_MiB": decode_bytes / 2**20, "window_GiB": (d["bytes_read"] - decode_bytes) / 2**30,
           "hit_pct": 100 * d["hits"] / max(1, d["hits"] + d["misses"])}
    if tokens:
        out.update(misses_per_token=d["misses"] / tokens, decode_MiB_per_token=decode_bytes / 2**20 / tokens)
    if rounds:
        out.update(misses_per_round=d["misses"] / rounds)
        if ms_round:
            out["implied_decode_GBps_if_serial"] = decode_bytes / rounds / (ms_round * 1e-3) / 1e9
    return out


def from_line(line: str) -> tuple[int, int, float | None]:
    get = lambda key: (re.search(rf"\b{re.escape(key)}=(\S+)", line or "") or [None, None])[1]   # noqa: E731
    ms = get("ms/round")
    return int(get("tokens") or 0), int(get("rounds") or 0), float(ms) if ms else None


def rows(path: str, expert_bytes: int) -> None:
    for line in open(path):
        row = json.loads(line)
        ex0, ex1 = row.get("ex0") or {}, row.get("ex1") or {}
        if "misses" not in ex0 or "misses" not in ex1:
            continue
        tokens, rounds, ms = from_line(row.get("srvline") or "")
        print(json.dumps({"tag": row.get("tag"), "sha": row.get("tok_sha"),
                          **{k: round(v, 3) if isinstance(v, float) else v
                             for k, v in delta(ex0, ex1, tokens, rounds, ms, expert_bytes).items()}}))


def health(url: str) -> dict:
    return json.loads(urllib.request.urlopen(f"{url}/health", timeout=10).read()).get("experts") or {}


def live(url: str, requests: list[str], expert_bytes: int) -> None:
    for path in requests:
        body = json.load(open(path))
        body["stream"] = False
        ex0 = health(url)
        started = time.time()
        reply = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}), timeout=3600).read())
        wall = time.time() - started
        ex1 = health(url)
        usage = reply.get("usage") or {}
        spec = reply.get("speculative") or {}
        print(json.dumps({"request": path, "wall_s": round(wall, 2),
                          **delta(ex0, ex1, int(usage.get("completion_tokens") or 0), int(spec.get("rounds") or 0),
                                  None, expert_bytes)}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows")
    r.add_argument("results")
    lv = sub.add_parser("live")
    lv.add_argument("url")
    lv.add_argument("requests", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "rows":
        rows(args.results, args.expert_bytes)
    else:
        live(args.url.rstrip("/"), args.requests, args.expert_bytes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
