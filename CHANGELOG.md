# What's new in TensorFold

`tensorfold update` prints the sections below that are newer than the version you had. Each release's page on
GitHub has the full notes and the measurements behind them.

## 0.3.1 (18 Oct 2026, MLXFW, the-shop fork)

A simplification release: opt-in features that measured no gain are removed. Each was off by default, so the
default path is the one 0.3.0 runs and replies are the same tokens, bit for bit: the numerics golden test passes
unchanged with `TF_PREFILL_LAYER_MAJOR` unset and with it `=1`, and `NUMERICS_VERSION` stays 1. Install with
`pip install "tensorfold[ssd] @ git+https://github.com/the-shop/TensorFold.git@v0.3.1"`.

Upgrading: drop the removed variables from launch scripts (they are now ignored). The first start compiles the
expert pool's host extension again.

### Removed

- **Decode prefetch and prompt seeding** (`TENSORFOLD_POOL_DECODE_PREFETCH`, `TENSORFOLD_POOL_PROMPT_SEED`,
  `TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER`). Either one also kept the pool on the Python serve loop. Measured on
  2026-10-09: prefetch lowered prefill (168 to 127 tok/s in one harness, 198-241 to 160-192 in the other) and
  moved drafted decode between -3% and +16%; seeding with prefetch was about 14% slower than prefetch alone.
- **The expert pack** (`TENSORFOLD_EXPERT_PACK`, `tools/pack_experts.py`, the packed mode of
  `tools/bench_pool_io.py`, `packed` in `/health` `experts`). It was slower at 40 and 60 GiB pools (decode 16-23%
  lower, prefill about 16-21% lower) and within 0-6% at 70 GiB, and took as much disk as the experts.
- **The n-gram row cache** (`--ple-cache-mib`, `ple_cache` in `/health`, `tools/bench_ple_cache.py`). It served
  about 99% of decode lookups with 512 MiB but did not make decode faster in either of two harnesses.
- **`TF_POOL_JOBLOG`**, the per-call expert-pool log. The decode profiler (`TENSORFOLD_DECODE_PROFILE`) records the
  same calls with more detail.
- Accessors only the tests used (`Streamer.window_layers`, the extension's `PoolServer.aligned`).

### Deprecated

- **`--ple-cache-mib`** is still accepted so existing commands start, but it does nothing: a value above 0 prints
  "--ple-cache-mib is ignored since 0.3.1 and will be removed in 0.4.0". It reserves no memory and is not part of
  the load options or any snapshot key.

### Internal

- The saved-conversation and prompt-snapshot pruning share one loop (older keys first, then oldest first), with the
  same order as before.

## 0.3.0 (12 Oct 2026, MLXFW, the-shop fork)

Faster prompts for Qwen3.8 Flash Next with SSD experts, the same tokens. Install with
`pip install "tensorfold[ssd] @ git+https://github.com/the-shop/TensorFold.git@v0.3.0"`.

Measured on an M5 Max (128 GB) serving the 8-bit Flash Next checkpoint (`--ssd-experts 60 --ple-on-ssd --context
77824 --parallel 1 --persist-turns --checkpoint-last-user`, 84 GiB budget), 0.2.0 and 0.3.0 back to back, two runs
each, page cache evicted before every start. All 0.3.0 numbers below were measured with `TF_PREFILL_LAYER_MAJOR=1`
(opt-in, off by default; 0.2.0 ignores it):
- **Cold 24K-token prompt:** first token in 292-323 s on 0.2.0, 33-39 s on 0.3.0; expert bytes read 1,452 GB to
  136 GB. This needs `TF_PREFILL_LAYER_MAJOR=1` (off by default). A separate 0.3.0 run with layer-major prompt
  groups and window reuse both off took 152.8 s; the default settings' cold time was not measured.
- **Resumed turns on the same server** (21.8-23.8K tokens cached): 5.3-6.9 s to 2.9-3.4 s. A fresh 0.3.0 server
  resuming a 21.8K-token system block from disk took 4.85 s. A turn adding 1.5K tokens: 27.2-29.9 s to 7.3-8.1 s.
- **Decode** (256-token replies): flat to +19% (15.00 to 14.95, 19.70 to 23.35, 21.55 to 22.85 tok/s).
- **Page cache:** after a cold prompt the expert files keep ~18 GB cached on 0.2.0, 25 MB on 0.3.0.
- **Memory:** peak process footprint 87.7 GB sampled (87.1 GB on 0.2.0); the kernel's lifetime peak, which also
  catches spikes between samples, was 87.9 GB (81.8 GiB). That is 2.3-2.5 GB under the 84 GiB (90.19 GB) budget;
  swap did not grow (0.2.0: +0.2 to +0.9 GB).
- **Bits:** every request with a reference token hash matched it on both builds (45 of 45); the one prompt without
  a reference gave the same hash on every run of both builds.

Upgrading: prompt snapshots written by 0.2.0 are not loaded (one re-prefill per saved system block or
conversation). The first start compiles the expert pool's host extension again.

### Performance

- **Page-cache-safe expert reads** (on by default; `TENSORFOLD_POOL_ALIGNED_READS=0` restores the old reads). On
  macOS, an `F_NOCACHE` read at an unaligned file offset leaves all but its first MiB in the page cache, so about a
  third of every expert's bytes stayed there. Unaligned expert parts over about 1 MiB are now read as the
  page-aligned range around them into an aligned buffer and copied out: the same bytes, counters and output tokens,
  and nothing cached. Credits: [strata-mlx](https://github.com/yibie/strata-mlx) (MIT License, Copyright (c) 2026
  yibie) for the finding that macOS `F_NOCACHE` reads still fill the page cache (`paged.py`, `ExpertStore.whole`).
  No strata-mlx code was copied; the alignment cause (file offset and buffer) and the aligned-superset read are
  TensorFold's own.
- **Prompt chunks reuse experts already in the decode pool** (on by default; `TENSORFOLD_POOL_WINDOW_REUSE=0`
  restores the 0.2.0 reads). With `--ssd-experts`, a prompt chunk's layer window copies each routed expert that the
  decode LRU already holds from its pool slot (memcpy) and reads only the rest from SSD. The same bytes land in the
  same slots and the kernels are unchanged, so replies are the same tokens; the decode LRU is not changed by a
  window. `/health` `experts` gains `window_reads`, `window_copies` and `copy_bytes`. Idea credit: resident-first
  expert routing in [oMLX](https://github.com/jundot/omlx)'s MoE expert offload (Apache-2.0); no oMLX code is used.
- **Layer-major prompt groups (`TF_PREFILL_LAYER_MAJOR=1`, off by default).** Flash Next's prompt chunks can run in
  groups, every chunk through a layer before the next layer, over the same chunk cuts and with the same per-call
  shapes, so replies, caches and saved prompt states are bit for bit those of chunk-by-chunk prefill while the SSD
  expert window is filled once per layer a group instead of once per layer a chunk. Checkpoints inside a group are
  assembled per layer as each layer passes the cut, and kept only once every expert-pool call of the group is
  answered without an error (a group whose SSD read failed keeps and saves none of them, as chunk-by-chunk prefill
  keeps nothing past a failed chunk). A group's carried activations are capped (`TF_PREFILL_GROUP_GIB`, default 1)
  and counted in admission; a cancelled group keeps no half-filled state. Known limit: a prompt cancelled while a
  group runs loses that whole group's progress (chunk by chunk keeps the last finished chunk), so its retry
  recomputes the group; bits are unaffected, only the retry is slower. Ideas credited, no code copied: staggered
  segments and per-chunk state marks of the upstream TensorFold native engine
  ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), `zig/src/core/segments.zig`,
  `zig/src/flashnext/marks.zig`), expert-major prefill with offloaded experts in oMLX, and strata-mlx's planned
  overlap of next-layer reads with compute.

### Correctness

- **Steady KV pricing for Flash Next's sparse indexer.** Index keys and pooled blocks are priced per allocated
  position, not per valid token, so one short request no longer inflates the admission profile and evicts or
  refuses later long prompts (measured on the 8-bit model: after one 15-token request, the old pricing projected a
  69,967-token prompt 4.03 GiB higher, 2.06 GB over the MLX budget, so it was refused; now it is served). At 84 GiB
  the startup still picks 2,048-token prompt chunks. Ported from upstream
  [ashhart/TensorFold](https://github.com/ashhart/TensorFold) commit 17e6390f (Apache-2.0; see
  THIRD_PARTY_NOTICES.md).
- **Out-of-bounds reads fixed in the expert down kernels.** The Flash Next expert down kernels (`expert_down_y`, the
  streamed pool's `down`) read past the activation row for expert widths below 512 and past both buffers above
  1,024: test models (width 64 or 128) gave different bits run to run. Narrower rows now take a variant whose idle
  lanes read nothing; widths above 1,024 or not a whole number of quantization groups (32 inputs at 4 bits, 64 at
  8) are refused. Widths 512 to 1,024 (the 8-bit Qwen checkpoint's 640) compile from the unchanged source, so their
  replies are the same tokens.
- **Token ids past the embedding are refused.** A rendered prompt holding an id the model's embedding lacks (a
  tokenizer larger than the checkpoint's vocabulary, e.g. a tiny test model) gets HTTP 400 (in a stream, an
  `invalid_request_error` event) instead of reading memory past the table; the server warns at load when the
  tokenizer has more ids than the embedding, and the Qwen4Exp prompt forward checks ids itself. The start-up memory
  probes (real text or their synthetic fallback), copied drafts (also negative ids) and MTP drafts leave such ids
  out, so a mismatched pair still starts and drafts.
- **Expert pool tripwire.** Expert-pool calls alternate between two regions of the routing box; the host clears a
  call's region once read and checks the next call's region is still clear before releasing it, so a call whose
  routing arrives before the previous call was answered fails the request ("calls ran out of order"). A call two
  ahead shares a region and is not caught by this check; by analysis it ends in a pool timeout error (slow, the
  request still fails). The engine orders its calls so neither happens; this is a backstop. The Python serve loop
  now keeps the first pool error (as the native server does) and clears a failed call's region, so the real cause
  is reported instead of "calls ran out of order".

### Diagnostics

- **Decode profiler (opt-in).** `TENSORFOLD_DECODE_PROFILE=<path.jsonl>` records, per decode round, host stamps at
  points the round already passes, each expert-pool call (GPU signal seen, SSD reads, release, misses, bytes) and
  the GPU start and end of the command buffers that carry the expert marks, with no added MLX op or eval (replies
  are the same tokens). `TENSORFOLD_DECODE_PROFILE_SYNC=1` adds eval barriers for exact per-part GPU times;
  `TENSORFOLD_DECODE_PROFILE_MAX_MB` (default 1024) caps the file, rotating it to `<path>.1`. The profiler never
  fails a reply or a model load: a write error or a C++ clock off Python's logs one line and turns it off. Prompt
  chunks of 16 rows or fewer take the decode rows path and count in `/health` misses; the report attributes them
  to the prompt. The `done` line's `draft=` and `post=` now report settle and after-read time for single-stream
  rounds. Tools: `tools/decode_profile_report.py` (per-round breakdown and each request's TTFT phases),
  `tools/decode_health_delta.py` and `tools/bench_flash_decode_parts.py`; see the Flash Next recipe. Idea credits
  (no code copied): the dense-versus-expert bytes-per-token analysis of strata-mlx (MIT) motivated the breakdown,
  and oMLX (Apache-2.0) GPU keep-warm and overlap observations were consulted while scoping it (no idle probe
  ships here).
- **Pool call log.** `TF_POOL_JOBLOG=path` logs every expert-pool call's host time, GPU gap and bytes (off by
  default).
- **`tools/page_cache.py`.** `evict MODEL_DIR` drops a model's files from the page cache without sudo
  (`msync(MS_INVALIDATE)`), and `resident MODEL_DIR [--experts]` reports how much of them is cached. Use it before
  cold benchmarks. Credits: the no-sudo eviction technique (`msync(MS_INVALIDATE)` over a shared mapping,
  `bench/drop_cache.py`) is strata-mlx's (MIT); no strata-mlx code was copied.

### Snapshots

- **Prompt snapshots survive bit-identical updates (Flash Next).** The snapshot key no longer hashes the family's
  source or names the release: it holds a declared `NUMERICS_VERSION`, the pinned environment (model path and
  weights, MLX and mlx-lm, prompt plan, prefill route with the fused-tail `TF_FLASH_DENSE`/`TF_FLASH_TILE_WIDE`
  choices) and the digest of a short canary prompt prefilled on the loaded model at start-up and read back through
  the snapshot store. A refactor, comment or release that computes the same bits keeps saved system blocks and
  conversations; a change that moves the canary's bits (including how states are stored and restored), or a
  version bump, never reads them. The canary reruns when the installed code, OS, GPU, `TF_*`/`TENSORFOLD_*`/
  `MLX_*`/`QWEN_*`/`GLM_*` variables (a test checks the package reads no others) or load options
  (`--ssd-experts`, `--ple-on-ssd`, `--ple-cache-mib`, drafts, `--lane-kernels`, `--parallel`, `--context`,
  memory) change, and is cached beside the snapshots otherwise; a malformed cache file is rewritten, and a canary
  measured while the code changed on disk is not recorded. `TF_SNAPSHOT_KEY=strict` also pins the exact code.
  Snapshots written by 0.2.0 are not loaded (one re-prefill).
- **Numerics guards for contributors.** The canary does not reach every path (long chunks, 1-4 row tails, sparse
  attention past 4096 keys, MTP, SSD expert reads), so two tests catch *accidental* numerics edits at commit time.
  `tests/test_numerics_golden.py` pins a tiny model's prompt-cache bits on cases across those sizes, per declared
  version and environment (GPU, macOS, MLX, mlx-lm); it never re-records an existing entry, and it records a new
  environment only for the code recorded as reviewed. `tests/test_numerics_reviewed.py` hashes, comments and
  docstrings aside, every file of the package except an allow-list of HTTP/admission code, argument parsing, the
  updater and docs, plus the golden's own generator; the allow-list and file list are in its fixture. It fails on
  any such edit until `NUMERICS_VERSION` is bumped or, with the golden passing unchanged, the edit is recorded
  with `TF_RECORD_NUMERICS_REVIEWED=1`. A deliberate re-record still takes only that command, so reviewers must
  treat any diff to `tests/fixtures/numerics_reviewed.json` or `tests/fixtures/numerics_golden.json` as a
  `NUMERICS_VERSION` decision.

### Docs

- The Flash Next recipe lists every new variable (`TF_PREFILL_LAYER_MAJOR`, `TF_PREFILL_GROUP_GIB`,
  `TENSORFOLD_POOL_ALIGNED_READS`, `TENSORFOLD_POOL_WINDOW_REUSE`, `TENSORFOLD_DECODE_PROFILE*`, `TF_POOL_JOBLOG`)
  and the 0.3.0 measurements; older tables are marked as measured before 0.3.0.
- **Licensing.** THIRD_PARTY_NOTICES.md now credits the two ports from upstream ashhart/TensorFold after its
  Apache-2.0 relicense that 0.2.0 shipped without a notice: the refused-checkpoint handling (commit e2a9ab01) and
  the cleanup of unfinished snapshot writes (commit 05b25ea5, by Isham Rashik). 0.3.0's own port, the sparse-indexer
  pricing (commit 17e6390f), is listed beside them, and each ported region carries a comment naming its upstream
  commit. These are the ports an exact-line comparison against upstream's post-relicense history found. The package
  metadata says `MIT AND Apache-2.0`; the fork's own code stays MIT.

The out-of-bounds, token-id and snapshot-key changes are own work, with no third-party code; all of 0.3.0 builds on
TensorFold by ashhart.

## 0.2.0 (10 Oct 2026)

Saved prompt states survive restarts and crashes, and an optional RAM cache sits in front of n-gram rows read from
SSD. Install with `pip install "tensorfold[ssd] @ git+https://github.com/the-shop/TensorFold.git@v0.2.0"`.

- **Safe snapshot files.** Every saved system block and conversation is sealed with a crc32 checksum and checked
  on load; a file whose bytes, header or length do not match is refused, removed and logged, and the next-longest
  saved state is tried. Writes are flushed (`F_FULLFSYNC` on macOS) before the rename and the directory after it;
  unfinished writes left by a crash are removed at start-up. The snapshot key now includes a fingerprint of the
  weights (headers, sizes and sampled bytes of every tensor), so changed weights at the same path never reuse old
  states; an edit confined to unsampled bytes is not detected. Snapshots written by 0.1.0 are not loaded.
- **`--persist-turns`.** Writes each turn's conversation checkpoint to the session directory behind the reply, so
  a crash or kill keeps it. When RAM has no room for a full host copy the checkpoint is streamed to disk in pieces
  of at most 64 MiB. Queued bytes count against the memory budget; a follow-up that needs them waits for the write
  instead of being refused. `/health` reports the writer's counters.
- **`--checkpoint-last-user`.** Keeps one more state where the last user message starts, for clients that replace
  a trailing reminder message every turn.
- **`--session-dir`.** Names the conversation directory (default `session-snapshots` beside `--snapshot-dir`).
- **`--background-warm`.** The background re-prefill of a system block saved under other kernel settings is now
  opt-in; it slowed foreground turns for many minutes after an upgrade.
- **Housekeeping.** A full disk removes this model's oldest saved conversations and retries once; states saved for
  older weights of the same model go first when a directory is over budget; a superseded prefix is evicted before
  others and a refused one is spilled to disk (ports of upstream fixes); `/health` counts write failures.
- **`--ple-cache-mib N`** (with `--ple-on-ssd`, default off). An LRU-like (CLOCK) cache of n-gram rows in a fixed
  arena counted in the memory budget; replies are the same tokens. `/health` reports its hits. On the 8-bit
  checkpoint it served ~99% of decode lookups but did not make decode faster, so it stays off by default.
- Measurements: see the Flash Next recipe (`docs/recipes/qwen3.8-flash-next.md`).

## 0.1.0 (10 Oct 2026, the-shop fork)

First release of [the-shop/TensorFold](https://github.com/the-shop/TensorFold), a fork of
[ashhart/TensorFold](https://github.com/ashhart/TensorFold) at release [v0.3.6.3](https://github.com/ashhart/TensorFold/tree/v0.3.6.3) (commit 1911880,
29 Sep 2026, MIT). The fork has its own version numbers from here on, starting at 0.1.0, with a git tag for each
release; `tensorfold update` looks for them on the-shop/TensorFold. It builds on upstream's SSD expert streaming
(`--ssd-experts`, [ashhart/TensorFold#17](https://github.com/ashhart/TensorFold/pull/17)) and adds the items below. Install with
`pip install "tensorfold[ssd] @ git+https://github.com/the-shop/TensorFold.git@v0.1.0"`.

- **Qwen3.8 Flash Next 8-bit on M5.** The Metal path serves MLX affine 8-bit checkpoints (group 64, n-gram tables
  group 32): 8-bit expert kernels (resident and SSD slot pool), token embedding and n-gram rows, hyper-connections
  through the lane matmul, per-module widths read off the checkpoint. `tools/convert_flash_next_q8.py` streams a BF16
  Hugging Face checkpoint into that layout under an MLX memory cap. 8-bit lane weights are read in place unless
  `TF_FLASH_TILE_WIDE=1`.
- **Native expert-pool server (default).** With upstream's `--ssd-experts`, a native server in the host-sync
  extension answers each layer's routing, keeps the LRU and reads misses on native threads, with no Python per MoE
  call.
  `TENSORFOLD_POOL_NATIVE=0` keeps the Python loop; `TENSORFOLD_POOL_THREADS` sets reader threads (default 8). At 40
  and 60 GiB pools it decodes 9-11% faster with drafts than the Python loop and prefills 6-8% faster.
- **Opt-in, experimental:** decode prefetch (`TENSORFOLD_POOL_DECODE_PREFETCH=K`) and prompt seeding
  (`TENSORFOLD_POOL_PROMPT_SEED=1`), which run the Python loop; a contiguous expert pack (`tools/pack_experts.py`,
  `TENSORFOLD_EXPERT_PACK`), checked against the checkpoint's files and bytes, slower at 40/60 GiB and within 0-6% at
  70 GiB. `tools/bench_pool_io.py` times miss reads.
- **Failures surface.** A failed expert-pool call (timeout, bad id, full pool, failed read, or a call released at
  close) now fails the request instead of returning a token computed on stale slots; `/health` reports pool hits,
  misses, bytes read and the error.
- **Short prompts on a tight budget.** A prompt of at most the fused-row count no longer pins rollback rows that
  admission could not reclaim, which refused it and every request after it.
- **Recipe:** [Running it on a 128 GB M5 Mac](docs/recipes/qwen3.8-flash-next.md#running-it-on-a-128-gb-m5-mac),
  with a 60 GiB pool recommended (40 GiB as the lighter option) and measured results for 40, 60 and 70 GiB pools.

# Upstream ashhart/TensorFold history

The sections below are upstream [ashhart/TensorFold](https://github.com/ashhart/TensorFold)'s own releases up to
0.3.6.3, the version this fork's 0.1.0 is based on. Their version numbers are upstream's, not this fork's.

## Upstream ashhart/TensorFold 0.3.6.3 (29 Sep 2026)

- **Images in, on Macs and Sparks.** `--vision` lets Qwen3.8-27B read up to four images a request, as data URLs or,
  with `--vision-urls`, HTTPS links. Image replies keep drafting and equal `"draft": false` ones (77.6 against 37.3
  tok/s serial on an M3 Ultra), and the first token comes as fast as mlx-vlm's. Thanks to @di37 for HTTPS-only
  fetching, bounded image preparation and redacted request logs (#64).
- **DeepSeek-V4-Flash on a 256 GB Mac.** The new `deepseek_v4` family serves `mlx-community/DeepSeek-V4-Flash-4bit`
  on the lane engine with DSpark or MTP drafts converted from DeepSeek's releases, and replies equal `"draft": false`
  ones. On an M3 Ultra it decodes 1.7-3.1x and reads prompts 1.8-2.0x as fast as mlx-lm PR #1797's server. Thanks to
  @jeffpeng3 (#14).
- **NVFP4 checkpoints on CUDA, as published.** Flash Next's NVFP4 exports (local-inference-lab's and RadixArk's)
  and NVIDIA's Qwen3.8-27B NVFP4 load through `tensorfold serve` and stay exact: drafted replies equal `"draft": false`
  ones, resumed prompts equal fresh ones, and 84 of 84 concurrent streams equal their solo runs. The experts read
  their routing on the GPU, so CUDA graphs replay any routing. On one Spark, Flash Next NVFP4 decodes 1.13-1.52x vLLM
  on the same checkpoint. Two ranks, `--ple-on-ssd` and images on NVFP4 checkpoints stop at startup with a message
  until they're qualified. Thanks to @tournierjc for the reader (#67).
- **Ternary Bonsai 2 27B** (prism-ml's 2-bit pack) serves on the lane engine with Qwen3.8-27B's DFlash2 drafts,
  exact. Against mlx_lm running the pack's own runtime on an M5 Max: decode 3.7-5.8x on code and 1.7-2.2x on chat,
  prompts 1.4-1.7x. Thanks to @gprot42 (#18).
- **CUDA server work from @nood-co1:** stop strings on every CUDA engine and `ignore_eos` on the 27B (#63);
  malformed requests get a 400, and failed ones a 500 or a stream error event (#61); a silent start explains itself
  with extension-build progress and stale-lock notes, and `kill -USR1 <pid>` prints every thread's stack, at start or
  while serving (#62); the 27B keeps its prompt cache entry one token before the prompt's end, so the next chat turn
  resumes from it (#65).
- **Replies keep decoding while long prompts prefill on Macs.** A prompt now fills one planned chunk at a time and
  running replies take rounds between its chunks, for `--decode-share` of each chunk's time (default 0.25, about a
  fifth of the time; 0 prefills whole prompts first, as before); every reply still equals its solo run. On an M3
  Ultra, a reply's longest pause while three 17K-token prompts arrive fell from 164 s to 7 s, and a request alone is
  unchanged. Thanks to @benwilson (#72).
- **Long conversations stay warm on Macs.** A checkpoint that can't fit no longer evicts the others first, a resumed
  turn no longer holds the stored prefix it copied, and the 27B's DFlash2 prompt taps take 64 KB a token instead of
  114. On a 64 GB budget (emulated on an M3 Ultra), one conversation grew to 143K tokens and each turn resumed in
  38-46 s, where 0.3.6.2 prefilled every turn past about 100K from the start. Startup names the longest request
  whose prompt is kept. Thanks to @sanjaibalajee (#74), and to @benwilson for the report and his
  64 GB measurements, now in the README labelled 0.3.5.1 (#71, #70).
- **Qwen3.6 on CUDA keeps serving past 8,192 rows** with expandable allocator segments: its graphs now capture into a
  new pool after the buffers grow, where every later request failed. Thanks to @philip-pentatonic (#78).
- **Typed tool arguments on CUDA:** a Qwen tool call's array, object, number and boolean parameters arrive as JSON
  values, as on the Mac. Thanks to @MiaAI-Lab (#75).
- **Flash Next reads a prompt chunk's n-gram rows on 16 threads,** same bytes, so prompt times vary less when the
  tables are not all in the page cache. Thanks to @MovieMaker93 (#73).
- **On M1-M4, the MoE prompt matmuls give each GPU tile one expert's rows,** the scheduling idea of MLX's gather_mm
  change (ml-explore/mlx#4567), written for 4-bit gathers. Same bits. On an M3 Ultra, Flash Next's prompts run 2-4%
  faster (1.02-1.20x oMLX's) and GLM-5.3's 3.4-3.8% faster (1.08-1.11x mlx-vlm's at 8k-32k).
- **Flash Next prompts on M1-M4:** the block scores run as one batched matmul, the GQA kernel scores two query heads
  a simdgroup and the top-512 selection finds its cuts with a simdgroup scan. Same bits, about 1% faster at 16k-64k;
  on an M3 Ultra it is level with oMLX at 32k and 64k.
- **MLX stays at 0.32.2 for now.** MLX 0.32.3 changed a kernel our prompt kernels build on, so fresh installs fell
  back to slower prompts with one log line; the next release follows the new kernel. Thanks to @ecohash-co (#88).

## Upstream ashhart/TensorFold 0.3.6.2 (28 Sep 2026)

- **EXL3 replies stop at the end of the turn.** Flash Next and 27B EXL3 packs list `<|im_end|>` only in
  `generation_config.json`, so replies ran past their turn and leaked tool calls and think tags. The CUDA engine now
  reads that file too. Thanks to @vcruz305 (#69).
- **pip installs serve 27B EXL3 packs.** The package was missing the 27B's CUDA sources; a test now checks that every
  CUDA source ships. Thanks to @taussoe (#66).
- **A quantized KV cache for Flash Next on CUDA.** `--kv-dtype int8` or `int4` holds about 1.7x or 2.6x the default
  window in the same memory, and drafted replies still equal serial ones. `--mtp-confidence` sets where MTP chains
  stop. Thanks to @vcruz305 (#47).
- **Flash Next on CUDA reaches the first token sooner.** The head runs on a prompt's final chunk only, and the prompt
  kernels load at startup, so the first 2k prompt takes 1.26 s instead of 1.79 s. Thanks to @MovieMaker93 (#40).
- **GLM on two Sparks holds 256k tokens** with a latent attention cache; its next-token loss is within 0.001 nats of
  the per-head cache. Thanks to @taussoe (#54).
- **Mixed-bit Qwen checkpoints** (4-bit with some 5- and 6-bit layers, such as oQ4) load on every lane backend,
  exact, with new row kernels for 2- to 8-bit weights. On an M3 Ultra, oQ4 27B decodes 114-120 tok/s on code and
  59-61 on chat, against mlx_lm's 34-36.
- **Concurrent 27B on CUDA:** `--parallel 16` serves 161.7 tok/s on one Spark in 25.4 GiB, each reply equal to its
  solo run (#38). **Qwen3.6-35B-A3B on CUDA,** exact: decode 1.36-1.49x vLLM with MTP, prompts 1.21-1.37x (#45).
- **Gemma 4 drafts (opt-in):** `--drafter z-lab/gemma-4-26B-A4B-it-DFlash` decodes 1.3-2.1x mlx_lm on an M3 Ultra,
  exact.
- **Tool calls:** `tool_choice: "required"` and a named tool are enforced on both servers (#52), and a complete tool
  call inside an unclosed think block comes back as a tool call (#60).
- **Conversations come back warm on Macs.** `--spill-gib N` writes a conversation pushed out of the prompt cache
  to disk, up to N GiB, and reads it back when the conversation returns. On a 48 GB budget a 35k-token
  conversation came back in 0.27 s on an M5 Max instead of 75 s, with the same reply. Off by default;
  `--checkpoint-slots` sets how many conversations stay in memory. Thanks to @gilby (#68, #55).
- **Memory:** `TENSORFOLD_MEMORY_LIMIT_GB` raises the budget above the default share, and Flash Next's memory check
  counts its host-mapped n-gram tables, so it starts on a 128 GB Mac. Thanks to @Chedrian07 (#49, #50).
- **CUDA server fixes from @nood-co1:** an abandoned request stops within a round (#57), keys and values stay within
  the admitted window (#58), and a failed admission no longer stops the scheduler (#59).
- **M1-M4:** a prompt split into parts attends exactly as it does in one piece at every length; two 8,192-key cases
  rounded differently in 0.3.6.

## Upstream ashhart/TensorFold 0.3.6.1 (28 Sep 2026)

- **CUDA builds inside NVIDIA's containers again.** Their `TORCH_CUDA_ARCH_LIST` names every architecture back to
  sm_80, so the kernels' thread-block clusters and FP8 MMA failed to compile for GPUs that lack them. Every extension
  now builds for the GPU that is present, and a GPU older than compute capability 9.0 gets a clear message. Thanks to
  @ss-cong for the report and the exact errors (#56).

## Upstream ashhart/TensorFold 0.3.6 (28 Sep 2026)

- **GLM-5.3-Flash on Macs** with 256 GB, drafted replies equal to serial ones. Prompts process at or above mlx-vlm
  from 2k to 32k tokens on an M3 Ultra, and tool calls parse in both servers. Thanks to @chadhurley25075-png (#9,
  #39) and @jeidbugs404 (#35).
- **Gemma 4 26B-A4B on the lanes,** exact at every width, with prompts at or above mlx_lm from 2k to 64k tokens on
  an M3 Ultra. Thanks to @cshintov (#10).
- **Bigger models on smaller Macs.** `--ple-on-ssd` reads Flash Next's n-gram tables from disk, so a 128 GB Mac holds
  it (#16). `--ssd-experts GIB` streams routed experts from the checkpoint into a GPU pool of that size, so Flash Next
  fits a 64 GB Mac and GLM a 128 GB one. Replies are the resident model's tokens; decode runs at 0.31-0.39x resident
  speed for Flash Next and 0.13-0.17x for GLM (measured on an M3 Ultra; `pip install "tensorfold[ssd]"` first) ([ashhart/TensorFold#17](https://github.com/ashhart/TensorFold/pull/17)).
- **EXL3 checkpoints on CUDA (experimental):** Qwen3.8-27B and Flash Next packs from turboderp, exact on the lanes.
  Decode runs 1.6-3.6x vLLM with MTP; prompt processing is about half the MLX checkpoints' speed for now, and the
  fix is next. Thanks to @vcruz305 (#42).
- **Faster prompts.** Flash Next sizes its prompt chunks to the memory it has, Nemotron takes up to 8,192 tokens a
  chunk on M5 GPUs, and the weights stay wired while a server runs.
- **Fixes:** GLM on two Sparks answered "!" past about 2,000 prompt tokens, and EXL3 GLM prompts past 128 tokens
  failed (#53). A reply that isn't a tool call comes back as content, not an HTTP 500 (#51). The server hands MLX's
  freed buffers back when it goes idle (@kingjamez, #44).
- **MLX 0.32.2 or newer** is required on Macs.
- **What's new after an update:** `tensorfold update` prints these notes when it finishes.
- **Known:** replies to prompts longer than one prompt chunk can differ between machines with different memory,
  because the chunk size follows the memory budget. Within one server, drafted replies always equal serial ones and
  resumed prompts equal fresh ones.

## Upstream ashhart/TensorFold 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## Upstream ashhart/TensorFold 0.3.5 (27 Sep 2026)

- **Concurrent requests share each verification round.** `--parallel auto` is on by default, and every stream's reply
  equals the same request served alone, on Metal and on CUDA.
- **Follow-up turns resume at the start of their newest messages,** with output identical to a fresh prompt. A 12-turn
  agent session with the 27B spent 14.9 s on first tokens instead of 31.9 s.
- **Memory that fits.** The whole process stays inside 70% of RAM, an omitted `--context` defaults to the window the
  machine can hold, and a prompt past it gets a clear 400.
- **Flash Next prefill** with sparse prompt attention, 1.1-1.4x faster than 0.3.4.1 on an M3 Ultra (@quigles1977, #29).
- **2- to 8-bit weights** on the lanes, so mixed-precision 27B checkpoints decode fully (@jasontitus, #34).
- **CUDA:** FP8 prefill and shared expert kernels, concurrent streams for the 27B and Flash Next, and admission from
  available memory before loading.
- **API:** raw `/v1/completions` prompts, `ignore_eos` and `stop`, request `reasoning_effort` and typed tool arguments
  (@chris247474, #28), `developer` messages, `parallel_tool_calls: false`, and cancellation when a client disconnects.

## Upstream ashhart/TensorFold 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## Upstream ashhart/TensorFold 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## Upstream ashhart/TensorFold 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## Upstream ashhart/TensorFold 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Mia-AiLab's EXL3 weights on two DGX Sparks (experimental).

## Upstream ashhart/TensorFold 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## Upstream ashhart/TensorFold 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## Upstream ashhart/TensorFold 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
