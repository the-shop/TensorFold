# What's new in TensorFold

`tensorfold update` prints the sections below that are newer than the version you had. Each release's page on
GitHub has the full notes and the measurements behind them.

## Unreleased

- **Page-cache-safe expert reads** (on by default; `TENSORFOLD_POOL_ALIGNED_READS=0` restores the old reads). On
  macOS, an `F_NOCACHE` read at an unaligned file offset leaves all but its first MiB in the page cache, so about a
  third of every expert's bytes stayed there. Unaligned expert parts over about 1 MiB are now read as the
  page-aligned range around them into an aligned buffer and copied out: the same bytes, counters and output tokens,
  and nothing cached. Each expert pool reads the variable once, when it is created (server start).
- **`tools/page_cache.py`.** `evict MODEL_DIR` drops a model's files from the page cache without sudo
  (`msync(MS_INVALIDATE)`), and `resident MODEL_DIR [--experts]` reports how much of them is cached. Use it before
  cold benchmarks.
- **Credits:** [strata-mlx](https://github.com/yibie/strata-mlx) (MIT License, Copyright (c) 2026 yibie) for two
  ideas: the finding that macOS `F_NOCACHE` reads still fill the page cache (`paged.py`, `ExpertStore.whole`), and
  the no-sudo eviction technique (`msync(MS_INVALIDATE)` over a shared mapping, `bench/drop_cache.py`) that
  `tools/page_cache.py evict` uses. No strata-mlx code was copied. The alignment cause (file offset and buffer) and
  the aligned-superset read are TensorFold's own.
- **Decode profiler (opt-in).** `TENSORFOLD_DECODE_PROFILE=<path.jsonl>` records, per decode round, host stamps at
  points the round already passes, each expert-pool call (GPU signal seen, SSD reads, release, misses, bytes) and
  the GPU start and end of the command buffers that carry the expert marks, with no added MLX op or eval (replies
  are the same tokens). `TENSORFOLD_DECODE_PROFILE_SYNC=1` adds eval barriers for exact per-part GPU times.
  `tools/decode_profile_report.py` turns the file into a per-round breakdown (dense GPU, routed-expert GPU, SSD
  stall, host answer, Python-bound idle, head tail, MTP, gaps) and the TTFT phases of each request;
  `tools/decode_health_delta.py` reads `/health` counters around requests; `tools/bench_flash_decode_parts.py`
  measures the parts on random weights of the checkpoint's shapes. The `done` line's `draft=` and `post=` now report
  settle and after-read time for single-stream rounds. The profiler never fails a reply or a model load: a write
  error (missing directory, full disk, permission) or a C++ clock off Python's logs one line and turns it off.
  `TENSORFOLD_DECODE_PROFILE_MAX_MB` (default 1024) caps the file, rotating it to `<path>.1`. Prompt chunks of 16
  rows or fewer take the decode rows path and count in `/health` misses; the report attributes them to the prompt.
  Idea credits (no code copied): the dense-versus-expert bytes-per-token analysis of
  [strata-mlx](https://github.com/yibie/strata-mlx) (MIT) motivated the breakdown, and
  [oMLX](https://github.com/jundot/omlx) (Apache-2.0) GPU keep-warm and overlap observations were consulted while
  scoping it (no idle probe ships here).
- **Steady KV pricing for Flash Next's sparse indexer.** Index keys and pooled blocks are priced per allocated
  position, not per valid token, so one short request no longer inflates the admission profile and evicts or
  refuses later long prompts (measured on the 8-bit model: after one 15-token request, the old pricing projected a
  69,967-token prompt 4.03 GiB higher, 2.06 GB over the MLX budget, so it was refused; now it is served). Ported from upstream
  [ashhart/TensorFold](https://github.com/ashhart/TensorFold) commit 17e6390f (Apache-2.0; see
  THIRD_PARTY_NOTICES.md). At 84 GiB the startup still picks 2,048-token prompt chunks.

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

## 0.3.6.3 (29 Sep 2026)

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

## 0.3.6.2 (28 Sep 2026)

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

## 0.3.6.1 (28 Sep 2026)

- **CUDA builds inside NVIDIA's containers again.** Their `TORCH_CUDA_ARCH_LIST` names every architecture back to
  sm_80, so the kernels' thread-block clusters and FP8 MMA failed to compile for GPUs that lack them. Every extension
  now builds for the GPU that is present, and a GPU older than compute capability 9.0 gets a clear message. Thanks to
  @ss-cong for the report and the exact errors (#56).

## 0.3.6 (28 Sep 2026)

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

## 0.3.5.1 (28 Sep 2026)

- Qwen3.8-27B loads on M1 and M2 Macs again. Kernels there fit Metal's per-kernel thread limit, with the same sums in
  the same order, so drafted output still equals serial output.
- M3, M4 and M5 run 0.3.5's machine code unchanged.
- Thanks to @hichaiuse, @simonmd, @gcarusso, @tonydehnke, @Cyb3r-Monk and @tinyapps for the reports and the repro.

## 0.3.5 (27 Sep 2026)

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

## 0.3.4.1 (27 Sep 2026)

- Prompt processing is back to MLX's speed on every model. Prompts prefill through MLX's own forward on a fixed
  2,048-token grid, so a resumed conversation still equals a fresh one byte for byte.
- Flash Next's peak memory stays within 20 GB of its weights up to a 196k-token prompt.

## 0.3.4 (26 Sep 2026, pre-release)

- Every model runs on the lane engine, and the serial engine is gone. Nemotron drafts on M1 to M4 with row-exact
  kernels.
- Qwen3.8-27B on M1 to M4 decodes at 1.9 to 4x serial, through a new 4-bit matmul on the simdgroup matrix units.
- CUDA: every default path verifies at least two rows a round.

## 0.3.3 (26 Sep 2026)

- Qwen3.8-27B verifies drafted tokens together on every M1 to M5 GPU, with output byte-identical to serial decoding.
- Streamed `/v1/completions` send plain text.

## 0.3.2 (26 Sep 2026)

- `tensorfold update` installs the newest release.
- GLM-5.3-Flash reads Mia-AiLab's EXL3 weights on two DGX Sparks (experimental).

## 0.3.1 (26 Sep 2026)

- `tensorfold info` shows how a checkpoint stores its weights and which backends read them. `serve` and `pull` refuse
  checkpoints no engine reads yet, before anything downloads.

## 0.3.0 (26 Sep 2026)

- NVIDIA GPUs: `tensorfold serve` picks CUDA on Linux and runs on one GPU or two (one rank per DGX Spark).
- A new family, GLM-5.3-Flash, on two Sparks.

## 0.2.0 (25 Sep 2026)

- A rewrite: `tensorfold serve`, `pull`, `models` and `info` for Nemotron 3.5 Lightning, Qwen3.8-27B and Qwen3.8
  Flash Next on Apple Silicon, with drafts that never change the output.
