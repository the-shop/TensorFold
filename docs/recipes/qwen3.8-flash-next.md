# Qwen3.8 Flash Next

The `qwen4_exp` family has Gated DeltaNet, sparse attention, MoE, hyper-connections and hashed n-gram
embeddings. The supported checkpoint uses MLX affine 4-bit weights in groups of 32 and includes an MTP head.

```bash
tensorfold pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --name bench
```

On MLX, a supported conversion without the head runs without MTP drafting. On CUDA, pass `--no-drafts`
for such a conversion; the default positive draft depth otherwise refuses the missing head.

## MLX execution

N-gram tables stay in host file mappings when the checkpoint's model files exceed 75% of the GPU's
recommended working set. `TF_NGRAM_HOST=1` forces this mode; `TF_NGRAM_HOST=0` keeps the tables in MLX.
Startup admission uses the same choice as the loader and subtracts the mapped weights, scales and biases
from the checkpoint's size. For the named 4-bit checkpoint, about 29.8 GiB of its 105.4 GiB is mapped,
leaving a conservative 75.6 GiB resident-weight estimate. The default command selects host mode on an
M4 Max with 128 GiB, whose process budget is 89.6 GiB, including the 3 GiB process reserve.

Mapped pages still use RAM while cached. The loader prefetches them after its initial forwards; macOS
can reclaim them, and subsequent lookups may read from disk. The remaining weights must fit the MLX
budget, and the server sizes context from runtime cache and workspace needs. `TENSORFOLD_MEMORY_LIMIT_GB`
can lower or raise the default budget, capped by physical RAM and the GPU's recommended working set.
For example, `TENSORFOLD_MEMORY_LIMIT_GB=110` gives a 128 GiB M4 Max a 110 GiB process budget and
107 GiB for MLX. An explicit context must still fit the startup estimate; a larger budget does not
establish full-window inference or keep every mapped page resident.
After measuring shared rounds, the runtime releases the probes' rollback buffers before sizing prompt
memory, so those unused states do not reduce the available context.

Fused kernels handle hyper-connections, routing, experts, recurrence and sparse attention. Row-exact
projections and stable routing ties keep each verify row independent of the other rows. M5 GPUs use
the lane matmul; M1 through M4 use the per-row projection and hyper-connection kernels by default. Rejected tails
restore recurrent state, n-gram history and attention state, including incomplete pooled blocks.
The load-time row check disables drafting when windows do not reproduce serial steps.

The prefill path uses sparse selected-key attention, fused hyper-connections and n-gram lookups,
stacked DeltaNet projections and sorted expert rows. It submits bounded groups of layers to limit live
workspace. Compatible prefill matmul kernels check against MLX; unsupported paths use MLX's kernels.
`TF_FLASH_PREFILL=0` selects the reference prefill path for comparison.

Prefill and decode can round differently. The chunk planner uses detected assistant-message starts and
the second message when at least 256 tokens follow the previous chunk start, otherwise cutting after
the chunk chosen at startup: the largest of 8,192 (with tensor units), 4,096 and 2,048 tokens whose
working memory leaves room for 128K tokens of context, or the model's window if smaller. Where none
fits, the prompt path queues one layer at a time instead of two. Cold and resumed prompts use the same
rendered-token boundaries, and reuse starts only at these cuts. Templates without detected markers use
the same fixed chunks. A chunk's sorted expert rows reach MLX's gather in slices of at most 32,768
(its M5 kernel keeps row offsets in 16 bits through MLX 0.32.2). Snapshots include the prefill
path, resolved matmul route and GPU identity; changing arithmetic requires a fresh cache.

Load-time shared-forward checks compare each stream with its own call. A failed check limits forwards
to one stream, while successful checks allow the lane engine to combine requests.

### 8-bit checkpoints (M5, experimental)

`tools/convert_flash_next_q8.py SRC DST` streams a BF16 Hugging Face checkpoint (with its `mtp.*` head)
into an MLX affine 8-bit one, a tensor (or row chunk) at a time under an MLX memory cap: every linear,
embedding and expert stack 8-bit in groups of 64, the n-gram tables (160 values a row) 8-bit in groups of
32, the routers bf16, the vision tower dropped. The result is 194 GB (experts 119.5 GiB, n-gram tables
53.6 GiB, the rest 7.5 GiB), so serve it with `--ple-on-ssd --ssd-experts GIB`.

The loader reads each module's width and group off its packed shapes. 8-bit experts run the 4-bit expert
kernels' loops with 16 bytes a lane (resident and slot pool alike); token and n-gram rows dequantize 4
values a word; the hyper-connections run their down and up products through the lane matmul (M5 only)
between the prompt path's normed / act / mix kernels. 8-bit lane weights are read in place: set
`TF_FLASH_TILE_WIDE=1` to keep tiled copies (faster reads, about 4 GiB more). Each pool size gives the
same tokens, and drafted replies equal `"draft": false` ones.

#### Running it on a 128 GB M5 Mac

You need:
- an M5 Mac with 128 GB of RAM;
- Xcode Command Line Tools (`xcode-select --install`);
- about 560 GB of free SSD: the 360 GB BF16 download plus the 194 GB 8-bit output.

SSD expert streaming (`--ssd-experts`) comes from upstream TensorFold 0.3.6
([ashhart/TensorFold#17](https://github.com/ashhart/TensorFold/pull/17)). The 8-bit Flash Next kernels and converter, the
native expert-pool server, the opt-in prefetch, prompt seeding and expert pack, the short-prompt admission fix and
failing a request on a pool error are in the-shop's fork, released as `v0.1.0` (or use `main` for the latest):

```bash
git clone -b v0.1.0 https://github.com/the-shop/TensorFold.git && cd TensorFold
```

Run everything below from that directory, in a Python 3.11+ virtual environment. The SSD extras build the expert
pool's host side, a small MLX extension, with cmake and nanobind the first time it is used. The `hf` command comes
from `huggingface_hub` 0.34 or later.

```bash
python -m pip install -e ".[ssd]" "huggingface_hub>=0.34"
hf download Qwen/Qwen3.8-Flash-Next --local-dir flashnext-bf16
python tools/convert_flash_next_q8.py flashnext-bf16 flashnext-q8 --limit-gb 14
```

To skip the download and conversion, the checkpoint measured below is published already converted:
`hf download the-shop/MLXFW-Qwen3.8-Flash-Next-Heretic-MLX-8bit --local-dir flashnext-q8` (194 GB). It is
`trohrbaugh/Qwen3.8-Flash-Next-heretic`, a decensored derivative of Qwen/Qwen3.8-Flash-Next made with
`p-e-w/heretic`, converted with this tool. It has the same tensor names, shapes and quantization as a conversion
of the original.

The converter opens the source and output files with `F_NOCACHE` and reads with `pread`, so it doesn't fill the
page cache. It also caps MLX at `--limit-gb`, so it can run alongside other work. `--dry-run` prints the
shard plan and writes nothing.

Serve with a 60 GiB expert pool and thinking off (recommended for heavy use):

```bash
TENSORFOLD_MEMORY_LIMIT_GB=76 tensorfold serve flashnext-q8 --name bench \
  --ssd-experts 60 --ple-on-ssd --context 16384 --no-thinking
```

The lighter option is `--ssd-experts 40` with `TENSORFOLD_MEMORY_LIMIT_GB=56`, which leaves about 75 GiB free.

The pool and the context window share the memory budget:

| `--ssd-experts` | `TENSORFOLD_MEMORY_LIMIT_GB` | `--context` | Longest prompt that keeps its cache between turns | Peak MLX (`/health`) | Peak process (`top`, native loop) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 56 | 16,384 | 13,184 tokens | 52.1 GiB | 54 GiB |
| 40 | 60 | 32,768 | 31,744 tokens | not measured | not measured |
| 40 | 62 | 65,536 | 55,077 tokens | not measured | not measured |
| 60 | 76 | 16,384 | 13,198 tokens | 72.1 GiB | 74 GiB |
| 70 | 86 | 16,384 | 13,204 tokens | 82.1 GiB | 84 GiB |
| 80 | 96 | - | - | macOS swapped during load | |

A longer prompt is served, but it is prefilled again on its next turn.

- **Largest pool:** a 70 GiB pool (86 GiB limit) runs; against 60 GiB (native) it decodes -2% (harness B) / +5%
  (harness A), prefills 9-15% lower, and needs 10 GiB more. An
  80 GiB pool on a 96 GiB limit made macOS swap during load, and a 32K context did not fit either. 60 GiB is the
  recommended pool on a 128 GB Mac.

To check that drafting changes nothing, compare the token hash the server returns with drafts on and off:

```bash
for d in true false; do
  curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' -d '{"model":"bench",
    "messages":[{"role":"user","content":"Write a merge_intervals function in Python."}],
    "max_tokens":128,"temperature":0,"draft":'$d'}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["tensorfold"]["token_sha"])'
done   # the two lines must match
```

The numbers below come from two independent private harnesses, not `tools/bench_openai.py`. That tool uses
64-token replies, two prompts and client-side timing, so its numbers aren't comparable to these. They were measured
on 2026-10-10 on an M5 Max MacBook Pro (128 GB, internal SSD) serving the published 8-bit checkpoint above. The
machine's other model servers and background jobs were paused during the runs, and swap stayed flat through every
60 and 70 GiB run.

Every run used `--context 16384 --ple-on-ssd --no-thinking`, temperature 0, replies of up to 128 tokens and
`--snapshot-dir none`, so every prefill was cold; the page cache was displaced before each measurement window. Each value is a
median of server-side tok/s.
- **Harness A:** 5 prompts, each sent 5 times back to back with drafts (the first run is a never-seen prompt, runs
  2-5 are repeats), then a sixth time with `"draft": false` (a repeat). Prefill on a 3,381-token prompt (3 runs).
- **Harness B:** 7 short prompts with drafts (first visit), then the same 7 with `"draft": false` (second visit,
  7 other prompts in between). Prefill on a 2,475-token prompt.

Decode and prefill in tok/s; load time in seconds (harness A / harness B):

| Pool | Loop | A: first run | A: repeats | A: serial | A: prefill | B: drafts | B: serial | B: prefill | Pool hits | Load |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 GiB | native (default) | 22.0 | 25.0 | 16.9 | 263 | 18.8 | 13.4 | 198 | 90% | 70 / 72 |
| 40 GiB | Python (`TENSORFOLD_POOL_NATIVE=0`) | 20.0 | 23.4 | 16.2 | 244 | 16.9 | 12.8 | 183 | 90% | 78 / 76 |
| 60 GiB | native (default) | 27.2 | 58.8 | 30.5 | 255 | 23.0 | 15.5 | 190 | 98% | 81 / 78 |
| 60 GiB | Python | 24.9 | 59.5 | 30.4 | 240 | 20.7 | 14.6 | 176 | 98% | 81 / 84 |
| 70 GiB | native (default) | 28.6 | 60.0 | 30.8 | 217 | 22.5 | 16.0 | 172 | 98% | 87 / 88 |
| 70 GiB | Python | 27.1 | 58.1 | 29.9 | 195 | 22.0 | 15.0 | 160 | 98% | 89 / 96 |

- **Native vs Python loop:** at 40 and 60 GiB the native server decodes 9-11% faster with drafts (harness A first
  runs +10% / +9%, harness B +11% / +11%) and prefills 6-8% faster; at 70 GiB decode +2-6%, prefill +8-11%.
- **Pool size:** on new prompts, the native server at 60 GiB decodes 22% (harness B) to 24% (harness A) faster than
  at 40 GiB. A rerun of the same prompt mostly hits the pool at 60 GiB, so harness A repeats reach 58.8 tok/s
  against 25.0 at 40 GiB. A 70 GiB pool (native) decodes -2% (harness B) / +5% (harness A) against 60 GiB, prefills
  9-15% slower, and needs 10 GiB more.
- **Hit rate:** 90% of expert reads at 40 GiB, 98% at 60 and 70 GiB (the pool's own hits). Expert and n-gram reads
  use `F_NOCACHE`, but on macOS a read whose file offset or buffer is not 4 KiB-aligned leaves every byte after its
  first MiB in the page cache. Checkpoint tensors are not aligned, so about a third of each expert's bytes (the tail
  of each 1.56 MiB weight) stays cached unless RAM is short, and later misses on those pages are served from RAM.
- **Output:** output tokens were identical across all 12 measured configurations (the six above, 16 reader threads
  and the expert pack at each pool size), with drafts on and off.

Two opt-in flags stream experts ahead of time. Both default to off, and either one runs the Python loop instead of
the native server. Their numbers are older (2026-10-09, Python loop).
- **`TENSORFOLD_POOL_DECODE_PREFETCH=K`** (0 to 16): runs the next layer's router on this layer's MoE input and
  reads its top-K experts while the GPU works.
  - **Prefetches used:** 76% at K=4, 61% at K=8.
  - **Drafted decode:** -3% in one harness, +8% to +16% in the other.
  - **Prefill:** fell from 168 to 127 tok/s in the first harness, and from 198-241 to 160-192 in the second.
- **`TENSORFOLD_POOL_PROMPT_SEED=1`**: loads each layer's most-used prompt experts at the first decode step. The
  number per layer is `TENSORFOLD_POOL_PROMPT_SEED_PER_LAYER`, default 32. Together with K=4 prefetch, it was about
  14% slower than prefetch alone.

Both flags change only which experts are resident, never the tokens.

With both flags off (the default), a native server in the host-sync extension answers each layer's routing: its
own thread waits for the GPU, keeps the LRU and reads misses on native threads, with no Python per layer call.
- **`TENSORFOLD_POOL_NATIVE=0`** keeps the Python serve loop. Both loops keep the same experts in the same slots.
- **`TENSORFOLD_POOL_THREADS=N`** (default 8) sets the reader threads of either loop. 16 threads raised prefill
  4-9% at 40 and 60 GiB but not at 70, and kept decode within ±3% at 40 and 60 GiB and ±4% at 70, except harness B serial at 40 GiB (-10%), so
  the default stays 8.
- **Expert pack (opt-in, experimental, off by default):** `python tools/pack_experts.py MODEL_DIR` writes
  `MODEL_DIR/experts.pack` and its index: each expert's parts back to back in one 16 KiB-aligned record, so the
  native server reads a miss with one `pread` instead of nine. It takes as much disk as the experts. It is used only
  with `TENSORFOLD_EXPERT_PACK=1` (`MODEL_DIR/experts.pack`) or `TENSORFOLD_EXPERT_PACK=PATH`, and a pack that does
  not match the checkpoint (layout, each file's size, inode and modification time, a sample of the expert bytes) is
  an error: rebuild it after any change to the checkpoint files, a copy included. It was slower at 40 and 60 GiB
  (decode 16-23% lower, prefill about 16-21% lower; the 40 GiB harness-A pack values come from an earlier
  window the same day) and within 0-6% at 70 GiB. `/health` reports `packed`, `preads`
  and `threads` with the pool's hits and misses.
- **`TENSORFOLD_POOL_ALIGNED_READS=1`** (default off) reads each unaligned expert part over about 1 MiB as the
  page-aligned range around it into an aligned buffer, then copies the part out: the same bytes and the same one
  `pread` a part, with nothing left in the page cache. Off keeps the leftover tails, which also serve repeat reads.
- `python tools/page_cache.py evict MODEL_DIR` drops the checkpoint files from the page cache (no sudo; run it with
  no server holding the model), and `python tools/page_cache.py resident MODEL_DIR [--experts]` prints how many of
  their bytes (or of the routed experts' bytes) are cached. Evict before every cold benchmark.
- `python tools/bench_pool_io.py MODEL_DIR --cold` times miss reads for the Python loop, the native server and
  the pack at 4, 8, 16 and 32 threads, on experts none of whose bytes are in the page cache.

## CUDA

Use the [container setup](../../RUNBOOK.md#nvidia-gpus). For two ranks, pull the checkpoint on both and
start rank 1 first:

```bash
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 1 --master 192.0.2.1
tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --rank 0 --master 192.0.2.1 --name bench --host 0.0.0.0
```

The default CUDA cap is six MTP drafts, with chains stopping below the configured confidence threshold.
`--mtp-drafts N` changes the cap; `--no-drafts` or `"draft": false` selects serial decoding.
Single-request serving uses CUDA graphs for verify windows and draft steps. Two-rank reductions add
gathered partials in rank order.

With one GPU, `--parallel N` enables eager shared forwards for up to N requests; CUDA
`--parallel auto` selects one request. Two ranks serve one request at a time and reject `--parallel N`
when N exceeds one. The single-request engine retains prompt and reply states for prefix reuse; the
concurrent decoder retains prompt snapshots per stream. Cache capacity is allocated at startup; inspect
the reported capacity rather than assuming an older fixed token limit.

Saved prompt states survive restarts. System blocks go to `--snapshot-dir` and conversations to `--session-dir`
(default `session-snapshots` beside it). Each file carries a checksum and the weights' fingerprint and is refused
and removed when either does not match. A graceful stop saves the open conversations; `--persist-turns` also
writes each turn's conversation behind the reply (streamed in pieces when RAM is tight), so a crash keeps it.
`--checkpoint-last-user` keeps one more state where the last user message starts, for clients such as agent
harnesses that replace a trailing reminder message on every turn. On the 8-bit checkpoint (`--ssd-experts 60
--ple-on-ssd`, 84 GiB budget) with ~25K-token agent prompts, two independent harnesses measured: after a graceful
restart the next turn's first token came in 6.0-7.0 s against 284-332 s cold; with `--checkpoint-last-user` a
follow-up whose reminder message changed came in 14-23 s against 54-64 s; replies were the same tokens in every
case, including after a kill -9 with `--persist-turns`.

N-gram tables are file-backed host data. On unified-memory GPUs they compete with weights and cache
allocations for RAM, so a checkpoint's GPU allocation alone does not describe its memory requirement.

`--ple-on-ssd` leaves the 29.8 GiB of n-gram tables in the checkpoint and reads each lookup's rows from SSD,
so a 128 GB Mac can hold Flash Next. It is an opt-in trade. On an M3 Ultra, replies were the same tokens,
decode was 3.5-8% slower across the four cells, prefill was unchanged, the peak footprint fell by 40 GiB
(135.1 to 95.4 GiB) and start-up halved (35.7 s to 17.8 s).

`--ple-cache-mib MIB` (with `--ple-on-ssd`, Metal only; default 0, off) keeps n-gram rows read recently, from
prompts and replies, in that many MiB of RAM and reads only the other rows of a lookup from SSD; a lookup whose
rows are all cached reads nothing. Rows are evicted in CLOCK order. The cached bytes are the checkpoint's own, so
replies are the same tokens. The rows (about 82% of the MiB) and their index are fixed arrays allocated at load,
all counted in the memory budget beside the 3 GiB process reserve. Each PLE layer's table gets an even share,
capped at the whole table and its index; a share too small for one row is refused at start-up. A full re-index of
the cache (when evictions have left too many tombstones) holds its lock for about 0.37 s at 512 MiB.
`tools/bench_ple_cache.py` times lookups with and without it on a synthetic table. On the 8-bit checkpoint with
`--ssd-experts 60` at an 84 GiB budget, 512 MiB of cache served about 99% of decode lookups and the replies were the
same tokens, but decode was not faster in either of two independent harnesses (one measured no difference, the
other 17.1 vs 15.4 tok/s over three runs), so leave it off unless the SSD is the bottleneck.

`--ssd-experts GIB` also leaves the routed experts (70.3 GiB) in the checkpoint and streams them into a GPU pool
of that many GiB; with `--ple-on-ssd` as well, a 64 GB Mac can hold Flash Next (`python -m pip install
"tensorfold[ssd]"` first: the pool's host side is a small MLX extension built on first use). The GPU hands each
MoE layer's picks to the host and waits while missing experts are read into the pool; the expert kernels are the
resident ones with only the weight address changed, so replies are the resident model's tokens.

On an M3 Ultra, each flag was held to a smaller Mac's budget and compared on the same machine:
- `--ple-on-ssd` at a 128 GB Mac's budget (89.6 GiB) peaked at 85.6 GiB. Decode was 0.91-1.03x the resident
  run and prefill 0.84-0.91x.
- Adding `--ssd-experts 24` at a 64 GB Mac's budget (44.8 GiB) peaked at 39.5 GiB. Replies were the same tokens as
  with the experts resident: 36 of 36 requests, drafted and serial. Decode ran at 36.8-42.5 tok/s against
  107.6-129.7 (0.31-0.39x), and prefill at 332-354 against 1,014-1,081 tok/s.
- A smaller budget also halves the prompt chunk, to 2,048 tokens. Prompts past that length then reply
  differently from a 256 GB Mac's run, whichever flags are set.

### KV cache

`--kv-dtype bf16` is the default. `--kv-dtype int8` and `--kv-dtype int4` store each attention layer's keys and
values as codes with one fp16 scale per 32 values, the arithmetic of ExLlamaV3's `-cq 8` and `-cq 4` (the
non-companded grid): each group of 32 is rotated by a 32-point Hadamard, its absmax is the scale, and the codes sit
on the midpoint grid. 8-bit stores `q - 128` as int8; 4-bit stores two unsigned codes a byte, low nibble first. The
query is rotated the same way and the merged attention output is rotated back, so the stored keys and values stay
rotated. Indexer keys and pooled block keys stay bf16.

A token costs 30,784 bytes in bf16, 18,304 in int8 and 11,648 in int4, counting the scales and the MTP head's own
cache: 1.68x and 2.64x smaller (the keys and values alone shrink 1.88x and 3.56x). The startup admission counts
those bytes, so an omitted `--context` admits a longer window at int8 and int4, and an explicit `--context` is
checked against the quantized cache. The dtype holds on every path: prompt chunks and decode windows, the MTP head,
`"draft": false` requests, `--parallel N` streams and their kept prompt ends, and both ranks of `--tp 2`, which
refuse to start with different `--kv-dtype` values.

A quantized cache changes the output, so its replies differ from bf16's. Drafted output still equals
`"draft": false` output at the same dtype, and a resumed prompt equals a fresh one. The MLX path and the other
families refuse `--kv-dtype` before any download.

`--mtp-confidence P`, from 0 to 1, sets the probability under which a chain stops before a later draft; the CUDA
default is 0.30. Only Flash Next's CUDA engine has this rule, so the MLX path and the other families refuse it.

### EXL3 checkpoints (experimental)

The CUDA engine also serves EXL3 packs of Flash Next (`quant_method: exl3`) on one GPU through the shared EXL3
module ([EXL3 weights](exl3.md)): any codebook, and a width per tensor, so mixed-K packs whose experts range from
2 to 8 bits within a layer load as they are. turboderp publishes a 3.05 bpw pack on the branch `3.05bpw_h5_ng5`.
Download that branch, then serve the folder:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('turboderp/Qwen3.8-Flash-Next-exl3', revision='3.05bpw_h5_ng5', local_dir='flashnext-exl3-3.05bpw')"
python -m tensorfold.cuda.exl3.inspect flashnext-exl3-3.05bpw   # bits per tensor, from the headers
tensorfold serve flashnext-exl3-3.05bpw --host 0.0.0.0 --port 8080
```

A pack maps its n-gram table from its own file and runs on one GPU: `--tp 2` and `--ple-on-ssd` are for the MLX
checkpoint, and an EXL3 pack refuses both.

- Dense projections (attention, DeltaNet, the MTP head's fc layers, the head) run on the row-invariant EXL3
  linear (`cuda/exl3/linear.py`). Routed experts and the shared expert (as expert 512 of the same table) run on
  the grouped EXL3 expert kernel (`cuda/exl3/experts.py`), each expert matrix at its own width. The fp32 router,
  its top-k with id tie-break, and the write-back in slot order are the MLX path's.
- The tensors a pack leaves unquantized (hyper-connection down / inject / up, DeltaNet `in_proj_a/b`, the n-gram
  key and value) run on an fp16 Triton matmul whose tiles and K split depend only on the shape. No cuBLAS.
- The packs store every centred RMSNorm weight as gamma - 1: the hyper-connection norms, q/k norms, the
  indexer's, the n-gram branch's three and the MTP head's two input norms. The loader detects this and adds 1
  in fp32. Checked tensor by tensor against the MLX checkpoint: after the offset the largest difference is
  0.031 (hyper-connection norms) and 0.004 (n-gram norms). DeltaNet's gated norm, `A_log`, `dt_bias`, the conv weights and the router equal the MLX values.
- The 128 `ple.ngram_embedding.shard_*` tensors are named `.trellis` but are not EXL3 tiles. They use
  ExLlamaV3's n-gram row codec: a row is one fp16 scale followed by 160 values of K bits in a tail-biting mul1
  trellis, and each head has an fp16 bias. The engine decodes a window's rows as ExLlamaV3's `ngram_dequant`
  does, bit for bit (tested at 2 to 8 bits). The rows stay in the checkpoint, memory-mapped.
- turboderp's pack keeps the MTP head's final mixer in `mtp_hyper_connection_mixer_patch.safetensors`,
  outside the index, and the reader loads it from there. A pack that carries the MTP layer drafts with it.
  For the draft head, the head's rows for the draft vocabulary are decoded through the EXL3 linear and
  requantized to 4-bit groups of 32. This affects only which drafts are proposed; verification uses the full
  EXL3 head.
- Prompt chunks decode each dense projection's weights once a chunk and multiply with a fixed-tile GEMM, so a
  prompt's rows do not depend on its chunking; routed experts run the grouped kernel in 1,024-row windows.

Measured on one DGX Spark (GB10) through `tensorfold serve`: the 3.05 bpw pack against the MLX 4-bit checkpoint
on the same engine and box, the [public benchmark command](README.md#measurements), medians of 10 runs a cell
(two sessions of each, alternated):

| Cell | EXL3 3.05 bpw | MLX 4-bit | vLLM MTP=3 |
| --- | ---: | ---: | ---: |
| Code, sampled | 80.8 tok/s | 76.5 tok/s | 42.4 tok/s |
| Chat, sampled | 59.4 tok/s | 62.7 tok/s | 33.2 tok/s |
| Code, greedy | 77.7 tok/s | 74.2 tok/s | 40.9 tok/s |
| Chat, greedy | 69.2 tok/s | 72.4 tok/s | 37.6 tok/s |

Drafted replies equal `"draft": false` ones (9 of 9), resumed prompts equal fresh ones, and four concurrent
streams equal their solo runs. Cold prefill runs 930-970 tok/s from 2k to 64k, about 0.4x the MLX checkpoint's
2,300-2,500: the routed experts decode their tiles again for every 16 rows. The pack's weights take 52 GB of GPU
memory against the MLX checkpoint's 81 GB, so on a Spark the default window is the full 262,144 tokens (57.6 GiB
allocated after loading) where the MLX checkpoint's is about 74,000; the 32.6 GB n-gram table stays in the page
cache.

#### Against ExLlamaV3 on the same pack

These comparisons were made when the path was written, on the 0.3.4.1 engine, with two packs: turboderp's
3.05 bpw and SAGE, a mixed-K 4.15 bpw pack the contributor built (2 to 8 bits a matrix; not published).

The first comparison is teacher-forced over 2,048 positions: the first 2,049 tokens of ExLlamaV3's
`eval/eval_texts`, all fed in one pass. TensorFold decodes in 64-row windows. ExLlamaV3 is the fork at 249f22a
with its mixed-K expert path. The token ids are the same for all three checkpoints, which share one
tokenizer.

| | top-1 agreement | TensorFold NLL | ExLlamaV3 NLL | TensorFold top-1 acc | ExLlamaV3 top-1 acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGE 4.15 bpw (mixed-K, 2-8 bits) | 0.954 | 1.0640 | 1.0642 | 0.718 | 0.720 |
| turboderp 3.05 bpw | 0.959 | 0.8510 | 0.8506 | 0.774 | 0.779 |
| TensorFold, MLX 4-bit (same tokens) | | 1.0574 | | 0.726 | |

The two engines' mean NLLs agree within 0.0005. The positions where they pick different tokens are close
calls. Where ExLlamaV3's top two logits are at least 0.5 apart, the engines agree on 98.8% (SAGE) and 99.1%
(3.05) of positions; at a gap of at least 1.0 they agree on 99.7%. ExLlamaV3 is no closer to itself when the
batch shape changes: rerunning the first 1,025 tokens alone, instead of as part of all 2,049, it agrees with its
own full pass on 96.4% (SAGE) and 97.2% (3.05) of positions. At the same positions TensorFold agrees with
ExLlamaV3's full pass on 95.5% and 97.0%. The per-position NLL difference has a median of 0.018 and 0.011.

On TensorFold's side, `tests/cuda/test_qwen4_exp_exl3.py` checks the following against the pack that
`TENSORFOLD_EXL3_FLASHNEXT` names, cut to 4 layers plus the head and the MTP head:

- a window of 1 to 128 rows gives one-row steps' logits bit for bit;
- MTP-drafted decoding gives serial decoding's tokens, eager and in CUDA graphs, greedy and sampled;
- a real layer's experts match an fp64 reference and are row-invariant.

Through `tensorfold serve` on the full packs, each reply was requested with `"draft": false` and again with
drafts. The prompts were the benchmark's two, the raw prompt as chat, and two 256-token chats, each greedy and
with seeds 1234 to 1238. On both packs, 30 of 30 replies had identical token-id SHA-256s. A drafted round
emitted 2.93 tokens on average on SAGE and 3.16 on 3.05.

```bash
TENSORFOLD_EXL3_FLASHNEXT=flashnext-exl3-3.05bpw pytest -q tests/cuda/test_qwen4_exp_exl3.py
```

#### Serial speed against ExLlamaV3

These are decode tokens per second after the first token, on one Spark. Each cell is one stream, 64-token replies,
and the median of seeds 1234 to 1238 after one warm-up. Every engine went through the same client:
`python tools/bench_openai.py http://127.0.0.1:PORT MODEL --tokens 64 --reps 5`. All TensorFold rows ran in
one exclusive GPU session, with every pack's page cache dropped before each server started. ExLlamaV3 ran behind a
minimal OpenAI shim around its `Generator`, serially: batch size 1, an 8,192-token cache, n-gram tables in host
RAM, no drafts. The ExLlamaV3 SAGE row comes from a second session. In that session it first failed to load
("Insufficient VRAM in split") while the previous server's pack was still in the page cache. The second session
also re-measured TensorFold on SAGE serially at 34.6 / 34.4 / 34.8 / 34.6, within 3.5% of the table.

| | Code, sampled | Chat, sampled | Code, greedy | Chat, greedy |
| --- | ---: | ---: | ---: | ---: |
| ExLlamaV3, SAGE 4.15 bpw, serial | 22.3 | 22.2 | 22.3 | 22.2 |
| TensorFold, SAGE 4.15 bpw, serial | 34.3 | 34.0 | 34.5 | 33.5 |
| ExLlamaV3, turboderp 3.05 bpw, serial | 37.0 | 36.9 | 37.1 | 36.9 |
| TensorFold, turboderp 3.05 bpw, serial | 34.9 | 34.9 | 35.1 | 35.0 |
| TensorFold, MLX 4-bit, serial | 36.8 | 36.7 | 37.0 | 36.8 |

Serially on the mixed-K pack, TensorFold is 1.51-1.55x ExLlamaV3. On the uniform 3.05 bpw pack
TensorFold's serial path is 5-6% behind ExLlamaV3's, and on both EXL3 packs it is 5-9% behind its own MLX path.
With drafts on the current engine (the table above), the 3.05 bpw pack decodes 1.6-2.2x ExLlamaV3's serial speed.

### NVFP4 checkpoints

The CUDA engine reads two published ModelOpt NVFP4 exports as they ship, MTP head included, on one GPU:

```bash
tensorfold serve local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8080
```

| Checkpoint (revision) | Routed experts | DeltaNet, attention, shared expert | n-gram table |
| --- | --- | --- | --- |
| `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (`7c4f1bc1`) | NVFP4 | MXFP8 | NVFP4 rows |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` (`7b719225`) | NVFP4 | bf16 | FP8 rows |

The loader reads each linear by its tensors. An NVFP4 weight is an E2M1 code times its e4m3 scale (a block of 16
inputs) times the tensor's fp32 `weight_scale_2`; an MXFP8 weight is an e4m3 byte times a power of two (a block of
32). Both products fit bf16 exactly, so decode multiplies them in bf16 MMAs, adds each block's products times its
scale in block order and applies the tensor's scale once; the K split depends on the shape alone, so drafted
windows keep serial decoding's bits. Prompts run the MXFP8 linears on the FP8 prompt matmul with the stored bytes.
Tests check the kernels against an fp64 reference built by an independent numpy dequantizer
(`tensorfold/cuda/nvfp4/format.py`). Both exports store their RMSNorm weights centred (gamma - 1), and the loader
tells centred from uncentred norms by their stored values. An n-gram table's shards must share one layout, or the
load stops.

The routed experts run on a grouped NVFP4 kernel that reads the step's routing plan on the GPU, so a decode graph
captured for one step's experts replays another step's. A test decodes a checkpoint whose expert picks change
every step with graphs on and off and compares the tokens; on both exports, replies with the decode graphs equal
replies without them.

Both exports were served with `tensorfold serve` and checked: drafted replies equal `"draft": false` ones (nine
pairs, 2k-16k-token prompts, greedy and sampled), resumed prompts equal fresh ones, and each reply of concurrent
requests (`--parallel 4`) equals the same request alone.

Not supported here:
- `--tp 2` and `--ple-on-ssd` on an NVFP4 checkpoint stop at startup: two ranks read the MLX checkpoint, and the
  NVFP4 exports' tables stay memory-mapped.
- `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` (186 GB; bf16 linears and n-gram table) has the RadixArk layout apart
  from its bf16 table, which the loader reads, but the whole checkpoint has not been loaded or served here, so it
  is not listed as tested.

## Draft vocabulary provenance

Both backends read the public list in `src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt`.
It contains 79,591 sorted IDs; MLX pads it with the lowest unused IDs to a multiple of 64 at load.
The target sampler still reads the full vocabulary, so the list affects proposals only.

The corpus is Homebrew CPython 3.14.5's standard-library `*.py` files, excluding `site-packages` and
`__pycache__`. It contains no repository or PyPI package text. Use `tokenizers==0.22.2` and
`tools/draft_vocab.py` with SHA-256
`1baf0dd08669355cf9cf6e32998e5436ce8712c3ab1bffe38d369f3fa3a86b56`.
The tokenizer JSON SHA-256 is:

```text
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3
```

Place that tokenizer at `tokenizer.json` and copy the clean stdlib into an empty `cpython` directory,
preserving relative paths. The [Nemotron corpus-copy example](nemotron-3.5.md#draft-vocabulary-provenance)
shows the file-selection rules; change its destination to `cpython`. Then run from the repository root:

```bash
TOKENIZERS_PARALLELISM=false python3 -B tools/draft_vocab.py tokenizer.json draft_vocab.txt --size 79591 --keep-below 65536 --min-count 1 --added-tokens 'cpython/**/*.py'
```

The generator retains all IDs below 65,536 and tokenizer-added IDs, then adds corpus IDs by frequency
and fills remaining places with the lowest unused IDs. It skips empty, unreadable and oversized files.
Expected output SHA-256:

```text
88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e
```

Check this hash before adopting a rebuild; a different stdlib distribution can change the corpus.
The current list replaces the older list whose PyPI corpus was not reproducible.

## Measurements

Use the [public benchmark command](README.md#measurements) with the server above; omit tensor-parallel
flags for one rank. The client supplies fixed public prompts, 64-token replies and seeds 1234 through
1238. Historical rates measured with the earlier draft list do not qualify the current list.

Use a separately pinned public long-context fixture when measuring prefill and reuse. Check fresh versus
resumed prompts across sparse-attention transitions and template changes, as well as drafted versus
serial output. Decode, cold/resumed latency, concurrent throughput and peak memory are
TBD [release-0.3.5].
