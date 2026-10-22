# Profiling and optimisation comparison

`d3-trt profile` is a matrix harness, not a single benchmark. It applies each
optimisation switch to an **already loaded** engine, runs the same synthetic request
shape, and records enough detail to explain the difference.

The point is to answer four separate questions:

1. How much does the state cache save on a cold request and on a warm one?
2. Which full-attention kernel actually served the request?
3. Is the GDN scan or the dense MLP the long pole at this state length?
4. Where does the time go inside the 64 decoder layers?

## What a variant is

A `Variant` is a set of overrides, not a fresh model. `apply_variant()` resets every
switch to the engine's base value first, then applies the override. That means
`core+cache+batch32,reference` really does run the reference path with the original
batch size, not a leaked batch size from the previous row.

### HF backend variants

| Name | What changes |
| --- | --- |
| `reference` | published `D3.run` state-once path; no cross-request cache |
| `core` | this package's `D3Core`; no state cache |
| `core+cache` | `D3Core` + content-addressed state cache |
| `core+cache+sdpa` | force PyTorch SDPA for state and questions |
| `core+cache+flash` | FlashAttention for state and questions (GPU only) |
| `core+cache+triton` | FlashAttention state + shared-prefix Triton questions (GPU only) |
| `core+cache+gdn-chunk` | force the chunked GDN scan for questions |
| `core+cache+gdn-recurrent` | recurrent GDN scan for question blocks `<= 256` tokens |
| `core+cache+batch1` | one question block per forward |
| `core+cache+batch32` | up to 32 question blocks per forward |
| `core+cache+compile` | `torch.compile` around each decoder layer |

### TRT-LLM backend variants

| Name | What changes |
| --- | --- |
| `trtllm` | TRT-LLM modules; no state cache |
| `trtllm+cache` | TRT-LLM modules + state cache |
| `trtllm+cache+sdpa` | SDPA comparison path |
| `trtllm+cache+flash` | FlashAttention for state and questions |
| `trtllm+cache+triton` | FlashAttention state + Triton questions |
| `trtllm+cache+gdn-recurrent` | recurrent GDN for short question blocks |
| `trtllm+cache+batch1` / `batch32` | question-block batch size |
| `trtllm+cache+compile` | compile each decoder layer |

The variant list comes from `d3trt.profile.default_variants()`. Add your own by
constructing `Variant(...)` objects and passing them to `profile_variants()`.

## Cold vs warm

The state cache changes the meaning of a request, so every cache-enabled variant is
run in both modes:

* **cold** — `cache.clear()` before each timed request. Every request pays the state
  pass; this is the first request from a new document.
* **warm** — warm-up populates the cache, then timed requests hit it. The state pass
  is skipped and `state_layers=0` in the kernel report.

The cache hit rate is reported as a **delta** over the measured window, not a
lifetime number, so a previous variant cannot contaminate it.

## Commands

### CPU smoke test

Use the tiny checkpoint and force the SDPA fallback. This validates the harness, not
GPU performance.

```bash
d3-trt profile --model .research/tiny/model --backend hf --device cpu --dtype float32 \
  --suite all --variants core+cache+sdpa --state-tokens 512 --questions 4 \
  --options 3 --repeats 2 --warmup 1 --cache-modes cold,warm \
  --json-out traces/profile-cpu.json --layers-out traces/layers-cpu.json \
  --trace-out traces/trace-cpu.json
```

### GPU optimisation matrix

```bash
d3-trt profile --model ./d3-turbo --backend trtllm --device cuda:0 \
  --suite variants --variants all --cache-modes cold,warm \
  --state-tokens 4096 --questions 16 --options 4 \
  --repeats 10 --warmup 2 --state-cache-gb 8 \
  --json-out traces/profile-trt.json
```

Run the same command with `--backend hf` for the reference/comparison matrix. Keep
the request shape, dtype, device and cache budget identical when comparing rows.

A single process cannot hold both 47.7 GiB BF16 checkpoints, so cross-backend
comparison is two runs plus a diff of the JSON reports:

```bash
d3-trt profile --model ./d3-turbo --backend hf --suite variants --variants all \
  --cache-modes cold,warm --state-tokens 4096 --questions 16 --warmup 2 \
  --repeats 10 --json-out traces/hf.json
d3-trt profile --model ./d3-turbo --backend trtllm --suite variants --variants all \
  --cache-modes cold,warm --state-tokens 4096 --questions 16 --warmup 2 \
  --repeats 10 --state-cache-gb 8 --json-out traces/trt.json
python - <<'PY'
import json
for name in ("hf", "trt"):
    rows = json.load(open(f"traces/{name}.json"))["variants"]["rows"]
    print(name, [(r["variant"], r["cache_mode"], r["latency_ms"]["p50"]) for r in rows])
PY
```

Compare only rows with the same variant name and cache mode. A TRT-LLM win on a
cold request but a loss on a warm request usually means the state cache or the
question kernel, not the linear layers.

### Per-layer breakdown

```bash
d3-trt profile --model ./d3-turbo --backend trtllm --suite layers \
  --variants trtllm+cache+triton --layer-variant trtllm+cache+triton \
  --state-tokens 4096 --questions 16 --layers-out traces/layers.json
```

For `D3Core`, the layer calls are timed directly with CUDA events (or
`perf_counter` on CPU). Both the state and question passes are included, and the
JSON separates `state_ms` from `question_ms` per layer. The reference path is timed
with hooks from `input_layernorm` to `mlp`; it reports `total_ms` only because the
reference does not expose the same phase split.

### Chrome / Perfetto trace

```bash
d3-trt profile --model ./d3-turbo --backend trtllm --suite trace \
  --variants trtllm+cache+triton --trace-variant trtllm+cache+triton \
  --state-tokens 4096 --questions 16 --trace-out traces/d3.json
```

Open `traces/d3.json` in `chrome://tracing` or <https://ui.perfetto.dev>. It shows
the CPU operator timeline and, on a GPU, the kernel timeline. The model also emits
`d3.state`, `d3.questions` and `d3.readout` NVTX ranges; those are primarily useful in
Nsight Systems, where they label the same phases on the GPU timeline.

### Nsight Systems / Nsight Compute

The profile report includes ready-to-run commands in `hints`. The shape is:

```bash
# timeline: CPU gaps, kernel overlap, memory copies
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o traces/d3-nsys python -m d3trt.cli bench --model ./d3-turbo \
  --backend trtllm --state-tokens 4096 --questions 16 --repeats 1

# one kernel at a time: occupancy, memory throughput, stalls
ncu --set full --target-processes all -o traces/d3-ncu \
  python -m d3trt.cli bench --model ./d3-turbo --backend trtllm \
  --state-tokens 1024 --questions 1 --repeats 1
```

Run `nsys` first: it answers whether the request is kernel-bound, launch-bound or
waiting on the CPU. Use `ncu` only after a trace identifies the kernel worth
optimising; full-set replay on a 25B-parameter model is expensive.

## Reading the report

Each row has:

* `latency_ms.mean/p50/p95/min/max` — wall time for one `system_one` call, including
  prompt preparation and response assembly. The engine is synchronized around the
  backend when `profile_sync=True`.
* `phase_ms.prepare` — tokenizer/prompt rendering.
* `phase_ms.backend` — the backbone forward and readout.
* `phase_ms.state` / `questions` / `readout` — the split inside the fast backend. A
  warm row with `state` near zero is a cache hit, not a missing measurement.
* `cache_hit_rate`, `cache_hits`, `cache_misses` — deltas over the measured window.
* `kernel.attention_state`, `kernel.attention_question`, `kernel.delta` — the kernel
  that actually ran. `sdpa` on a GPU means FlashAttention was requested but not
  available (or the dtype was unsupported). `torch-chunk` means FLA was not used.
* `cuda_memory.allocated_gib` / `reserved_gib` — peak **request** memory after
  resetting the peak counter, not total model weight memory.
* `samples` — every timed request, so a long first request (compilation, lazy
  allocation) is visible rather than hidden by the mean.

Compare rows only when the model, device, dtype, state length, question count,
option count and cache budget match. A 16-question warm run and a 1-question cold run
are different workloads.

## Recommended GPU-day sequence

1. `d3-trt check --model ./d3-turbo` — confirm TRT-LLM, FLA, Triton and
   FlashAttention are importable.
2. `d3-trt parity --backend trtllm --cases tests/golden/text.json` — correctness
   before speed.
3. `d3-trt profile --suite variants --variants all --cache-modes cold,warm` — pick
   the best attention/GDN/batch configuration.
4. `d3-trt profile --suite layers` on the winning configuration — find the long pole.
5. `d3-trt profile --suite trace` — confirm kernel launch gaps and copies.
6. `nsys` on one cold and one warm request — separate state-pass cost from
   question-pass cost.
7. `ncu` only on the kernel identified in step 5/6.

Record the software versions alongside the report: `torch`, `tensorrt_llm`,
`flash-attn`, `flash-linear-attention`, `triton`, CUDA driver, GPU model and power
limit. A kernel comparison without those is not reproducible.

## Caveats

* `torch.profiler` and per-layer CUDA events add overhead. Use them for attribution,
  not for the headline latency; use the plain variants matrix for that.
* CPU numbers validate logic only. SDPA on CPU is not a proxy for FlashAttention or
  Triton on GPU.
* `--allow-fallback` runs GPU-only variants on CPU with fallback kernels. The row is
  marked `fallback_on_cpu=true`; do not treat it as a GPU result.
* The default profile cache is 2 GiB. A 32K-token state is roughly 2.2 GiB of KV plus
  GDN state, so raise `--state-cache-gb` if you want the warm row to hold it.
* `compile` variants pay `torch.compile` cost during warm-up. A zero-warmup run
  measures compilation, not steady state.
