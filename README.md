# d3-trt

A prefill-only serving runtime for [`vllm-sr/d3-turbo`](https://huggingface.co/vllm-sr/d3-turbo) that keeps the
reference d3 prompt/answer contract and replaces only the backbone forward pass.

The first-class fast path is **TensorRT-LLM** (NVIDIA, CUDA). The stock HuggingFace
Transformers path is both the parity oracle and the CPU / media fallback. A vLLM
backend is the intended second fast path; the backend boundary is already in place.

> Status: the HF path is validated on the tiny checkpoint and the reference HTTP
> contract is green. The TensorRT-LLM path is implemented end to end but has **not
> been run on a GPU in this workspace**; see [docs/VALIDATION.md](docs/VALIDATION.md)
> for the GPU-day checklist. Text only on the TRT path for now; images, videos and
> Box/Track requests use the HF fallback.

## Why a custom runner

d3 is not a chat model. It does one prefill per request and never decodes:

* the *state* block attends bidirectionally inside itself;
* each *question* block attends to the whole state plus itself, also bidirectionally;
* the output is a 255-way readout at selected positions, with an FP32 softmax;
* up to 255 question blocks share the same state, so the state pass should run once.

TensorRT-LLM's generation loop, sampling, stop criteria and paged-KV policy are the
wrong abstraction for that. This package uses TRT-LLM's optimized modules and kernels
(`Linear`, `Embedding`, `RMSNorm`) inside a custom state-once scheduler:

```
client -> reference-compatible FastAPI -> Frontend (d3_format / d3_runtime semantics)
                                              |
                                              +-- HF backend (oracle, media, CPU)
                                              +-- TRT-LLM backend (text fast path)
                                                     state pass once -> state cache
                                                     batched question passes
```

The reference `d3_server.py` is reused unchanged through a `ServedModel` adapter, so
the HTTP contract (`/health`, `/v1/models`, `/v1/systemone`, 422 handling) is the
published one.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,server]"
```

For the GPU path install a TensorRT-LLM wheel matching the CUDA/driver stack first,
then:

```bash
pip install -e ".[cuda,server]"
# flash-attn, flash-linear-attention, causal-conv1d, triton
```

`tensorrt-llm` is not on PyPI for every platform; use NVIDIA's index or the container
that ships the wheel. `d3-trt check --model <dir>` reports what is importable.

A d3 package directory must contain `config.json`, `decision_config.json`,
`readout.safetensors`, the tokenizer, the model safetensors and the reference
runtime files (`d3_runtime.py`, `d3_format.py`, `d3_compose.py`, `d3_server.py`,
`modeling_d3.py`). The public `vllm-sr/d3-turbo` package has all of them.

## Quick start

```bash
# inspect the package and which backends are importable
d3-trt check --model ./d3-turbo

# compare a backend to the published reference runtime
d3-trt parity --model ./d3-turbo --backend hf --cases tests/golden/text.json

# serve the reference System One API
d3-trt serve --model ./d3-turbo --backend hf --host 0.0.0.0 --port 8000

# TensorRT-LLM on a GPU node
d3-trt serve --model ./d3-turbo --backend trtllm --device cuda:0 \
  --state-cache-gb 8 --fallback --fallback-device cpu

# benchmark a request shape
d3-trt bench --model ./d3-turbo --backend trtllm --state-tokens 4096 \
  --questions 16 --repeats 5 --roofline

# compare optimisation switches, per-layer timing and a Chrome trace
d3-trt profile --model ./d3-turbo --backend trtllm --suite all \
  --state-tokens 4096 --questions 16 --trace-out traces/d3.json \
  --json-out traces/profile.json
```

## Profiling

`d3-trt profile` applies runtime switches to an already loaded engine, so weights are
not reloaded between variants. It reports wall latency, the state/question/readout
split, cache hit rate, the kernel that actually ran, and peak GPU memory. See
[docs/PROFILING.md](docs/PROFILING.md) for the variant list, cold vs warm cache
semantics, per-layer timings, `torch.profiler` traces and `nsys`/`ncu` commands.

## Contract and parity gates

The reference d3 semantics are not reimplemented. The frontend is built from the
reference runtime via an `object.__new__(D3)` skeleton, so prompt rendering, answer
assembly and Box/Track readout offsets stay byte-for-byte identical.

Gates (the same ones as the plan):

| Gate | Threshold |
| --- | --- |
| Per-answer probability absolute difference | `<= 1e-2` |
| Argmax agreement | `100%` |
| Jev Decision Index | within `±0.3` of 70.2 text / 72.9 vision |

The HF backend currently passes the probability and argmax gates on the tiny test
checkpoint. The Decision Index requires the official 0.3 kit and the real checkpoint
on a GPU; it is a GPU-day item, not a unit test.

## Documentation

* [docs/TENSORRT.md](docs/TENSORRT.md) — TRT-LLM design, weight loading, kernels,
  GPU-day install and the TP/quantization roadmap.
* [docs/VALIDATION.md](docs/VALIDATION.md) — parity harness, golden sets, tiny
  checkpoint, gates and the command sequence for a GPU day.
* [docs/PROFILING.md](docs/PROFILING.md) — optimisation matrix, per-layer profile,
  traces, external profiler commands and how to read the numbers.

## Limitations

* The TRT-LLM path is text-only; media and composed requests need the HF fallback.
* TRT-LLM build supports `tp_size=1`; FP8/NVFP4 checkpoint loading is not implemented.
* The default 24 GiB state cache is too large for an 80 GB BF16 card; use
  `--state-cache-gb 8` or less. State memory is ~64 KiB per state token for KV plus
  ~150 MiB of GDN state per snapshot.
* BF16 weights are ~47.7 GiB; one 80 GB GPU is the minimum for BF16. FP8 would fit
  48 GB but is not implemented yet.
* The test suite runs on a tiny random checkpoint with the real tokenizer/config
  contract; it cannot validate numerics against the real 27B weights.
