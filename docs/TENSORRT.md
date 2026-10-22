# TensorRT-LLM backend

The TRT-LLM path is a **custom prefill-only runner**, not `trtllm-serve` and not a
compiled TensorRT engine. It reuses TRT-LLM's optimized PyTorch modules and NVIDIA
kernels, while keeping d3's state-once schedule, bidirectional attention and
position-wise readout.

## Why not `LLM.generate` / a compiled engine

* d3 never decodes or samples. Generation-loop features are dead weight.
* The state pass must run once and then be reused by up to 255 question blocks.
* Full-attention state and question blocks are non-causal; GDN stays causal.
* The output is `hidden @ readout.T` at selected positions, not a vocabulary
  distribution.
* The state boundary is arbitrary, so a block-aligned paged-KV cache is not enough;
  the runner needs an explicit snapshot of KV + GDN recurrent/conv state.

## Module stack

`d3trt/model/trtllm_stack.py` builds:

* `Embedding` for the token embedding;
* 64 decoder layers: 16 `_TrtllmAttention` + 48 `_TrtllmGDN`, each with
  `input_layernorm`, `post_attention_layernorm`, and a fused gate/up MLP;
* a final zero-centered RMSNorm;
* the FP32 `[255, hidden]` readout matrix loaded from `readout.safetensors`.

`D3Core` (the same code used by the HF backend) drives the math. It expects the
module attributes, not a particular module class, so the HF and TRT-LLM stacks run
identical Python control flow. The differences are the module implementations and
the kernels selected by `d3trt/model/ops.py`.

Weight loading is one layer at a time (`ShardedWeights`), so the full 47.7 GiB BF16
checkpoint is never mirrored in host RAM while the GPU copy is built. HF names are
mapped to TRT-LLM module names, including concatenating `gate_proj` + `up_proj`
into `gate_up_proj`.

## Kernels

| Path | Implementation | Fallback |
| --- | --- | --- |
| Full attention, state | FlashAttention 3/2 via `flash_attn_func` | PyTorch SDPA |
| Full attention, questions | shared-prefix Triton kernel (`kernels/prefix_bidir.py`) | FlashAttention, then SDPA |
| GDN scan | FLA `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` | Transformers reference PyTorch |
| GDN convolution | `causal-conv1d` | `F.conv1d` |
| Linear / norm | TRT-LLM `Linear`, `RMSNorm` | not used on this backend |

The Triton kernel reads the shared state K/V directly for every question instead of
copying it per question. It expects `[H, S, D]` K/V in token-major memory (head stride
`D`, token stride `H*D`) and normalizes a contiguous `[H, S, D]` view if needed.

The active kernel is recorded in `CoreStats` after each pass. A request that asks for
`flash` but gets `sdpa` means FlashAttention was not importable or the dtype was not
supported; use `d3-trt check` and the profiler's `kernel` field to confirm.

## State cache

`StateCache` is content-addressed by state token ids and byte-budgeted LRU. One entry
holds the per-layer snapshot:

* K/V for the 16 full-attention layers: about 64 KiB per state token;
* conv tail + recurrent state for the 48 GDN layers: about 150 MiB per snapshot.

A 32K-token state is roughly 2.2 GiB; 128K is roughly 8.3 GiB. The package default is
24 GiB, which is too aggressive on an 80 GB BF16 card because the weights already
take ~47.7 GiB. Use `--state-cache-gb 8` or lower on 80 GB, and raise it on H200/B200.

Cold versus warm behavior is described in [PROFILING.md](PROFILING.md). The cache is
per process and currently not shared across TP ranks.

## Install on a GPU node

The wheel must match the CUDA, driver and PyTorch versions in the container. A
typical H100/H200 setup:

```bash
# inside the NVIDIA PyTorch / TRT-LLM container, or after installing the matching wheel
pip install tensorrt-llm
pip install flash-attn flash-linear-attention causal-conv1d triton
pip install -e ".[server]"
d3-trt check --model ./d3-turbo
```

`d3-trt check` prints a `trtllm.available` boolean and the import error if it fails.
It does not load the 47.7 GiB checkpoint.

## GPU-day sequence

```bash
# 1. contract and parity first
d3-trt parity --model ./d3-turbo --backend hf --cases tests/golden/text.json
d3-trt parity --model ./d3-turbo --backend trtllm --cases tests/golden/text.json

# 2. cold/warm matrix
d3-trt profile --model ./d3-turbo --backend trtllm --suite variants \
  --variants all --cache-modes cold,warm --state-tokens 4096 --questions 16 \
  --repeats 10 --warmup 2 --state-cache-gb 8 --json-out traces/trt.json

# 3. per-layer and trace on the winner
d3-trt profile --model ./d3-turbo --backend trtllm --suite all \
  --variants trtllm+cache+triton --layer-variant trtllm+cache+triton \
  --trace-variant trtllm+cache+triton --state-tokens 4096 --questions 16 \
  --state-cache-gb 8 --json-out traces/trt-win.json \
  --layers-out traces/trt-layers.json --trace-out traces/trt-trace.json

# 4. benchmark and roofline
d3-trt bench --model ./d3-turbo --backend trtllm --state-tokens 4096 \
  --questions 16 --repeats 10 --roofline --device-profile h100-sxm-bf16
```

Run `nsys`/`ncu` from the hints in the profile report, as described in
[PROFILING.md](PROFILING.md).

## Current limitations

* **No GPU validation yet.** The code path is complete and import-safe on a Mac, but
  the TRT-LLM wheel, CUDA kernels and weight layouts have not been exercised on a
  GPU in this workspace.
* **Text only.** `prepared.images` / `prepared.videos` raise `MediaNotSupported`.
  Use `--fallback --fallback-device cpu` to serve media through the HF backend.
* **TP=1 only.** `TrtllmBackend(tp_size>1)` raises `BackendUnavailable`. The mapping
  and weight-copy code is isolated in `trtllm_stack.py` for the next step.
* **No FP8/NVFP4 loading.** The stack loads BF16 HF weights. A quantized checkpoint
  needs a quantized module recipe and a parity/index gate.
* **No compiled TensorRT engines / CUDA graphs.** The first version favors
  flexibility; graph capture is a later optimization after the profile identifies
  launch-bound regions.
* **Vision tower not ported.** The HF fallback owns images/videos and Box/Track
  composition until a TensorRT vision path is justified by measurements.
