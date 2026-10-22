# Validation and parity gates

Correctness is checked before speed. The reference runtime is the oracle; the fast
backends must reproduce its probabilities within the published budget.

## Gates

| Gate | Threshold |
| --- | --- |
| Per-answer probability absolute difference | `<= 1e-2` |
| Argmax agreement | `100%` |
| Jev Decision Index | within `±0.3` of 70.2 text / 72.9 vision |

The first two are tested by `d3-trt parity`. The Decision Index needs the official
0.3 kit, the real checkpoint and a GPU; it is not a unit test. The index is the
release gate, not a substitute for per-answer parity.

## Tiny checkpoint

The repository tests do not download the 27B model. `.research/tiny/model` is a tiny
random checkpoint with the real tokenizer and config contract (2 layers, hidden 128,
same layer types, real `decision_config.json`). Regenerate it with:

```bash
python scripts/make_tiny_model.py
```

The tests skip automatically if the checkpoint is absent; set `D3_TEST_MODEL` to
another directory to use a different tiny model. The tiny model validates control
flow, prompt rendering, state splitting, readout offsets, the HTTP contract, cache
behavior and the profiling harness. It cannot validate 27B numerics.

## HF parity

```bash
PYTHONPATH=. python -m pytest -q
d3-trt parity --model .research/tiny/model --backend hf \
  --cases tests/golden/text.json
```

`tests/golden/text.json` contains text choice / noul cases with full reference
probabilities. The parity harness:

1. loads the published `D3` and the selected backend;
2. prepares the same request through the reference frontend;
3. compares every answer probability, not just the argmax;
4. reports `max_abs_diff`, `choice_agreement` and per-case detail.

The HF core path uses the reference chunk GDN kernel on the parity path
(`recurrent_max_tokens=0`) so differences come from attention/kernel selection, not
from a different scan order.

## TRT-LLM parity on a GPU

```bash
d3-trt check --model ./d3-turbo
d3-trt parity --model ./d3-turbo --backend hf --cases tests/golden/text.json
d3-trt parity --model ./d3-turbo --backend trtllm --cases tests/golden/text.json \
  --json-out traces/parity-trt.json
```

Run the HF parity first on the same GPU/dtype. If HF passes and TRT-LLM does not, the
gap is in the TRT-LLM module stack or kernel selection, not in the frontend.

Recommended order when a case fails:

1. Run the same case with `--backend hf`, `--dtype bfloat16`, `--device cuda:0`.
2. Run `core+cache+sdpa` variants on both backends to remove kernel differences.
3. Compare per-layer hidden states if the gap remains; then check the weight mapping
   (`q_norm`/`k_norm` zero-centered RMSNorm, `gate_up_proj` concatenation order,
   GDN `A_log`/`dt_bias`/`conv1d` layout).
4. Only then test FlashAttention/Triton/FLA kernels.

## Golden set coverage

The release golden set should include:

* text `choice`, `noul`, `score`, varied option counts (2 to 255);
* long states (8K, 32K, near the 131K long-context limit);
* images, videos, images inside options, `assets` placeholders;
* Box and Track questions, including track windows;
* permutation-average on and off;
* near-limit and over-limit requests (`max_length_exceeded` must never truncate).

The tiny `tests/golden/text.json` is a smoke set, not the release golden set.

## Profiling as a validation tool

Parity says *whether* a variant is correct; the profiler says *what it costs*. Use
them together:

```bash
# correctness first
d3-trt parity --backend trtllm --cases tests/golden/text.json

# then speed on the same request shape
d3-trt profile --backend trtllm --suite variants --variants all \
  --cache-modes cold,warm --state-tokens 4096 --questions 16 \
  --repeats 10 --warmup 2 --state-cache-gb 8
```

A fast variant that changes the argmax is not a candidate. A correct variant that is
slower than the HF path at the target shape is also not a candidate. Record the
software stack (CUDA driver, PyTorch, TRT-LLM, FlashAttention, FLA, Triton, GPU
model and power limit) with every report.

## HTTP contract

`tests/test_server.py` drives the reference `d3_server.py` through the
`DecisionEngine` adapter and checks:

* `GET /health`;
* `GET /v1/models`;
* `POST /v1/systemone` for a valid request;
* `422` for an invalid question shape.

The adapter changes only the backbone; prompt rendering, answer codes and response
assembly remain the published implementation.
