"""Benchmark and roofline tools for the d3 serving runtime.

Two questions the GPU-day run has to answer:

1. How much did state-once + the shared-prefix kernels actually save?
   ``run_benchmark`` reports per-request latency, question throughput, token
   throughput and the state-cache hit rate for a synthetic request shape.
2. What should the number be?
   ``roofline`` estimates the dense FLOPs, the quadratic attention FLOPs, the
   bytes moved and the resulting lower bound at a given peak. A measured number
   far above the bound means the kernels are still leaving something on the
   table; a number close to it means the next win must come from fewer FLOPs
   (quantization) or a better schedule (batching, prefix reuse).

The estimates are intentionally explicit and config-derived, not tuned magic
constants.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import D3Config

# Common device envelopes. Override on the command line for the actual part.
PEAKS = {
    "h100-sxm-bf16": (989.0, 3.35),
    "h100-sxm-fp8": (1979.0, 3.35),
    "h200-sxm-bf16": (989.0, 4.8),
    "b200-bf16": (2250.0, 8.0),
    "b200-fp8": (4500.0, 8.0),
    "rtx-6000-blackwell-bf16": (250.0, 1.79),
    "rtx-6000-blackwell-fp8": (500.0, 1.79),
    "a100-80g-bf16": (312.0, 2.0),
    "a100-80g-int8": (624.0, 2.0),
    "a100-40g-bf16": (312.0, 1.555),
    "l40-bf16": (181.0, 0.864),
    "l40-fp8": (362.0, 0.864),
    "l40s-bf16": (362.0, 0.864),
    "l40s-fp8": (733.0, 0.864),
}


@dataclass
class Params:
    total: int
    embedding: int
    per_layer_dense: int
    attention_per_layer: int
    gdn_per_layer: int
    mlp_per_layer: int
    full_layers: int
    linear_layers: int


def estimate_params(config: D3Config) -> Params:
    h = config.hidden_size
    inter = config.intermediate_size
    nh = config.num_heads
    nkv = config.num_kv_heads
    hd = config.head_dim
    full = len(config.full_attention_layers)
    linear = len(config.linear_attention_layers)

    attention = 2 * nh * hd * h + 2 * nkv * hd * h + nh * hd * h
    mlp = 3 * h * inter
    key_dim = config.num_k_heads * config.head_k_dim
    value_dim = config.num_v_heads * config.head_v_dim
    conv_dim = 2 * key_dim + value_dim
    gdn = (
        h * conv_dim
        + h * value_dim
        + 2 * h * config.num_v_heads
        + value_dim * h
        + conv_dim * config.conv_kernel_size
        + config.num_v_heads * 3
    )
    per_layer = attention + mlp + (gdn if linear else 0)
    embedding = config.vocab_size * h
    total = full * (attention + mlp) + linear * (mlp + gdn) + embedding + h
    return Params(
        total=total,
        embedding=embedding,
        per_layer_dense=mlp + attention + gdn,
        attention_per_layer=attention,
        gdn_per_layer=gdn,
        mlp_per_layer=mlp,
        full_layers=full,
        linear_layers=linear,
    )


def roofline(
    config: D3Config,
    *,
    state_tokens: int,
    question_tokens: int,
    questions: int,
    peak_tflops: float = 989.0,
    bandwidth_gbps: float = 3350.0,
    weight_dtype_bytes: int = 2,
) -> dict[str, Any]:
    """Analytic lower bound for one request with ``state_tokens`` reused."""
    params = estimate_params(config)
    s = int(state_tokens)
    q = int(question_tokens)
    n = int(questions)
    total_q = q * n
    hd = config.head_dim
    nh = config.num_heads

    dense_state = 2 * params.total * s
    dense_questions = 2 * params.total * total_q
    # Full attention: QK^T and AV, each 2 * heads * head_dim per query-key pair.
    full = max(1, params.full_layers)
    attention_state = full * 2 * 2 * nh * hd * s * s
    attention_questions = full * 2 * 2 * nh * hd * (s * total_q + q * q * n / 2)
    readout = 2 * config.max_options * config.hidden_size * n
    flops = dense_state + dense_questions + attention_state + attention_questions + readout

    weights_bytes = params.total * weight_dtype_bytes
    kv_bytes = s * config.kv_bytes_per_state_token
    gdn_bytes = config.gdn_state_bytes
    activation_bytes = total_q * config.hidden_size * weight_dtype_bytes * 4
    bytes_moved = weights_bytes + kv_bytes + gdn_bytes + activation_bytes

    compute_seconds = flops / (peak_tflops * 1e12)
    memory_seconds = bytes_moved / (bandwidth_gbps * 1e9)
    bound = max(compute_seconds, memory_seconds)
    return {
        "params": params.total,
        "params_billions": round(params.total / 1e9, 3),
        "state_tokens": s,
        "question_tokens": q,
        "questions": n,
        "flops": flops,
        "flops_tflops": round(flops / 1e12, 3),
        "dense_state_tflops": round(dense_state / 1e12, 3),
        "dense_question_tflops": round(dense_questions / 1e12, 3),
        "attention_state_tflops": round(attention_state / 1e12, 3),
        "attention_question_tflops": round(attention_questions / 1e12, 3),
        "readout_gflops": round(readout / 1e9, 3),
        "weights_gib": round(weights_bytes / 1024**3, 3),
        "state_kv_mib": round(kv_bytes / 1024**2, 3),
        "gdn_state_kib": round(gdn_bytes / 1024, 3),
        "bytes_moved_gib": round(bytes_moved / 1024**3, 3),
        "arithmetic_intensity": round(flops / bytes_moved, 3),
        "compute_seconds": compute_seconds,
        "memory_seconds": memory_seconds,
        "bound_seconds": bound,
        "bound_ms": round(bound * 1000, 3),
        "bottleneck": "compute" if compute_seconds >= memory_seconds else "memory",
        "peak_tflops": peak_tflops,
        "bandwidth_gbps": bandwidth_gbps,
        "state_cache_slots": config.state_cache_slots(),
        "layer_counts": {
            "full_attention": params.full_layers,
            "linear_attention": params.linear_layers,
        },
    }


# ---------------------------------------------------------------------------
# benchmark harness
# ---------------------------------------------------------------------------


def synthetic_cases(
    frontend: Any,
    *,
    state_tokens: int = 2048,
    questions: int = 8,
    options: int = 4,
) -> list[dict[str, Any]]:
    """A repeatable request shape with a long shared state and mixed questions."""
    sentence = (
        "The service handles requests from many regions, the database is replicated, "
        "and the on-call engineer is investigating a latency regression. "
    )
    state = ""
    # Tokenizers differ; aim for at least the requested count and let the
    # measured usage report the true number.
    while len(frontend.d3.tokenizer(state, add_special_tokens=False)["input_ids"]) < state_tokens:
        state += sentence
    qs: dict[str, Any] = {}
    for index in range(questions):
        if index % 4 == 3:
            qs[f"q{index}"] = {
                "type": "noul",
                "instructions": f"Is condition {index} present?",
                "criteria": {"true": "the condition is present", "false": "it is absent"},
            }
        else:
            criteria = {
                f"option{j}": f"the {j}th possible answer for question {index}"
                for j in range(options)
            }
            qs[f"q{index}"] = {
                "type": "choice",
                "instructions": f"Pick the best option for question {index}.",
                "criteria": criteria,
            }
    return [{"id": f"s{state_tokens}-q{questions}", "state": state, "questions": qs}]


def run_benchmark(
    engine: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    if not engine._started:
        engine.start()
    for _ in range(warmup):
        for case in cases:
            engine.system_one(
                state=case.get("state"),
                questions=case["questions"],
                images=case.get("images"),
                videos=case.get("videos"),
                assets=case.get("assets"),
            )

    samples: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for _ in range(repeats):
        for case in cases:
            started = time.perf_counter()
            response = engine.system_one(
                state=case.get("state"),
                questions=case["questions"],
                images=case.get("images"),
                videos=case.get("videos"),
                assets=case.get("assets"),
            )
            elapsed = time.perf_counter() - started
            timings = response.get("timings", {})
            samples.append(
                {
                    "case": case.get("id", ""),
                    "seconds": elapsed,
                    "input_tokens": response.get("usage", {}).get("input_tokens", 0),
                    "questions": len(response.get("answers", {})),
                    "prepare_ms": timings.get("prepare_ms"),
                    "backend_ms": timings.get("backend_ms"),
                    "respond_ms": timings.get("respond_ms"),
                }
            )
    wall = time.perf_counter() - total_started
    latencies = [s["seconds"] for s in samples]
    tokens = sum(s["input_tokens"] for s in samples)
    questions = sum(s["questions"] for s in samples)
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]
    payload: dict[str, Any] = {
        "backend": engine.backend_name,
        "requests": len(samples),
        "repeats": repeats,
        "warmup": warmup,
        "wall_seconds": round(wall, 4),
        "latency_ms": {
            "mean": round(1000 * statistics.fmean(latencies), 3),
            "p50": round(1000 * statistics.median(latencies), 3),
            "p95": round(1000 * p95, 3),
            "min": round(1000 * min(latencies), 3),
            "max": round(1000 * max(latencies), 3),
        },
        "input_tokens": tokens,
        "tokens_per_second": round(tokens / wall, 2) if wall else 0.0,
        "questions": questions,
        "questions_per_second": round(questions / wall, 2) if wall else 0.0,
        "samples": [
            {**s, "seconds": round(s["seconds"], 6)} for s in samples
        ],
        "info": engine.info(),
    }
    if hasattr(engine, "backend") and getattr(engine.backend, "cache", None) is not None:
        payload["state_cache"] = engine.backend.cache.stats()
    try:
        import torch

        if torch.cuda.is_available():
            payload["cuda_memory"] = {
                "allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            }
    except Exception:  # noqa: BLE001
        pass
    return payload


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
