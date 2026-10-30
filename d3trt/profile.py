"""Profiling harness: compare d3 optimisations on the same request shape.

The harness is built around *variants*: a small set of runtime switches that can
be applied to a loaded engine without reloading weights.

For the HF backend:

* ``reference``      the published ``D3.run`` state-once path
* ``core``           this package's ``D3Core`` path, no state cache
* ``core+cache``     ``D3Core`` + content-addressed state cache
* ``+sdpa`` / ``+flash`` / ``+triton``  full-attention kernel selection
* ``+gdn-recurrent`` short-question recurrent GDN instead of the chunk kernel
* ``+batch1/8/32``   question blocks per forward pass
* ``+compile``       ``torch.compile`` around each decoder layer

For the TensorRT-LLM backend the same switches apply to the TRT-LLM module
stack. Media variants are out of scope here (the HF reference path owns media).

Every row records wall latency, the state/question/readout split, state-cache
hits, the kernel that actually ran (not just the one requested), and peak GPU
memory. ``profile_layers`` gives a per-layer breakdown, ``trace`` exports a
Chrome/Perfetto trace, and ``nsys``/``ncu`` command hints are included for
kernel-level work.

On CPU the GPU-only variants are skipped unless ``allow_fallback=True``; when
they run, the report records the fallback kernel so a CPU number is never
mistaken for a GPU kernel measurement.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .bench import synthetic_cases
from .model.cache import StateCache

DEFAULT_PROFILE_CACHE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Variant:
    """One switch configuration that can be applied to a live engine."""

    name: str
    description: str = ""
    use_core: bool | None = None
    cache_bytes: int | None = None
    attention_state: str | None = None
    attention_question: str | None = None
    recurrent_max_tokens: int | None = None
    batch_size: int | None = None
    compile_layers: bool | str | None = None
    gpu_only: bool = False


def default_variants(
    engine: Any,
    *,
    cache_bytes: int = DEFAULT_PROFILE_CACHE_BYTES,
) -> list[Variant]:
    name = getattr(engine, "backend_name", "hf")
    if name == "hf":
        return [
            Variant(
                "reference",
                "published D3.run state-once; no cross-request cache",
                use_core=False,
                cache_bytes=0,
            ),
            Variant("core", "D3Core, no state cache", use_core=True, cache_bytes=0),
            Variant(
                "core+cache",
                "D3Core + content-addressed state cache",
                use_core=True,
                cache_bytes=cache_bytes,
            ),
            Variant(
                "core+cache+sdpa",
                "force PyTorch SDPA attention",
                use_core=True,
                cache_bytes=cache_bytes,
                attention_state="sdpa",
                attention_question="sdpa",
            ),
            Variant(
                "core+cache+flash",
                "FlashAttention for state and questions",
                use_core=True,
                cache_bytes=cache_bytes,
                attention_state="flash",
                attention_question="flash",
                gpu_only=True,
            ),
            Variant(
                "core+cache+triton",
                "FlashAttention state + shared-prefix Triton questions",
                use_core=True,
                cache_bytes=cache_bytes,
                attention_state="flash",
                attention_question="triton",
                gpu_only=True,
            ),
            Variant(
                "core+cache+gdn-chunk",
                "chunked GDN kernel for questions",
                use_core=True,
                cache_bytes=cache_bytes,
                recurrent_max_tokens=0,
            ),
            Variant(
                "core+cache+gdn-recurrent",
                "recurrent GDN kernel for question blocks <= 256 tokens",
                use_core=True,
                cache_bytes=cache_bytes,
                recurrent_max_tokens=256,
            ),
            Variant(
                "core+cache+batch1",
                "one question block per forward",
                use_core=True,
                cache_bytes=cache_bytes,
                batch_size=1,
            ),
            Variant(
                "core+cache+batch32",
                "up to 32 question blocks per forward",
                use_core=True,
                cache_bytes=cache_bytes,
                batch_size=32,
            ),
            Variant(
                "core+cache+compile",
                "torch.compile each decoder layer",
                use_core=True,
                cache_bytes=cache_bytes,
                compile_layers=True,
            ),
        ]
    if name in ("trtllm", "native"):
        modules = "TRT-LLM" if name == "trtllm" else "native PyTorch"
        return [
            Variant(name, f"{modules} modules, no state cache", cache_bytes=0),
            Variant(f"{name}+cache", f"{modules} modules + state cache", cache_bytes=cache_bytes),
            Variant(
                f"{name}+cache+flash",
                "FlashAttention state and questions",
                cache_bytes=cache_bytes,
                attention_state="flash",
                attention_question="flash",
                gpu_only=True,
            ),
            Variant(
                f"{name}+cache+triton",
                "FlashAttention state + shared-prefix Triton questions",
                cache_bytes=cache_bytes,
                attention_state="flash",
                attention_question="triton",
                gpu_only=True,
            ),
            Variant(
                f"{name}+cache+sdpa",
                "force PyTorch SDPA (reference comparison)",
                cache_bytes=cache_bytes,
                attention_state="sdpa",
                attention_question="sdpa",
            ),
            Variant(
                f"{name}+cache+gdn-recurrent",
                "recurrent GDN for short question blocks",
                cache_bytes=cache_bytes,
                recurrent_max_tokens=256,
            ),
            Variant(
                f"{name}+cache+batch1",
                "one question block per forward",
                cache_bytes=cache_bytes,
                batch_size=1,
            ),
            Variant(
                f"{name}+cache+batch32",
                "up to 32 question blocks per forward",
                cache_bytes=cache_bytes,
                batch_size=32,
            ),
            Variant(
                f"{name}+cache+compile",
                "torch.compile each decoder layer",
                cache_bytes=cache_bytes,
                compile_layers=True,
            ),
            Variant(
                f"{name}+cache+compile-state",
                "torch.compile the state pass only",
                cache_bytes=cache_bytes,
                compile_layers="state",
            ),
        ]
    raise ValueError(f"no profile variants for backend {name!r}")


def _core(engine: Any) -> Any:
    stack = getattr(engine.backend, "stack", None)
    return getattr(stack, "core", None)


def _capture_base(engine: Any) -> dict[str, Any]:
    """Snapshot the switches every variant starts from.

    Variants describe overrides, not a full state. Without a base, selecting
    ``batch1`` and then ``reference`` would silently keep batch size 1 on the
    reference path. The base is captured once, after the engine has started.
    """
    existing = getattr(engine, "_profile_base", None)
    if existing is not None:
        return existing
    backend = engine.backend
    core = _core(engine)
    base: dict[str, Any] = {
        "batch_size": int(engine.batch_size),
        "cache": getattr(backend, "cache", None),
    }
    if hasattr(backend, "use_core"):
        base["use_core"] = bool(backend.use_core)
    if core is not None:
        base["attention_state"] = core.attention_state
        base["attention_question"] = core.attention_question
        base["recurrent_max_tokens"] = int(core.recurrent_max_tokens)
        base["compile_layers"] = core.compile_layers
    engine._profile_base = base
    return base


def apply_variant(engine: Any, variant: Variant) -> None:
    """Apply a variant in place. Weights are untouched; only switches change."""
    if not engine._started:
        engine.start()
    base = _capture_base(engine)
    backend = engine.backend
    core = _core(engine)

    if "use_core" in base:
        backend.use_core = base["use_core"]
    if variant.use_core is not None and hasattr(backend, "use_core"):
        backend.use_core = variant.use_core

    if variant.cache_bytes is not None and hasattr(backend, "cache"):
        backend.cache = StateCache(variant.cache_bytes)
    else:
        backend.cache = base["cache"]

    batch = int(variant.batch_size) if variant.batch_size is not None else base["batch_size"]
    engine.batch_size = batch
    engine.frontend.d3.batch_size = batch

    if core is not None:
        core.attention_state = (
            variant.attention_state if variant.attention_state is not None else base["attention_state"]
        )
        core.attention_question = (
            variant.attention_question
            if variant.attention_question is not None
            else base["attention_question"]
        )
        core.recurrent_max_tokens = (
            int(variant.recurrent_max_tokens)
            if variant.recurrent_max_tokens is not None
            else base["recurrent_max_tokens"]
        )
        compile_layers = (
            variant.compile_layers
            if variant.compile_layers is not None
            else base["compile_layers"]
        )
        if compile_layers != core.compile_layers:
            core._compiled.clear()
        core.compile_layers = compile_layers


def select_variants(variants: Sequence[Variant], names: str | Sequence[str] | None) -> list[Variant]:
    if names is None or names == "" or names == "all":
        return list(variants)
    if isinstance(names, str):
        wanted = [n.strip() for n in names.split(",") if n.strip()]
    else:
        wanted = list(names)
    by_name = {v.name: v for v in variants}
    missing = [n for n in wanted if n not in by_name]
    if missing:
        raise ValueError(f"unknown variants: {missing}; choose from {sorted(by_name)}")
    return [by_name[n] for n in wanted]


# ---------------------------------------------------------------------------
# single variant
# ---------------------------------------------------------------------------


def _model_device(engine: Any) -> Any:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    stack = getattr(engine.backend, "stack", None)
    device = getattr(stack, "device", None)
    if device is None:
        device = getattr(getattr(engine.frontend, "d3", None), "device", None)
    return torch.device(device) if device is not None else None


def _sync(device: Any = None) -> None:
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.synchronize(device)
        elif (device is None or str(device).startswith("mps")) and getattr(
            torch.backends, "mps", None
        ) is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _call(engine: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    return engine.system_one(
        state=case.get("state"),
        questions=case["questions"],
        images=case.get("images"),
        videos=case.get("videos"),
        assets=case.get("assets"),
    )


def run_variant(
    engine: Any,
    variant: Variant,
    cases: Sequence[Mapping[str, Any]],
    *,
    repeats: int = 5,
    warmup: int = 1,
    cache_mode: str = "cold",
    allow_fallback: bool = False,
) -> dict[str, Any] | None:
    """Measure one variant. Returns None when a GPU-only variant is skipped."""
    if cache_mode not in ("cold", "warm"):
        raise ValueError(f"cache_mode must be 'cold' or 'warm', got {cache_mode!r}")
    if variant.gpu_only and not _cuda_available() and not allow_fallback:
        return None
    apply_variant(engine, variant)
    if not engine._started:
        engine.start()
    engine.profile_sync = True
    backend = engine.backend
    device = _model_device(engine)
    cache = getattr(backend, "cache", None)

    if cache_mode == "cold" and cache is not None:
        cache.clear()
    for _ in range(max(0, warmup)):
        for case in cases:
            _call(engine, case)
    _sync(device)

    core = _core(engine)
    if core is not None:
        core.reset_stats()
    if cache is not None and cache_mode == "cold":
        cache.clear()
    before = cache.stats() if cache is not None else None
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:  # noqa: BLE001
        pass

    samples: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    for _ in range(max(1, repeats)):
        for case in cases:
            if cache_mode == "cold" and cache is not None:
                cache.clear()
            _sync(device)
            t0 = time.perf_counter()
            response = _call(engine, case)
            _sync(device)
            t1 = time.perf_counter()
            timings = dict(response.get("timings", {}) or {})
            profile = dict(timings.get("backend_profile", {}) or {})
            samples.append(
                {
                    "seconds": t1 - t0,
                    "input_tokens": int(response.get("usage", {}).get("input_tokens", 0)),
                    "questions": len(response.get("answers", {})),
                    "prepare_ms": timings.get("prepare_ms"),
                    "backend_ms": timings.get("backend_ms"),
                    "respond_ms": timings.get("respond_ms"),
                    "state_ms": profile.get("state_ms"),
                    "question_ms": profile.get("question_ms"),
                    "readout_ms": profile.get("readout_ms"),
                    "cache_hit": profile.get("cache_hit"),
                    "batches": profile.get("batches"),
                    "state_tokens": profile.get("state_tokens"),
                    "question_tokens": profile.get("question_tokens"),
                }
            )
    wall = time.perf_counter() - wall_started
    after = cache.stats() if cache is not None else None

    latencies = [s["seconds"] for s in samples]
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered) + 0.5)) - 1))
    tokens = sum(s["input_tokens"] for s in samples)
    questions = sum(s["questions"] for s in samples)
    kernel = None
    core_stats = None
    if core is not None and getattr(engine.backend, "use_core", True):
        from dataclasses import asdict

        core_stats = asdict(core.stats)
        kernel = {
            "attention_state": core.stats.attention_state,
            "attention_question": core.stats.attention_question,
            "state_layers": core.stats.state_layers,
            "question_layers": core.stats.question_layers,
            "delta": core.stats.delta,
        }
    row: dict[str, Any] = {
        "variant": variant.name,
        "description": variant.description,
        "cache_mode": cache_mode,
        "requests": len(samples),
        "latency_ms": {
            "mean": round(1000 * statistics.fmean(latencies), 3),
            "p50": round(1000 * statistics.median(latencies), 3),
            "p95": round(1000 * ordered[p95_index], 3),
            "min": round(1000 * min(latencies), 3),
            "max": round(1000 * max(latencies), 3),
        },
        "phase_ms": {
            "prepare": _mean(samples, "prepare_ms"),
            "backend": _mean(samples, "backend_ms"),
            "state": _mean(samples, "state_ms"),
            "questions": _mean(samples, "question_ms"),
            "readout": _mean(samples, "readout_ms"),
            "respond": _mean(samples, "respond_ms"),
        },
        "cache_hit_rate": _cache_delta_rate(before, after),
        "cache_hits": (after["hits"] - before["hits"]) if before and after else 0,
        "cache_misses": (after["misses"] - before["misses"]) if before and after else 0,
        "input_tokens": tokens,
        "tokens_per_second": round(tokens / wall, 2) if wall else 0.0,
        "questions_per_second": round(questions / wall, 2) if wall else 0.0,
        "kernel": kernel,
        "core": core_stats,
        "samples": [
            {
                **{
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in sample.items()
                }
            }
            for sample in samples
        ],
        "fallback_on_cpu": bool(variant.gpu_only and not _cuda_available()),
    }
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            row["cuda_memory"] = {
                "allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
                "reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 3),
            }
    except Exception:  # noqa: BLE001
        pass
    return row


def _mean(samples: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [s.get(key) for s in samples if s.get(key) is not None]
    return round(statistics.fmean(values), 3) if values else None


def _cache_delta_rate(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> float | None:
    if not before or not after:
        return None
    hits = after["hits"] - before["hits"]
    misses = after["misses"] - before["misses"]
    total = hits + misses
    return round(hits / total, 4) if total else None


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------


def profile_variants(
    engine: Any,
    cases: Sequence[Mapping[str, Any]],
    *,
    variants: Sequence[Variant] | None = None,
    names: str | Sequence[str] | None = None,
    repeats: int = 5,
    warmup: int = 1,
    cache_modes: Sequence[str] = ("cold", "warm"),
    allow_fallback: bool = False,
    cache_bytes: int = DEFAULT_PROFILE_CACHE_BYTES,
) -> dict[str, Any]:
    choices = list(variants) if variants is not None else default_variants(engine, cache_bytes=cache_bytes)
    selected = select_variants(choices, names)
    modes = [str(mode) for mode in cache_modes]
    bad = [mode for mode in modes if mode not in ("cold", "warm")]
    if bad:
        raise ValueError(f"cache_modes must be cold and/or warm, got {bad}")
    previous_sync = getattr(engine, "profile_sync", False)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for variant in selected:
        if variant.gpu_only and not _cuda_available() and not allow_fallback:
            skipped.append(variant.name)
            continue
        expected_modes = (
            modes
            if variant.cache_bytes and variant.cache_bytes > 0 and variant.use_core is not False
            else ["cold"]
        )
        for mode in expected_modes:
            row = run_variant(
                engine,
                variant,
                cases,
                repeats=repeats,
                warmup=warmup,
                cache_mode=mode,
                allow_fallback=allow_fallback,
            )
            if row is not None:
                rows.append(row)
    engine.profile_sync = previous_sync
    return {
        "backend": engine.backend_name,
        "cases": [c.get("id", "") for c in cases],
        "repeats": repeats,
        "warmup": warmup,
        "rows": rows,
        "skipped": skipped,
        "hints": profiling_hints(engine),
        "config": engine.config.describe(),
    }


def format_table(report: Mapping[str, Any]) -> str:
    headers = ["variant", "mode", "p50 ms", "p95 ms", "state", "quest", "readout", "cache hit", "tok/s", "kernel"]
    lines = ["  ".join(f"{h:>12}" for h in headers)]
    for row in report.get("rows", []):
        phase = row.get("phase_ms") or {}
        kernel = row.get("kernel") or {}
        values = [
            row["variant"],
            row["cache_mode"],
            _fmt(row["latency_ms"]["p50"]),
            _fmt(row["latency_ms"]["p95"]),
            _fmt(phase.get("state")),
            _fmt(phase.get("questions")),
            _fmt(phase.get("readout")),
            _fmt(row.get("cache_hit_rate")),
            _fmt(row.get("tokens_per_second")),
            _kernel_label(row),
        ]
        lines.append("  ".join(f"{str(v):>12}" for v in values))
    if report.get("skipped"):
        lines.append(f"\nskipped (GPU-only): {', '.join(report['skipped'])}")
    return "\n".join(lines)


def _kernel_label(row: Mapping[str, Any]) -> str:
    kernel = row.get("kernel") or {}
    label = kernel.get("attention_question") or "-"
    if row.get("cache_mode") == "warm" and kernel.get("state_layers") == 0:
        label = f"{label} (state cached)"
    return label


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}" if value >= 10 else f"{value:.3f}"
    return str(value)


# ---------------------------------------------------------------------------
# per-layer profile
# ---------------------------------------------------------------------------


def profile_layers(
    engine: Any,
    case: Mapping[str, Any],
    *,
    variant: Variant | None = None,
) -> dict[str, Any]:
    """Per-layer wall time for one request (state + question passes combined).

    The D3Core path times its own layer calls, so the numbers cover exactly the
    module work the fast path performs. The published reference path is timed
    with hooks on ``input_layernorm`` -> ``mlp`` inside each decoder layer.
    GPU timings use CUDA events and are synchronised once at the end; CPU
    timings use ``perf_counter``.
    """
    if variant is not None:
        apply_variant(engine, variant)
    if not engine._started:
        engine.start()
    engine.profile_sync = True
    device = _model_device(engine)
    _call(engine, case)  # warm kernels
    _sync(device)
    cache = getattr(engine.backend, "cache", None)
    if cache is not None:
        cache.clear()  # make sure the measured request includes the state pass

    core = _core(engine)
    use_core = bool(getattr(engine.backend, "use_core", True))
    if core is not None and use_core:
        core.begin_layer_profile()
        started = time.perf_counter()
        try:
            _call(engine, case)
            _sync(device)
        finally:
            result = core.end_layer_profile()
        total_ms = (time.perf_counter() - started) * 1000
        return {
            "total_ms": round(total_ms, 3),
            "layers": result["layers"],
            "raw": result["raw"],
            "gpu": result["gpu"],
            "source": "core",
            "variant": variant.name if variant else None,
        }

    # Reference path: the decoder layer's forward is input_layernorm -> attn ->
    # residual -> post_attention_layernorm -> mlp. Hooking the first and last
    # submodule covers the layer without depending on private signatures.
    layers = _layers(engine)
    cuda = device is not None and device.type == "cuda"
    spans: list[list[tuple[Any, Any, float]]] = [[] for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        first = getattr(layer, "input_layernorm", None)
        last = getattr(layer, "mlp", None)
        if first is None or last is None:
            continue

        def pre_hook(module, args, index=index):
            if cuda:
                import torch

                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(device))
                spans[index].append((event, None, 0.0))
            else:
                spans[index].append((None, None, time.perf_counter()))

        def post_hook(module, args, output, index=index):
            if cuda:
                import torch

                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(device))
                start, _, _ = spans[index][-1]
                spans[index][-1] = (start, event, 0.0)
            else:
                start, _, started_cpu = spans[index][-1]
                spans[index][-1] = (start, None, time.perf_counter() - started_cpu)

        handles.append(first.register_forward_pre_hook(pre_hook))
        handles.append(last.register_forward_hook(post_hook))

    started = time.perf_counter()
    try:
        _call(engine, case)
        _sync(device)
    finally:
        for handle in handles:
            handle.remove()
    total_ms = (time.perf_counter() - started) * 1000

    rows = []
    for index, spans_for_layer in enumerate(spans):
        elapsed_ms = 0.0
        for start_event, end_event, cpu_seconds in spans_for_layer:
            if start_event is not None and end_event is not None:
                elapsed_ms += float(start_event.elapsed_time(end_event))
            else:
                elapsed_ms += cpu_seconds * 1000
        spec = engine.config.layer_specs[index] if index < len(engine.config.layer_specs) else None
        rows.append(
            {
                "index": index,
                "kind": spec.kind if spec else "",
                "calls": len(spans_for_layer),
                "total_ms": round(elapsed_ms, 3),
                "state_ms": None,
                "question_ms": None,
            }
        )
    return {
        "total_ms": round(total_ms, 3),
        "layers": rows,
        "raw": [],
        "gpu": cuda,
        "source": "hooks",
        "variant": variant.name if variant else None,
    }


def _layers(engine: Any) -> list[Any]:
    core = _core(engine)
    if core is not None:
        return list(core.layers)
    backbone = getattr(engine.frontend.d3, "backbone", None)
    if backbone is not None:
        return list(backbone.language_model.layers)
    raise RuntimeError("no loaded layers to profile")


# ---------------------------------------------------------------------------
# traces and external profiler hints
# ---------------------------------------------------------------------------


def trace(
    engine: Any,
    case: Mapping[str, Any],
    path: str | Path,
    *,
    variant: Variant | None = None,
) -> dict[str, Any]:
    """Export a Chrome/Perfetto trace for one request."""
    import torch

    if variant is not None:
        apply_variant(engine, variant)
    if not engine._started:
        engine.start()
    engine.profile_sync = True
    device = _model_device(engine)
    _call(engine, case)
    _sync(device)
    cache = getattr(engine.backend, "cache", None)
    if cache is not None:
        cache.clear()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if _cuda_available() and device is not None and device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        _call(engine, case)
        _sync(device)
    prof.export_chrome_trace(str(target))
    sort_by = (
        "cuda_time_total"
        if torch.profiler.ProfilerActivity.CUDA in activities
        else "cpu_time_total"
    )
    return {
        "trace": str(target),
        "table": prof.key_averages().table(sort_by=sort_by, row_limit=30),
        "activities": [str(a) for a in activities],
    }


def profiling_hints(engine: Any) -> dict[str, str]:
    model = str(engine.config.root)
    backend = getattr(engine, "backend_name", "hf")
    return {
        "torch_trace": "d3-trt profile --suite trace --trace-out traces/d3.json",
        "nsys": (
            "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none "
            f"-o traces/d3-nsys python -m d3trt.cli bench --model {model} --backend {backend} "
            "--state-tokens 4096 --questions 16 --repeats 1"
        ),
        "ncu": (
            "ncu --set full --target-processes all -o traces/d3-ncu "
            f"python -m d3trt.cli bench --model {model} --backend {backend} "
            "--state-tokens 1024 --questions 1 --repeats 1"
        ),
    }


# ---------------------------------------------------------------------------
# convenience
# ---------------------------------------------------------------------------


def profile_request_shape(
    engine: Any,
    *,
    state_tokens: int = 2048,
    questions: int = 8,
    options: int = 4,
    repeats: int = 5,
    warmup: int = 1,
    variants: Sequence[Variant] | None = None,
    names: str | Sequence[str] | None = None,
    cache_modes: Sequence[str] = ("cold", "warm"),
    allow_fallback: bool = False,
) -> dict[str, Any]:
    cases = synthetic_cases(
        engine.frontend,
        state_tokens=state_tokens,
        questions=questions,
        options=options,
    )
    return profile_variants(
        engine,
        cases,
        variants=variants,
        names=names,
        repeats=repeats,
        warmup=warmup,
        cache_modes=cache_modes,
        allow_fallback=allow_fallback,
    )
