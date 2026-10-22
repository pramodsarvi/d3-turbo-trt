"""Command line entry point: ``d3-trt``.

Examples::

    d3-trt check --model ./d3-turbo
    d3-trt serve --model ./d3-turbo --backend trtllm --device cuda:0
    d3-trt parity --model ./d3-turbo --backend trtllm --cases tests/golden/text.json
    d3-trt bench  --model ./d3-turbo --backend trtllm --state-tokens 4096 --questions 16
    d3-trt profile --model ./d3-turbo --backend trtllm --suite all --trace-out traces/d3.json
    d3-trt roofline --model ./d3-turbo --state-tokens 4096 --questions 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .bench import PEAKS


def _gb(value: str | None) -> int | None:
    if value is None:
        return None
    return int(float(value) * 1024**3)


def _backend_options(args: argparse.Namespace) -> dict[str, Any] | None:
    options: dict[str, Any] = {}
    if getattr(args, "recurrent_max_tokens", None) is not None:
        options["recurrent_max_tokens"] = args.recurrent_max_tokens
    if getattr(args, "tp_size", None) is not None:
        options["tp_size"] = args.tp_size
    if getattr(args, "quant", None) is not None:
        from . import backends

        name = backends.detect() if args.backend == "auto" else args.backend
        if name != "trtllm":
            raise SystemExit(f"--quant applies to the trtllm backend only (backend is {name})")
        options["quant"] = args.quant
    return options or None


def _engine(args: argparse.Namespace):
    from .engine import DecisionEngine

    return DecisionEngine(
        args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        reference_dir=args.reference_dir,
        state_cache_bytes=_gb(getattr(args, "state_cache_gb", None)),
        fallback=getattr(args, "fallback", False),
        fallback_device=getattr(args, "fallback_device", "cpu"),
        backend_options=_backend_options(args),
    )


def _add_common(parser: argparse.ArgumentParser, *, model_required: bool = True) -> None:
    parser.add_argument("--model", required=model_required, help="d3 package directory")
    parser.add_argument("--reference-dir", help="directory holding d3_runtime.py if not in the package")
    parser.add_argument("--backend", default="auto", choices=["auto", "hf", "trtllm"])
    parser.add_argument("--device", help="torch device (default: cuda:0 if available, else cpu)")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch-size", type=int, default=8, help="question blocks per forward pass")
    parser.add_argument(
        "--quant",
        choices=["auto", "none", "fp8", "int8"],
        help="trtllm weight format: fp8 (sm_89+: L40, RTX Pro 6000), int8 (A100), "
        "none (BF16, needs 80 GB), auto (BF16 if it fits, else the best 8-bit; default)",
    )


def cmd_check(args: argparse.Namespace) -> int:
    from . import backends
    from .config import load_config
    from .model.trtllm_stack import trtllm_available

    config = load_config(args.model, state_cache_bytes=_gb(args.state_cache_gb))
    ok, reason = trtllm_available()
    payload = {
        "config": config.describe(),
        "selected_backend": backends.detect(),
        "backends": list(backends.available()),
        "trtllm": {"available": ok, "reason": reason},
        "decision": {
            "attention_mode": config.attention_mode,
            "prompt": config.prompt,
            "max_options": config.max_options,
            "validated_length": config.validated_length,
            "max_length": config.context_limit,
            "temperature": config.temperature,
            "image_placements": config.image_placements,
            "question_types": config.question_types,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    engine = _engine(args)
    engine.start()
    serve(
        engine,
        args.host,
        args.port,
        api_key=os.environ.get("DECISION_API_KEY") or None,
        warmup=not args.no_warmup,
    )
    return 0


def cmd_parity(args: argparse.Namespace) -> int:
    from .parity import load_cases, run_parity

    cases = load_cases(args.cases)
    report = run_parity(
        args.model,
        cases,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        reference_dir=args.reference_dir,
        batch_size=args.batch_size,
        tolerance=args.tolerance,
        reference_verify=args.reference_verify,
        backend_options=_backend_options(args),
        reference_device=args.reference_device,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    summary = payload["summary"]
    print(
        f"\nparity: max_abs_diff={summary['max_abs_diff']:.6g} "
        f"choice_agreement={summary['choice_agreement']:.4f} "
        f"passed={summary['passed']}",
        file=sys.stderr,
    )
    return 0 if summary["passed"] else 1


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import PEAKS, roofline, run_benchmark, synthetic_cases, write_report

    engine = _engine(args)
    engine.start()
    cases = synthetic_cases(
        engine.frontend,
        state_tokens=args.state_tokens,
        questions=args.questions,
        options=args.options,
    )
    report = run_benchmark(engine, cases, repeats=args.repeats, warmup=args.warmup)
    if args.roofline:
        peak, bandwidth = PEAKS.get(args.device_profile, (args.peak_tflops, args.bandwidth_gbps))
        report["roofline"] = roofline(
            engine.config,
            state_tokens=args.state_tokens,
            question_tokens=args.question_tokens,
            questions=args.questions,
            peak_tflops=peak,
            bandwidth_gbps=bandwidth,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        write_report(report, args.json_out)
    engine.close()
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from .bench import synthetic_cases
    from .profile import (
        DEFAULT_PROFILE_CACHE_BYTES,
        default_variants,
        format_table,
        profile_layers,
        profile_variants,
        select_variants,
        trace,
    )

    engine = _engine(args)
    engine.start()
    cache_bytes = _gb(args.state_cache_gb) or DEFAULT_PROFILE_CACHE_BYTES
    cases = synthetic_cases(
        engine.frontend,
        state_tokens=args.state_tokens,
        questions=args.questions,
        options=args.options,
    )
    variants = default_variants(engine, cache_bytes=cache_bytes)
    selected = select_variants(variants, args.variants)
    payload: dict[str, Any] = {
        "state_tokens": args.state_tokens,
        "questions": args.questions,
        "options": args.options,
    }
    if args.suite in ("variants", "all"):
        report = profile_variants(
            engine,
            cases,
            variants=selected,
            repeats=args.repeats,
            warmup=args.warmup,
            cache_modes=[m.strip() for m in args.cache_modes.split(",") if m.strip()],
            allow_fallback=args.allow_fallback,
        )
        payload["variants"] = report
        print(format_table(report), file=sys.stderr)

    if args.suite in ("layers", "all"):
        chosen = _pick_variant(selected, args.layer_variant, engine.backend_name)
        if chosen is None:
            print("layer profile: no variant selected", file=sys.stderr)
        else:
            layer_report = profile_layers(engine, cases[0], variant=chosen)
            payload["layers"] = layer_report
            if args.layers_out:
                Path(args.layers_out).write_text(json.dumps(layer_report, indent=2, sort_keys=True))
            print(
                f"layers: total={layer_report['total_ms']:.1f} ms "
                f"({len(layer_report['layers'])} layers, variant={chosen.name})",
                file=sys.stderr,
            )

    if args.suite in ("trace", "all"):
        chosen = _pick_variant(selected, args.trace_variant, engine.backend_name)
        if chosen is None:
            print("trace: no variant selected", file=sys.stderr)
        else:
            trace_path = args.trace_out or "traces/d3-profile.json"
            trace_report = trace(engine, cases[0], trace_path, variant=chosen)
            payload["trace"] = trace_report
            print(trace_report["table"], file=sys.stderr)
            print(f"trace: {trace_report['trace']}", file=sys.stderr)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    engine.close()
    return 0


def _pick_variant(variants: list[Any], name: str | None, backend: str) -> Any:
    if name:
        for variant in variants:
            if variant.name == name:
                return variant
        raise SystemExit(f"unknown variant {name!r}")
    preferred = "trtllm+cache" if backend == "trtllm" else "core+cache"
    for variant in variants:
        if variant.name == preferred:
            return variant
    return variants[0] if variants else None


def cmd_roofline(args: argparse.Namespace) -> int:
    from .bench import PEAKS, roofline
    from .config import load_config

    config = load_config(args.model)
    peak, bandwidth = PEAKS.get(args.device_profile, (args.peak_tflops, args.bandwidth_gbps))
    payload = roofline(
        config,
        state_tokens=args.state_tokens,
        question_tokens=args.question_tokens,
        questions=args.questions,
        peak_tflops=peak,
        bandwidth_gbps=bandwidth,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_state_key(args: argparse.Namespace) -> int:
    from .config import load_config

    config = load_config(args.model)
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception:  # noqa: BLE001
        pass
    ids = tokenizer(args.text, add_special_tokens=False)["input_ids"] if tokenizer else []
    print(json.dumps({"tokens": len(ids), "state_key": config.state_key(ids)}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="d3-trt", description=__doc__)
    parser.add_argument("--version", action="version", version="d3-trt 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="parse a package and report backend availability")
    check.add_argument("--model", required=True)
    check.add_argument("--state-cache-gb", type=str)
    check.set_defaults(func=cmd_check)

    serve = sub.add_parser("serve", help="serve the reference System One HTTP API")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--state-cache-gb", type=str)
    serve.add_argument("--fallback", action="store_true", help="lazy-load HF for media/composed requests")
    serve.add_argument("--fallback-device", default="cpu")
    serve.add_argument("--no-warmup", action="store_true")
    serve.set_defaults(func=cmd_serve)

    parity = sub.add_parser("parity", help="compare a backend against the reference runtime")
    _add_common(parity)
    parity.add_argument("--cases", required=True, help="JSON list of cases")
    parity.add_argument("--tolerance", type=float, default=1e-2)
    parity.add_argument("--reference-verify", default="none", choices=["none", "fast", "full"])
    parity.add_argument(
        "--reference-device",
        help="device for the BF16 oracle (default: --device); use cpu when the BF16 weights do not fit the GPU",
    )
    parity.add_argument("--json-out")
    parity.set_defaults(func=cmd_parity)

    bench = sub.add_parser("bench", help="benchmark a request shape")
    _add_common(bench)
    bench.add_argument("--state-tokens", type=int, default=2048)
    bench.add_argument("--question-tokens", type=int, default=96)
    bench.add_argument("--questions", type=int, default=8)
    bench.add_argument("--options", type=int, default=4)
    bench.add_argument("--repeats", type=int, default=3)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--state-cache-gb", type=str)
    bench.add_argument("--roofline", action="store_true")
    bench.add_argument("--device-profile", default="h100-sxm-bf16", choices=sorted(PEAKS))
    bench.add_argument("--peak-tflops", type=float, default=989.0)
    bench.add_argument("--bandwidth-gbps", type=float, default=3350.0)
    bench.add_argument("--json-out")
    bench.add_argument("--recurrent-max-tokens", type=int)
    bench.add_argument("--tp-size", type=int)
    bench.set_defaults(func=cmd_bench)

    prof = sub.add_parser(
        "profile",
        help="compare optimisations, dump per-layer timings and export a Chrome trace",
    )
    _add_common(prof)
    prof.add_argument("--suite", default="variants", choices=["variants", "layers", "trace", "all"])
    prof.add_argument("--variants", default="all", help="comma-separated variant names, or all")
    prof.add_argument("--cache-modes", default="cold,warm", help="cold, warm, or both")
    prof.add_argument("--state-tokens", type=int, default=2048)
    prof.add_argument("--questions", type=int, default=8)
    prof.add_argument("--options", type=int, default=4)
    prof.add_argument("--repeats", type=int, default=5)
    prof.add_argument("--warmup", type=int, default=1)
    prof.add_argument("--state-cache-gb", type=str)
    prof.add_argument("--allow-fallback", action="store_true", help="run GPU-only variants with CPU fallback kernels")
    prof.add_argument("--layer-variant", help="variant for the per-layer pass")
    prof.add_argument("--trace-variant", help="variant for the Chrome trace")
    prof.add_argument("--layers-out", help="write the per-layer JSON here")
    prof.add_argument("--trace-out", help="write the Chrome/Perfetto trace here")
    prof.add_argument("--json-out")
    prof.add_argument("--recurrent-max-tokens", type=int)
    prof.add_argument("--tp-size", type=int)
    prof.set_defaults(func=cmd_profile)

    roof = sub.add_parser("roofline", help="analytic lower bound without loading weights")
    roof.add_argument("--model", required=True)
    roof.add_argument("--state-tokens", type=int, default=4096)
    roof.add_argument("--question-tokens", type=int, default=96)
    roof.add_argument("--questions", type=int, default=8)
    roof.add_argument("--device-profile", default="h100-sxm-bf16", choices=sorted(PEAKS))
    roof.add_argument("--peak-tflops", type=float, default=989.0)
    roof.add_argument("--bandwidth-gbps", type=float, default=3350.0)
    roof.set_defaults(func=cmd_roofline)

    key = sub.add_parser("state-key", help="content hash for a state string")
    key.add_argument("--model", required=True)
    key.add_argument("--text", required=True)
    key.set_defaults(func=cmd_state_key)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
