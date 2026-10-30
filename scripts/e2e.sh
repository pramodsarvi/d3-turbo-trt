#!/usr/bin/env bash
# End-to-end test in one go.
#
#   scripts/e2e.sh tiny              # CPU/Mac: venv, tiny checkpoint, pytest, HF parity, HTTP smoke
#   scripts/e2e.sh gpu [MODEL_DIR]   # GPU box: check, HF + TRT parity, HTTP smoke, bench, profile
#
# Env: BACKEND (hf|trtllm, default hf for tiny / trtllm for gpu), PORT (8000),
#      STATE_CACHE_GB (8), STATE_TOKENS (4096), QUESTIONS (16), HF_TOKEN
# Results land in traces/e2e-<mode>-<timestamp>/ ; exit code is non-zero if any step failed.
set -uo pipefail

MODE="${1:-tiny}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8000}"
OUT="traces/e2e-${MODE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
FAILED=()
# Empty-array expansion is unsafe under `set -u` on bash < 4.4 (macOS), so always guard it.

# Run a command (or shell function), tee its output to a log, record failures.
run() {
  local name="$1"; shift
  echo; echo "=== $name ==="
  "$@" 2>&1 | tee "$OUT/$name.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then FAILED+=("$name"); echo "--> FAILED ($rc)"; else echo "--> ok"; fi
}

# Start the server, hit it with smoke_http.py, always stop it. Runs in a pipeline subshell,
# so its own EXIT trap is what guarantees cleanup.
serve_and_smoke() { # serve_and_smoke <model> <backend> [extra serve args...]
  local model="$1" backend="$2"; shift 2
  d3-trt serve --model "$model" --backend "$backend" --port "$PORT" "$@" > "$OUT/server.log" 2>&1 &
  local pid=$!
  trap 'kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null' EXIT
  python scripts/smoke_http.py --base "http://127.0.0.1:$PORT" --wait "${SERVER_WAIT:-900}" --pid "$pid"
}

make_tiny() {
  python scripts/make_tiny_model.py --tokenizer-from reference --out .research/tiny/model
}

case "$MODE" in
tiny)
  BACKEND="${BACKEND:-hf}"
  MODEL=".research/tiny/model"
  if [[ ! -d .venv ]] || ! .venv/bin/python -c "import torch" 2>/dev/null; then
    run setup bash -c 'python3 -m venv .venv && . .venv/bin/activate && pip install -q -U pip && pip install -q -e ".[dev,server]"'
    [[ " ${FAILED[*]:-} " == *" setup "* ]] && { echo "setup failed"; exit 1; }
  fi
  # shellcheck disable=SC1091
  . .venv/bin/activate
  [[ -f "$MODEL/config.json" ]] || run make-tiny make_tiny
  export D3_TEST_MODEL="$MODEL"
  run pytest python -m pytest -q tests
  run check d3-trt check --model "$MODEL"
  run parity d3-trt parity --model "$MODEL" --backend "$BACKEND" --device cpu --cases tests/golden/text.json \
      --json-out "$OUT/parity.json"
  run http-smoke serve_and_smoke "$MODEL" "$BACKEND" --device cpu --dtype float32 --no-warmup
  run bench d3-trt bench --model "$MODEL" --backend "$BACKEND" --device cpu --dtype float32 --state-tokens 256 \
      --questions 4 --repeats 2 --json-out "$OUT/bench.json"
  ;;
gpu)
  BACKEND="${BACKEND:-trtllm}"
  MODEL="${2:-${MODEL:-./d3-turbo}}"
  command -v nvidia-smi >/dev/null && nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
  if [[ ! -f "$MODEL/config.json" ]]; then
    run download python -c "from huggingface_hub import snapshot_download as s; s('vllm-sr/d3-turbo', local_dir='$MODEL')"
  fi
  command -v d3-trt >/dev/null || run install pip install -e . --no-deps
  SC="${STATE_CACHE_GB:-8}"
  [[ -f .research/tiny/model/config.json ]] || run make-tiny make_tiny
  # the oracle is loaded first and freed before the candidate; REF_DEVICE=cpu if BF16 does not fit
  REF_ARGS=(); [[ -n "${REF_DEVICE:-}" ]] && REF_ARGS=(--reference-device "$REF_DEVICE")
  # candidate args: --quant is trtllm-only (the CLI rejects it for hf); QUANT=fp8|int8|none|auto
  CAND=(--device cuda:0 --state-cache-gb "$SC")
  [[ "$BACKEND" == "trtllm" && -n "${QUANT:-}" ]] && CAND+=(--quant "$QUANT")
  run pytest python -m pytest -q tests
  run check d3-trt check --model "$MODEL" --state-cache-gb "$SC"
  run parity-hf d3-trt parity --model "$MODEL" --backend hf --device cuda:0 --dtype bfloat16 \
      ${REF_ARGS[@]+"${REF_ARGS[@]}"} --cases tests/golden/text.json --json-out "$OUT/parity-hf.json"
  if [[ "$BACKEND" != "hf" ]]; then
    run parity-$BACKEND d3-trt parity --model "$MODEL" --backend "$BACKEND" "${CAND[@]}" \
        ${REF_ARGS[@]+"${REF_ARGS[@]}"} --cases tests/golden/text.json --json-out "$OUT/parity-$BACKEND.json"
  fi
  run http-smoke serve_and_smoke "$MODEL" "$BACKEND" "${CAND[@]}" --fallback
  run bench d3-trt bench --model "$MODEL" --backend "$BACKEND" "${CAND[@]}" \
      --state-tokens "${STATE_TOKENS:-4096}" --questions "${QUESTIONS:-16}" --repeats 5 --roofline \
      --json-out "$OUT/bench.json"
  run profile d3-trt profile --model "$MODEL" --backend "$BACKEND" "${CAND[@]}" --suite all \
      --state-tokens "${STATE_TOKENS:-4096}" --questions "${QUESTIONS:-16}" \
      --trace-out "$OUT/trace.json" --json-out "$OUT/profile.json"
  ;;
*)
  echo "usage: $0 tiny | gpu [MODEL_DIR]"; exit 2 ;;
esac

echo
echo "Results: $OUT"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED steps: ${FAILED[*]}"; exit 1
fi
echo "ALL STEPS PASSED"
