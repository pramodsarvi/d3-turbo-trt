#!/usr/bin/env bash
# Run the GPU end-to-end test on a remote NVIDIA box from your Mac, inside the Docker image.
#
#   REMOTE=user@host IMAGE=yourname/d3-trt-deps:cu130 scripts/remote_gpu.sh
#
# Env: REMOTE (required), IMAGE (required), SSH_PORT (22), REMOTE_DIR (~/d3-trt),
#      HF_TOKEN (forwarded), BACKEND (hf|trtllm; default hf because the image has no TensorRT-LLM),
#      QUANT, REF_DEVICE, STATE_CACHE_GB, STATE_TOKENS, QUESTIONS (forwarded to e2e.sh)
# Steps: rsync repo (no weights) -> docker run --gpus all ... e2e.sh gpu -> rsync traces/ back.
# The model is cached in $REMOTE_DIR/models on the box, so only the first run downloads ~52 GB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${REMOTE:?set REMOTE=user@host}"
: "${IMAGE:?set IMAGE=<dockerhub-user>/d3-trt-deps:<tag>}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-d3-trt}"
SSH=(ssh -p "$SSH_PORT")

rsync -az --delete -e "ssh -p $SSH_PORT" \
  --exclude .venv --exclude .git --exclude .research --exclude traces --exclude __pycache__ \
  --exclude .pytest_cache --exclude '*.egg-info' \
  ./ "$REMOTE:$REMOTE_DIR/"

FWD=""
for v in HF_TOKEN QUANT REF_DEVICE STATE_CACHE_GB STATE_TOKENS QUESTIONS; do
  if [[ -n "${!v:-}" ]]; then FWD+=" -e $v=${!v}"; fi
done

rc=0
"${SSH[@]}" "$REMOTE" bash -s <<EOF || rc=$?
set -euo pipefail
cd $REMOTE_DIR
mkdir -p models traces
docker pull $IMAGE
docker run --rm --gpus all --ipc=host $FWD -e BACKEND=${BACKEND:-hf} \
  -v "\$PWD":/workspace/d3 -v "\$PWD/models":/models -w /workspace/d3 \
  $IMAGE bash -c 'pip install -q -e . --no-deps && scripts/e2e.sh gpu /models/d3-turbo'
EOF

mkdir -p traces
rsync -az -e "ssh -p $SSH_PORT" "$REMOTE:$REMOTE_DIR/traces/" traces/ || true
echo "Results copied to ./traces"
exit $rc
