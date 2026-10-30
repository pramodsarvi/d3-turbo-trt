#!/usr/bin/env bash
# Build the dependency image for NVIDIA GPUs and push it to Docker Hub.
#
#   DOCKERHUB_USER=myuser scripts/docker_push.sh [tag]
#
# Env: DOCKERHUB_USER (required), IMAGE (default d3-trt-deps), ARCHS (default "8.0;8.9", add 9.0 for H100),
#      WITH_FLASH_ATTN (1|0), DOCKERHUB_TOKEN (optional; else run `docker login` first)
# Mac note: the image is linux/amd64, so buildx cross-builds under emulation. flash-attn compile
# is very slow that way; use WITH_FLASH_ATTN=0 or build on an x86 box / CI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${DOCKERHUB_USER:?set DOCKERHUB_USER}"
TAG="${1:-cu130}"
IMAGE="${IMAGE:-d3-trt-deps}"
REF="$DOCKERHUB_USER/$IMAGE:$TAG"

if [[ -n "${DOCKERHUB_TOKEN:-}" ]]; then
  echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
fi

docker buildx build --platform linux/amd64 \
  --build-arg "TORCH_CUDA_ARCH_LIST=${ARCHS:-8.0;8.9}" \
  --build-arg "WITH_FLASH_ATTN=${WITH_FLASH_ATTN:-1}" \
  -t "$REF" --push .

echo "Pushed $REF"
echo "Run:  docker run --gpus all -it $REF bash"
