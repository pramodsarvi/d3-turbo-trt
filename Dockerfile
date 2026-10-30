# Dependency-only image for d3-trt on NVIDIA GPUs (A100/L40/H100). No project code and no
# model weights are baked in, so it is safe to publish.
#
#   docker buildx build --platform linux/amd64 -t <user>/d3-trt-deps:cu130 .
#   add --build-arg WITH_FLASH_ATTN=0 to skip the long flash-attn compile
#
# The torch build matches the one validated on the A100 box (2.14.1+cu130).
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_XET=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv git curl ca-certificates \
        openssh-server rsync tmux htop build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130

# Same dependency set as pyproject.toml (core + dev + server) plus the GPU kernels.
RUN python -m pip install \
        "transformers>=5.0" "safetensors>=0.4" "numpy>=1.24" \
        "pytest>=8" "accelerate>=1.0" \
        "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2" \
        "huggingface_hub>=1.0" pillow \
        "triton>=3.1" "flash-linear-attention>=0.3"

# causal-conv1d and flash-attn compile from source (no cu130 wheels expected). sm_80 covers
# A100, sm_89 covers L40/RTX 40xx; add 9.0 for H100 (each arch lengthens the build).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9"
ARG WITH_FLASH_ATTN=1
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN MAX_JOBS=4 python -m pip install causal-conv1d --no-build-isolation \
    && if [ "$WITH_FLASH_ATTN" = "1" ]; then \
         MAX_JOBS=4 python -m pip install "flash-attn>=2.7" --no-build-isolation; \
       fi

WORKDIR /workspace
# Usage on the box:  rsync the repo to /workspace/d3-turbo-trt, then
#   pip install -e . --no-deps
CMD ["sleep", "infinity"]
