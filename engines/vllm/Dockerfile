# syntax=docker/dockerfile:1.7
#
# vLLM engine image for NVIDIA DGX Spark (GB10, aarch64, CUDA 13, sm_121).
#
# Stages
#   base                 CUDA devel + torch stack + NCCL built from source
#   flashinfer-builder   FlashInfer wheels
#   flashinfer-export    scratch image holding those wheels (for --output type=local)
#   vllm-builder         vLLM wheel with this repo's patch queue applied
#   vllm-export          scratch image holding that wheel
#   flashinfer_wheels    defaults to flashinfer-export; override with --build-context
#   vllm_wheels          defaults to vllm-export;       override with --build-context
#   runner               final image
#
# Every source is pinned through build args; the values that matter live in
# engine.yaml and are turned into --build-arg flags by scripts/build_args.py.
# Derived from eugr/spark-vllm-docker (MIT); see patches/NOTICE.

ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG BUILD_JOBS=4
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG TORCHVISION_VERSION=0.28.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG CUTLASS_DSL_VERSION=4.7.0
ARG TORCH_CUDA_ARCH_LIST=12.1a
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

ARG NCCL_REPO=https://github.com/NVIDIA/nccl.git
ARG NCCL_REF=master
ARG FLASHINFER_REPO=https://github.com/flashinfer-ai/flashinfer.git
ARG FLASHINFER_REF=main
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a

# Variant switches
ARG B12X_REPO=""
ARG B12X_REF=""
ARG VLLM_PRESERVE_SM12X_TARGET=0
ARG VLLM_PATCH_B12X_C128A_ALIGNMENT=0

# Provenance written into /workspace/build-metadata.yaml
ARG ENGINE_VERSION=0.0.0
ARG ENGINE_VARIANT=default
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

# =========================================================
# base
# =========================================================
FROM ${BASE_IMAGE} AS base
ARG BUILD_JOBS
ARG TORCH_VERSION
ARG TORCH_INDEX
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUTLASS_DSL_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG NCCL_NVCC_GENCODE
ARG NCCL_REPO
ARG NCCL_REF

ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=0 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    UV_PYTHON_DOWNLOADS=never \
    VLLM_BASE_DIR=/workspace/vllm \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    PATH=/usr/lib/ccache:$PATH \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=20G \
    CCACHE_COMPRESS=1 \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl cmake build-essential ninja-build \
      libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
      python3-dev python3-pip git wget \
      libibverbs1 libibverbs-dev rdma-core \
      ccache devscripts debhelper fakeroot && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    set -eu; \
    TORCHAUDIO_SPEC=""; \
    if [ "$TORCHAUDIO_VERSION" != "none" ]; then TORCHAUDIO_SPEC="torchaudio==$TORCHAUDIO_VERSION"; fi; \
    uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" $TORCHAUDIO_SPEC triton \
        --index-url "$TORCH_INDEX" && \
    uv pip install nvidia-nvshmem-cu13 \
        "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" \
        "apache-tvm-ffi==0.1.12" filelock packaging pynvml requests tqdm

WORKDIR ${VLLM_BASE_DIR}

# NCCL from source so the library carries sm_121 code and the Spark mesh fixes.
RUN set -eu; \
    git clone "$NCCL_REPO" nccl && cd nccl && git checkout --detach "$NCCL_REF" && \
    make -j "${BUILD_JOBS}" src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make pkg.debian.build && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./build/pkg/deb/*.deb && \
    git rev-parse HEAD > /workspace/.nccl-commit

# =========================================================
# flashinfer-builder
# =========================================================
FROM base AS flashinfer-builder
ARG FLASHINFER_REPO
ARG FLASHINFER_REF
ARG TORCH_CUDA_ARCH_LIST
# FlashInfer ships a `.python-version` pinning 3.10, and uv honours it ahead of
# whatever interpreter the image has — so `uv build` demanded a 3.10 that this
# CUDA base (3.12) does not carry and failed in under a tenth of a second. Every
# uv build below names the interpreter explicitly rather than letting the
# checkout choose one for us.
ENV FLASHINFER_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_PYTHON=python3

RUN set -eu; \
    git clone --recursive "$FLASHINFER_REPO" /workspace/flashinfer && \
    cd /workspace/flashinfer && git checkout --detach "$FLASHINFER_REF" && \
    git submodule update --init --recursive

WORKDIR /workspace/flashinfer
COPY patches/flashinfer_cache.patch /tmp/flashinfer_cache.patch

# flashinfer-cubin fetches ~42,700 cubins one HTTP request at a time from
# NVIDIA's artifactory, takes around 25 minutes, and fails the whole build if a
# single one of them fails — upstream collects the results and asserts
# `all(...)` with no retry. A CI run has already reached 42749/42749 and still
# raised "Failed to download cubins".
#
# This is what flashinfer_cache.patch is for. It makes the loop skip any cubin
# already on disk that verifies against its checksum, so a retry inside the same
# RUN — where the previous attempt's downloads still exist — re-fetches only the
# handful that failed instead of starting over.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    set -eu; \
    if git apply --reverse --check /tmp/flashinfer_cache.patch >/dev/null 2>&1; then \
        echo "flashinfer_cache.patch already present"; \
    else \
        git apply /tmp/flashinfer_cache.patch; \
    fi && \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --python "$UV_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels && \
    (cd flashinfer-cubin && for attempt in 1 2 3; do \
        uv build --python "$UV_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels && break; \
        if [ "$attempt" = 3 ]; then exit 1; fi; \
        echo "cubin download attempt $attempt failed; retrying, already-verified files are skipped"; \
    done) && \
    (cd flashinfer-jit-cache && uv build --python "$UV_PYTHON" --no-build-isolation --wheel . --out-dir=/workspace/wheels) && \
    git rev-parse HEAD > /workspace/wheels/.flashinfer-commit && \
    printf '%s\n' "$FLASHINFER_CUDA_ARCH_LIST" > /workspace/wheels/.flashinfer-arch

FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /

# =========================================================
# vllm-builder
# =========================================================
FROM base AS vllm-builder
ARG VLLM_REPO
ARG VLLM_REF
ARG DEEPGEMM_REPO
ARG DEEPGEMM_REF
ARG CUTLASS_DSL_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG VLLM_PRESERVE_SM12X_TARGET
ARG VLLM_PATCH_B12X_C128A_ALIGNMENT
ARG RUSTUP_TOOLCHAIN=stable
ENV RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:$PATH \
    PROTOC_INCLUDE=/usr/include \
    DEEPGEMM_SRC_DIR=/workspace/DeepGEMM

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates pkg-config protobuf-compiler libprotobuf-dev && \
    rm -rf /var/lib/apt/lists/* && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain ${RUSTUP_TOOLCHAIN} --no-modify-path

RUN set -eu; \
    git clone --recursive "$VLLM_REPO" "$VLLM_BASE_DIR/vllm" && \
    cd "$VLLM_BASE_DIR/vllm" && git checkout --detach "$VLLM_REF" && \
    git submodule update --init --recursive && \
    git clone --recursive "$DEEPGEMM_REPO" "$DEEPGEMM_SRC_DIR" && \
    cd "$DEEPGEMM_SRC_DIR" && git checkout --detach "$DEEPGEMM_REF" && \
    git submodule update --init --recursive

WORKDIR ${VLLM_BASE_DIR}/vllm
COPY patches/ /tmp/vllm-patches/

# Patch queue. Each script inspects the source and is a no-op when the fix is
# already upstream. *.patch files, if any, are applied with git apply first.
RUN set -eu; \
    git config --global user.email builder@spark-pulse.local; \
    git config --global user.name "spark-pulse-engine"; \
    for p in /tmp/vllm-patches/*.patch; do \
        [ -e "$p" ] || continue; case "$p" in *flashinfer_cache.patch) continue;; esac; \
        if git apply --reverse --check "$p" >/dev/null 2>&1; then echo "already applied: $p"; \
        else echo "applying: $p"; git apply --3way "$p"; fi; \
    done; \
    python3 /tmp/vllm-patches/patch_vllm_mrv2_speculator_cudagraph_pool.py . && \
    VLLM_PATCH_B12X_C128A_ALIGNMENT="$VLLM_PATCH_B12X_C128A_ALIGNMENT" \
        python3 /tmp/vllm-patches/patch_vllm_b12x_c128a_topk_alignment.py . && \
    python3 /tmp/vllm-patches/patch_vllm_flashinfer_b12x_swigluoai.py . && \
    python3 /tmp/vllm-patches/patch_vllm_topk_softplus_sqrt_control_flow.py . && \
    VLLM_PRESERVE_SM12X_TARGET="$VLLM_PRESERVE_SM12X_TARGET" \
        python3 /tmp/vllm-patches/patch_vllm_preserve_sm12x_target.py . && \
    python3 /tmp/vllm-patches/patch_vllm_diffusion_tensor_causal.py . && \
    python3 /tmp/vllm-patches/patch_vllm_gemma4_mtp_embedding_share.py . && \
    python3 /tmp/vllm-patches/patch_vllm_sm120_cooperative_topk.py . && \
    python3 /tmp/vllm-patches/patch_vllm_autogptq_symmetric_moe_qzeros.py . && \
    python3 /tmp/vllm-patches/patch_vllm_disable_minimax_qk_rmsnorm_ipc.py . && \
    python3 /tmp/vllm-patches/patch_vllm_routed_experts_weight_shape.py . && \
    python3 /tmp/vllm-patches/patch_vllm_spark_kv_cache_cleanup.py .

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 /tmp/vllm-patches/pin_cutlass_dsl.py "$CUTLASS_DSL_VERSION" --expected-count 1 requirements/cuda.txt && \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt "setuptools-rust>=1.9.0"

RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=cargo-registry,target=/opt/cargo/registry \
    --mount=type=cache,id=cargo-git,target=/opt/cargo/git \
    VLLM_REQUIRE_RUST_FRONTEND=1 CARGO_BUILD_JOBS=${MAX_JOBS} \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels && \
    git rev-parse HEAD > /workspace/wheels/.vllm-commit && \
    git -C "$DEEPGEMM_SRC_DIR" rev-parse HEAD > /workspace/wheels/.deepgemm-commit && \
    printf '%s\n' "$TORCH_CUDA_ARCH_LIST" > /workspace/wheels/.vllm-arch

FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# Overridable wheel sources. `docker build --build-context vllm_wheels=./wheels/vllm`
# replaces the stage below with a local directory and skips the builder entirely.
FROM flashinfer-export AS flashinfer_wheels
FROM vllm-export AS vllm_wheels

# =========================================================
# runner
# =========================================================
FROM ${BASE_IMAGE} AS runner
ARG BUILD_JOBS
ARG TORCH_VERSION
ARG TORCH_INDEX
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG CUTLASS_DSL_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG B12X_REPO
ARG B12X_REF
ARG ENGINE_VERSION
ARG ENGINE_VARIANT
ARG BUILD_DATE
ARG GIT_SHA
ARG VLLM_REF
ARG FLASHINFER_REF
ARG NCCL_REF
ARG BASE_IMAGE

ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=0 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    VLLM_BASE_DIR=/workspace/vllm

RUN --mount=type=bind,from=base,source=/workspace/vllm/nccl/build/pkg/deb,target=/workspace/nccl-pkg \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev curl git wget \
      libcudnn9-cuda-13 libibverbs1 libibverbs-dev rdma-core libxcb1 earlyoom && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages /workspace/nccl-pkg/*.deb && \
    rm -rf /var/lib/apt/lists/* && \
    pip install uv

WORKDIR ${VLLM_BASE_DIR}

RUN mkdir -p tiktoken_encodings && \
    wget -qO tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -qO tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    set -eu; \
    TORCHAUDIO_SPEC=""; \
    if [ "$TORCHAUDIO_VERSION" != "none" ]; then TORCHAUDIO_SPEC="torchaudio==$TORCHAUDIO_VERSION"; fi; \
    uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" $TORCHAUDIO_SPEC triton \
        --index-url "$TORCH_INDEX" && \
    uv pip install nvidia-nvshmem-cu13 "nvidia-cutlass-dsl[cu13]==$CUTLASS_DSL_VERSION" "apache-tvm-ffi==0.1.12"

# Pins go through --override so transitive deps can never swap the CUDA torch
# for a CPU wheel. fastapi <0.137 avoids a prometheus instrumentator break.
RUN --mount=type=bind,from=flashinfer_wheels,target=/workspace/flashinfer-wheels \
    --mount=type=bind,from=vllm_wheels,target=/workspace/vllm-wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    set -eu; \
    { echo "torch==$(python3 -c 'import torch; print(torch.__version__)')"; \
      echo "torchvision==$(python3 -c 'import importlib.metadata as m; print(m.version("torchvision"))')"; \
      python3 -c 'import importlib.metadata as m; print("torchaudio=="+m.version("torchaudio"))' 2>/dev/null || true; \
      echo "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}"; \
      echo "fastapi[standard]>=0.115.0,<0.137.0"; } > /tmp/override.txt && \
    uv pip install /workspace/flashinfer-wheels/*.whl /workspace/vllm-wheels/*.whl --override /tmp/override.txt && \
    uv pip install fastsafetensors instanttensor --override /tmp/override.txt && \
    for f in flashinfer-wheels/.flashinfer-commit vllm-wheels/.vllm-commit vllm-wheels/.deepgemm-commit; do \
        n=$(basename "$f"); if [ -f "/workspace/$f" ]; then cp "/workspace/$f" "/workspace/$n"; else echo prebuilt > "/workspace/$n"; fi; \
    done; \
    for d in flashinfer vllm; do if [ -f "/workspace/$d-wheels/manifest.json" ]; then cp "/workspace/$d-wheels/manifest.json" "/workspace/.$d-wheels-manifest.json"; fi; done

# Optional B12X kernel package (variant b12x). Installed without deps so the
# image-wide cutlass-dsl pin wins; kernels are JIT-compiled on first use.
COPY patches/pin_cutlass_dsl.py /tmp/pin_cutlass_dsl.py
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    set -eu; \
    if [ -n "$B12X_REPO" ]; then \
        git clone "$B12X_REPO" /tmp/b12x-source && git -C /tmp/b12x-source checkout --detach "$B12X_REF" && \
        python3 /tmp/pin_cutlass_dsl.py "$CUTLASS_DSL_VERSION" --expected-count 5 /tmp/b12x-source/pyproject.toml && \
        uv pip install --reinstall --no-deps /tmp/b12x-source && \
        git -C /tmp/b12x-source rev-parse HEAD > /workspace/.b12x-commit && \
        rm -rf /tmp/b12x-source; \
    fi

# Point the pip-shipped NCCL at the one we built.
RUN set -eu; \
    SP=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
    rm -f "$SP/nvidia/nccl/lib/libnccl.so.2" && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 "$SP/nvidia/nccl/lib/libnccl.so.2"

COPY --from=base /workspace/.nccl-commit /workspace/.nccl-commit

RUN set -eu; \
    { echo "engine: vllm"; \
      echo "variant: ${ENGINE_VARIANT}"; \
      echo "version: ${ENGINE_VERSION}"; \
      echo "build_date: ${BUILD_DATE}"; \
      echo "git_sha: ${GIT_SHA}"; \
      echo "base_image: ${BASE_IMAGE}"; \
      echo "gpu_arch: ${TORCH_CUDA_ARCH_LIST}"; \
      echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"; \
      echo "vllm: $(python3 -c 'import importlib.metadata as m; print(m.version("vllm"))')"; \
      echo "vllm_ref: ${VLLM_REF}"; \
      echo "vllm_commit: $(cat /workspace/.vllm-commit)"; \
      echo "flashinfer_ref: ${FLASHINFER_REF}"; \
      echo "flashinfer_commit: $(cat /workspace/.flashinfer-commit)"; \
      echo "deepgemm_commit: $(cat /workspace/.deepgemm-commit)"; \
      echo "nccl_ref: ${NCCL_REF}"; \
      echo "nccl_commit: $(cat /workspace/.nccl-commit)"; \
      if [ -f /workspace/.b12x-commit ]; then echo "b12x_commit: $(cat /workspace/.b12x-commit)"; fi; \
    } > /workspace/build-metadata.yaml && cat /workspace/build-metadata.yaml

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings \
    PATH=/workspace/vllm:$PATH

LABEL org.opencontainers.image.title="spark-pulse-engine/vllm" \
      org.opencontainers.image.source="https://github.com/kharkevich-engineering-lab/spark-pulse-engine" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.spark-pulse.engine="vllm" \
      io.spark-pulse.variant="${ENGINE_VARIANT}"

# No ENTRYPOINT/CMD on purpose: the control plane starts an idle container and
# execs the serve command.
