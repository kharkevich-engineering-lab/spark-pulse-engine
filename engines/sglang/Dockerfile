# syntax=docker/dockerfile:1.7
#
# SGLang engine image for NVIDIA DGX Spark (GB10, aarch64, CUDA 13).
#
# The upstream lmsysorg/sglang cu130 image runs on GB10 without patches
# (verified solo and 2-node, see spark-pulse docs/native-runtime-plan.md 1.5).
# This stage adds what the control plane expects from every engine image:
# offline tiktoken files, earlyoom, provenance metadata, no entrypoint.

ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.10.post1-cu130

ARG ENGINE_VERSION=0.0.0
ARG ENGINE_VARIANT=default
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

FROM ${SGLANG_IMAGE} AS runner
ARG SGLANG_IMAGE
ARG ENGINE_VERSION
ARG ENGINE_VARIANT
ARG BUILD_DATE
ARG GIT_SHA

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends earlyoom curl wget libibverbs1 rdma-core && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/tiktoken_encodings && \
    wget -qO /workspace/tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -qO /workspace/tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

RUN set -eu; \
    { echo "engine: sglang"; \
      echo "variant: ${ENGINE_VARIANT}"; \
      echo "version: ${ENGINE_VERSION}"; \
      echo "build_date: ${BUILD_DATE}"; \
      echo "git_sha: ${GIT_SHA}"; \
      echo "base_image: ${SGLANG_IMAGE}"; \
      echo "sglang: $(python3 -c 'import importlib.metadata as m; print(m.version("sglang"))')"; \
      echo "torch: $(python3 -c 'import torch; print(torch.__version__)')"; \
    } > /workspace/build-metadata.yaml && cat /workspace/build-metadata.yaml

ENV TIKTOKEN_ENCODINGS_BASE=/workspace/tiktoken_encodings

LABEL org.opencontainers.image.title="spark-pulse-engine/sglang" \
      org.opencontainers.image.source="https://github.com/kharkevich-engineering-lab/spark-pulse-engine" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.spark-pulse.engine="sglang" \
      io.spark-pulse.variant="${ENGINE_VARIANT}"

WORKDIR /workspace
ENTRYPOINT []
CMD []
