# syntax=docker/dockerfile:1.7
#
# TensorFold engine image for NVIDIA DGX Spark (GB10, aarch64, CUDA 13).
#
# Neither built from source nor wrapped: TensorFold publishes a native release
# archive per platform, and the Linux aarch64 one is the engine whole -- one
# Zig executable, `tensorfold-native`, with the CUDA kernels embedded at build
# time (RUNTIME.md: "CUDA `.cu` fatbins are embedded at build time from the
# qualified directory supplied to the release build"), plus `libtfgrammar` for
# structured output. Rebuilding it here would mean Zig, a qualified kernel
# directory and the release owner's gates, to arrive at the bytes the release
# already ships, so the archive is fetched at the pinned tag and checked
# against the sha256 the release publishes beside it.
#
# What the archive deliberately does not carry is everything a driver-only
# host lacks, and that is what this image adds (RUNTIME.md, "Optional shared
# libraries"):
#   - NCCL, for tensor-parallel ranks. The engine refuses anything older than
#     2.30.0 before rank setup (CHANGELOG 1.1.2, #1060), and the CUDA 13.0
#     runtime image's own libnccl2 is 2.28.3, so NVIDIA's nvidia-nccl-cu13
#     wheel is unpacked into /opt/nccl and put first on LD_LIBRARY_PATH. The
#     build arg is NCCL_WHEEL_VERSION, not NCCL_VERSION: the base image sets
#     ENV NCCL_VERSION=2.28.3-1, and an inherited ENV beats an ARG of the
#     same name inside RUN.
#   - cuBLASLt 13, for --vision. The CUDA runtime image already has it.
#   - libibverbs and the mlx5 provider, for NCCL's IB transport over the
#     ConnectX RoCE ports and TensorFold's own RoCE gathers. Without them NCCL
#     drops to TCP sockets at INFO level, which is the slow path nobody sees.
# And `procps`, because the control plane watches the serve process with
# `pgrep -f`; an image without it reads as "could not ask" for the whole
# readiness window.
#
# JPEG and WebP decoding need pinned libjpeg-turbo 3.1.4.1 and libwebp 1.6.0,
# which Ubuntu does not package at those versions. They are left out: images
# are not part of what this image is for yet (see engine.yaml), and without
# them the server refuses those formats by name while PNG still serves.

ARG RUNTIME_IMAGE=nvidia/cuda:13.0.3-runtime-ubuntu24.04
ARG TENSORFOLD_REPO=https://github.com/ashhart/TensorFold
ARG TENSORFOLD_REF=v1.3.3
#: The release's own `tensorfold-<version>-linux-aarch64.tar.gz.sha256`.
ARG TENSORFOLD_SHA256=95e4388aba4af1f41a4f50265fd19f7636fa34007f8392a8723d8b2e1612491c
ARG NCCL_WHEEL_VERSION=2.32.6

ARG ENGINE_VERSION=0.0.0
ARG ENGINE_VARIANT=default
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

FROM ${RUNTIME_IMAGE} AS fetch
ARG TENSORFOLD_REPO
ARG TENSORFOLD_REF
ARG TENSORFOLD_SHA256
ARG NCCL_WHEEL_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl python3-pip unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /fetch
# The archive is named for the version without its `v`; the tag keeps it.
RUN set -eu; \
    version="${TENSORFOLD_REF#v}"; \
    archive="tensorfold-${version}-linux-aarch64.tar.gz"; \
    curl -fsSL -o "$archive" "${TENSORFOLD_REPO}/releases/download/${TENSORFOLD_REF}/${archive}"; \
    echo "${TENSORFOLD_SHA256}  ${archive}" | sha256sum -c -; \
    mkdir -p /opt/tensorfold; \
    tar -xzf "$archive" -C /opt/tensorfold --strip-components=1; \
    cd /opt/tensorfold && sha256sum -c SHA256SUMS

# `pip download` installs nothing, so the system interpreter's PEP 668 guard
# does not apply; the wheel is a zip holding the shared library and nothing
# that needs Python at run time.
RUN set -eu; \
    python3 -m pip download --no-deps --only-binary=:all: --dest /fetch/nccl \
      "nvidia-nccl-cu13==${NCCL_WHEEL_VERSION}"; \
    unzip -q /fetch/nccl/*.whl -d /fetch/nccl/wheel; \
    mkdir -p /opt/nccl/lib; \
    cp -P /fetch/nccl/wheel/nvidia/nccl/lib/libnccl.so* /opt/nccl/lib/

FROM ${RUNTIME_IMAGE} AS runner
ARG RUNTIME_IMAGE
ARG TENSORFOLD_REF
ARG NCCL_WHEEL_VERSION
ARG ENGINE_VERSION
ARG ENGINE_VARIANT
ARG BUILD_DATE
ARG GIT_SHA

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl procps libibverbs1 ibverbs-providers rdma-core && \
    rm -rf /var/lib/apt/lists/*

COPY --from=fetch /opt/tensorfold/ /opt/tensorfold/
COPY --from=fetch /opt/nccl/ /opt/nccl/
ENV PATH=/opt/tensorfold/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/nccl/lib:/opt/tensorfold/lib:${LD_LIBRARY_PATH}

# `capabilities --json` reports which optional libraries the loader found, so
# the build records the answer for this image rather than leaving it to the
# first deploy that needs NCCL.
RUN set -eu; \
    mkdir -p /workspace; \
    { echo "engine: tensorfold"; \
      echo "variant: ${ENGINE_VARIANT}"; \
      echo "version: ${ENGINE_VERSION}"; \
      echo "build_date: ${BUILD_DATE}"; \
      echo "git_sha: ${GIT_SHA}"; \
      echo "base_image: ${RUNTIME_IMAGE}"; \
      echo "tensorfold_ref: ${TENSORFOLD_REF}"; \
      echo "nccl_wheel_version: ${NCCL_WHEEL_VERSION}"; \
      echo "tensorfold_native: $(tensorfold-native --version 2>&1 | head -1)"; \
    } > /workspace/build-metadata.yaml && cat /workspace/build-metadata.yaml

ENV HF_HOME=/home/spark/.cache/huggingface
ENV HF_HUB_OFFLINE=1

LABEL org.opencontainers.image.title="spark-pulse-engine/tensorfold" \
      org.opencontainers.image.source="https://github.com/kharkevich-engineering-lab/spark-pulse-engine" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.spark-pulse.engine="tensorfold" \
      io.spark-pulse.variant="${ENGINE_VARIANT}"

WORKDIR /workspace
ENTRYPOINT []
CMD []
