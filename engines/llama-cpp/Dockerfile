# syntax=docker/dockerfile:1.7
#
# llama.cpp engine image for NVIDIA DGX Spark (GB10, aarch64, CUDA 13).
#
# Built from source rather than wrapped, which is the opposite of what
# engines/sglang does, for one reason: llama.cpp compiles in minutes, so
# building it ourselves costs almost nothing and buys the thing that matters
# here -- CUDA 13 with kernels generated for sm_121 and nothing else. The
# published upstream images target older architectures and general CUDA
# versions; a GB10 either falls back to PTX JIT or has no kernel at all.
#
# The cmake invocation is upstream's own (.devops/cuda.Dockerfile) with the
# architecture pinned, so the flags are theirs and not a guess. The node stage
# that builds the bundled web UI is dropped: nothing in the control plane
# serves it, and it is a whole toolchain for a page nobody opens.

ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:13.0.2-runtime-ubuntu24.04
ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_CPP_REF=73ab7599b553c03f6f5d2db24a18ad76f2eb36a3
#: GB10 is sm_121. Not "all supported archs": every other one is dead weight in
#: a build that exists for one machine.
ARG CUDA_DOCKER_ARCH=121
ARG GCC_VERSION=14

ARG ENGINE_VERSION=0.0.0
ARG ENGINE_VARIANT=default
ARG BUILD_DATE=unknown
ARG GIT_SHA=unknown

FROM ${BASE_IMAGE} AS build
ARG LLAMA_CPP_REPO
ARG LLAMA_CPP_REF
ARG CUDA_DOCKER_ARCH
ARG GCC_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake git ca-certificates \
      libcurl4-openssl-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /src
# A shallow fetch of the one commit: the history is 6000 builds deep and none
# of it is needed to compile a pinned ref.
RUN git init -q . && \
    git remote add origin ${LLAMA_CPP_REPO} && \
    git fetch -q --depth 1 origin ${LLAMA_CPP_REF} && \
    git checkout -q FETCH_HEAD

# LLAMA_CURL is what makes `-hf <repo>` resolve a GGUF from HuggingFace, which
# is how a model reaches this engine at all: llama.cpp reads GGUF, and the
# control plane's snapshots are HuggingFace repositories.
RUN cmake -B build \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=ON \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH} \
      -DLLAMA_CURL=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j"$(nproc)"

RUN mkdir -p /out/bin /out/lib && \
    find build -name "*.so*" -exec cp -P {} /out/lib \; && \
    cp build/bin/llama-server build/bin/llama-cli /out/bin/ && \
    printf '%s\n' "${LLAMA_CPP_REF}" > /out/llama-cpp-ref

FROM ${RUNTIME_IMAGE} AS runner
ARG BASE_IMAGE
ARG LLAMA_CPP_REF
ARG CUDA_DOCKER_ARCH
ARG ENGINE_VERSION
ARG ENGINE_VARIANT
ARG BUILD_DATE
ARG GIT_SHA

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libcurl4 curl earlyoom && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/bin/ /usr/local/bin/
COPY --from=build /out/lib/ /usr/local/lib/llama.cpp/
ENV LD_LIBRARY_PATH=/usr/local/lib/llama.cpp:${LD_LIBRARY_PATH}
# GGML loads its backends as shared objects; without this it finds the CPU one
# next to the binary and never the CUDA one.
ENV GGML_BACKEND_PATH=/usr/local/lib/llama.cpp

RUN set -eu; \
    mkdir -p /workspace; \
    { echo "engine: llama-cpp"; \
      echo "variant: ${ENGINE_VARIANT}"; \
      echo "version: ${ENGINE_VERSION}"; \
      echo "build_date: ${BUILD_DATE}"; \
      echo "git_sha: ${GIT_SHA}"; \
      echo "base_image: ${BASE_IMAGE}"; \
      echo "llama_cpp_ref: ${LLAMA_CPP_REF}"; \
      echo "cuda_arch: ${CUDA_DOCKER_ARCH}"; \
      echo "llama_server: $(llama-server --version 2>&1 | head -1)"; \
    } > /workspace/build-metadata.yaml && cat /workspace/build-metadata.yaml

ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_CACHE=/root/.cache/llama.cpp
ENV HF_HOME=/root/.cache/huggingface

LABEL org.opencontainers.image.title="spark-pulse-engine/llama-cpp" \
      org.opencontainers.image.source="https://github.com/kharkevich-engineering-lab/spark-pulse-engine" \
      org.opencontainers.image.version="${ENGINE_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      io.spark-pulse.engine="llama-cpp" \
      io.spark-pulse.variant="${ENGINE_VARIANT}"

WORKDIR /workspace
ENTRYPOINT []
CMD []
