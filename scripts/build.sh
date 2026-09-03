#!/usr/bin/env bash
# Build one engine image locally (on a Spark or any arm64 docker host).
#
#   scripts/build.sh engines/vllm            # full build, tags <image>:<version> and legacy tags
#   scripts/build.sh engines/vllm --wheels   # only export wheels to ./wheels/<engine>/{flashinfer,vllm}
#   scripts/build.sh engines/vllm --from-wheels   # runner stage from ./wheels/<engine>
#   scripts/build.sh engines/vllm --prebuilt      # download upstream wheels (engine.yaml `wheels:`), then runner stage
#   scripts/build.sh engines/sglang
#
# Uses docker when present, otherwise podman (override with CONTAINER_TOOL).
# Extra args after `--` go to the build command.
set -euo pipefail
cd "$(dirname "$0")/.."
TOOL="${CONTAINER_TOOL:-$(command -v docker >/dev/null 2>&1 && echo docker || echo podman)}"
build() { if [ "$TOOL" = docker ]; then docker buildx build "$@"; else podman build "$@"; fi; }

ENGINE_DIR="${1:?engine dir, e.g. engines/vllm}"; shift || true
MODE=full
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --wheels) MODE=wheels ;;
    --from-wheels) MODE=from-wheels ;;
    --prebuilt) MODE=prebuilt ;;
    --) shift; EXTRA=("$@"); break ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

YAML="$ENGINE_DIR/engine.yaml"
NAME=$(python3 scripts/build_args.py "$YAML" --get name)
TAG=$(python3 scripts/build_args.py "$YAML" --get tag)
IMAGE=$(python3 scripts/build_args.py "$YAML" --get image)
CONTEXT=$(python3 scripts/build_args.py "$YAML" --get context)
DOCKERFILE=$(python3 scripts/build_args.py "$YAML" --get dockerfile)
mapfile -t ARGS < <(python3 scripts/build_args.py "$YAML")
LEGACY=$(python3 -c 'import yaml,sys; print(" ".join(yaml.safe_load(open(sys.argv[1])).get("legacy_tags", [])))' "$YAML")

WHEELS="wheels/$NAME"
LOAD=""; [ "$TOOL" = docker ] && LOAD="--load"
COMMON=(--platform linux/arm64 -f "$DOCKERFILE" "${ARGS[@]}" "${EXTRA[@]}")

case "$MODE" in
  wheels)
    mkdir -p "$WHEELS"
    build "${COMMON[@]}" --target flashinfer-export --output "type=local,dest=$WHEELS/flashinfer" "$CONTEXT"
    build "${COMMON[@]}" --target vllm-export --output "type=local,dest=$WHEELS/vllm" "$CONTEXT"
    ;;
  from-wheels)
    build "${COMMON[@]}" --target runner \
      --build-context "flashinfer_wheels=$WHEELS/flashinfer" \
      --build-context "vllm_wheels=$WHEELS/vllm" \
      -t "$TAG" -t "$IMAGE:latest" $LOAD "$CONTEXT"
    ;;
  prebuilt)
    python3 scripts/fetch_wheels.py "$YAML" --out "$WHEELS"
    build "${COMMON[@]}" --target runner \
      --build-context "flashinfer_wheels=$WHEELS/flashinfer" \
      --build-context "vllm_wheels=$WHEELS/vllm" \
      -t "$TAG" -t "$IMAGE:latest" $LOAD "$CONTEXT"
    ;;
  full)
    build "${COMMON[@]}" --target runner -t "$TAG" -t "$IMAGE:latest" $LOAD "$CONTEXT"
    ;;
esac

if [ "$MODE" != wheels ]; then
  for legacy in $LEGACY; do "$TOOL" tag "$TAG" "$legacy"; done
  echo "built $TAG${LEGACY:+ (also tagged: $LEGACY)}"
fi
