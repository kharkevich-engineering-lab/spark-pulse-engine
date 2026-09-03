#!/usr/bin/env bash
# Smoke test an engine image on a DGX Spark. Needs a GPU, so this does not run
# in GitHub CI; run it by hand (or from a self-hosted runner) after a build.
#
#   scripts/smoke-test.sh engines/vllm  [IMAGE_REF] [MODEL]
#   scripts/smoke-test.sh engines/sglang [IMAGE_REF] [MODEL]
#
# Starts an idle container the way Spark Pulse does, execs the engine's serve
# command with a small model, waits for the readiness endpoint, hits
# /v1/models and one chat completion, then tears down.
set -euo pipefail
cd "$(dirname "$0")/.."
TOOL="${CONTAINER_TOOL:-$(command -v docker >/dev/null 2>&1 && echo docker || echo podman)}"

ENGINE_DIR="${1:?engine dir}"
YAML="$ENGINE_DIR/engine.yaml"
IMAGE="${2:-$(python3 scripts/build_args.py "$YAML" --get tag)}"
MODEL="${3:-Qwen/Qwen2.5-0.5B-Instruct}"
NAME="spark-pulse-smoke-$$"
TIMEOUT="${TIMEOUT:-900}"

py() { python3 -c "import yaml,sys; e=yaml.safe_load(open('$YAML')); print($1)"; }
SERVE=$(py 'e["runtime"]["serve"]')
MODEL_ARG=$(py 'e["runtime"].get("model_arg","positional")')
READY=$(py 'e["runtime"]["readiness"]')
PORT=$(py 'e["runtime"]["ports"]["api"]')
PRIV=$(py 'str(e["runtime"].get("container",{}).get("privileged",False)).lower()')
KEEP=$(py 'e["runtime"].get("container",{}).get("keepalive","sleep infinity")')
PORT_FLAG=$(py 'e["runtime"].get("param_flags",{}).get("port","--port")')
HOST_FLAG=$(py 'e["runtime"].get("param_flags",{}).get("host","--host")')

RUN=("$TOOL" run -d --rm --name "$NAME" --gpus all --network host --ipc=host --entrypoint= \
     -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
     -e HF_TOKEN="${HF_TOKEN:-}")
if [ "$PRIV" = true ]; then RUN+=(--privileged --ulimit nofile=1048576:1048576)
else RUN+=(--device=/dev/infiniband --ulimit memlock=-1 --shm-size=32g); fi
for m in $(py '" ".join(e["runtime"].get("cache_mounts",[]))'); do
  host="${m/#\~/$HOME}"; mkdir -p "$host"; RUN+=(-v "$host:${m/#\~//root}")
done
for kv in $(py '" ".join(f"{k}={v}" for k,v in e["runtime"].get("env",{}).items())'); do RUN+=(-e "$kv"); done

cleanup() { "$TOOL" rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== starting idle container from $IMAGE"
"${RUN[@]}" "$IMAGE" $KEEP >/dev/null
"$TOOL" exec "$NAME" cat /workspace/build-metadata.yaml || true

if [ "$MODEL_ARG" = positional ]; then CMD="$SERVE $MODEL"; else CMD="$SERVE $MODEL_ARG $MODEL"; fi
CMD="$CMD $HOST_FLAG 0.0.0.0 $PORT_FLAG $PORT ${EXTRA_ARGS:-}"
echo "== exec: $CMD"
"$TOOL" exec -d "$NAME" bash -c "$CMD >> /proc/1/fd/1 2>&1"

echo "== waiting for http://127.0.0.1:$PORT$READY (timeout ${TIMEOUT}s)"
for ((i=0; i<TIMEOUT; i+=5)); do
  if curl -fsS "http://127.0.0.1:$PORT$READY" >/dev/null 2>&1; then echo "ready after ${i}s"; break; fi
  if ! "$TOOL" ps -q -f "name=$NAME" | grep -q .; then echo "container died"; "$TOOL" logs "$NAME" | tail -50; exit 1; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$PORT$READY" >/dev/null || { echo "timeout"; "$TOOL" logs "$NAME" | tail -100; exit 1; }

echo "== /v1/models"
curl -fsS "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool | head -20
echo "== chat completion"
curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in three words.\"}],\"max_tokens\":16}" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
echo "== PASS $IMAGE"
