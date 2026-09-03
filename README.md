# spark-pulse-engine

Engine images for [Spark Pulse](https://github.com/kharkevich-engineering-lab/spark-pulse)
on NVIDIA DGX Spark (GB10, aarch64, CUDA 13, `sm_121`).

Every serving engine that Spark Pulse can run comes from here: a Dockerfile,
pinned sources, and an `engine.yaml` describing how the control plane should
run the image. Spark Pulse never builds images itself; it reads the published
index and pulls by digest.

| Engine dir | Image | What it is |
|---|---|---|
| `engines/vllm` | `ghcr.io/kharkevich-engineering-lab/spark-pulse-engine/vllm` | vLLM built from source with the Spark patch queue |
| `engines/vllm-b12x` | `.../spark-pulse-engine/vllm-b12x` | vLLM from the local-inference-lab fork plus B12X kernels |
| `engines/sglang` | `.../spark-pulse-engine/sglang` | SGLang, wrapping the upstream cu130 image |

Index: `ghcr.io/kharkevich-engineering-lab/spark-pulse-engine/index:latest`
(OCI artifact holding `index.yaml`).

## Layout

```
engines/<name>/Dockerfile      build definition (vllm-b12x reuses engines/vllm)
engines/<name>/engine.yaml     pinned sources, runtime contract, capabilities, hardware evidence
engines/vllm/patches/          patch queue applied to the pinned vLLM ref (see NOTICE)
spark-engine.schema.json       schema for engine.yaml
scripts/build_args.py          engine.yaml -> --build-arg flags
scripts/validate.py            schema + pin checks
scripts/inventory.py           engines/*/engine.yaml -> index.yaml
scripts/build.sh               local build helper
scripts/smoke-test.sh          run on a Spark: idle container, exec serve, probe readiness
.github/workflows/             validate, build-vllm, build-sglang, publish-index
```

## engine.yaml

`sources` pins every input (git SHA or tag, package version, base image).
`runtime` is the contract Spark Pulse relies on: serve command, how the model
is passed, readiness and metrics paths, ports, cache mounts, base env, the
container profile (privileged or not, ipc, shm, devices, ulimits) and the
multi-node style. `capabilities` says what the control plane may do with the
image. `verified` records hardware evidence. `legacy_tags` maps v1 recipe
`container:` names such as `vllm-node` onto the image.

## Building

Locally on any arm64 docker host (a Spark is fine):

```bash
pip install pyyaml jsonschema
python3 scripts/validate.py
scripts/build.sh engines/sglang
scripts/build.sh engines/vllm                 # full from-source build, several hours
scripts/build.sh engines/vllm --wheels        # only export wheels to ./wheels/vllm
scripts/build.sh engines/vllm --from-wheels   # runner stage from those wheels
```

In CI the vLLM build runs on GitHub's native `ubuntu-24.04-arm` runners, split
into FlashInfer wheels, vLLM wheel, and runner image jobs.

`wheels.mode` in `engine.yaml` decides where the wheels come from:

- `prebuilt` (default for `engines/vllm`): the wheel jobs download the
  `*.whl` assets of the named GitHub releases (currently upstream's daily
  `prebuilt-vllm-current` and `prebuilt-flashinfer-current`, built from vLLM
  main with the same patch queue and torch pin) and the runner stage installs
  them. A full run takes well under an hour. The wheel's embedded commit and
  the release manifest land in `/workspace/build-metadata.yaml`.
- `source` (`engines/vllm-b12x`): the Dockerfile clones the pinned refs and
  compiles. Multi-hour on a hosted runner; wheels are cached by pinned ref plus
  a hash of the Dockerfile and patch queue, so bumping only metadata rebuilds
  the runner stage.

Locally, `scripts/build.sh engines/vllm --prebuilt` does the same download and
runner build with docker or podman. Builds are pushed with tags `<version>`, `<version>-<sha>` and
`latest`; `publish-index` then regenerates `index.yaml` with digests and
pushes it as an OCI artifact.

Hosted arm64 runners have no GPU. Smoke testing happens on a Spark:

```bash
scripts/smoke-test.sh engines/vllm ghcr.io/kharkevich-engineering-lab/spark-pulse-engine/vllm:0.1.0
scripts/smoke-test.sh engines/sglang
```

Add the result to `verified` in the engine's `engine.yaml`. Once a Spark is
available as a self-hosted runner, the smoke test can gate the `latest` tag.

## Bumping vLLM

1. Change `sources.vllm.ref` (and `flashinfer`, `nccl` as needed) in `engines/vllm/engine.yaml`, bump `version`.
2. Run `scripts/validate.py --strict`. Branch names are rejected; use a SHA or tag.
3. Build. Every script in `patches/` inspects the source and skips itself when the fix is upstream; a script that fails means the source shape changed and the patch needs attention or removal.
4. Smoke test on a Spark, record it under `verified`.

## Credits

The vLLM Dockerfile and the patch queue derive from
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT). The
SGLang runtime contract follows the findings in
[mark-ramsey-ri/sglang-dgx-spark](https://github.com/mark-ramsey-ri/sglang-dgx-spark).
