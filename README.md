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
| `engines/llama-cpp` | `.../spark-pulse-engine/llama-cpp` | llama.cpp's `llama-server`, built from source with CUDA 13 kernels for `sm_121` only |
| `engines/llama-cpp-prism` | `.../spark-pulse-engine/llama-cpp-prism` | The same build from [PrismML's llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) (branch `prism`), which is the only one that runs the `PQ2_0`/`PTQ1_0` ternary GGUFs |
| `engines/trtllm` | `nvcr.io/nvidia/tensorrt-llm/release` | TensorRT-LLM, NVIDIA's own DGX Spark release image (external) |
| `engines/modular-max` | `docker.io/modular/max-nvidia-full` | Modular MAX, `max serve` (external) |
| `engines/atlas` | `docker.io/azeezish/atlas-gb10` | Atlas, a pure-Rust server, from the image its quick-start publishes (external) |

Index: `ghcr.io/kharkevich-engineering-lab/spark-pulse-engine/index:latest`
(OCI artifact holding `index.yaml`).

## Layout

```
engines/<name>/Dockerfile      build definition (vllm-b12x reuses engines/vllm,
                               llama-cpp-prism reuses engines/llama-cpp)
engines/<name>/engine.yaml     pinned sources, runtime contract, capabilities, hardware evidence
engines/vllm/patches/          patch queue applied to the pinned vLLM ref (see NOTICE)
spark-engine.schema.json       schema for engine.yaml
scripts/build_args.py          engine.yaml -> --build-arg flags
scripts/validate.py            schema + pin checks
scripts/inventory.py           engines/*/engine.yaml -> index.yaml
scripts/build.sh               local build helper
scripts/smoke-test.sh          run on a Spark: idle container, exec serve, probe readiness
scripts/resolve_digests.py     ask each registry what is actually published
.github/workflows/             validate, build-vllm, build-sglang, build-llama-cpp, publish-index
```

## Engines we build, and engines we point at

Most engine directories here hold a Dockerfile: we build the image, push it to
our own registry, and the index pins it by digest. Four do not. `build:
{external: true}` says this repo builds nothing for that engine — the image is
somebody else's, it is pulled as it is, and the `engine.yaml` is only the
runtime contract plus a record of what it points at.

That is the right shape when rebuilding would change nothing and cost a great
deal. NVIDIA's TensorRT-LLM release image for the Spark is 21 GB compressed
across 95 layers; wrapping it on a hosted 4-core arm64 runner to add a label is
not a trade worth making. Modular and Atlas publish images for this hardware
themselves. Nothing is added by copying them.

An external engine names its upstream tag in `tag`, because a publisher's tag
is rarely our semver — `version` stays the version of the *definition*.
`resolve_digests.py` resolves `image:tag` against whatever registry it lives on
(following the registry's own auth challenge, since ghcr.io, Docker Hub and
nvcr.io each put their token endpoint somewhere different), so the index pins a
digest even for a tag as slippery as `latest`, and an image nobody can pull
comes out `available: false` rather than as a deploy that 403s.

Everything an external engine loses is worth naming: no patch queue, no
provenance we control, and no guarantee the publisher will not move the tag
under the next index run.

### Two llama.cpp builds

`llama-cpp-prism` is a variant, not a second engine: it reuses
`engines/llama-cpp/Dockerfile` unchanged and differs in one pin, the llama.cpp
it clones. PrismML's ternary packings — `PQ2_0` and `PTQ1_0`, which Bonsai 2
27B ships — are not llama.cpp types. Stock llama.cpp rejects them as unknown,
and loads a `Q2_0` file with no warning and answers nonsense, because it has no
Hadamard activation runtime; the fork's kernels are the whole engine. Nothing
recovers this with a flag, so the fork gets its own image rather than the
default variant moving off upstream.

Both builds carry `-DGGML_RPC=ON` and `ggml-rpc-server`. That is what lets the
prism variant declare `multi_node: {style: llama-rpc}` and `cluster: true`: a
worker rank exposes its GPU over TCP and the head passes `--rpc host:50052`.
It costs one small backend and one binary in the default variant, which is
cheaper than the two builds diverging over a cmake flag.

## engine.yaml

`sources` pins every input (git SHA or tag, package version, base image); for
an external engine it is the upstream image itself, digest-pinned where one has
been published.
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

## Availability

`index.yaml` carries `available` per engine. It is false when the registry
serves no digest for that image, meaning it has never been published.
`publish-index` asks the registry rather than reading build artifacts, so the
answer cannot go stale and publishing one variant cannot mark another
unpublished. Consumers should not offer an unavailable engine, because pulling
it returns a 403 rather than an image.

`vllm-b12x` has now been built and run: the from-source build completes and the
image serves on a GB10 — see `verified` in its `engine.yaml`. That was a local
build on the Spark, because the source path is far too slow to iterate on a
hosted runner; it stays `available: false` until CI publishes an image from the
same pins.

**Building the source path takes real time.** On a GB10 (20 cores) FlashInfer
is about 63 minutes and the vLLM wheel a similar order; on GitHub's 4-core arm64
runners expect several hours per stage, against a 6-hour job cap. The
`prebuilt` wheel mode exists precisely to avoid this, and it is why
`engines/vllm` finishes in under an hour.
