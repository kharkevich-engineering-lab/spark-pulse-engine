#!/usr/bin/env python3
"""Turn an engine.yaml into `docker build` arguments.

Mapping from `sources`:
  <key>.repo    -> <KEY>_REPO
  <key>.ref     -> <KEY>_REF
  <key>.version -> <KEY>_VERSION
  <key>.index   -> <KEY>_INDEX
  <key>.image   -> <KEY>_IMAGE
Entries under `build_args` are passed verbatim, then, with `--platform`, the
entries under `platform_build_args.<platform>` override them: a multi-arch
image compiles for different GPUs on each platform. ENGINE_VERSION,
ENGINE_VARIANT, BUILD_DATE and GIT_SHA are always added.

Usage:
  build_args.py engines/vllm/engine.yaml            # prints one --build-arg per line
  build_args.py engines/vllm/engine.yaml --json     # prints a JSON object
  build_args.py engines/vllm/engine.yaml --platform linux/amd64   # that platform's args
  build_args.py engines/vllm/engine.yaml --get image|version|context|dockerfile|tag
  build_args.py engines/vllm/engine.yaml --get platforms           # one `arch` entry per line
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys

import yaml


def load(path: pathlib.Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:12]


def platforms(engine: dict) -> list[str]:
    """The platforms an engine is published for: its `arch`, arm64 by default."""
    return list(engine.get("arch") or ["linux/arm64"])


def build_args(engine: dict, platform: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, src in (engine.get("sources") or {}).items():
        prefix = key.upper().replace("-", "_")
        for field, suffix in (("repo", "REPO"), ("ref", "REF"), ("version", "VERSION"), ("index", "INDEX"), ("image", "IMAGE")):
            if field in src:
                out[f"{prefix}_{suffix}"] = str(src[field])
    for key, value in (engine.get("build_args") or {}).items():
        out[key] = str(value).lower() if isinstance(value, bool) else str(value)
    if platform:
        if platform not in platforms(engine):
            raise SystemExit(f"{engine['engine']}/{engine['variant']} is not built for {platform}")
        overrides = (engine.get("platform_build_args") or {}).get(platform) or {}
        for key, value in overrides.items():
            out[key] = str(value).lower() if isinstance(value, bool) else str(value)
    out["ENGINE_VERSION"] = engine["version"]
    out["ENGINE_VARIANT"] = engine["variant"]
    out["BUILD_DATE"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["GIT_SHA"] = git_sha()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_yaml", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--platform", help="apply this platform's platform_build_args")
    ap.add_argument("--get", choices=["image", "version", "context", "dockerfile", "tag", "name", "platforms"])
    ns = ap.parse_args()

    engine = load(ns.engine_yaml)
    build = engine.get("build") or {}
    context = build.get("context") or str(ns.engine_yaml.parent)
    dockerfile = build.get("dockerfile") or str(pathlib.Path(context) / "Dockerfile")

    if ns.get == "platforms":
        print("\n".join(platforms(engine)))
        return 0

    if ns.get:
        # An engine built here is published at the version we gave it; one we
        # only point at (`build.external`) keeps whatever tag its publisher
        # uses, which is rarely our semver. `--get tag` has to be the thing a
        # pull would actually resolve, because that is what it is used for.
        tag = str(engine.get("tag") or engine["version"])
        value = {
            "image": engine["image"],
            "version": engine["version"],
            "context": context,
            "dockerfile": dockerfile,
            "tag": f'{engine["image"]}:{tag}',
            "name": ns.engine_yaml.parent.name,
        }[ns.get]
        print(value)
        return 0

    args = build_args(engine, ns.platform)
    if ns.json:
        json.dump(args, sys.stdout, indent=2)
        print()
    else:
        for key, value in args.items():
            print(f"--build-arg={key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
