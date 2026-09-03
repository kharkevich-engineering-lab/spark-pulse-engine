#!/usr/bin/env python3
"""Turn an engine.yaml into `docker build` arguments.

Mapping from `sources`:
  <key>.repo    -> <KEY>_REPO
  <key>.ref     -> <KEY>_REF
  <key>.version -> <KEY>_VERSION
  <key>.index   -> <KEY>_INDEX
  <key>.image   -> <KEY>_IMAGE
Entries under `build_args` are passed verbatim. ENGINE_VERSION, ENGINE_VARIANT,
BUILD_DATE and GIT_SHA are always added.

Usage:
  build_args.py engines/vllm/engine.yaml            # prints one --build-arg per line
  build_args.py engines/vllm/engine.yaml --json     # prints a JSON object
  build_args.py engines/vllm/engine.yaml --get image|version|context|dockerfile|tag
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


def build_args(engine: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, src in (engine.get("sources") or {}).items():
        prefix = key.upper().replace("-", "_")
        for field, suffix in (("repo", "REPO"), ("ref", "REF"), ("version", "VERSION"), ("index", "INDEX"), ("image", "IMAGE")):
            if field in src:
                out[f"{prefix}_{suffix}"] = str(src[field])
    for key, value in (engine.get("build_args") or {}).items():
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
    ap.add_argument("--get", choices=["image", "version", "context", "dockerfile", "tag", "name"])
    ns = ap.parse_args()

    engine = load(ns.engine_yaml)
    build = engine.get("build") or {}
    context = build.get("context") or str(ns.engine_yaml.parent)
    dockerfile = build.get("dockerfile") or str(pathlib.Path(context) / "Dockerfile")

    if ns.get:
        value = {
            "image": engine["image"],
            "version": engine["version"],
            "context": context,
            "dockerfile": dockerfile,
            "tag": f'{engine["image"]}:{engine["version"]}',
            "name": ns.engine_yaml.parent.name,
        }[ns.get]
        print(value)
        return 0

    args = build_args(engine)
    if ns.json:
        json.dump(args, sys.stdout, indent=2)
        print()
    else:
        for key, value in args.items():
            print(f"--build-arg={key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
