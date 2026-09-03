#!/usr/bin/env python3
"""Validate every engines/*/engine.yaml against spark-engine.schema.json.

Beyond the schema it checks that git sources are pinned to a full SHA or a
tag (never a branch name) for the default branch build, and that the
Dockerfile referenced by `build` exists.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import jsonschema
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^v?\d+(\.\d+)*([-.+][0-9A-Za-z.]+)?$")
BRANCHY = {"main", "master", "dev", "develop", "latest", "HEAD"}


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    schema = json.loads((ROOT / "spark-engine.schema.json").read_text())
    validator = jsonschema.Draft7Validator(schema)
    files = sorted(ROOT.glob("engines/*/engine.yaml"))
    if not files:
        print("no engines/*/engine.yaml found", file=sys.stderr)
        return 1

    failed = False
    for path in files:
        engine = yaml.safe_load(path.read_text())
        errors = sorted(validator.iter_errors(engine), key=lambda e: list(e.path))
        for err in errors:
            failed = True
            loc = "/".join(str(p) for p in err.path) or "<root>"
            print(f"{path}: {loc}: {err.message}")

        build = engine.get("build") or {}
        dockerfile = ROOT / (build.get("dockerfile") or f"{path.parent.relative_to(ROOT)}/Dockerfile")
        if not dockerfile.exists():
            failed = True
            print(f"{path}: dockerfile not found: {dockerfile.relative_to(ROOT)}")

        for key, src in (engine.get("sources") or {}).items():
            ref = src.get("ref")
            if ref is None:
                continue
            pinned = bool(SHA_RE.match(ref) or TAG_RE.match(ref))
            if not pinned and (strict or ref in BRANCHY):
                msg = f"{path}: sources/{key}/ref '{ref}' is not a SHA or tag"
                if strict:
                    failed = True
                    print(msg)
                else:
                    print(f"warning: {msg}")
            if "image" in src and "@sha256:" not in src["image"] and strict:
                print(f"warning: {path}: sources/{key}/image is not digest-pinned")

        if not errors:
            print(f"ok: {path.relative_to(ROOT)} ({engine['engine']}/{engine['variant']} {engine['version']})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
