#!/usr/bin/env python3
"""Validate every engines/*/engine.yaml against spark-engine.schema.json.

Beyond the schema it checks that git sources are pinned to a full SHA or a
tag (never a branch name) for the default branch build, that the Dockerfile
referenced by `build` exists, and that every file in a patch queue is at least
well formed — a `*.patch` git can parse, a `*.py` that compiles.

That last check exists because it was missing. `flashinfer_cache.patch` had a
hunk header claiming nine lines over a body of eight, so `git apply` ran off
the end of the file. Nothing noticed for as long as it did because the only
variant CI built used prebuilt wheels, which skip the builder stages the patch
queue belongs to; the corruption surfaced thirteen minutes into the first
source-mode build ever run.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import jsonschema
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^v?\d+(\.\d+)*([-.+][0-9A-Za-z.]+)?$")
BRANCHY = {"main", "master", "dev", "develop", "latest", "HEAD"}


def check_patch_queue(directory: pathlib.Path) -> list[str]:
    """Structural problems with the patch files in one engine directory.

    This cannot say a patch will *apply* — that needs the source tree, which
    is cloned inside the build. It says the file is parseable, which is the
    failure that is otherwise invisible until a build is well under way.
    """
    problems: list[str] = []
    patches = directory / "patches"
    if not patches.is_dir():
        return problems

    for path in sorted(patches.glob("*.patch")):
        # --numstat parses every hunk header against its body and needs no
        # working tree, so it is the cheapest honest check available here.
        result = subprocess.run(
            ["git", "apply", "--numstat", str(path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            problems.append(
                f"{path.relative_to(ROOT)}: {detail[0] if detail else 'unparseable'}"
            )

    for path in sorted(patches.glob("*.py")):
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(ROOT)}: line {exc.lineno}: {exc.msg}")

    return problems


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

        # The patch queue lives with the Dockerfile, which a variant may share.
        problems = check_patch_queue(dockerfile.parent)
        for problem in problems:
            failed = True
            print(f"{path}: {problem}")

        if not errors and not problems:
            print(f"ok: {path.relative_to(ROOT)} ({engine['engine']}/{engine['variant']} {engine['version']})")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
