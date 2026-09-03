#!/usr/bin/env python3
"""Generate index.yaml from engines/*/engine.yaml.

The index is what Spark Pulse fetches to populate its engine registry. It
carries the full engine.yaml of every image plus the resolved image reference,
so a consumer never needs this repo checked out.

Usage: inventory.py [--out index.yaml] [--digests digests.json]
  --digests  JSON mapping "<image>:<version>" -> "sha256:..." produced by the
             build workflow; when present, `ref` uses the digest form.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "index.yaml")
    ap.add_argument("--digests", type=pathlib.Path)
    ns = ap.parse_args()

    digests: dict[str, str] = {}
    if ns.digests and ns.digests.exists():
        digests = json.loads(ns.digests.read_text())

    entries = []
    for path in sorted(ROOT.glob("engines/*/engine.yaml")):
        engine = yaml.safe_load(path.read_text())
        tag = f'{engine["image"]}:{engine["version"]}'
        digest = digests.get(tag)
        entry = {
            "id": path.parent.name,
            "engine": engine["engine"],
            "variant": engine["variant"],
            "version": engine["version"],
            "image": engine["image"],
            "tag": tag,
            "ref": f'{engine["image"]}@{digest}' if digest else tag,
            "digest": digest,
            "legacy_tags": list(engine.get("legacy_tags", [])),
            "description": engine.get("description", ""),
            "capabilities": dict(engine.get("capabilities", {})),
            "spec": copy.deepcopy(engine),
        }
        entries.append(entry)

    index = {
        "apiVersion": "spark-pulse.io/v1",
        "kind": "EngineIndex",
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engines": entries,
    }
    class _Dumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    ns.out.write_text(yaml.dump(index, Dumper=_Dumper, sort_keys=False))
    print(f"wrote {ns.out} with {len(entries)} engines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
