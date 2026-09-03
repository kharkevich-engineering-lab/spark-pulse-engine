#!/usr/bin/env python3
"""Download prebuilt wheels declared under `wheels:` in an engine.yaml.

engine.yaml:
  wheels:
    mode: prebuilt
    flashinfer: {release: eugr/spark-vllm-docker@prebuilt-flashinfer-current}
    vllm:       {release: eugr/spark-vllm-docker@prebuilt-vllm-current}

Each entry is fetched into <out>/<entry>/ with every *.whl asset of that
GitHub release, plus provenance the Dockerfile's runner stage reads:
  .<entry>-commit   git sha parsed from the wheel version (g<sha>) or "prebuilt"
  manifest.json     release tag, published_at, asset names, sizes, digests

Usage:
  fetch_wheels.py engines/vllm/engine.yaml --out wheels/vllm
  fetch_wheels.py engines/vllm/engine.yaml --keys     # print cache keys, one per line as <entry>=<key>
Set GITHUB_TOKEN to avoid API rate limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import urllib.request

import yaml

API = "https://api.github.com"


def api(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def release(spec: str) -> dict:
    repo, _, tag = spec.partition("@")
    if not tag:
        raise SystemExit(f"release spec must be owner/repo@tag, got {spec!r}")
    return api(f"{API}/repos/{repo}/releases/tags/{tag}")


def key_for(rel: dict) -> str:
    h = hashlib.sha256()
    for a in sorted(rel["assets"], key=lambda a: a["name"]):
        h.update(f'{a["name"]}:{a.get("digest") or a["updated_at"]}:{a["size"]}'.encode())
    return f'{rel["tag_name"]}-{h.hexdigest()[:16]}'


def download(url: str, dest: pathlib.Path) -> str:
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    h = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_yaml", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--keys", action="store_true")
    ns = ap.parse_args()

    engine = yaml.safe_load(ns.engine_yaml.read_text())
    wheels = engine.get("wheels") or {}
    if wheels.get("mode") != "prebuilt":
        print("wheels.mode is not prebuilt; nothing to do", file=sys.stderr)
        return 0
    entries = {k: v for k, v in wheels.items() if k != "mode"}

    if ns.keys:
        for name, spec in entries.items():
            print(f"{name}={key_for(release(spec['release']))}")
        return 0

    if not ns.out:
        ap.error("--out is required unless --keys")

    for name, spec in entries.items():
        rel = release(spec["release"])
        dest = ns.out / name
        dest.mkdir(parents=True, exist_ok=True)
        manifest = {"release": spec["release"], "tag": rel["tag_name"], "published_at": rel["published_at"], "assets": []}
        commit = "prebuilt"
        for asset in rel["assets"]:
            if not asset["name"].endswith(".whl"):
                continue
            target = dest / asset["name"]
            print(f"downloading {asset['name']} ({asset['size'] // 1048576} MB)")
            digest = download(asset["browser_download_url"], target)
            expected = asset.get("digest")
            if expected and expected != digest:
                raise SystemExit(f"digest mismatch for {asset['name']}: {expected} != {digest}")
            manifest["assets"].append({"name": asset["name"], "size": asset["size"], "digest": digest})
            m = re.search(r"\+g([0-9a-f]{7,40})", asset["name"])
            if m and asset["name"].startswith(name.replace("-", "_")):
                commit = m.group(1)
        (dest / f".{name}-commit").write_text(commit + "\n")
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(manifest['assets'])} wheels from {rel['tag_name']} ({rel['published_at']}), commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
