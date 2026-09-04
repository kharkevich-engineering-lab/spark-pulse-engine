#!/usr/bin/env python3
"""Ask the registry what is actually published, for every engine.

The index says whether an engine is `available` and pins it by digest. The
build workflow knows the digest of what it just pushed, but only for the
variants *that run* built — so collecting digests from build artifacts loses
every engine the latest run did not touch, and publishing one variant would
quietly mark the others unpublished.

The registry is the authority and it never goes stale, so ask it. For each
engines/*/engine.yaml this resolves `<image>:<version>` to the manifest digest
the registry serves, or reports it absent.

Usage:
  resolve_digests.py [--out digests.json] [--engines engines]
Auth: anonymous by default (public packages). Set GITHUB_TOKEN — or pass
--username/--password — for a private one.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Every manifest media type a registry might answer with. Omitting the list
#: types makes a multi-arch image resolve to one platform's manifest, which is
#: a different digest from the one a `docker pull` records.
ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def split_ref(image: str) -> tuple[str, str]:
    """``ghcr.io/owner/name`` -> (registry host, repository path)."""
    host, _, repository = image.partition("/")
    if "." not in host and ":" not in host and host != "localhost":
        raise ValueError(f"image without a registry host: {image}")
    return host, repository


def bearer(host: str, repository: str, username: str, password: str) -> str:
    """A pull token for one repository, anonymous when no password is given."""
    url = (
        f"https://{host}/token?scope=repository:{repository}:pull"
        f"&service={host}"
    )
    request = urllib.request.Request(url)
    if password:
        import base64

        raw = base64.b64encode(f"{username}:{password}".encode()).decode()
        request.add_header("Authorization", f"Basic {raw}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return str(json.load(response).get("token") or "")
    except urllib.error.URLError as exc:
        # A repository that does not exist and a private one we may not read
        # both answer this way. Either means "no digest to record", but only
        # the second is a misconfiguration, so say which case this could be.
        how = "with credentials" if password else "anonymously"
        print(
            f"warning: no pull token for {repository} {how} ({exc}); "
            "treating it as unpublished",
            file=sys.stderr,
        )
        return ""


def digest_of(image: str, tag: str, username: str, password: str) -> str:
    """The digest the registry serves for ``image:tag``, or "" when absent."""
    host, repository = split_ref(image)
    token = bearer(host, repository, username, password)
    request = urllib.request.Request(
        f"https://{host}/v2/{repository}/manifests/{tag}", method="HEAD"
    )
    request.add_header("Accept", ACCEPT)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return str(response.headers.get("Docker-Content-Digest") or "")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return ""
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "digests.json")
    ap.add_argument("--engines", type=pathlib.Path, default=ROOT / "engines")
    ap.add_argument("--username", default=os.environ.get("GITHUB_ACTOR", ""))
    ap.add_argument("--password", default=os.environ.get("GITHUB_TOKEN", ""))
    ns = ap.parse_args()

    digests: dict[str, str] = {}
    missing: list[str] = []
    for path in sorted(ns.engines.glob("*/engine.yaml")):
        engine = yaml.safe_load(path.read_text())
        image, version = engine["image"], str(engine["version"])
        digest = digest_of(image, version, ns.username, ns.password)
        if digest:
            digests[f"{image}:{version}"] = digest
            print(f"{path.parent.name}: {version} -> {digest}")
        else:
            missing.append(path.parent.name)
            print(f"{path.parent.name}: {version} -> not published")

    ns.out.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    print(f"wrote {ns.out} with {len(digests)} published engine(s)")
    if missing:
        print(f"note: not published: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
