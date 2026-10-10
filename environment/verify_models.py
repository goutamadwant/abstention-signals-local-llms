#!/usr/bin/env python3
"""Verify local GGUF files against models.json (stdlib only).

Usage: python3 verify_models.py --models-dir <LM Studio models directory> [--manifest models.json]

Each model's file is looked up as <models-dir>/<publisher_repo>/<file>, hashed with SHA-256 and compared with the
manifest (size first, then digest). One line per model (OK / MISMATCH / MISSING); exit status 1 on any problem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(path: Path, bytes_expected: int, sha_expected: str) -> tuple[str, str]:
    if not path.is_file():
        return "MISSING", "file not found"
    size = path.stat().st_size
    if size != bytes_expected:
        return "MISMATCH", f"size {size} != {bytes_expected}"
    actual = sha256_file(path)
    if actual != sha_expected:
        return "MISMATCH", f"sha256 {actual} != {sha_expected}"
    return "OK", f"{size} bytes, sha256 {actual[:16]}..."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", required=True, type=Path, help="LM Studio models directory (contains <publisher>/<repository>/<file>.gguf)")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "models.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    host, server = manifest.get("host") or {}, manifest.get("server") or {}
    print(f"manifest collected_at: {manifest.get('collected_at')}")
    print(f"host: {json.dumps(host, sort_keys=True)}")
    print(f"server: {json.dumps(server, sort_keys=True)}")

    failed = False
    for model in manifest["models"]:
        files = [(model["file"], model["bytes"], model["sha256"])]
        files += [(a["file"], a["bytes"], a["sha256"]) for a in model.get("auxiliary_files") or []]
        for name, size, digest in files:
            status, detail = check(args.models_dir / model["publisher_repo"] / name, size, digest)
            failed |= status != "OK"
            print(f"{status:8} {model['id']}  {model['publisher_repo']}/{name}  {detail}")
    print("RESULT:", "FAIL" if failed else "all model files match the manifest")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
