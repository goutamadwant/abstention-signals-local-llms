#!/usr/bin/env python3
"""Verify the benchmark checkouts the runner reads against protocol.yaml `model_pins.benchmarks` (stdlib + PyYAML).

Usage: python3 check_benchmarks.py [--teleqna <TeleQnA.txt>] [--netconfeval <conext24-NetConfEval checkout>]

The paths default to what artifact/run_cell.py uses: TELEQNA_PATH (the unzipped TeleQnA.txt; its directory is the
TeleQnA checkout) and NETCONFEVAL_PATH (the NetConfEval checkout), falling back to
<repo>/runtime/ground_truth/TeleQnA/TeleQnA.txt and <repo>/runtime/ground_truth/NetConfEval.

For each benchmark: the checkout's HEAD commit must start with the pinned commit, and no file the runner reads may
carry local modifications. TeleQnA.txt is not tracked by git (it is unzipped from TeleQnA.zip), so its size and CRC-32
are compared with the zip entry (no password needed). Prints the found commit and the SHA-256 of every file the runner
reads, plus one combined digest per benchmark. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

LANE = Path(__file__).resolve().parents[1]
REPO = LANE.parents[1]
PROTOCOL = LANE / "protocol" / "protocol.yaml"

# files artifact/run_cell.py (load_netconfeval_t1, netconfeval_score) imports or opens, relative to the checkout
NETCONFEVAL_FILES = [
    "assets/step_1_policies.csv",
    "netconfeval/__init__.py",
    "netconfeval/common/__init__.py",
    "netconfeval/common/utils.py",
    "netconfeval/prompts/__init__.py",
    "netconfeval/prompts/step_1_reachability.py",
    "netconfeval/prompts/step_1_reachability_waypoint.py",
    "netconfeval/prompts/step_1_reachability_waypoint_load.py",
]
# TeleQnA: the runner reads TeleQnA.txt; the zip is what the commit pins; evaluation_tools.py defines the semantics
# that artifact/audit_scorer.py checks the option-id scorer against
TELEQNA_FILES = ["TeleQnA.txt", "TeleQnA.zip", "evaluation_tools.py"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_commits() -> dict[str, str]:
    import yaml
    pins = ((yaml.safe_load(PROTOCOL.read_text(encoding="utf-8")) or {}).get("model_pins") or {}).get("benchmarks") or {}
    out = {}
    for name, text in pins.items():
        m = re.search(r"commit\s+([0-9a-f]{7,40})", str(text))
        if m:
            out[str(name)] = m.group(1)
    return out


def git(checkout: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(checkout), *args], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def check_checkout(name: str, checkout: Path, pinned: str, files: list[str]) -> bool:
    ok = True
    print(f"== {name}: {checkout}")
    head = git(checkout, "rev-parse", "HEAD")
    if head is None:
        print(f"MISMATCH commit: not a git checkout (expected {pinned})")
        ok = False
    elif not head.startswith(pinned):
        print(f"MISMATCH commit: found {head}, pinned {pinned}")
        ok = False
    else:
        print(f"OK       commit: found {head} (pinned {pinned})")
    combined = hashlib.sha256()
    for rel in files:
        path = checkout / rel
        if not path.is_file():
            print(f"MISSING  {rel}")
            ok = False
            continue
        digest = sha256_file(path)
        combined.update(rel.encode()); combined.update(digest.encode())
        dirty = git(checkout, "status", "--porcelain", "--", rel) if head is not None else None
        tracked = git(checkout, "ls-files", "--error-unmatch", "--", rel) if head is not None else None
        state = "no git" if head is None else "modified" if dirty and tracked else ("untracked" if not tracked else "clean")
        if state == "modified":
            ok = False
        print(f"{'MODIFIED' if state == 'modified' else 'OK':8} {rel}  sha256 {digest}  ({state}, {path.stat().st_size} bytes)")
    print(f"combined sha256 ({name} files above): {combined.hexdigest()}")
    return ok


def check_teleqna_unzipped(checkout: Path, txt: Path) -> bool:
    """TeleQnA.txt is the password-protected zip's only member: compare size and CRC-32 with the zip directory entry."""
    zpath = checkout / "TeleQnA.zip"
    if not zpath.is_file() or not txt.is_file():
        print("MISMATCH TeleQnA.txt vs TeleQnA.zip: file missing (unzip TeleQnA.zip with the password from the TeleQnA README)")
        return False
    with zipfile.ZipFile(zpath) as z:
        info = next((i for i in z.infolist() if Path(i.filename).name == "TeleQnA.txt"), None)
    if info is None:
        print("MISMATCH TeleQnA.zip has no TeleQnA.txt entry")
        return False
    crc = 0
    with txt.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    size = txt.stat().st_size
    if size != info.file_size or crc != info.CRC:
        print(f"MISMATCH TeleQnA.txt: size {size} crc32 {crc:08x}; zip entry size {info.file_size} crc32 {info.CRC:08x}")
        return False
    print(f"OK       TeleQnA.txt equals the TeleQnA.zip entry (size {size}, crc32 {crc:08x})")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--teleqna", type=Path, default=Path(os.environ.get("TELEQNA_PATH", str(REPO / "runtime" / "ground_truth" / "TeleQnA" / "TeleQnA.txt"))), help="TeleQnA.txt the runner reads (default: TELEQNA_PATH)")
    parser.add_argument("--netconfeval", type=Path, default=Path(os.environ.get("NETCONFEVAL_PATH", str(REPO / "runtime" / "ground_truth" / "NetConfEval"))), help="NetConfEval checkout (default: NETCONFEVAL_PATH)")
    args = parser.parse_args(argv)
    pins = pinned_commits()
    print(f"pinned in protocol.yaml model_pins.benchmarks: {pins}")
    ok = True
    for name in ("TeleQnA", "NetConfEval"):
        if name not in pins:
            print(f"MISMATCH {name}: no commit pinned in protocol.yaml")
            ok = False
    teleqna_dir = args.teleqna.resolve().parent
    ok &= check_checkout("TeleQnA", teleqna_dir, pins.get("TeleQnA", "?"), TELEQNA_FILES)
    ok &= check_teleqna_unzipped(teleqna_dir, args.teleqna.resolve())
    ok &= check_checkout("NetConfEval", args.netconfeval.resolve(), pins.get("NetConfEval", "?"), NETCONFEVAL_FILES)
    print("RESULT:", "all benchmark checkouts match the protocol pins" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
