#!/usr/bin/env python3
"""Verify downloaded FPV-LLM-Navigation release assets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


EXPECTED_JSONL_COUNTS = {
    "human-fpv-deidentified.jsonl.gz": 8263,
    "human-gv-deidentified.jsonl.gz": 9581,
    "llm-primary-fpv.jsonl.gz": 8100,
    "llm-posthoc-fpv.jsonl.gz": 1800,
    "llm-trace-and-memory-ablation.jsonl.gz": 720,
    "llm-interventions.jsonl.gz": 1400,
    "llm-global-view.jsonl.gz": 1080,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    args = parser.parse_args()

    checksums = load_checksums(args.checksums)
    for name, expected_digest in checksums.items():
        path = args.asset_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing asset: {name}")
        actual = sha256_file(path)
        if actual != expected_digest:
            raise SystemExit(f"SHA-256 mismatch: {name}")
        print(f"checksum ok: {name}")

    for name, expected in EXPECTED_JSONL_COUNTS.items():
        path = args.asset_dir / name
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SystemExit(f"Non-object record in {name} at line {count + 1}")
                count += 1
        if count != expected:
            raise SystemExit(f"Record-count mismatch for {name}: {count} != {expected}")
        print(f"jsonl ok: {name} ({count:,} records)")

    for name in ("baseline-agent-outputs.tar.gz", "analysis-and-figure-source.tar.gz"):
        with tarfile.open(args.asset_dir / name, "r:gz") as archive:
            members = archive.getmembers()
        print(f"archive ok: {name} ({len(members):,} files)")

    print("All release checks passed.")


if __name__ == "__main__":
    main()
