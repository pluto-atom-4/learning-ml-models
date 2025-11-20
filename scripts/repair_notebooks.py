#!/usr/bin/env python3
"""
scripts/repair_notebooks.py

Scan .ipynb files under learning_modules and repair files where the notebook
JSON was embedded as a single JSON string inside a raw cell (common when
notebooks are generated incorrectly). For each such file, replace the file
contents with the parsed JSON object so the notebook renders properly.

Usage:
  python scripts/repair_notebooks.py [--dry-run]

This script will backup the original file to file.bak before overwriting.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_GLOB = "learning_modules/**/*.ipynb"


def is_embedded_json(nb_path: Path) -> bool:
    try:
        j = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    # Check if first cell exists and is raw with a source that appears to be a JSON string
    cells = j.get("cells")
    if not cells or not isinstance(cells, list):
        return False
    first = cells[0]
    if first.get("cell_type") != "raw":
        return False
    source = first.get("source")
    if not source or not isinstance(source, list):
        return False
    # Join source into one string and test if it's valid JSON object
    joined = "".join(source).strip()
    if not joined:
        return False
    try:
        parsed = json.loads(joined)
        if isinstance(parsed, dict) and "cells" in parsed:
            return True
    except Exception:
        return False
    return False


def repair_file(nb_path: Path, dry_run: bool = False) -> bool:
    text = nb_path.read_text(encoding="utf-8")
    obj = json.loads(text)
    first = obj["cells"][0]
    joined = "".join(first["source"]).strip()
    parsed = json.loads(joined)
    if dry_run:
        print(f"Would repair: {nb_path}")
        return True
    bak = nb_path.with_suffix(nb_path.suffix + ".bak")
    print(f"Repairing: {nb_path} -> backup {bak}")
    nb_path.replace(bak)
    nb_path.write_text(json.dumps(parsed, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    nbs = list(ROOT.glob(NB_GLOB))
    repaired = 0
    for nb in nbs:
        if is_embedded_json(nb):
            if repair_file(nb, dry_run=args.dry_run):
                repaired += 1
    print(f"Done. Repaired: {repaired} files.")


if __name__ == "__main__":
    main()

