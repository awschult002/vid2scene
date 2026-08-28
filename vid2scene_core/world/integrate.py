"""Helpers for integrate-mode pair lists (world MegaLoc + intra-clip EigenPlaces)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def merge_pair_files(*paths: Path, dest: Path) -> int:
    seen = set()
    lines = []
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        for raw in Path(path).read_text().splitlines():
            parts = raw.split()
            if len(parts) < 2:
                continue
            key = tuple(sorted((parts[0], parts[1])))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{parts[0]} {parts[1]}\n")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(lines))
    return len(lines)
