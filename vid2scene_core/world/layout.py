"""On-disk layout for a persistent world map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


def cell_id_for_xyz(x: float, y: float, z: float, size: float = 16.0) -> str:
    """Axis-aligned cell key. 16 m default fits rooms/yard; shrink later for fittings."""
    if size <= 0:
        raise ValueError("cell size must be positive")
    ix = int(x // size)
    iy = int(y // size)
    iz = int(z // size)
    return f"{ix}_{iy}_{iz}"


@dataclass(frozen=True)
class WorldLayout:
    root: Path
    cell_size: float = 16.0

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @property
    def control_dir(self) -> Path:
        return self.root / "control"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def cells_dir(self) -> Path:
        return self.root / "cells"

    @property
    def captures_dir(self) -> Path:
        return self.root / "captures"

    @property
    def sparse_dir(self) -> Path:
        return self.root / "sparse"

    @property
    def cameras_jsonl(self) -> Path:
        return self.index_dir / "cameras.jsonl"

    @property
    def descriptors_h5(self) -> Path:
        return self.index_dir / "global-feats-megaloc.h5"

    def capture_dir(self, capture_id: str) -> Path:
        return self.captures_dir / capture_id

    def cell_dir(self, cell_id: str) -> Path:
        return self.cells_dir / cell_id

    def ensure(self) -> None:
        for path in (
            self.control_dir,
            self.index_dir,
            self.cells_dir,
            self.captures_dir,
            self.sparse_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def neighbor_ids(self, cell_id: str) -> Iterable[str]:
        parts = cell_id.split("_")
        if len(parts) != 3:
            raise ValueError(f"bad cell id: {cell_id}")
        ix, iy, iz = map(int, parts)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield f"{ix + dx}_{iy + dy}_{iz + dz}"

    def cell_for_camera_center(self, xyz: Tuple[float, float, float]) -> Path:
        return self.cell_dir(cell_id_for_xyz(*xyz, size=self.cell_size))
