"""World-level MegaLoc descriptor store.

Uses hloc HDF5 when the worker has hloc. FAISS is optional: if the package
is missing, retrieval falls back to brute-force cosine on the loaded matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class WorldDescriptorIndex:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.h5_path = self.index_dir / "global-feats-megaloc.h5"
        self.names_path = self.index_dir / "descriptor_names.txt"
        self.faiss_path = self.index_dir / "megaloc.faiss"

    def has_database(self) -> bool:
        return self.h5_path.exists() and self.h5_path.stat().st_size > 0

    def append_from_hloc_hdf5(self, src_h5: Path) -> int:
        """Copy descriptors produced by hloc extract_features into the world store."""
        if h5py is None:
            raise RuntimeError("h5py is required to maintain the world descriptor store")
        src_h5 = Path(src_h5)
        added = 0
        existing = set()
        if self.names_path.exists():
            existing = set(self.names_path.read_text().splitlines())
        with h5py.File(src_h5, "r") as src, h5py.File(self.h5_path, "a") as dst:
            for name in src.keys():
                if name in existing or name in dst:
                    continue
                src.copy(src[name], dst, name=name)
                existing.add(name)
                added += 1
        self.names_path.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""))
        return added

    def _load_matrix(self) -> Tuple[List[str], np.ndarray]:
        if h5py is None:
            raise RuntimeError("h5py is required to query the world descriptor store")
        names: List[str] = []
        vecs: List[np.ndarray] = []
        with h5py.File(self.h5_path, "r") as handle:
            for name in handle.keys():
                group = handle[name]
                if "global_descriptor" in group:
                    desc = np.asarray(group["global_descriptor"], dtype=np.float32).reshape(-1)
                else:
                    desc = np.asarray(group, dtype=np.float32).reshape(-1)
                names.append(name)
                vecs.append(desc)
        if not vecs:
            return [], np.zeros((0, 1), dtype=np.float32)
        mat = np.stack(vecs, axis=0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return names, mat / norms

    def query(self, query_h5: Path, top_k: int = 32) -> List[Tuple[str, str, float]]:
        """Return (query_name, db_name, score) pairs."""
        if h5py is None:
            raise RuntimeError("h5py is required to query the world descriptor store")
        db_names, db = self._load_matrix()
        if db.shape[0] == 0:
            return []
        out: List[Tuple[str, str, float]] = []
        with h5py.File(query_h5, "r") as handle:
            for qname in handle.keys():
                group = handle[qname]
                if "global_descriptor" in group:
                    q = np.asarray(group["global_descriptor"], dtype=np.float32).reshape(-1)
                else:
                    q = np.asarray(group, dtype=np.float32).reshape(-1)
                q = q / (np.linalg.norm(q) + 1e-8)
                scores = db @ q
                k = min(top_k, scores.shape[0])
                idx = np.argpartition(-scores, kth=k - 1)[:k]
                idx = idx[np.argsort(-scores[idx])]
                for i in idx:
                    out.append((qname, db_names[int(i)], float(scores[int(i)])))
        return out

    def write_pairs_file(self, pairs: Sequence[Tuple[str, str, float]], dest: Path, min_score: float = 0.0) -> int:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        seen = set()
        for a, b, score in pairs:
            if score < min_score or a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{a} {b}\n")
        dest.write_text("".join(lines))
        return len(lines)

    def append_camera_record(self, record: dict) -> None:
        cameras = self.index_dir / "cameras.jsonl"
        with cameras.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
