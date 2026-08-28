"""Resolve and sync a persistent world map for the GPU worker.

The job workspace is deleted after each run. The world must outlive that.
Local source of truth: WORLD_ROOT/{world_id}/ (default /data/worlds).
Optional mirror in Django storage under worlds/{world_id}/ so another worker
can pull sparse + descriptors if the local disk is empty.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from django.core.files.storage import default_storage

logger = logging.getLogger(__name__)

WORLD_RELATIVE_FILES = (
    "sparse/0/cameras.bin",
    "sparse/0/images.bin",
    "sparse/0/points3D.bin",
    "sparse/0/project.ini",
    "index/global-feats-megaloc.h5",
    "index/descriptor_names.txt",
    "index/cameras.jsonl",
    "index/megaloc.faiss",
)


def world_root() -> Path:
    return Path(os.environ.get("WORLD_ROOT", "/data/worlds"))


def local_world_dir(world_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in world_id).strip("-._")
    if not safe:
        raise ValueError("world_id is empty after sanitizing")
    path = world_root() / safe
    path.mkdir(parents=True, exist_ok=True)
    (path / "sparse").mkdir(exist_ok=True)
    (path / "index").mkdir(exist_ok=True)
    (path / "captures").mkdir(exist_ok=True)
    (path / "cells").mkdir(exist_ok=True)
    (path / "control").mkdir(exist_ok=True)
    return path


def world_has_model(world_dir: Path) -> bool:
    sparse0 = world_dir / "sparse" / "0"
    return (sparse0 / "images.bin").exists() or (sparse0 / "images.txt").exists()


def resolve_mode(requested: Optional[str], world_dir: Path) -> str:
    requested = (requested or "").strip().lower()
    if requested in ("bootstrap", "integrate"):
        if requested == "integrate" and not world_has_model(world_dir):
            logger.warning("integrate requested but world %s has no sparse model; using bootstrap", world_dir)
            return "bootstrap"
        return requested
    return "integrate" if world_has_model(world_dir) else "bootstrap"


def pull_world_from_storage(world_id: str, world_dir: Path) -> int:
    prefix = f"worlds/{world_id}"
    pulled = 0
    for rel in WORLD_RELATIVE_FILES:
        storage_name = f"{prefix}/{rel}"
        dest = world_dir / rel
        if dest.exists():
            continue
        if not default_storage.exists(storage_name):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with default_storage.open(storage_name, "rb") as src, dest.open("wb") as out:
                out.write(src.read())
            pulled += 1
            logger.info("Pulled world file %s -> %s", storage_name, dest)
        except Exception:
            logger.exception("Failed to pull %s", storage_name)
    return pulled


def push_world_to_storage(world_id: str, world_dir: Path) -> int:
    prefix = f"worlds/{world_id}"
    pushed = 0
    for rel in WORLD_RELATIVE_FILES:
        src = world_dir / rel
        if not src.exists() or not src.is_file():
            continue
        storage_name = f"{prefix}/{rel}"
        try:
            if default_storage.exists(storage_name):
                default_storage.delete(storage_name)
            with src.open("rb") as handle:
                default_storage.save(storage_name, handle)
            pushed += 1
            logger.info("Pushed world file %s", storage_name)
        except Exception:
            logger.exception("Failed to push %s", storage_name)
    return pushed


def apply_job_world_env(spj) -> Optional[Path]:
    """Set V2S_WORLD_* so generate_sfm_hloc.run_sfm sees this job's world.

    Isolated jobs (no world_id) clear the env so leftover state cannot leak.
    """
    world_id = getattr(spj, "world_id", None)
    if not world_id:
        os.environ.pop("V2S_WORLD_DIR", None)
        os.environ.pop("V2S_WORLD_MODE", None)
        os.environ.pop("V2S_CAPTURE_ID", None)
        return None
    world_dir = local_world_dir(world_id)
    pulled = pull_world_from_storage(world_id, world_dir)
    mode = resolve_mode(getattr(spj, "world_mode", ""), world_dir)
    capture_id = getattr(spj, "capture_id", None) or str(spj.id)
    os.environ["V2S_WORLD_DIR"] = str(world_dir)
    os.environ["V2S_WORLD_MODE"] = mode
    os.environ["V2S_CAPTURE_ID"] = str(capture_id)
    logger.info(
        "World hook world_id=%s mode=%s dir=%s pulled=%s capture=%s",
        world_id, mode, world_dir, pulled, capture_id,
    )
    return world_dir


def persist_job_world(spj, world_dir: Optional[Path]) -> None:
    if not getattr(spj, "world_id", None) or world_dir is None:
        return
    pushed = push_world_to_storage(spj.world_id, Path(world_dir))
    logger.info("Persisted world %s (%s files)", spj.world_id, pushed)
