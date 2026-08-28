# World-scale plan (branch `world-scale`)

Vid2Scene stays the **clip worker**. The system of record is a persistent `world/` store so a house can grow into a neighborhood without a second architecture.

## Decisions

- One world frame. Jobs are captures into that world (`bootstrap` or `integrate`).
- Intra-clip pair mining stays **EigenPlaces + ALIKED + LightGlue + GLOMAP** (stock Vid2Scene).
- World retrieval for a *new* capture uses **MegaLoc** (`hloc` conf `megaloc`), then the same local matcher and `colmap image_registrator`.
- Trainer stays **gsplat MCMC**. FastGS is a later optional cell backend, not a Phase 0 swap.
- Densify only cells that new close-up cameras see. Do not run one global densification over the whole world.

## Layout

```
worlds/{world_id}/
  control/
  index/                 # MegaLoc descriptors + cameras.jsonl + FAISS when available
  cells/{ix}_{iy}_{iz}/  # landmarks, gaussians, cameras
  captures/{capture_id}/ # original frames / video
  sparse/                # living COLMAP model for the world (bootstrap + integrates)
```

On-disk helpers: `vid2scene_core/world/`.

## Phases

| Phase | Status on this branch | Proof |
|---|---|---|
| 0 Stock clip | Unchanged default `run_sfm()` | One video still produces `sparse/` + splat |
| 1 World prefix | `WorldLayout` + capture/cell paths | First clip written under `worlds/{id}/` |
| 2 Integrate | `run_sfm(..., mode="integrate", world_dir=...)` | Second clip registers without wiping sparse |
| 3 Cell gsplat | `cell_id_for_xyz` + scoped export (next) | Fridge orbit densifies one cell |
| 4 Viewer | Still one PLY/SPZ (merge cells as stopgap) | Walk yard to fitting |
| 5 Multi-sensor | Same integrate path for drone / second phone | Roof + interior in one world |

## How to run (worker)

Bootstrap (empty world, same as today plus world paths):

```python
from world.layout import WorldLayout
from generate_sfm_hloc import run_sfm

world = WorldLayout("/data/worlds/house")
world.ensure()
run_sfm(image_dir, world.sparse_dir, reconstruction_method="glomap", mode="bootstrap")
```

Integrate (world already has cameras):

```python
run_sfm(
    image_dir,
    world.capture_dir(capture_id) / "sfm",
    reconstruction_method="glomap",
    mode="integrate",
    world_dir=world.root,
)
```

Do not point `mode="integrate"` at an empty world. Bootstrap first.
