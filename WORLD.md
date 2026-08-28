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
WORLD_ROOT/{world_id}/          # default WORLD_ROOT=/data/worlds (worker disk)
  control/
  index/                        # MegaLoc descriptors
  cells/{ix}_{iy}_{iz}/
  captures/{capture_id}/
  sparse/0/                     # living COLMAP model

blob: worlds/{world_id}/sparse/0/* and index/*   # worker pull/push mirror
```

On-disk helpers: `vid2scene_core/world/`.
Worker hook: `video_processor.world_task.process_video_task`.

## Worker hook

1. Set `world_id` (and optional `world_mode`, `capture_id`) on `SceneProcessingJob`.
2. RQ runs `video_processor.world_task.process_video_task`.
3. That sets `V2S_WORLD_DIR` / `V2S_WORLD_MODE`, pulls `worlds/{id}/` from blob if the local dir is empty, then calls the stock `tasks.process_video_task`.
4. `generate_sfm_hloc.run_sfm` reads those env vars. Isolated jobs leave `world_id` blank and get the original clip pipeline.
5. After the job, sparse + index are pushed back to blob storage.

```bash
# worker disk
export WORLD_ROOT=/data/worlds
# migrate
python manage.py migrate video_processor
```

Admin: set World ID on a job to `house`, leave mode blank (auto). First job bootstraps. Second job with the same World ID integrates.

## Phases

| Phase | Status on this branch | Proof |
|---|---|---|
| 0 Stock clip | Unchanged when `world_id` is empty | One video still produces `sparse/` + splat |
| 1 World prefix | `WorldLayout` + worker hook | First clip written under `WORLD_ROOT/{id}/` |
| 2 Integrate | `run_sfm` + MegaLoc world pairs | Second clip registers without wiping sparse |
| 3 Cell gsplat | `cell_id_for_xyz` only | Fridge orbit densifies one cell |
| 4 Viewer | Still one PLY/SPZ | Walk yard to fitting |
| 5 Multi-sensor | Same integrate path | Roof + interior in one world |
