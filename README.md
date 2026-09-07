# vid2scene

Turn a phone video into a 3D Gaussian-Splat scene. Open-source self-hostable version of [vid2scene.com](https://vid2scene.com).

**Primary path: local CLI** - one command, no web stack.

```bash
./scripts/vid2scene /path/to/video.mp4 -o /path/to/outdir
# or:
python -m vid2scene_core /path/to/video.mp4 -o /path/to/outdir
```

An optional Docker Compose web UI (Django + Redis + Postgres + Azurite + GPU worker) remains in the repo for the full browser experience - see [Optional: full web UI](#optional-full-web-ui). **`vid2scene_server/` was not deleted**; the CLI is simply the recommended happy path.

- **Apache-2.0** licensed
- Default SfM is GLOMAP (no model weights / no Hugging Face token)
- Optional VGGT / SAM3 paths need an HF token (gated models)

---

## CLI local usage (recommended)

### 1. Clone and submodules

```bash
git clone https://github.com/awschult002/vid2scene.git
cd vid2scene
git submodule update --init --recursive
chmod +x scripts/vid2scene
```

Submodules needed for a full run include at least `gsplat`, `glomap`, `Hierarchical-Localization`, and (for SPZ export) `spz`.

### 2. GPU / conda environment

The pipeline needs an NVIDIA GPU, CUDA-capable PyTorch, COLMAP, and the Python deps used by the worker image.

Easiest parity with Docker:

```bash
# Conda env used by Worker_Dockerfile
conda env create -f env_cuda_colmap.yml
conda activate env_cuda_colmap

# Python deps (same set the worker installs - see Worker_Dockerfile)
pip install -r requirements_worker.txt
pip install -r gsplat/examples/requirements.txt
pip install -e ./gsplat
pip install -e ./Hierarchical-Localization
# Optional: vggt/, sam3/ if you use those reconstruction paths
```

Alternatively, build/run inside the existing [`Worker_Dockerfile`](Worker_Dockerfile) image and invoke the CLI there - it already compiles glomap, gsplat, and `spz_convert`.

```bash
nvidia-smi   # host driver must work first
```

### 3. GSPLAT_SCRIPT

gsplat training is launched via `examples/simple_trainer.py`. The worker image sets:

```text
GSPLAT_SCRIPT=/app/gsplat/examples/simple_trainer.py
```

**Locally you usually do not need to set it.** The CLI auto-resolves, in order:

1. `$GSPLAT_SCRIPT` if set and the file exists
2. `<repo>/gsplat/examples/simple_trainer.py`
3. `/app/gsplat/examples/simple_trainer.py` (inside the worker container)
4. `./gsplat/examples/simple_trainer.py` relative to cwd

Override explicitly if needed:

```bash
export GSPLAT_SCRIPT="/absolute/path/to/gsplat/examples/simple_trainer.py"
# or:
python -m vid2scene_core video.mp4 -o out --gsplat-script /path/to/simple_trainer.py
```

### 4. Run

```bash
./scripts/vid2scene /path/to/video.mp4 -o /path/to/outdir
python -m vid2scene_core /path/to/video.mp4 -o /path/to/outdir
```

If `-o` / `--output` is omitted, output goes to `./vid2scene_out_<video-stem>`.

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--target-framecount` | `600` | Frames extracted from the video |
| `--reconstruction-method` | `glomap` | `glomap` / `colmap` / `vggt` / `quest` |
| `--training-num-steps` | `30000` | gsplat steps |
| `--training-max-num-gaussians` | `1000000` | MCMC cap |
| `--equirectangular` | off | 360 / equirect input |
| `--remove-background` | off | Background removal before SfM |
| `--mock` | off | Skip pipeline; write a tiny mock PLY |
| `--skip-postprocess` | off | Skip prune / SPZ after training |

```bash
python -m vid2scene_core --help
```

### 5. Output layout

After success, `outdir` holds temporary scene data and the final splat(s):

```text
outdir/
  sfm_output/          # frames + SfM sparse model
  results/             # gsplat training run artifacts
  preview_data/        # training preview frames / camera JSON
  ply/
    splat.ply          # full trained splat
    splat_pruned.ply   # alpha-pruned (when spz_convert is available)
    splat_pruned.spz   # SPZ export (when spz_convert is available)
```

Prune + SPZ are **best-effort**: if `spz/build_native/bin/spz_convert` is missing, the run still succeeds with `ply/splat.ply`. Build the converter with `spz/build_native.sh` (see `Worker_Dockerfile`) to enable them.

### Optional Hugging Face token

Only needed for `reconstruction_method=vggt` or SAM3-based panorama background removal. Agree to each gated model license on huggingface.co, then:

```bash
export HF_TOKEN=<your-huggingface-token>
```

Default `glomap` needs no token.

---

## Optional: full web UI

The Django web app, RQ worker, Redis, Postgres, and Azurite stack remain in-tree for the browser upload -> process -> viewer flow. **Nothing under `vid2scene_server/` was removed** in the CLI work; treat it as optional.

### Architecture

```
   browser
     |  HTTP + direct SAS to blob
     v
   web (Django + gunicorn :8000)
     | enqueue (RQ) / DB / blob I/O
     v
   redis + postgres + azurite
     |
     v
   worker (GPU)  -- runs vid2scene_core.process_video_to_scene
```

The ML pipeline lives in [`vid2scene_core/`](vid2scene_core/). The worker downloads from blob storage, calls `process_video_to_scene`, then uploads PLY/SPZ/SOG. The local CLI bypasses that glue and writes straight to disk.

### Setup (Docker Compose)

**Hardware:** Linux (or Windows + WSL2) with an NVIDIA GPU, Docker Engine 24+ with Compose, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

git submodule update --init --recursive
cp .env.example .env    # optional knobs
docker compose up --build
```

Open http://localhost:8000 (default `admin` / `admin`). Services use `network_mode: host` so Azurite SAS URLs work for browser and worker alike.

See [`.env.example`](.env.example) for `SECRET_KEY`, `HF_TOKEN`, `BILLING_ENABLED`, Postgres credentials, etc.

### Django admin / frontend dev

- Admin: `/admin/`
- Bare-metal Django: `cd vid2scene_server && pip install -r requirements.txt && ENVIRONMENT=development python manage.py runserver`
- Viewer (Svelte/Vite): `cd vid2scene_server/viewer && npm install && npm run dev`

---

## Project layout

```
vid2scene_core/        - ML pipeline + local CLI (python -m vid2scene_core)
  __main__.py          - Friendly CLI entrypoint
  vid2scene.py         - process_video_to_scene(...) + legacy argparse
scripts/vid2scene      - Thin launcher -> python -m vid2scene_core
vid2scene_server/      - Optional Django web app (REST API, viewer, RQ tasks)
Worker_Dockerfile      - GPU worker image (CUDA + COLMAP/glomap/gsplat/spz)
Web_Dockerfile         - Django web image
docker-compose.yaml    - Optional full self-host web stack
env_cuda_colmap.yml    - Conda env for COLMAP / CUDA tooling
requirements_worker.txt
```

Submodules (vendored at clone time): glomap, Hierarchical-Localization, spz, sam3, gsplat, vggt, ply-to-sog, 3dgs-autolod, quest-3d-reconstruction, and others. See [NOTICE](NOTICE) for attribution.

---

## Security

The self-host **web** build is meant for a single trusted machine (DEBUG on, insecure defaults). The production hosted-service hardening is not included. The **CLI path** never starts Django, Redis, Postgres, or Azurite.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE).
