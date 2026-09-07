"""CLI entrypoint: python -m vid2scene_core VIDEO -o OUTDIR

Runs the local pipeline without Django / Redis / Postgres / Azurite.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _core_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _core_dir().parent


def _ensure_import_path() -> None:
    """vid2scene.py uses sibling imports; put package dir on sys.path like the worker."""
    core = str(_core_dir())
    if core not in sys.path:
        sys.path.insert(0, core)


def resolve_gsplat_script() -> str:
    """Return GSPLAT_SCRIPT, auto-resolving repo-relative paths if unset.

    Worker_Dockerfile sets GSPLAT_SCRIPT=/app/gsplat/examples/simple_trainer.py
    """
    existing = os.environ.get("GSPLAT_SCRIPT")
    if existing:
        path = Path(existing).expanduser()
        if path.is_file():
            return str(path.resolve())
        logging.getLogger(__name__).warning(
            "GSPLAT_SCRIPT=%s missing on disk; trying defaults", existing
        )

    candidates = [
        _repo_root() / "gsplat" / "examples" / "simple_trainer.py",
        Path("/app/gsplat/examples/simple_trainer.py"),
        Path.cwd() / "gsplat" / "examples" / "simple_trainer.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            resolved = str(candidate.resolve())
            os.environ["GSPLAT_SCRIPT"] = resolved
            logging.getLogger(__name__).info("Auto-resolved GSPLAT_SCRIPT=%s", resolved)
            return resolved

    raise SystemExit(
        "GSPLAT_SCRIPT is not set and could not be auto-resolved.\n"
        "Expected gsplat/examples/simple_trainer.py under the repo root\n"
        "(git submodule update --init --recursive), or export GSPLAT_SCRIPT=..."
    )


def _spz_convert_path() -> Path:
    return (_repo_root() / "spz" / "build_native" / "bin" / "spz_convert").resolve()


def _maybe_prune_and_spz(ply_path: str, logger: logging.Logger) -> dict:
    """Best-effort prune + SPZ into outdir/ply/. Never fails the run."""
    import vid2scene as v2s

    results = {"ply": ply_path, "ply_pruned": None, "spz": None}
    spz_bin = _spz_convert_path()
    if not spz_bin.is_file():
        logger.info(
            "spz_convert not found at %s - skipping prune/SPZ. PLY still available.",
            spz_bin,
        )
        return results

    v2s.SPZ_TO_PLY_EXECUTABLE_PATH = str(spz_bin)
    ply = Path(ply_path)
    pruned = ply.with_name(ply.stem + "_pruned.ply")
    try:
        ok = v2s.prune_ply(str(ply), str(pruned), alpha_prune_threshold=12)
        if ok and pruned.is_file():
            results["ply_pruned"] = str(pruned)
            spz_path = pruned.with_suffix(".spz")
            ok_spz = v2s.convert_ply_to_spz(str(pruned), str(spz_path))
            if ok_spz and spz_path.is_file():
                results["spz"] = str(spz_path)
            else:
                logger.warning("PLY->SPZ did not produce %s", spz_path)
        else:
            logger.warning("PLY prune did not produce %s", pruned)
    except Exception as exc:
        logger.warning("Post-process prune/SPZ skipped: %s", exc)
    return results


def _pilgram_choices():
    try:
        from apply_pilgram import list_available_filters
        return list_available_filters()
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vid2scene",
        description=(
            "Convert a video to a 3D Gaussian splat locally "
            "(no web / Django / redis / postgres / azurite)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "video", nargs="?",
        help="Input video path. Optional if --image-dir/--sfm-dir/--quest-project-dir set.",
    )
    parser.add_argument("-o", "--output", dest="output",
                        help="Output directory. Default: ./vid2scene_out_<stem>")
    parser.add_argument("--image-dir", dest="image_dir",
                        help="Use existing images instead of extracting frames.")
    parser.add_argument("--sfm-dir", dest="sfm_dir",
                        help="Precomputed SfM model dir (must contain images/).")
    parser.add_argument("--target-framecount", type=int, default=600,
                        help="Target frames to extract from video.")
    parser.add_argument("--reconstruction-method",
                        choices=["glomap", "colmap", "vggt", "quest"], default="glomap",
                        help="SfM / reconstruction backend.")
    parser.add_argument("--training-num-steps", type=int, default=30_000,
                        help="gsplat training steps.")
    parser.add_argument("--training-max-num-gaussians", type=int, default=1_000_000,
                        help="Max Gaussians during gsplat MCMC training.")
    parser.add_argument("--equirectangular", action="store_true",
                        help="Treat input as equirectangular / 360 video.")
    parser.add_argument("--remove-background", action="store_true",
                        help="Remove image backgrounds before SfM.")
    parser.add_argument("--use-background-sphere", action="store_true",
                        help="Inject fibonacci background sphere into SfM points.")
    filter_choices = _pilgram_choices()
    parser.add_argument("--apply-pilgram-filter",
                        choices=filter_choices if filter_choices else None,
                        metavar="FILTER" if not filter_choices else None,
                        help="Optional Pilgram color filter for frames.")
    parser.add_argument("--quest-project-dir",
                        help="Quest project dir (required for reconstruction-method quest).")
    parser.add_argument("--apriltag-size", type=float,
                        help="AprilTag outer-border size in meters for scale calibration.")
    parser.add_argument("--mock", action="store_true",
                        help="Skip pipeline; copy a tiny mock splat.ply.")
    parser.add_argument("--gsplat-script",
                        help="Override path to gsplat examples/simple_trainer.py.")
    parser.add_argument("--skip-postprocess", action="store_true",
                        help="Do not attempt prune / SPZ after training.")
    return parser


def _default_output_dir(video):
    stem = Path(video).stem if video else "scene"
    stem = stem or "scene"
    return Path.cwd() / f"vid2scene_out_{stem}"


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger("vid2scene")

    _ensure_import_path()
    import vid2scene as v2s

    args = build_parser().parse_args(argv)

    if not args.video and not args.image_dir and not args.sfm_dir and not args.quest_project_dir:
        build_parser().error(
            "provide a video path, or --image-dir / --sfm-dir / --quest-project-dir"
        )
    if args.video and not Path(args.video).is_file():
        build_parser().error(f"video not found: {args.video}")

    output_dir = Path(args.output) if args.output else _default_output_dir(args.video)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gsplat_script:
        os.environ["GSPLAT_SCRIPT"] = str(Path(args.gsplat_script).expanduser().resolve())
    gsplat_script = resolve_gsplat_script()
    logger.info("Using GSPLAT_SCRIPT=%s", gsplat_script)
    logger.info("Output directory: %s", output_dir.resolve())

    ply_path = v2s.process_video_to_scene(
        video_path=args.video,
        image_dir=args.image_dir,
        output_dir=str(output_dir),
        sfm_dir=args.sfm_dir,
        target_framecount=args.target_framecount,
        preview_data_handler=None,
        remove_background_from_images=args.remove_background,
        equirectangular=args.equirectangular,
        use_background_sphere=args.use_background_sphere,
        apply_pilgram_filter_name=args.apply_pilgram_filter,
        training_max_num_gaussians=args.training_max_num_gaussians,
        training_num_steps=args.training_num_steps,
        kill_check=None,
        reconstruction_method=args.reconstruction_method,
        apriltag_size_meters=args.apriltag_size,
        mock=args.mock,
        quest_project_dir=args.quest_project_dir,
    )

    if not ply_path:
        logger.error("Pipeline returned no PLY path (terminated early?).")
        return 1

    artifacts = {"ply": ply_path, "ply_pruned": None, "spz": None}
    if not args.skip_postprocess and not args.mock:
        artifacts = _maybe_prune_and_spz(ply_path, logger)

    print()
    print("=" * 60)
    print("vid2scene - done")
    print("=" * 60)
    print(f"  output dir : {output_dir.resolve()}")
    print(f"  sfm        : {(output_dir / 'sfm_output').resolve()}")
    print(f"  results    : {(output_dir / 'results').resolve()}")
    print(f"  preview    : {(output_dir / 'preview_data').resolve()}")
    print(f"  splat PLY  : {artifacts['ply']}")
    if artifacts.get("ply_pruned"):
        print(f"  pruned PLY : {artifacts['ply_pruned']}")
    if artifacts.get("spz"):
        print(f"  splat SPZ  : {artifacts['spz']}")
    else:
        print("  splat SPZ  : (not produced - build spz/ or install spz_convert)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
