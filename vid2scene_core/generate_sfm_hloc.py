import multiprocessing
from pathlib import Path
import shutil
import argparse
import logging
from typing import Any, Dict, Optional
import pycolmap
import torch
import gc
from run_command import run_command

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    triangulation,
    pairs_from_retrieval
)

try:
    from world.layout import WorldLayout
    from world.index import WorldDescriptorIndex
    from world.integrate import merge_pair_files
except ImportError:
    WorldLayout = None  # type: ignore
    WorldDescriptorIndex = None  # type: ignore
    merge_pair_files = None  # type: ignore


def custom_estimation_and_geometric_verification(database_path: Path, pairs_path: Path, verbose: bool = False):
    # launch colmap verification using colmap subprocess. This is because pycolmap doesn't support GPU for verify_geometry.
    run_command(["colmap", "matches_importer",
                 "--database_path", str(database_path),
                 "--match_list_path", str(pairs_path),
                 "--TwoViewGeometry.min_inlier_ratio", "0.1",
                 "--TwoViewGeometry.max_num_trials", "20000"])

triangulation.estimation_and_geometric_verification = custom_estimation_and_geometric_verification


def custom_import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    image_list: Optional[list] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """Custom import_images function that properly converts arguments for pycolmap."""
    logger.info("Importing images into the database...")
    if options is None:
        options = {}

    image_reader_options = pycolmap.ImageReaderOptions()
    for key, value in options.items():
        if hasattr(image_reader_options, key):
            setattr(image_reader_options, key, value)

    images = list(image_dir.iterdir())
    if len(images) == 0:
        raise IOError(f"No images found in {image_dir}.")

    with pycolmap.ostream():
        pycolmap.import_images(
            str(database_path),
            str(image_dir),
            camera_mode,
            options=image_reader_options,
        )

import hloc.reconstruction
hloc.reconstruction.import_images = custom_import_images

def custom_run_reconstruction(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None
):
    models_path = sfm_dir / "sparse"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction...")
    if options is None:
        options = {}
    options = {"num_threads": min(multiprocessing.cpu_count(), 16), **options}
    with triangulation.OutputCapture(verbose):
        with pycolmap.ostream():
            reconstructions = pycolmap.incremental_mapping(
                str(database_path), str(image_dir), str(models_path), options=options
            )
    if len(reconstructions) == 0:
        logger.error("Could not reconstruct any model!")
        return None
    logger.info(f"Reconstructed {len(reconstructions)} model(s).")
    return reconstructions[0]


hloc.reconstruction.run_reconstruction = custom_run_reconstruction

logger = logging.getLogger(__name__)


def _intra_clip_pairs(image_dir: Path, output_dir: Path, num_matched: int = 32) -> Path:
    retrieval_conf = extract_features.confs["eigenplaces"]
    sfm_pairs = output_dir / "pairs-eigenplaces.txt"
    logger.info("Doing intra-clip retrieval (EigenPlaces)")
    retrieval_path = extract_features.main(retrieval_conf, image_dir, output_dir)
    logger.info("Doing intra-clip pairs")
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_matched)
    return sfm_pairs


def _world_pairs(image_dir: Path, output_dir: Path, world_dir: Path, num_matched: int = 32) -> Optional[Path]:
    if WorldLayout is None or WorldDescriptorIndex is None:
        logger.warning("world package missing; skipping MegaLoc world retrieval")
        return None
    world = WorldLayout(world_dir)
    world.ensure()
    index = WorldDescriptorIndex(world.index_dir)
    if not index.has_database():
        logger.warning("World MegaLoc index is empty; integrate will rely on intra-clip pairs only")
        return None
    retrieval_conf = extract_features.confs.get("megaloc")
    if retrieval_conf is None:
        logger.warning("hloc has no megaloc conf; falling back to eigenplaces for world retrieval")
        retrieval_conf = extract_features.confs["eigenplaces"]
    logger.info("Doing world retrieval (%s)", retrieval_conf.get("output", "megaloc"))
    query_h5 = extract_features.main(retrieval_conf, image_dir, output_dir)
    pairs = index.query(query_h5, top_k=num_matched)
    dest = output_dir / "pairs-world-megaloc.txt"
    index.write_pairs_file(pairs, dest)
    return dest


def _ingest_world_descriptors(image_dir: Path, output_dir: Path, world_dir: Path) -> None:
    if WorldLayout is None or WorldDescriptorIndex is None:
        return
    world = WorldLayout(world_dir)
    world.ensure()
    index = WorldDescriptorIndex(world.index_dir)
    retrieval_conf = extract_features.confs.get("megaloc") or extract_features.confs["eigenplaces"]
    h5 = extract_features.main(retrieval_conf, image_dir, output_dir)
    added = index.append_from_hloc_hdf5(h5)
    logger.info("Appended %s descriptors to world index %s", added, world.index_dir)


def run_sfm(
    image_dir,
    output_dir,
    kill_check=None,
    reconstruction_method="glomap",
    mode="bootstrap",
    world_dir=None,
):
    """SfM for a clip.

    mode="bootstrap": stock Vid2Scene (EigenPlaces pairs + GLOMAP/COLMAP).
    mode="integrate": also retrieve against the world MegaLoc index and register
    new images into world_dir/sparse when that model exists.
    """
    logger.info("Running HLOC SfM pipeline mode=%s method=%s", mode, reconstruction_method)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_conf = extract_features.confs["aliked-n16"]
    matcher_conf = match_features.confs["aliked+lightglue"]
    sfm_dir = output_dir

    clip_pairs = _intra_clip_pairs(image_dir, output_dir)
    if kill_check and kill_check():
        logger.info("Job was deleted after pairs, stopping")
        return None

    sfm_pairs = clip_pairs
    if mode == "integrate" and world_dir is not None:
        world_pairs = _world_pairs(image_dir, output_dir, Path(world_dir))
        if world_pairs is not None and merge_pair_files is not None:
            merged = output_dir / "pairs-merged.txt"
            n = merge_pair_files(clip_pairs, world_pairs, dest=merged)
            logger.info("Merged %s intra-clip + world pairs", n)
            sfm_pairs = merged

    logger.info("Doing features")
    feature_path = extract_features.main(feature_conf, image_dir, output_dir)
    logger.info("Doing matches")
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf["output"], output_dir)

    if kill_check and kill_check():
        logger.info("Job was deleted after retrieving, stopping")
        return None

    sparse_dir = sfm_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(exist_ok=True, parents=True)

    world_sparse = None
    if world_dir is not None:
        world_sparse = Path(world_dir) / "sparse" / "0"

    if reconstruction_method == "glomap":
        database = sfm_dir / "database.db"
        camera_mode = pycolmap.CameraMode.SINGLE
        reconstruction.create_empty_db(database)
        custom_import_images(image_dir, database, camera_mode, None, None)
        image_ids = reconstruction.get_image_ids(database)
        reconstruction.import_features(image_ids, database, feature_path)
        reconstruction.import_matches(
            image_ids,
            database,
            sfm_pairs,
            match_path,
            None,
            False
        )
        triangulation.estimation_and_geometric_verification(database, sfm_pairs, True)

        if mode == "integrate" and world_sparse is not None and world_sparse.exists():
            logger.info("Integrating into existing world model at %s", world_sparse)
            dest = sparse_dir / "0"
            dest.mkdir(parents=True, exist_ok=True)
            run_command([
                "colmap", "image_registrator",
                "--database_path", str(database),
                "--input_path", str(world_sparse),
                "--output_path", str(dest),
            ], kill_check=kill_check)
        else:
            run_command(["glomap", "mapper",
                         "--database_path", str(database),
                         "--image_path", str(image_dir),
                         "--output_path", str(sparse_dir),
                         "--ba_iteration_num", "5",
                         "--skip_pruning", "0",
                         "--GlobalPositioning.max_num_iterations", "300",
                         "--BundleAdjustment.max_num_iterations", "500",
                         "--Thresholds.max_epipolar_error_E=0.5",
                         "--Thresholds.max_epipolar_error_F=1.5",
                         "--Thresholds.max_epipolar_error_H=1.5",
                         "--Thresholds.min_inlier_num=50",
                         "--Thresholds.min_inlier_ratio=0.4",
                         "--Thresholds.max_rotation_error=5"
                        ], kill_check=kill_check)
            run_command(["colmap", "image_registrator",
                        "--database_path", str(database),
                        "--input_path",  str(sparse_dir / "0"),
                        "--output_path", str(sparse_dir / "0")], kill_check=kill_check)
    else:
        incremental_mapper_options = pycolmap.IncrementalMapperOptions()
        incremental_mapper_options.init_min_tri_angle = 5
        incremental_pipeline_options = {"mapper": incremental_mapper_options}
        reconstruction.main(sfm_dir, image_dir, sfm_pairs, feature_path, match_path, mapper_options=incremental_pipeline_options)

    if world_dir is not None:
        try:
            _ingest_world_descriptors(image_dir, output_dir, Path(world_dir))
        except Exception:
            logger.exception("Failed to append descriptors to world index")
        world_root = Path(world_dir)
        world_sparse_parent = world_root / "sparse"
        world_sparse_parent.mkdir(parents=True, exist_ok=True)
        produced = sparse_dir / "0"
        if produced.exists() and mode == "bootstrap":
            dest = world_sparse_parent / "0"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(produced, dest)
            logger.info("Wrote bootstrap world model to %s", dest)
        elif produced.exists() and mode == "integrate":
            dest = world_sparse_parent / "0"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(produced, dest)
            logger.info("Updated world model from integrate at %s", dest)

    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    return str(sparse_dir)


def main():
    parser = argparse.ArgumentParser(
        description="3D SfM pointmap generation using COLMAP CLI."
    )
    parser.add_argument("image_dir", help="Directory containing the image frames.")
    parser.add_argument("output_dir", help="Directory to store the output SfM model.")
    parser.add_argument("--mode", choices=["bootstrap", "integrate"], default="bootstrap")
    parser.add_argument("--world-dir", default=None, help="Persistent world root (enables descriptor ingest / integrate).")
    parser.add_argument("--reconstruction-method", default="glomap")

    args = parser.parse_args()

    model = run_sfm(
        args.image_dir,
        args.output_dir,
        kill_check=None,
        reconstruction_method=args.reconstruction_method,
        mode=args.mode,
        world_dir=args.world_dir,
    )

    if model:
        logger.info(f"3D pointmap generation completed. Model saved to: {model}")
    else:
        logger.warning("No model generated.")


if __name__ == "__main__":
    main()
