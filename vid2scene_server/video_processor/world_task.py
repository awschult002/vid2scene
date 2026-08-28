"""RQ entrypoint that binds a job to a persistent world, then runs the stock worker."""

import logging

from .models import SceneProcessingJob
from .tasks import process_video_task as stock_process_video_task
from .world_store import apply_job_world_env, persist_job_world

logger = logging.getLogger(__name__)


def process_video_task(scene_processing_job_id):
    world_dir = None
    try:
        spj = SceneProcessingJob.objects.get(id=scene_processing_job_id)
    except SceneProcessingJob.DoesNotExist:
        logger.error("Job %s missing before world hook", scene_processing_job_id)
        raise

    try:
        world_dir = apply_job_world_env(spj)
        return stock_process_video_task(scene_processing_job_id)
    finally:
        try:
            persist_job_world(spj, world_dir)
        except Exception:
            logger.exception("Failed to persist world for job %s", scene_processing_job_id)
