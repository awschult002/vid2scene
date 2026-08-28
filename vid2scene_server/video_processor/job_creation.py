"""
Core job creation and upload functionality for vid2scene.

This module contains the shared logic for:
- Generating Azure blob storage SAS URLs for video uploads
- Creating and enqueueing scene processing jobs
- Handling differences between web API and dev API workflows
"""

import datetime
import re
import logging
import django_rq
from django.conf import settings
from django.core.mail import send_mail
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

from .models import SceneProcessingJob
from .utils import validate_user_settings
from subscriptions.utils import get_subscription_tier_string, user_can_generate_premium_scene
from subscriptions.models import SubscriptionTier, CreditTransaction

logger = logging.getLogger(__name__)

RQ_PROCESS_FUNC = "video_processor.world_task.process_video_task"


def validate_blob(blob_name, max_age_minutes=60, max_size_bytes=None):
    """
    Check if blob was created within the last N minutes and is under the size limit.
    Returns (True, None) if valid, (False, error_message) if invalid.
    Uses UTC timestamps so daylight savings doesn't affect it.
    """
    try:
        from datetime import timedelta, timezone
        
        blob_service_client = BlobServiceClient.from_connection_string(
            settings.STORAGES["default"]["OPTIONS"]["connection_string"]
        )
        container_name = settings.STORAGES["default"]["OPTIONS"]["azure_container"]
        
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        blob_properties = blob_client.get_blob_properties()
        creation_time = blob_properties.creation_time
        blob_size = blob_properties.size
        
        if max_size_bytes and blob_size > max_size_bytes:
            return False, f"File size ({blob_size / (1024**3):.2f} GB) exceeds the maximum allowed limit ({max_size_bytes / (1024**3):.2f} GB)."
        
        current_time = datetime.datetime.now(timezone.utc)
        age = current_time - creation_time
        
        if age > timedelta(minutes=max_age_minutes):
            return False, 'Blob too old. Please upload a new video.'
            
        return True, None
        
    except Exception:
        return False, 'Blob not found. Please verify the upload.'


def create_upload_sas_url(file_extension):
    blob_name = f"videos/{SceneProcessingJob.generate_uuid()}.{file_extension}" if hasattr(SceneProcessingJob, 'generate_uuid') else None
    if not blob_name:
        import uuid
        blob_name = f"videos/{uuid.uuid4()}.{file_extension}"

    blob_service_client = BlobServiceClient.from_connection_string(
        settings.STORAGES["default"]["OPTIONS"]["connection_string"]
    )
    container_name = settings.STORAGES["default"]["OPTIONS"]["azure_container"]

    sas_token = generate_blob_sas(
        account_name=blob_service_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(write=True, create=True),
        expiry=datetime.datetime.utcnow() + datetime.timedelta(minutes=120),
    )

    sas_url = f"{blob_service_client.url}{container_name}/{blob_name}?{sas_token}"

    return {'sas_url': sas_url, 'blob_name': blob_name}


def create_processing_job(user, title, blob_name, public=False, allow_as_example=False, 
                         reconstruction_method=None, training_max_num_gaussians=None, 
                         training_num_steps=None, remove_background=False,
                         equirectangular=False, use_background_sphere=False, pilgram_filter=None,
                         apriltag_size_mm=None, camera_type=None,
                         is_api_call=False,
                         world_id=None, world_mode=None, capture_id=None):
    if not blob_name:
        raise ValueError('No blob_name provided.')
    
    if not re.match(r'^videos/[a-f0-9\-]{36}\.[a-z0-9]+$', blob_name):
        raise ValueError('Invalid blob_name format.')
    
    reconstruction_method = reconstruction_method or SceneProcessingJob.ReconstructionMethod.GLOMAP
    
    user_has_premium_access = user_can_generate_premium_scene(user, api=is_api_call) if user else False
    if reconstruction_method == SceneProcessingJob.ReconstructionMethod.GENERATE_LOD:
        max_size_bytes = 6 * 1024 * 1024 * 1024
    elif reconstruction_method == SceneProcessingJob.ReconstructionMethod.QUEST:
        max_size_bytes = settings.MAX_QUEST_FILE_SIZE_PRO if user_has_premium_access else settings.MAX_QUEST_FILE_SIZE_FREE
    else:
        max_size_bytes = settings.MAX_VIDEO_FILE_SIZE_PRO if user_has_premium_access else settings.MAX_VIDEO_FILE_SIZE_FREE

    is_valid, error_msg = validate_blob(blob_name, max_age_minutes=60, max_size_bytes=max_size_bytes)
    if not is_valid:
        raise ValueError(error_msg)
    
    valid_methods = [choice[0] for choice in SceneProcessingJob.ReconstructionMethod.choices]
    if reconstruction_method not in valid_methods:
        raise ValueError(f'Invalid reconstruction method. Must be one of: {", ".join(valid_methods)}')
    
    user_subscription_tier = get_subscription_tier_string(user)
    
    gets_premium_treatment = user_has_premium_access
    consumption_transaction = None
    
    if (user.is_authenticated and hasattr(user, 'subscription') 
        and user.subscription.tier == SubscriptionTier.ENTERPRISE_PERSCENE):
        if is_api_call and user.subscription.api_credits_remaining < 1:
            raise ValueError('Insufficient API credits to process scene.')
        if gets_premium_treatment:
            consumption_transaction = CreditTransaction.objects.create(
                user=user,
                transaction_type=CreditTransaction.TransactionType.CONSUMPTION,
                credits_amount=-1,
                auto_processed=True,
                status=CreditTransaction.TransactionStatus.PENDING
            )

    validation_result = validate_user_settings(
        user,
        training_max_num_gaussians=training_max_num_gaussians,
        training_num_steps=training_num_steps,
        reconstruction_method=reconstruction_method,
        apriltag_size_mm=apriltag_size_mm,
        is_api_call=is_api_call,
        user_subscription_tier=user_subscription_tier,
    )
    
    if not validation_result['valid']:
        if validation_result['adjusted_values']:
            training_max_num_gaussians = validation_result['adjusted_values'].get('training_max_num_gaussians', training_max_num_gaussians)
            training_num_steps = validation_result['adjusted_values'].get('training_num_steps', training_num_steps)
            reconstruction_method = validation_result['adjusted_values'].get('reconstruction_method', reconstruction_method)
            apriltag_size_mm = validation_result['adjusted_values'].get('apriltag_size_mm', apriltag_size_mm)
        else:
            raise ValueError('; '.join(validation_result['errors']))
    
    job_data = {
        'title': title or blob_name.split('/')[-1],
        'public': public,
        'allow_as_example': allow_as_example,
        'user': user if user.is_authenticated else None,
        'reconstruction_method': reconstruction_method,
        'training_max_num_gaussians': training_max_num_gaussians or SceneProcessingJob.DEFAULT_NUM_GAUSSIANS,
        'training_num_steps': training_num_steps or SceneProcessingJob.DEFAULT_NUM_STEPS,
    }

    job_data['video_file'] = blob_name
    job_data['remove_background'] = remove_background
    job_data['equirectangular'] = equirectangular
    job_data['use_background_sphere'] = use_background_sphere
    if pilgram_filter:
        job_data['pilgram_filter'] = pilgram_filter
    if apriltag_size_mm is not None:
        job_data['apriltag_size_mm'] = apriltag_size_mm
    if world_id:
        job_data['world_id'] = world_id
    if world_mode:
        job_data['world_mode'] = world_mode
    if capture_id:
        job_data['capture_id'] = capture_id
    if camera_type:
        job_data['camera_data'] = {
            "lookAt": {"x": 0.0, "y": 0.0, "z": 0.0},
            "position": {"x": 0.0, "y": 0.0, "z": -3.0},
            "up": {"x": 0.0, "y": 1.0, "z": 0.0},
            "cameraType": camera_type
        }

    spj = SceneProcessingJob.objects.create(**job_data)

    if consumption_transaction:
        consumption_transaction.scene_processing_job = spj
        consumption_transaction.job_title = spj.title[:255] if spj.title else ""
        consumption_transaction.job_created_at = spj.uploaded_at
        consumption_transaction.save()

    queue_name = "default"
    if user.is_authenticated and user.is_superuser:
        queue_name = "internal"
    elif (user.is_authenticated and hasattr(user, 'subscription') 
          and user.subscription is not None
          and user.subscription.tier in [SubscriptionTier.ENTERPRISE, SubscriptionTier.ENTERPRISE_PERSCENE]):
        queue_name = "enterprise"
    elif gets_premium_treatment:
        queue_name = "high"
    queue = django_rq.get_queue(queue_name)

    from django.db import transaction
    from rq.job import Job

    job = Job.create(
        func=RQ_PROCESS_FUNC,
        args=(spj.id,),
        connection=queue.connection
    )
    
    spj.rq_job_id = job.id
    spj.save()

    def enqueue_precreated_job():
        queue.enqueue_job(job)
        if consumption_transaction:
            success = consumption_transaction.fulfill(admin_notes="Automatic consumption - job successfully queued")
            if not success:
                logger.error(f"Failed to fulfill credit transaction {consumption_transaction.id} for job {spj.id}")
                
    transaction.on_commit(enqueue_precreated_job)
    
    try:
        user_info = f"User: {user.username}" if user.is_authenticated else "User: Anonymous"
        queue_info = f"Queue: {queue_name}"
        from django.urls import reverse
        redirect_url = reverse("check_status", kwargs={'spj_id': spj.id})
        email_body = f"""
A new video processing job has been enqueued.

Job Details:
- Job ID: {spj.id}
- Title: {title}
- {user_info}
- {queue_info}
- Public: {public}
- World ID: {world_id or ''}
- World mode: {world_mode or 'auto'}
- Max Gaussians: {spj.training_max_num_gaussians}
- Training Steps: {spj.training_num_steps}
- Remove Background: {remove_background}
- Equirectangular: {equirectangular}
- Reconstruction Method: {reconstruction_method}
- RQ Job ID: {job.id}

Status URL: {redirect_url}
""".strip()
        send_mail(
            subject='New Video Processing Job Enqueued',
            message=email_body,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[settings.STATUS_EMAIL],
            fail_silently=True,
        )
    except Exception:
        pass
    
    return {
        'spj': spj,
        'job': job,
        'queue_name': queue_name
    }
