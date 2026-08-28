import uuid
import json
from django.core.exceptions import ValidationError
from django.db import models
from django.conf import settings
from django.core.cache import cache
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from django.core.validators import MinValueValidator, MaxValueValidator

import os

# Path to the JSON schema
BASE_DIR = settings.BASE_DIR
CAMERA_DATA_SCHEMA_PATH = os.path.join(BASE_DIR, 'video_processor', 'schemas', 'camera_data_schema.json')

def load_json_schema():
    with open(CAMERA_DATA_SCHEMA_PATH, 'r') as schema_file:
        return json.load(schema_file)

CAMERA_DATA_SCHEMA = load_json_schema() 

def validate_camera_data(value):
    """
    Validates the camera_data JSON using the defined JSON schema.
    Django model validators should not return values, only raise ValidationError if invalid.
    """
    if value is None:
        return  # None is valid for nullable fields
    
    try:
        # If value is already a dict/list, validate it directly
        # If it's a string, parse it first
        json_value = value if isinstance(value, (dict, list)) else json.loads(value)
        validate(instance=json_value, schema=CAMERA_DATA_SCHEMA)
        # If we get here, validation passed - don't return anything
    except JsonSchemaValidationError as e:
        raise ValidationError(f"Invalid camera_data: {e.message}")
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in camera_data: {e}")
    


class SceneProcessingJob(models.Model):

    # Define constants for the number of Gaussians and steps
    MIN_NUM_GAUSSIANS = 100_000
    MAX_NUM_GAUSSIANS = 2_000_000
    DEFAULT_NUM_GAUSSIANS = 1_000_000
    MAX_NUM_GAUSSIANS_FREE = 1_200_000

    MIN_NUM_STEPS = 5_000
    MAX_NUM_STEPS = 40_000
    MAX_NUM_STEPS_FREE = 30_000

    DEFAULT_NUM_STEPS = 25_000
    
    # Pilgram filter choices
    PILGRAM_FILTER_CHOICES = [
        (None, 'None'),
        ('_1977', '_1977'),
        ('aden', 'Aden'),
        ('brannan', 'Brannan'),
        ('brooklyn', 'Brooklyn'),
        ('clarendon', 'Clarendon'),
        ('earlybird', 'Earlybird'),
        ('gingham', 'Gingham'),
        ('hudson', 'Hudson'),
        ('inkwell', 'Inkwell'),
        ('kelvin', 'Kelvin'),
        ('lark', 'Lark'),
        ('lofi', 'Lo-Fi'),
        ('maven', 'Maven'),
        ('mayfair', 'Mayfair'),
        ('moon', 'Moon'),
        ('nashville', 'Nashville'),
        ('perpetua', 'Perpetua'),
        ('reyes', 'Reyes'),
        ('rise', 'Rise'),
        ('slumber', 'Slumber'),
        ('stinson', 'Stinson'),
        ('toaster', 'Toaster'),
        ('valencia', 'Valencia'),
        ('walden', 'Walden'),
        ('willow', 'Willow'),
        ('xpro2', 'X-Pro II'),
    ]

    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to="videos/")

    ply_file = models.FileField(
        upload_to="ply_files/", null=True, blank=True
    )
    spz_file = models.FileField(
        upload_to="spz_files/", null=True, blank=True
    )
    sog_file = models.FileField(
        upload_to="sog_files/", null=True, blank=True,
        help_text="Points to meta.json of unbundled SOG directory"
    )
    lod_file = models.FileField(
        upload_to="lod_files/", null=True, blank=True,
        help_text="Points to lod-meta.json of LOD octree directory"
    )
    preview_image = models.ImageField(
        upload_to="preview_images/", null=True, blank=True
    )
    rq_job_id = models.CharField(max_length=255, unique=True, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.CASCADE)  # Associates the job with a user
    public = models.BooleanField(default=False, verbose_name="Publicly Accessible")  # Controls visibility to other users
    example = models.BooleanField(default=False)  # Controls if the scene is an example on the examples page
    example_sort_order = models.IntegerField(default=0)  # Controls the order of the example on the examples page
    allow_as_example = models.BooleanField( # Controls if the scene can be featured as an example on the examples page
        default=False,
        verbose_name="Allow as Example"
    )
    camera_data = models.JSONField(
        null=True,
        blank=True,
        validators=[validate_camera_data],
        help_text="Contains 'lookAt', 'position', and 'up' vectors."
    )
    remove_background = models.BooleanField(default=False)
    equirectangular = models.BooleanField(default=False, verbose_name="360° Video")
    use_background_sphere = models.BooleanField(default=False, verbose_name="Use Gaussian Fibonacci Sphere")
    pilgram_filter = models.CharField(
        max_length=128,
        choices=PILGRAM_FILTER_CHOICES,
        null=True,
        blank=True,
        verbose_name="Instagram-style Filter",
        help_text="Apply an Instagram-style filter to all frames before processing"
    )
    training_max_num_gaussians = models.PositiveIntegerField(default=DEFAULT_NUM_GAUSSIANS,
                                                             validators=[MinValueValidator(MIN_NUM_GAUSSIANS, "Maximum number of Gaussians must be at least 100,000."),
                                                                         MaxValueValidator(MAX_NUM_GAUSSIANS, "Maximum number of Gaussians must be at most 2,000,000.")])
    training_num_steps = models.PositiveIntegerField(default=DEFAULT_NUM_STEPS, 
                                                     validators=[MinValueValidator(MIN_NUM_STEPS, "Number of steps must be at least 5,000."),
                                                                 MaxValueValidator(MAX_NUM_STEPS, "Number of steps must be at most 40,000.")])
    class ReconstructionMethod(models.TextChoices):
        GLOMAP = 'glomap', 'GLOMAP'
        COLMAP = 'colmap', 'COLMAP'
        VGGT = 'vggt', 'VGGT'
        QUEST = 'quest', 'QUEST'
        GENERATE_LOD = 'generate_lod', 'GENERATE_LOD'
        # Future methods can be added here
        
    reconstruction_method = models.CharField(
        max_length=20,
        choices=ReconstructionMethod.choices,
        default=ReconstructionMethod.GLOMAP,
        verbose_name="3D Reconstruction Method",
        help_text="Choose the reconstruction algorithm for creating the 3D point cloud"
    )
    
    # AprilTag Scale Calibration (Enterprise feature)
    apriltag_size_mm = models.FloatField(
        null=True,
        blank=True,
        verbose_name="AprilTag Size (mm)",
        help_text="Physical size of AprilTag in millimeters (measured from inner white square). If provided, enables automatic scale calibration (Enterprise only)",
        validators=[MinValueValidator(1.0, "Tag size must be at least 1mm"),
                    MaxValueValidator(1000.0, "Tag size must be at most 1000mm")]
    )

    world_id = models.CharField(
        max_length=128,
        null=True,
        blank=True,
        verbose_name="World ID",
        help_text="Persistent world this capture belongs to. Empty = isolated stock scene.",
    )
    WORLD_MODE_CHOICES = [
        ("", "auto"),
        ("bootstrap", "bootstrap"),
        ("integrate", "integrate"),
    ]
    world_mode = models.CharField(
        max_length=16,
        choices=WORLD_MODE_CHOICES,
        default="",
        blank=True,
        verbose_name="World mode",
        help_text="bootstrap seeds an empty world. integrate registers into an existing world. blank = auto.",
    )
    capture_id = models.CharField(
        max_length=128,
        null=True,
        blank=True,
        verbose_name="Capture ID",
        help_text="Optional capture id inside the world. Defaults to the job UUID.",
    )

    def __str__(self):
        return f"Processing Job {self.id}: {self.title}"


@receiver([post_save, post_delete], sender=SceneProcessingJob)
def invalidate_examples_cache(sender, instance, **kwargs):
    """
    Signal handler to invalidate the examples cache when example-related fields change.
    """
    # Only invalidate if this job is or was an example, or if example fields changed
    if hasattr(instance, 'example') and instance.example:
        cache.delete('examples:list')


def is_publicly_shareable(spj: SceneProcessingJob) -> bool:
    return spj.public or spj.user == None or spj.example
