from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("video_processor", "0030_alter_sceneprocessingjob_reconstruction_method"),
    ]

    operations = [
        migrations.AddField(
            model_name="sceneprocessingjob",
            name="world_id",
            field=models.CharField(
                blank=True,
                help_text="Persistent world this capture belongs to. Empty = isolated stock scene.",
                max_length=128,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="sceneprocessingjob",
            name="world_mode",
            field=models.CharField(
                blank=True,
                choices=[("", "auto"), ("bootstrap", "bootstrap"), ("integrate", "integrate")],
                default="",
                help_text="bootstrap seeds an empty world. integrate registers into an existing world. blank = auto.",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="sceneprocessingjob",
            name="capture_id",
            field=models.CharField(
                blank=True,
                help_text="Optional capture id inside the world. Defaults to the job UUID.",
                max_length=128,
                null=True,
            ),
        ),
    ]
