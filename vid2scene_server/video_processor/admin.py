from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from django import forms
from .models import SceneProcessingJob

class SceneProcessingJobAdminForm(forms.ModelForm):
    # Allow typing Azure blob path directly instead of file upload
    lod_file = forms.CharField(
        required=False,
        help_text="Azure blob path, e.g. lod_files/your-scene-id/lod-meta.json",
        widget=forms.TextInput(attrs={'size': 120, 'style': 'width: 100%;'}),
    )

    class Meta:
        model = SceneProcessingJob
        fields = '__all__'

    def clean_lod_file(self):
        """Accept empty string as None."""
        value = self.cleaned_data.get('lod_file')
        return value if value else ''

def make_requeue_action(queue_name):
    def requeue_jobs(self, request, queryset):
        import django_rq
        from rq.job import Job
        from subscriptions.utils import get_subscription_tier_string

        count = 0
        skipped = []
        for spj in queryset.order_by('uploaded_at'):
            if spj.ply_file:
                skipped.append(str(spj.id))
                continue
            actual_queue_name = queue_name
            if queue_name == 'automatic':
                actual_queue_name = "default"
                user = spj.user
                if user is not None:
                    if user.is_superuser:
                        actual_queue_name = "internal"
                    else:
                        tier = get_subscription_tier_string(user)
                        if tier in ['enterprise', 'enterprise_perscene']:
                            actual_queue_name = "enterprise"
                        elif tier == 'pro':
                            actual_queue_name = "high"
            
            queue = django_rq.get_queue(actual_queue_name)
            
            job = Job.create(
                func="video_processor.world_task.process_video_task",
                args=(spj.id,),
                connection=queue.connection
            )
            
            spj.rq_job_id = job.id
            spj.save(update_fields=['rq_job_id'])
            
            queue.enqueue_job(job)
            count += 1
            
        if queue_name == 'automatic':
            self.message_user(request, f"Successfully requeued {count} job(s) to their automatic queues.")
        else:
            self.message_user(request, f"Successfully requeued {count} job(s) to the '{queue_name}' queue.")
        if skipped:
            self.message_user(
                request,
                f"Skipped {len(skipped)} job(s) that already have a ply_file: {', '.join(skipped)}",
                level='WARNING',
            )
    
    requeue_jobs.__name__ = f"requeue_{queue_name}"
    if queue_name == 'automatic':
        requeue_jobs.short_description = "Requeue selected jobs automatically depending on user subscription"
    else:
        requeue_jobs.short_description = f"Requeue selected jobs to '{queue_name}' queue"
    return requeue_jobs

@admin.register(SceneProcessingJob)
class SceneProcessingJobAdmin(admin.ModelAdmin):
    form = SceneProcessingJobAdminForm
    list_display = ('title', 'user', 'world_id', 'world_mode', 'reconstruction_method', 'example', 'allow_as_example', 'view_link', 'uploaded_at', 'rq_job_id', 'ply_file', 'lod_file')
    list_filter = ('reconstruction_method', 'world_mode', 'example', 'uploaded_at', 'user')
    search_fields = ('title', 'user__username', 'id', 'world_id', 'capture_id')
    actions = ['make_example', 'remove_example', 'requeue_automatic', 'requeue_default', 'requeue_high', 'requeue_internal', 'requeue_enterprise']

    def make_example(self, request, queryset):
        queryset.update(example=True)
    make_example.short_description = "Mark selected jobs as examples"

    def remove_example(self, request, queryset):
        queryset.update(example=False)
    remove_example.short_description = "Unmark selected jobs as examples"

    requeue_automatic = make_requeue_action("automatic")
    requeue_default = make_requeue_action("default")
    requeue_high = make_requeue_action("high")
    requeue_internal = make_requeue_action("internal")
    requeue_enterprise = make_requeue_action("enterprise")

    def view_link(self, obj):
        url = reverse('splat_viewer_spj_id', args=[obj.id])
        return format_html('<a href="{}" target="_blank">View Scene</a>', url)
    view_link.short_description = "Viewer Link"
