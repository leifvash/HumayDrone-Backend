# Generated migration adding farm association fields to DroneFlight

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_cloudinferenceresult_review_notes_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='droneflight',
            name='farm_id',
            field=models.UUIDField(
                blank=True,
                db_index=True,
                help_text='Optional linked Supabase public.farms UUID',
                null=True,
            ),
        ),
        migrations.AddField(
            model_name='droneflight',
            name='farm_name',
            field=models.CharField(
                blank=True,
                default='',
                help_text='Optional cached farm code/name for display',
                max_length=255,
            ),
        ),
    ]
