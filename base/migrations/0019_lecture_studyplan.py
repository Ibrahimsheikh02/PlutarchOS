# Generated by Django 4.2.1 on 2023-08-24 19:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0018_discussionmessages_parent"),
    ]

    operations = [
        migrations.AddField(
            model_name="lecture",
            name="studyplan",
            field=models.FileField(blank=True, null=True, upload_to="study_plans/"),
        ),
    ]
