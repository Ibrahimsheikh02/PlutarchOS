# Generated by Django 4.2.1 on 2023-06-04 03:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0002_message_lecture"),
    ]

    operations = [
        migrations.CreateModel(
            name="LectureTranscript",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("embeddings", models.BinaryField(blank=True, null=True)),
                ("lecture_text", models.TextField(blank=True, null=True)),
            ],
        ),
    ]
