# Generated by Django 4.2.1 on 2023-12-21 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0025_lecture_visible"),
    ]

    operations = [
        migrations.AddField(
            model_name="message", name="reply", field=models.TextField(null=True),
        ),
    ]
