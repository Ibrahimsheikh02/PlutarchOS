# Generated by Django 4.2.1 on 2023-09-04 23:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0024_remove_lecture_professor_course_professor"),
    ]

    operations = [
        migrations.AddField(
            model_name="lecture",
            name="visible",
            field=models.BooleanField(default=False),
        ),
    ]
