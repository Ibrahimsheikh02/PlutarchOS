# Generated by Django 4.2.1 on 2023-09-04 16:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0023_lecture_professor"),
    ]

    operations = [
        migrations.RemoveField(model_name="lecture", name="professor",),
        migrations.AddField(
            model_name="course",
            name="professor",
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
