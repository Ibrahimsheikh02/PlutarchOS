# Generated by Django 4.2.1 on 2023-06-04 19:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0007_lecture_transcript_embeddings_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="email",
            field=models.EmailField(max_length=254, unique=True),
        ),
    ]
