# Generated by Django 4.2.1 on 2023-06-25 02:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0010_alter_user_expenditure"),
    ]

    operations = [
        migrations.AddField(
            model_name="message",
            name="is_deleted",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="user",
            name="questions_asked",
            field=models.IntegerField(default=0),
        ),
    ]
