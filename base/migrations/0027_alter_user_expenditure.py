# Generated by Django 4.2.1 on 2023-12-21 15:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("base", "0026_message_reply"),
    ]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="expenditure",
            field=models.DecimalField(decimal_places=10, default=0.0, max_digits=15),
        ),
    ]
