# Generated by Django 3.1.5 on 2021-03-28 13:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("keyboard", "0003_algorithmmodelinst"),
    ]

    operations = [
        migrations.AddField(
            model_name="algorithmmodelinst",
            name="is_latest",
            field=models.BooleanField(default=False, verbose_name="算法模型是否最新"),
        ),
        migrations.AddField(
            model_name="algorithmmodelinst",
            name="is_ready",
            field=models.BooleanField(default=False, verbose_name="算法模型是否可用"),
        ),
    ]