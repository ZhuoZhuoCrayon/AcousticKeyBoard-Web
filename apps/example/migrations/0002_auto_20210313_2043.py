# Generated by Django 3.1.5 on 2021-03-13 12:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("example", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="book",
            name="publication_date",
            field=models.DateField(verbose_name="出版日期"),
        ),
    ]