# Generated by Django 3.1.5 on 2021-03-23 16:01

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Dataset",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("verbose_name", models.CharField(max_length=128, verbose_name="数据集别名")),
                ("dataset_name", models.CharField(max_length=128, verbose_name="数据集名称")),
                (
                    "data_type",
                    models.CharField(
                        choices=[
                            ("all-1micro", "1号麦克风-30个键位"),
                            ("all-0micro", "0号麦克风-30个键位"),
                            ("all-1micro-far12", "1号麦克风-远间隔12个键位"),
                            ("all-0micro-far12", "0号麦克风-远间隔12个键位"),
                            ("all-1micro-near12", "1号麦克风-近间隔12个键位"),
                            ("all-0micro-near12", "0号麦克风-近间隔12个键位"),
                        ],
                        max_length=32,
                        verbose_name="数据集格式",
                    ),
                ),
                (
                    "project_type",
                    models.CharField(choices=[("single", "单键"), ("single", "组合键")], max_length=32, verbose_name="数据类型"),
                ),
                ("description", models.CharField(max_length=256, verbose_name="数据集组合说明")),
                ("description_more", models.TextField(blank=True, null=True, verbose_name="数据集具体描述")),
                ("length", models.IntegerField(verbose_name="信号长度")),
                ("fs", models.IntegerField(verbose_name="采样频率")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="创建时间")),
                ("updated_at", models.DateTimeField(auto_now=True, null=True, verbose_name="更新时间")),
            ],
            options={
                "verbose_name": "数据集",
                "verbose_name_plural": "数据集",
                "ordering": ["id"],
            },
        ),
        migrations.CreateModel(
            name="DatasetMfccFeature",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("dataset_id", models.IntegerField(db_index=True, verbose_name="数据集ID")),
                ("label", models.CharField(db_index=True, max_length=32, verbose_name="数据标签")),
                ("mfcc_feature", models.JSONField(default=list, verbose_name="MFCC特征")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="创建时间")),
                ("updated_at", models.DateTimeField(auto_now=True, null=True, verbose_name="更新时间")),
            ],
            options={
                "verbose_name": "mfcc特征",
                "verbose_name_plural": "mfcc特征",
                "ordering": ["id"],
            },
        ),
        migrations.CreateModel(
            name="DatasetOriginalData",
            fields=[
                ("id", models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("dataset_id", models.IntegerField(db_index=True, verbose_name="数据集ID")),
                ("label", models.CharField(db_index=True, max_length=32, verbose_name="数据标签")),
                ("original_data", models.JSONField(default=list, verbose_name="数据集原数据")),
                ("created_at", models.DateTimeField(auto_now_add=True, verbose_name="创建时间")),
                ("updated_at", models.DateTimeField(auto_now=True, null=True, verbose_name="更新时间")),
            ],
            options={
                "verbose_name": "数据集原数据",
                "verbose_name_plural": "数据集原数据",
                "ordering": ["id"],
            },
        ),
    ]
