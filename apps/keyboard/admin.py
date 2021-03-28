from django.contrib import admin

from apps.keyboard import models


@admin.register(models.Dataset)
class KeyBoardDatasetAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.Dataset._meta.get_fields()]
    search_fields = ["verbose_name", "dataset_name", "data_type", "project_type"]
    list_filter = ["verbose_name", "dataset_name", "data_type", "project_type"]


@admin.register(models.DatasetMfccFeature)
class KeyBoardDatasetMfccFeatureAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.DatasetMfccFeature._meta.get_fields()]
    search_fields = ["dataset_id", "label"]
    list_filter = ["dataset_id", "label"]


@admin.register(models.DatasetOriginalData)
class KeyBoardDatasetOriginalDataAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.DatasetOriginalData._meta.get_fields()]
    search_fields = ["dataset_id", "label"]
    list_filter = ["dataset_id", "label"]


@admin.register(models.AlgorithmModelInst)
class KeyBoardAlgorithmModelInstAdmin(admin.ModelAdmin):
    list_display = [field.name for field in models.AlgorithmModelInst._meta.get_fields()]
    search_fields = ["dataset_id", "algorithm", "save_path"]
    list_filter = ["dataset_id", "algorithm", "save_path"]
