from django.db import models
from django.utils.translation import ugettext_lazy as _

from apps.keyboard import constants


class Dataset(models.Model):
    verbose_name = models.CharField(verbose_name=_("数据集别名"), max_length=128)
    dataset_name = models.CharField(verbose_name=_("数据集名称"), max_length=128)
    data_type = models.CharField(verbose_name=_("数据集格式"), max_length=32, choices=constants.DataType.get_choices())
    project_type = models.CharField(verbose_name=_("数据类型"), max_length=32, choices=constants.ProjectType.get_choices())
    description = models.CharField(verbose_name=_("数据集组合说明"), max_length=256)
    description_more = models.TextField(verbose_name=_("数据集具体描述"), null=True, blank=True)
    length = models.IntegerField(verbose_name=_("信号长度"))
    fs = models.IntegerField(verbose_name=_("采样频率"))

    # 基础字段
    created_at = models.DateTimeField(verbose_name=_("创建时间"), auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name=_("更新时间"), blank=True, null=True, auto_now=True)

    class Meta:
        verbose_name = _("数据集")
        verbose_name_plural = _("数据集")
        ordering = ["id"]


class DatasetMfccFeature(models.Model):
    dataset_id = models.IntegerField(verbose_name=_("数据集ID"), db_index=True)
    label = models.CharField(verbose_name=_("数据标签"), db_index=True, max_length=32)
    label_type = models.CharField(
        verbose_name=_("标签类型"), db_index=True, max_length=16, choices=constants.LabelType.get_choices()
    )
    mfcc_feature = models.JSONField(verbose_name=_("MFCC特征"), default=list)

    # 基础字段
    created_at = models.DateTimeField(verbose_name=_("创建时间"), auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name=_("更新时间"), blank=True, null=True, auto_now=True)

    class Meta:
        verbose_name = _("mfcc特征")
        verbose_name_plural = _("mfcc特征")
        ordering = ["id"]


class DatasetOriginalData(models.Model):
    dataset_id = models.IntegerField(verbose_name=_("数据集ID"), db_index=True)
    label = models.CharField(verbose_name=_("数据标签"), db_index=True, max_length=32)
    label_type = models.CharField(
        verbose_name=_("标签类型"), db_index=True, max_length=16, choices=constants.LabelType.get_choices()
    )

    original_data = models.JSONField(verbose_name=_("数据集原数据"), default=list)

    # 基础字段
    created_at = models.DateTimeField(verbose_name=_("创建时间"), auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name=_("更新时间"), blank=True, null=True, auto_now=True)

    class Meta:
        verbose_name = _("数据集原数据")
        verbose_name_plural = _("数据集原数据")
        ordering = ["id"]
