import logging
import os

import tensorflow as tf
from django.conf import settings
from django.db import models
from django.utils.translation import ugettext_lazy as _

from apps.keyboard import constants, exceptions
from apps.keyboard.core.algorithm import base, tf_models
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


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
        verbose_name=_("标签类型"), db_index=True, max_length=16, choices=constants.LabelType.get_choices(), null=True
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
        verbose_name=_("标签类型"), db_index=True, max_length=16, choices=constants.LabelType.get_choices(), null=True
    )

    original_data = models.JSONField(verbose_name=_("数据集原数据"), default=list)

    # 基础字段
    created_at = models.DateTimeField(verbose_name=_("创建时间"), auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name=_("更新时间"), blank=True, null=True, auto_now=True)

    class Meta:
        verbose_name = _("数据集原数据")
        verbose_name_plural = _("数据集原数据")
        ordering = ["id"]


class AlgorithmModelInst(models.Model):
    dataset_id = models.IntegerField(verbose_name=_("数据集ID"), db_index=True)
    algorithm = models.CharField(verbose_name=_("算法名称"), max_length=64, db_index=True)
    save_path = models.CharField(verbose_name=_("模型存放路径"), max_length=256, db_index=True, default="")
    train_info = models.JSONField(verbose_name=_("训练信息"), default=dict)
    extra_info = models.JSONField(verbose_name=_("额外信息"), default=dict)

    is_ready = models.BooleanField(verbose_name=_("算法模型是否可用"), default=False)
    is_latest = models.BooleanField(verbose_name=_("算法模型是否最新"), default=False)

    # 基础字段
    created_at = models.DateTimeField(verbose_name=_("创建时间"), auto_now_add=True)
    updated_at = models.DateTimeField(verbose_name=_("更新时间"), blank=True, null=True, auto_now=True)

    class Meta:
        verbose_name = _("算法模型实例（已训练）")
        verbose_name_plural = _("算法模型实例（已训练）")
        ordering = ["id"]

    def save_model(self, tf_model: base.TfBaseModel, **kwargs):

        # 该数据集-算法相关的所有模型实例变更为非最新
        AlgorithmModelInst.objects.filter(dataset_id=self.dataset_id, algorithm=self.algorithm, is_latest=True).update(
            is_latest=False
        )

        save_root = os.path.join(settings.MODEL_INST_ROOT, f"dataset_id:{self.dataset_id}:algorithm:{self.algorithm}")
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, f"inst_id:{self.id}")

        tf_model.model.save(filepath=save_path, overwrite=True)

        self.save_path = save_path
        self.extra_info.update(
            {
                **kwargs.get("extra_info", {}),
                "input_shape_list": list(tf_model.input_shape),
                "max_label_num": tf_model.max_label_num,
            }
        )
        # 变更为最新的模型
        self.is_latest = True
        # 变更模型为可用状态
        self.is_ready = True
        self.save()

    def load_model(self) -> base.TfBaseModel:
        if not self.is_ready:
            raise exceptions.ModelInstNotReadyExc(context={"inst_id": self.id})
        name_tf_model_class_map = {
            tf_models.BLstmModel.MODEL_NAME: tf_models.BLstmModel,
            tf_models.RnnModel.MODEL_NAME: tf_models.RnnModel,
            tf_models.LstmModel.MODEL_NAME: tf_models.LstmModel,
        }
        model: tf.keras.Sequential = tf.keras.models.load_model(self.save_path)

        logger.info(
            f"AlgorithmModelInst.load_model_inst: dataset_id -> {self.dataset_id}, algorithm -> {self.algorithm}, "
            f"inst_id -> {self.id}, model -> \n"
        )

        return name_tf_model_class_map[self.algorithm](
            max_label_num=self.extra_info["max_label_num"],
            input_shape=tuple(self.extra_info["input_shape_list"]),
            model=model,
        )
