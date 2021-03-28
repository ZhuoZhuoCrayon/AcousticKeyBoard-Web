# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from apps.keyboard import constants, exceptions, models
from apps.keyboard.tests import mock_data


class DatasetModelSer(serializers.ModelSerializer):
    class Meta:
        model = models.Dataset
        fields = "__all__"


class ImportDatasetRequestSer(serializers.Serializer):
    class DatasetField(serializers.FileField):
        def to_internal_value(self, data):
            data = super().to_internal_value(data)
            file_name: str = data.name
            if file_name.endswith(constants.FileType.PICKLE):
                return data
            raise ValidationError(_("仅支持{support_types}格式的数据集").format(support_types=constants.FileType.get_names()))

    dataset = DatasetField(label=_("数据集"))
    verbose_name = serializers.CharField(label=_("数据库别名"), max_length=128)
    description_more = serializers.CharField(label=_("数据集具体描述"), required=False, default="")

    class Meta:
        swagger_schema_fields = {"example": mock_data.API_DATASET_IMPORT_DATASET.request_data}


class ImportDataseResponseSer(serializers.Serializer):
    class Meta:
        swagger_schema_fields = {"example": mock_data.API_DATASET_IMPORT_DATASET.response_data}


class TrainRequestSer(serializers.Serializer):
    dataset_id = serializers.IntegerField(label=_("数据集id"))
    per_train_num = serializers.IntegerField(label=_("每个标签的训练个数"), min_value=1, required=False)
    per_train_rate = serializers.FloatField(label=_("每个标签的训练比例"), min_value=0.1, required=False)

    def validate(self, attrs):
        if not ("per_train_num" in attrs or "per_train_rate" in attrs):
            raise ValidationError(_("per_train_num（每个标签的训练个数）, per_train_rate（每个标签的训练比例）必须包含一个"))
        if not models.Dataset.objects.filter(id=attrs["dataset_id"]).exists():
            raise exceptions.DatasetNotFoundExc(context={"dataset_id": attrs["dataset_id"]})
        return attrs

    class Meta:
        swagger_schema_fields = {"example": mock_data.API_DATASET_TRAIN.request_data}


class TrainResponseSer(serializers.Serializer):
    class Meta:
        swagger_schema_fields = {"example": mock_data.API_DATASET_TRAIN.response_data}
