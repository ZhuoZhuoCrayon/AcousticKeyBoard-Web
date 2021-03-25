# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from apps.keyboard import constants, models
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
