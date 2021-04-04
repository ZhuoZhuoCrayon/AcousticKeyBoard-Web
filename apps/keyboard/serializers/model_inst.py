# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from apps.keyboard import constants, exceptions, models
from apps.keyboard.tests import mock_data


class ModelInstModelSer(serializers.ModelSerializer):
    class Meta:
        model = models.AlgorithmModelInst
        fields = "__all__"


class PredictRequestSer(serializers.Serializer):
    inst_id = serializers.IntegerField(label=_("算法模型实例ID"), required=False)
    dataset_id = serializers.IntegerField(label=_("数据集id"), required=False)
    algorithm = serializers.ChoiceField(label=_("算法模型"), choices=constants.AlgorithmModel.get_choices(), required=False)
    # 16-bit short int array
    signal = serializers.ListField(label=_("敲击信号"), child=serializers.IntegerField(), required=True, min_length=1000)

    def validate(self, attrs):
        if "inst_id" in attrs:
            model_inst = models.AlgorithmModelInst.objects.filter(id=attrs["inst_id"]).first()
        else:
            if not ("dataset_id" in attrs and "algorithm" in attrs):
                raise ValidationError(_("必须包含字段：dataset_id（数据集id），algorithm（算法模型）"))
            model_inst = models.AlgorithmModelInst.objects.filter(
                dataset_id=attrs["dataset_id"], algorithm=attrs["algorithm"], is_latest=True
            ).first()
        if not model_inst:
            raise exceptions.ModelInstNotFoundExc({"inst_id": attrs.get("inst_id")})
        if not model_inst.is_ready:
            raise exceptions.ModelInstNotReadyExc({"inst_id": model_inst.id})

        # 回填实例ID，用于业务逻辑
        attrs["inst_id"] = model_inst.id
        return attrs

    class Meta:
        swagger_schema_fields = {"example": mock_data.API_MODEL_INST_PREDICT.request_data}


class PredictResponseSer(serializers.Serializer):
    class Meta:
        swagger_schema_fields = {"example": mock_data.API_MODEL_INST_PREDICT.response_data}


class DebugRequestSer(PredictRequestSer):
    label = serializers.CharField(label=_("标签"))


class DebugResponseSer(PredictResponseSer):
    pass


class CorrectRequestSer(serializers.Serializer):
    dataset_id = serializers.IntegerField(label=_("数据集id"))
    expect_label = serializers.CharField(label=_("标签"))
    cache_key = serializers.CharField(label=_("缓存MFCC特征的key值"))

    class Meta:
        swagger_schema_fields = {"example": mock_data.API_MODEL_INST_CORRECT.request_data}


class CorrectResponseSer(serializers.Serializer):
    class Meta:
        swagger_schema_fields = {"example": mock_data.API_MODEL_INST_CORRECT.response_data}
