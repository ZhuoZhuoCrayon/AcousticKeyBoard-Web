# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.keyboard import models
from apps.keyboard.handler import model_inst as handler
from apps.keyboard.serializers import model_inst as serializers
from djangocli.utils.drf import view


class ModelInstViews(view.DjangoCliModelViewSet):
    model = models.AlgorithmModelInst
    serializer_class = serializers.ModelInstModelSer

    # 移除`post` `patch` `delete`等内置视图，模型实例设置可读
    http_method_names = ["get", "head", "post"]

    def get_queryset(self):
        return self.model.objects.all()

    @swagger_auto_schema(
        operation_summary=_("信号预测"),
        tags=["model_inst"],
        request_body=serializers.PredictRequestSer(),
        responses={status.HTTP_200_OK: serializers.PredictResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.PredictRequestSer)
    def predict(self, request, *args, **kwargs):
        return Response(
            handler.ModelInstHandler.predict(inst_id=self.query_data["inst_id"], signal=self.query_data["signal"])
        )

    @swagger_auto_schema(
        operation_summary=_("安卓Debug接口"),
        tags=["model_inst"],
        request_body=serializers.DebugRequestSer(),
        responses={status.HTTP_200_OK: serializers.DebugResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.DebugRequestSer)
    def debug(self, request, *args, **kwargs):
        handler.ModelInstHandler.debug(
            inst_id=self.query_data["inst_id"], signal=self.query_data["signal"], label=self.query_data["label"]
        )
        return Response({})
