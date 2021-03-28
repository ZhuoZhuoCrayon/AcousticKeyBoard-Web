# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.keyboard import models
from apps.keyboard.handler import dataset as handler
from apps.keyboard.serializers import dataset as serializers
from djangocli.utils.drf import view


class DatasetViews(view.DjangoCliModelViewSet):
    model = models.Dataset
    serializer_class = serializers.DatasetModelSer

    def get_queryset(self):
        return self.model.objects.all()

    @swagger_auto_schema(
        operation_summary=_("导入数据集"),
        tags=["dataset"],
        request_body=serializers.ImportDatasetRequestSer(),
        responses={status.HTTP_200_OK: serializers.ImportDataseResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.ImportDatasetRequestSer)
    def import_dataset(self, request, *args, **kwargs):
        task_id = handler.DatasetHandler.import_dataset(
            dataset_file=self.query_data["dataset"],
            verbose_name=self.query_data["verbose_name"],
            description_more=self.query_data["description_more"],
        )
        return Response({"task_id": task_id})

    @swagger_auto_schema(
        operation_summary=_("训练数据集"),
        tags=["dataset"],
        request_body=serializers.TrainRequestSer(),
        responses={status.HTTP_200_OK: serializers.TrainResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.TrainRequestSer)
    def train(self, request, *args, **kwargs):
        handler.DatasetHandler.train(
            dataset_id=self.query_data["dataset_id"], per_train_rate=self.query_data["per_train_rate"]
        )
        return Response({})
