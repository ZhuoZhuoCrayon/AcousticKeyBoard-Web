# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.keyboard import models
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
        return Response({"verbose_name": self.query_data["verbose_name"]})
