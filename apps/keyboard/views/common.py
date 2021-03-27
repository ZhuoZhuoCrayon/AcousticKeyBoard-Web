# -*- coding: utf-8 -*-

import logging

from django.utils.translation import ugettext_lazy as _
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.keyboard.handler import common as handler
from apps.keyboard.serializers import common as serializers
from djangocli.constants import LogModule
from djangocli.utils.drf import view

# Create your views here.

logger = logging.getLogger(LogModule.APPS)


class CommonViews(view.DjangoCliGenericViewSet):
    @swagger_auto_schema(
        operation_summary=_("获取Celery执行结果"),
        tags=["common"],
        request_body=serializers.BatchCeleryResultsRequestSer(),
        responses={status.HTTP_200_OK: serializers.BatchCeleryResultsResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.BatchCeleryResultsRequestSer)
    def batch_celery_results(self, request, *args, **kwargs):
        return Response(handler.CommonHandler.batch_celery_results(task_ids=self.query_data["task_ids"]))
