# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _
from rest_framework import serializers

from apps.keyboard.tests import mock_data


class BatchCeleryResultsRequestSer(serializers.Serializer):

    task_ids = serializers.ListField(
        required=True, child=serializers.CharField(help_text=_("celery task id")), min_length=1
    )

    def validate(self, attrs):
        attrs = super().validate(attrs)
        attrs["task_ids"] = list(set(attrs["task_ids"]))
        return attrs

    class Meta:
        swagger_schema_fields = {"example": mock_data.API_COMMON_BATCH_CELERY_RESULTS_DELAY.request_data}


class BatchCeleryResultsResponseSer(serializers.Serializer):
    class Meta:
        swagger_schema_fields = {"example": mock_data.API_COMMON_BATCH_CELERY_RESULTS_DELAY.response_data}
