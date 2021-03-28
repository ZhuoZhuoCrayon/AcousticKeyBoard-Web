# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _

from apps.exceptions import AppModuleErrorCode
from djangocli.exceptions import DjangoCliBaseException


class ExampleAppBaseException(DjangoCliBaseException):
    MODULE_CODE = AppModuleErrorCode.KEYBOARD


class DatasetNotFoundExc(ExampleAppBaseException):
    FUNCTION_ERROR_CODE = "00"
    MESSAGE_TEMPLATE = _("数据集：dataset_id -> {dataset_id} 不存在")
    MESSAGE = _("数据集不存在")
