# -*- coding: utf-8 -*-
from django.utils.translation import ugettext_lazy as _

from apps.exceptions import AppModuleErrorCode
from djangocli.exceptions import DjangoCliBaseException


class KeyBoardAppBaseException(DjangoCliBaseException):
    MODULE_CODE = AppModuleErrorCode.KEYBOARD


class DatasetNotFoundExc(KeyBoardAppBaseException):
    FUNCTION_ERROR_CODE = "00"
    MESSAGE_TEMPLATE = _("数据集：dataset_id -> {dataset_id} 不存在")
    MESSAGE = _("数据集不存在")


class ModelInstNotReadyExc(KeyBoardAppBaseException):
    FUNCTION_ERROR_CODE = "01"
    MESSAGE_TEMPLATE = _("算法模型实例：inst_id -> {inst_id} 不可用")
    MESSAGE = _("算法模型实例不可用")


class ModelInstNotFoundExc(KeyBoardAppBaseException):
    FUNCTION_ERROR_CODE = "02"
    MESSAGE_TEMPLATE = _("算法模型实例：inst_id -> {inst_id} 不存在")
    MESSAGE = _("算法模型实例不存在")
