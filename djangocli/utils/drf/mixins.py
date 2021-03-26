# -*- coding: utf-8 -*-
import json
import logging
from collections import OrderedDict

from django.conf import settings
from django.http import Http404, JsonResponse
from rest_framework import status
from rest_framework.exceptions import APIException
from rest_framework.response import Response

from djangocli import exceptions
from djangocli.constants import LogModule
from djangocli.utils.drf import base

logger = logging.getLogger(LogModule.APPS)


class ViewSetExceptionHandlerMixin:
    """统一异常处理"""

    def initialize_request(self, request, *args, **kwargs):
        # 打印请求日志
        # 默认content_type == multipart/form-data时传递文件，不打印相关body信息
        body = json.dumps({"type": "file"}) if request.content_type in ["multipart/form-data"] else request.body
        request_info = {"headers": dict(request.headers), "body": json.loads(body)}
        logging.getLogger(LogModule.API).info(
            f"{settings.APP_NAME} receive request: "
            f"api -> {request.path}, request_info -> \n {json.dumps(request_info, indent=2, ensure_ascii=False)}"
        )
        return super().initialize_request(request, *args, **kwargs)

    def dispatch(self, request, *args, **kwargs):
        """
        `.dispatch()` is pretty much the same as Django's regular dispatch,
        but with extra hooks for startup, finalize, and exception handling.
        """
        self.args = args
        self.kwargs = kwargs
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = self.default_response_headers  # deprecate?

        try:
            self.initial(request, *args, **kwargs)

            # Get the appropriate handler method
            if request.method.lower() in self.http_method_names:
                handler = getattr(self, request.method.lower(), self.http_method_not_allowed)
            else:
                handler = self.http_method_not_allowed

            response = handler(request, *args, **kwargs)

        except (exceptions.DjangoCliBaseException, APIException, Http404) as exc:
            response = self.handle_exception(exc)
        except Exception as exc:
            params = self.request.query_params if self.request.method == "GET" else self.request.data
            # logging.exception自带记录堆栈的功能，具体 -> https://stackoverflow.com/questions/1508467/log-exception-with-traceback
            logger.exception(f"Uncaptured error: api -> {request.path}, params -> {params}")
            response = self.handle_exception(exc)

        self.response = self.finalize_response(request, response, *args, **kwargs)
        return self.response

    def handle_exception(self, exc):
        """将异常格式化为统一格式"""
        if isinstance(exc, exceptions.DjangoCliBaseException):
            return super().handle_exception(exc)

        if isinstance(exc, APIException):
            exc = exceptions.DjangoCliRestApiException(exc)
        elif isinstance(exc, Http404):
            exc = exceptions.DjangoCliHttp404Exception(exc)
        else:
            exc = exceptions.DjangoCliSystemBaseException(exc)

        return super().handle_exception(exc)


class ViewSetResponseMixin:
    """统一返回格式"""

    def finalize_response(self, request, response, *args, **kwargs):
        """统一接口返回格式"""
        actual_status_code = response.status_code
        if isinstance(response, Response):
            response_data = base.build_response_dict(
                result=not response.exception,
                data=response.data or {},
                code=getattr(response, "code", base.SUCCESS_CODE),
                message=getattr(response, "message", "") if response.exception else base.SUCCESS_MSG,
            )
            response.data = response_data
            response.status_code = status.HTTP_200_OK
        elif isinstance(response, JsonResponse):
            response_data = json.loads(response.content)
        else:
            # 不可能有这种情况
            response_data = {"error": "not Response and  JsonResponse!"}

        # 打印接口返回日志
        logging.getLogger(LogModule.API).info(
            f"{settings.APP_NAME} response: api -> {request.path}, actual_status_code -> {actual_status_code}, "
            f"data -> \n{json.dumps(response_data, indent=2, ensure_ascii=False)}"
        )
        return super().finalize_response(request, response, *args, **kwargs)


class ViewSetValidationMixin:
    @property
    def query_data(self) -> OrderedDict:
        _query_data = getattr(self, "_query_data", None)

        if _query_data:
            return _query_data

        original_data = self.request.query_params if self.request.method == "GET" else self.request.data

        serializer_class = self.serializer_class or self.get_serializer_class()

        serializer_inst = serializer_class(data=original_data)
        serializer_inst.is_valid(raise_exception=True)

        self._query_data = serializer_inst.validated_data
        return self._query_data
