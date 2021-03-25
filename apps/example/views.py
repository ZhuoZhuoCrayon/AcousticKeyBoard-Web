import logging

from django.utils.translation import ugettext_lazy as _
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.example import exceptions, handler, models, serializers
from djangocli.constants import LogModule
from djangocli.utils.drf import view

# Create your views here.

logger = logging.getLogger(LogModule.APPS)


class ExampleBookViews(view.DjangoCliModelViewSet):
    model = models.Book
    serializer_class = serializers.BookModelSer

    def get_queryset(self):
        return self.model.objects.all()

    @swagger_auto_schema(
        operation_summary=_("查询书籍"),
        tags=["book"],
        request_body=serializers.BookSearchRequestSer(),
        responses={status.HTTP_200_OK: serializers.BookSearchResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.BookSearchRequestSer)
    def search(self, request, *args, **kwargs):

        return Response({"query_data": self.query_data, "list": self.get_queryset().values()})


class ExampleAuthorViews(view.DjangoCliModelViewSet):
    model = models.Author
    serializer_class = serializers.AuthorModelSer

    def get_queryset(self):
        return self.model.objects.all()


class ExamplePublisherView(view.DjangoCliModelViewSet):
    model = models.Publisher
    serializer_class = serializers.PublisherModelSer

    def get_queryset(self):
        return self.model.objects.all()


class ExampleCommonViews(view.DjangoCliGenericViewSet):
    @swagger_auto_schema(
        operation_summary=_("主动抛出预期异常"),
        tags=["common"],
        request_body=serializers.CommonExceptionRequestSer(),
        responses={status.HTTP_200_OK: serializers.CommonExceptionResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.CommonExceptionRequestSer)
    def expected_exception(self, request, *args, **kwargs):
        logger.warning(_("用户 -> {user} 即将主动抛出了一个异常").format(user=request.user))
        raise exceptions.CommonExceptionException(context={"user": request.user})

    @swagger_auto_schema(
        operation_summary=_("系统错误（非预期）异常"),
        tags=["common"],
        request_body=serializers.CommonUnExceptionRequestSer(),
        responses={status.HTTP_200_OK: serializers.CommonUnExceptionResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.CommonUnExceptionRequestSer)
    def unexpected_exception(self, request, *args, **kwargs):
        return Response({"name": request.user["no_name"]})

    @swagger_auto_schema(
        operation_summary=_("序列化器校验异常"),
        tags=["common"],
        request_body=serializers.CommonValidateExceptionRequestSer(),
        responses={status.HTTP_200_OK: serializers.CommonValidateExceptionResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.CommonValidateExceptionRequestSer)
    def validate_exception(self, request, *args, **kwargs):
        return Response({"query_data": self.query_data})

    @swagger_auto_schema(
        operation_summary=_("Celery异步任务"),
        tags=["common"],
        request_body=serializers.CommonCeleryDelayRequestSer(),
        responses={status.HTTP_200_OK: serializers.CommonCeleryDelayResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.CommonCeleryDelayRequestSer)
    def celery_delay(self, request, *args, **kwargs):
        return Response(
            {
                "task_id": handler.CommonHandler.celery_delay(
                    left_val=self.query_data["left_val"],
                    right_val=self.query_data["right_val"],
                    operate=self.query_data["operate"],
                )
            }
        )

    @swagger_auto_schema(
        operation_summary=_("获取Celery执行结果"),
        tags=["common"],
        request_body=serializers.CommonBatchCeleryResultsRequestSer(),
        responses={status.HTTP_200_OK: serializers.CommonBatchCeleryResultsResponseSer()},
    )
    @action(methods=["POST"], detail=False, serializer_class=serializers.CommonBatchCeleryResultsRequestSer)
    def batch_celery_results(self, request, *args, **kwargs):
        return Response(handler.CommonHandler.batch_celery_results(task_ids=self.query_data["task_ids"]))
