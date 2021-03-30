# -*- coding: utf-8 -*-

from django.conf.urls import url
from django.urls import include
from rest_framework import routers

from apps.keyboard.views import common as common_views
from apps.keyboard.views import dataset as dataset_views
from apps.keyboard.views import model_inst as model_inst_views

router = routers.DefaultRouter(trailing_slash=True)
router.register(prefix="dataset", viewset=dataset_views.DatasetViews, basename="dataset")
router.register(prefix="common", viewset=common_views.CommonViews, basename="common")
router.register(prefix="model_inst", viewset=model_inst_views.ModelInstViews, basename="model_inst")

urlpatterns = [url(r"", include(router.urls))]
