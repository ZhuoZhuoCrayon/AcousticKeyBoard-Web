# -*- coding: utf-8 -*-

from django.conf.urls import url
from django.urls import include
from rest_framework import routers

from apps.keyboard.views import dataset as dataset_views

router = routers.DefaultRouter(trailing_slash=True)
router.register(prefix="dataset", viewset=dataset_views.DatasetViews, basename="dataset")


urlpatterns = [url(r"", include(router.urls))]
