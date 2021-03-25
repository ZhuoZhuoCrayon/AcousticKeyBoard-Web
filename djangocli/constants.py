# -*- coding: utf-8 -*-


class TimeUnit:
    SECOND = 1
    MINUTE = SECOND * 60
    HOUR = MINUTE * 60
    DAY = MINUTE * 24


class LogModule:
    DJANGO_CLI = "dc"
    APPS = "apps"
    DJANGO = "django"
    DJANGO_REQUEST = "django.request"
    DJANGO_SERVER = "django.server"
