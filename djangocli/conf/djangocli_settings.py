# -*- coding: utf-8 -*-

from djangocli.conf.default_settings import *  # noqa
from djangocli.constants import LogModule
from djangocli.utils import string

INSTALLED_APPS.extend(
    [
        "rest_framework",
        "drf_yasg",  # swagger's support
    ]
)

REST_FRAMEWORK = {
    "DATETIME_FORMAT": "%Y-%m-%d %H:%M:%S",
    # "DEFAULT_AUTHENTICATION_CLASSES": (
    #     "rest_framework.authentication.BasicAuthentication",
    #     "rest_framework.authentication.SessionAuthentication",
    #     'rest_framework.authentication.TokenAuthentication',
    # ),
    # "DEFAULT_RENDERER_CLASSES": (
    #     # 自定义渲染器
    #     # "apps.utils.drf.CustomJsonRender",
    #     "rest_framework.renderers.JSONRenderer",
    # ),
    "EXCEPTION_HANDLER": "djangocli.utils.drf.view.django_cli_exception_handler",
    # djangocli.utils.drf.view.DjangoCliGenericViewSet 中亦可指定分页器
    "DEFAULT_PAGINATION_CLASS": "djangocli.utils.drf.filter.DjangoCliPageNumberPagination",
}


ALLOWED_HOSTS = ["*"]

# REDIS
DEFAULT_REDIS_PORT = 6379

# Cache
CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": string.get_redis_url(
            host=os.getenv("DC_REDIS_HOST", "localhost"), port=os.getenv("DC_REDIS_PORT", 6379), db_index=0
        ),
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "PASSWORD": os.getenv("DC_REDIS_PASSWORD", ""),
            # 最大连接数量
            "CONNECTION_POOL_KWARGS": {"max_connections": 100},
        },
        "KEY_FUNCTION": "djangocli.utils.redis.django_cache_key_maker",
    }
}


# Celery's config

CELERY_BROKER_URL = string.get_redis_url(
    password=os.getenv("DC_REDIS_PASSWORD", ""),
    host=os.getenv("DC_REDIS_HOST", "localhost"),
    port=os.getenv("DC_REDIS_PORT", 6379),
    db_index=1,
)

#: Only add pickle to this list if your broker is secured
#: from unwanted access (see userguide/security.html)
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
CELERY_TASK_SERIALIZER = "json"


# LOG

LOGGING_FILE_ROOT = os.path.join(BASE_DIR, "logs")

LOGGING_FILE = {
    "apps_log_file": os.path.join(LOGGING_FILE_ROOT, "apps.log"),
    "dc_log_file": os.path.join(LOGGING_FILE_ROOT, "dc.log"),
    "api_log_file": os.path.join(LOGGING_FILE_ROOT, "api.log"),
    "django_request_log_file": os.path.join(LOGGING_FILE_ROOT, "django_request.log"),
    "django_server_log_file": os.path.join(LOGGING_FILE_ROOT, "django_server.log"),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(levelname)s [%(asctime)s] %(name)s | %(funcName)s | %(lineno)d %(message)s"},
        "verbose": {
            "format": "%(levelname)s [%(asctime)s] %(name)s | %(funcName)s | %(lineno)d | %(pathname)s"
            "\n %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "django_api": {
            "format": "%(levelname)s [%(asctime)s] %(name)s | %(funcName)s | %(lineno)d | %(pathname)s "
            " \n %(message)s status_code:%(status_code)d",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {"level": "INFO", "class": "logging.StreamHandler", "formatter": "simple"},
        "django_request_log_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGGING_FILE["django_request_log_file"],
            "formatter": "django_api",
        },
        "django_server_log_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGGING_FILE["django_server_log_file"],
            "formatter": "django_api",
        },
        "apps_log_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGGING_FILE["apps_log_file"],
            "formatter": "verbose",
        },
        "dc_log_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGGING_FILE["dc_log_file"],
            "formatter": "verbose",
        },
        "api_log_file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGGING_FILE["api_log_file"],
            "formatter": "verbose",
        },
    },
    "loggers": {
        LogModule.DJANGO: {
            "handlers": ["console"],
            "propagate": True,
        },
        LogModule.DJANGO_REQUEST: {
            "handlers": ["django_request_log_file"],
            "level": "INFO",
            "propagate": True,
        },
        LogModule.DJANGO_SERVER: {
            "handlers": ["django_server_log_file"],
            "level": "INFO",
            "propagate": True,
        },
        LogModule.DJANGO_CLI: {"handlers": ["console", "dc_log_file"], "level": "INFO"},
        LogModule.APPS: {"handlers": ["console", "apps_log_file"], "level": "INFO"},
        LogModule.API: {"handlers": ["api_log_file"], "level": "INFO"},
    },
}
