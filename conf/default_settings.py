# -*- coding: utf-8 -*-
import platform

from djangocli.conf.djangocli_settings import *  # noqa

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.mysql",
        "NAME": APP_NAME,
        "USER": os.getenv("DC_MYSQL_NAME", "root"),
        "PASSWORD": os.getenv("DC_MYSQL_PASSWORD", ""),
        "HOST": os.getenv("DC_MYSQL_HOST", "localhost"),
        "PORT": os.getenv("DC_MYSQL_PORT", 3306),
        "TEST": {
            "NAME": f"{APP_NAME}_test",
            "CHARSET": "utf8mb4",
            "COLLATION": "utf8mb4_unicode_ci",
        },
    }
}

INSTALLED_APPS.extend(
    [
        # "apps.example",    # 用不到example
        "apps.keyboard"
    ]
)

AK_ROOT = os.getenv("AK_ROOT", BASE_DIR)

# 文件临时存储目录
TMP_ROOT = os.getenv("TMP_ROOT", "c:/" if platform.system() == "Windows" else "/tmp")
