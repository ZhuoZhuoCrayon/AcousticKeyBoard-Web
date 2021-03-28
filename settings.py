# -*- coding: utf-8 -*-

import os
from pathlib import Path

import dotenv

from scripts import check_and_create_log_file, dir_checker


class EnvType:
    DEV = "dev"
    PROD = "prod"
    STAG = "stag"


# 对于本地开发环境，通过建立dc_dev.env文件，可以在启动时便导入环境变量，解决envfile不支持在Pycharm Terminal / Python Console
# 中导入环境变量的问题，当然也可以将env/script的脚本分别加入到PyCharm的启用脚本（Terminal仍不支持前置命令；）
# 为防止信息泄漏，禁止生产环境通过push .env文件到源码
dev_env_file_path = f"{Path(__file__).resolve().parent}/env/dc_dev.env"
if os.path.exists(dev_env_file_path):
    dotenv.load_dotenv(dotenv_path=dev_env_file_path)

ENV = os.getenv("DJANGO_CLI_ENV", EnvType.DEV)

DJANGO_CONF_MODULE = "conf.{env}".format(env=ENV)

try:
    _module = __import__(DJANGO_CONF_MODULE, globals(), locals(), ["*"])
except ImportError as error:
    raise ImportError(f"Could not import config {DJANGO_CONF_MODULE} (Is it on sys.path?): {error}")

for _setting in dir(_module):
    if _setting == _setting.upper():
        locals()[_setting] = getattr(_module, _setting)

# 需要覆盖全局的变量
BASE_DIR = Path(__file__).resolve().parent


# check and create log file
check_and_create_log_file.check_and_create_log_file(log_file_paths=list(_module.LOGGING_FILE.values()))
dir_checker.dir_checker(dir_roots=[_module.MODEL_INST_ROOT])
