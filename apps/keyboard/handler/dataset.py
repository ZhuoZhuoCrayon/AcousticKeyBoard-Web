# -*- coding: utf-8 -*-
import logging
import os
import shutil
import uuid

from django.conf import settings

from apps.keyboard.core import load

logger = logging.getLogger("apps")


class DatasetHandler:
    @classmethod
    def import_dataset(cls, dataset_file, verbose_name: str, description_more: str):
        tmp_dir = os.path.join(settings.TMP_ROOT, uuid.uuid4().hex)
        os.mkdir(tmp_dir)
        tmp_path = os.path.join(tmp_dir, dataset_file.name)
        logger.info(f"created tmp path -> [{tmp_path}]")
        with open(tmp_path, "wb") as tmp_file:
            for chunk in dataset_file.chunks():
                tmp_file.write(chunk)
        logger.info(f"moved dataset -> [{dataset_file.name}] to tmp path -> [{tmp_path}]")

        load.load_dataset_unit(tmp_path)

        shutil.rmtree(tmp_dir)
        logger.info(f"load dataset unit success, rm -rf tmp dir -> [{tmp_dir}]")
