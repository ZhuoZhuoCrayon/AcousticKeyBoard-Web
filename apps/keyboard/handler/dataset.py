# -*- coding: utf-8 -*-
import logging
import os
import uuid

from django.conf import settings

from apps.keyboard import tasks
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


class DatasetHandler:
    @classmethod
    def import_dataset(cls, dataset_file, verbose_name: str, description_more: str):
        tmp_dir = os.path.join(settings.TMP_ROOT, uuid.uuid4().hex)
        os.mkdir(tmp_dir)
        tmp_path = os.path.join(tmp_dir, dataset_file.name)
        logger.info(f"created tmp path -> {tmp_path}")
        with open(tmp_path, "wb") as tmp_file:
            for chunk in dataset_file.chunks():
                tmp_file.write(chunk)
        logger.info(f"moved dataset -> {dataset_file.name} to tmp path -> {tmp_path}")

        # 暂时不存储原数据
        task_id = tasks.import_dataset.delay(
            dataset_save_path=tmp_path,
            verbose_name=verbose_name,
            description_more=description_more,
            save_original_data=False,
            tmp_dir=tmp_dir,
        ).task_id
        logger.info(f"import dataset to db: task_id -> {task_id}")
        return task_id

    @classmethod
    def train(cls, dataset_id: int, per_train_rate: float):
        task_id = tasks.train_dataset.delay(dataset_id=dataset_id, per_train_rate=per_train_rate).task_id
        logger.info(f"import dataset to db: task_id -> {task_id}")
        return task_id
