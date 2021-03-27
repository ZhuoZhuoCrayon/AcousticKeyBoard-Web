# -*- coding: utf-8 -*-
import logging
import shutil
from typing import Any, Dict

from celery import shared_task

from apps.keyboard.core import load, parse
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


@shared_task
def import_dataset(
    dataset_save_path, verbose_name: str, description_more: str, save_original_data=True, *args, **kwargs
) -> Dict[str, Any]:
    logger.info(f"celery task: task -> import_dataset, dataset_save_path -> {dataset_save_path} begin")
    dataset_unit = load.load_dataset_unit(save_path=dataset_save_path)

    if "tmp_dir" in kwargs:
        shutil.rmtree(kwargs["tmp_dir"])
        logger.info(f"load dataset unit success, rm -rf tmp dir -> {kwargs['tmp_dir']}")

    dataset_parse = parse.DatasetParse(
        dataset_unit=dataset_unit, verbose_name=verbose_name, description_more=description_more
    )
    dataset_id = dataset_parse.import_db(save_original_data=save_original_data)

    logger.info(f"celery task: task -> import_dataset, dataset_id -> {dataset_id} finished")
    return {"dataset_id": dataset_id}
