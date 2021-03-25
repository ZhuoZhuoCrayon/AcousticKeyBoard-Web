# -*- coding: utf-8 -*-
import logging
import pickle

from djangocli.constants import LogModule
from in_python.core.pre.load import DataSetUnit

logger = logging.getLogger(LogModule.APPS)


def load_dataset_unit(save_path: str) -> DataSetUnit:
    with open(file=save_path, mode="rb") as ds_unit_reader:
        dataset_unit = pickle.load(ds_unit_reader)
        logger.info(f"read dataset_unit: {dataset_unit}")
    return dataset_unit
