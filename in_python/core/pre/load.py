# -*- coding: utf-8 -*-
import datetime
import logging
import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy

from apps.keyboard import constants
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


class DataSetUnitBase:
    SAVE_FILE_SUFFIX = f".{constants.FileType.PICKLE}"

    SEP = "#"

    def __init__(
        self,
        dataset_name: str,
        project_type: str,
        data_type: str,
        description: str,
        length: int,
        fs: int,
    ):
        self.dataset_name = dataset_name
        self.project_type = project_type
        self.data_type = data_type
        self.description = description
        self.length = length
        self.fs = fs

    def get_save_path(self, get_root: bool = False):
        """
        project_type
            - dataset_name  [一般用日期]
                - {data_type}|{description}
                    - length[SAVE_FILE_SUFFIX]
        """
        save_root = os.path.join(
            constants.PREPROCESSED_DATA_ROOT,
            self.project_type,
            self.dataset_name,
            self.SEP.join([self.data_type, self.description]),
        )
        return save_root if get_root else os.path.join(save_root, f"{self.length}{self.SAVE_FILE_SUFFIX}")


class DataSetUnit(DataSetUnitBase):
    def __init__(
        self,
        train_data: Dict[str, numpy.ndarray],
        test_data: Dict[str, numpy.ndarray],
        dataset_name: str,
        project_type: str,
        data_type: str,
        description: str,
        length: int,
        fs: int,
    ):

        super(DataSetUnit, self).__init__(
            dataset_name=dataset_name,
            project_type=project_type,
            data_type=data_type,
            description=description,
            length=length,
            fs=fs,
        )

        self.train_data = train_data
        self.test_data = test_data
        self.labels = list(train_data.keys())

    def __str__(self):
        return (
            f"\n {'-' * 80} \n "
            f"DATASET_NAME: {self.dataset_name} \n "
            f"PROJECT_TYPE: {self.project_type} \n "
            f"DATA_TYPE: {self.data_type} \n "
            f"DESCRIPTION: {self.description} \n "
            f"FS: {self.fs}\n "
            f"LENGTH: {self.length}\n "
            f"LABELS: {self.labels} \n "
            f"TRAIN_DATA_INFO: {self.get_data_info(self.train_data)} \n "
            f"TEST_DATA_INFO: {self.get_data_info(self.test_data)} \n "
            f"SAVE_PATH: {self.get_save_path()} \n "
            f"{'-' * 80} \n "
        )

    def get_result_save_path(
        self,
        accuracy: float,
        save_type: str,
        file_format: str,
        algorithm: str,
        select_policy: Tuple[float, float] = (1.0, 1.0),
    ) -> str:
        ds_save_root = self.get_save_path(get_root=True)
        base_root = ds_save_root.replace(constants.PREPROCESSED_DATA_ROOT, constants.RESULT_ROOT)
        select_policy_str = f"train{self.SEP}{select_policy[0]}{self.SEP}test{self.SEP}{select_policy[1]}"
        file_name = f"{str(datetime.date.today())}{self.SEP}{round(accuracy, 2)}.{file_format}"
        save_root = os.path.join(base_root, algorithm, str(self.length), select_policy_str, save_type)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        return os.path.join(save_root, file_name)

    @staticmethod
    def get_data_info(data: Dict[str, numpy.ndarray]):
        labels = list(data.keys())
        label_data = data[labels[random.randint(0, len(labels) - 1)]]
        label_data_number, length = label_data.shape
        return {"count": label_data_number, "length": length}

    @staticmethod
    def plot_data(data):
        plt.figure()
        plt.plot(data)
        plt.show()
