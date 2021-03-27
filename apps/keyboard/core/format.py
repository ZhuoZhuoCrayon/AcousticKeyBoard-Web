# -*- coding: utf-8 -*-
from typing import Dict, List

import numpy as np

from apps.keyboard.core.mfcc.mfcc_transform import mfcc_transform
from apps.keyboard.utils import concurrent


def gen_label_data_mfcc(label: str, label_data: np.ndarray, fs: int):
    return label, [mfcc_transform(vec, fs, 20, 6, 0.96, 28, 14, 1000, 10) for vec in label_data]


def batch_gen_mfcc_feature(data: Dict[str, np.ndarray], fs: int) -> Dict[str, List[np.ndarray]]:
    """
    multi_proc - 2.5s
    multi_thread - 16s
    serial - 8s
    :param data:
    :param fs:
    :return:
    """
    feature_data_items = concurrent.batch_call(
        func=gen_label_data_mfcc,
        params_list=[{"label": label, "label_data": label_data, "fs": fs} for label, label_data in data.items()],
        get_data=lambda result: result,
    )
    return dict(feature_data_items)
