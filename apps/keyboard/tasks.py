# -*- coding: utf-8 -*-
import itertools
import logging
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from celery import shared_task
from django.db import transaction

from apps.keyboard import constants, models
from apps.keyboard.core import load, parse
from apps.keyboard.core.algorithm import tf_models
from apps.keyboard.utils import tools
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


def get_model_data(
    dataset_id: int, label_type: str, per_action_rate: float
) -> Tuple[Dict[str, List[int]], Dict[str, List[np.ndarray]]]:
    mfcc_feature_infos = models.DatasetMfccFeature.objects.filter(dataset_id=dataset_id, label_type=label_type).values(
        "id", "label"
    )

    feature_id_gby_label = defaultdict(list)
    for mfcc_feature_info in mfcc_feature_infos:
        feature_id_gby_label[mfcc_feature_info["label"]].append(mfcc_feature_info["id"])

    model_data_info = tools.data_select(original_data=feature_id_gby_label, per_action_rate=per_action_rate)

    mfcc_feature_objs = models.DatasetMfccFeature.objects.filter(
        id__in=list(itertools.chain(*list(model_data_info.values())))
    )

    feature_gby_label = defaultdict(list)
    for mfcc_feature_obj in mfcc_feature_objs:
        feature_gby_label[mfcc_feature_obj.label].append(np.asarray(mfcc_feature_obj.mfcc_feature, dtype=np.float64))

    return model_data_info, feature_gby_label


@shared_task
def train_dataset(dataset_id: int, per_train_rate: float, **kwargs):
    logger.info(f"celery task: task -> train_dataset, dataset_id -> {dataset_id} begin")

    train_info, train_data = get_model_data(dataset_id, constants.LabelType.TRAIN, per_train_rate)
    _, test_data = get_model_data(dataset_id, constants.LabelType.TEST, per_action_rate=0.4)

    labels = list(train_info.keys())
    shape = train_data[labels[0]][0].shape

    logger.info(
        f"dataset_id -> {dataset_id}, train_labels -> {labels}, shape -> {shape} " f"train_info -> \n {train_info}"
    )

    blstm_model = tf_models.BLstmModel(max_label_num=len(constants.ALL_CLASS_LABELS), input_shape=shape)

    # first_labels = [chr(chr_ord) for chr_ord in range(ord("A"), ord("M"))]
    # # 先训练一部分数据
    # blstm_model.train({label: train_data[label] for label in train_data if label in first_labels})
    # blstm_model.test(test_data)
    # # 训练余下数据
    # blstm_model.train({label: train_data[label] for label in train_data if label not in first_labels})
    # blstm_model.test(test_data)

    blstm_model.train(train_data)

    with transaction.atomic():
        model_inst = models.AlgorithmModelInst.objects.create(
            dataset_id=dataset_id, train_info=train_info, algorithm=blstm_model.MODEL_NAME
        )
        model_inst.save_model(blstm_model)

        blstm_model_stored = model_inst.load_model()

        # 测试模型精度
        blstm_model_stored.test(test_data)

    logger.info(f"celery task: task -> train_dataset, dataset_id -> {dataset_id} finished")
