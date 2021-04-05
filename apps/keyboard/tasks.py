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
from djangocli.constants import LogModule, TimeUnit
from djangocli.utils import redis

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
    dataset_id: int, label_type: str, per_action_rate: float, top_num: int = None
) -> Tuple[Dict[str, List[int]], Dict[str, List[np.ndarray]]]:
    mfcc_feature_infos = models.DatasetMfccFeature.objects.filter(dataset_id=dataset_id, label_type=label_type).values(
        "id", "label"
    )

    feature_id_gby_label = defaultdict(list)
    for mfcc_feature_info in mfcc_feature_infos:
        feature_id_gby_label[mfcc_feature_info["label"]].append(mfcc_feature_info["id"])

    if not top_num:
        model_data_info = tools.data_select(original_data=feature_id_gby_label, per_action_rate=per_action_rate)
    else:
        # 取后插入的样本
        model_data_info = {
            label: sorted(feature_ids)[0 if top_num > len(feature_ids) else len(feature_ids) - top_num :]
            for label, feature_ids in feature_id_gby_label.items()
        }

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
    _, test_data = get_model_data(dataset_id, constants.LabelType.TEST, per_action_rate=per_train_rate)

    labels = list(train_info.keys())
    shape = train_data[labels[0]][0].shape

    logger.info(
        f"dataset_id -> {dataset_id}, train_labels -> {labels}, shape -> {shape} " f"train_info -> \n {train_info}"
    )

    blstm_model = tf_models.BLstmModel(max_label_num=len(constants.ALL_CLASS_LABELS), input_shape=shape)

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


def correct_model_lock(correct_model_func):
    def wrapper(dataset_id: int, **kwargs):
        lock_name = f":dataset:{dataset_id}:correct"
        with redis.RedisLock(lock_name=lock_name, lock_expire=30 * TimeUnit.MINUTE) as identifier:
            if identifier is None:
                # 如果被低耗时任务占用或者有其他模型矫正任务触发，结束，等待下一次触发
                logger.info(f"celery task: task -> correct_model, dataset_id -> {dataset_id}, 被低耗时任务占用，结束，等待下一次触发")
                return
            return correct_model_func(dataset_id, **kwargs)

    return wrapper


@shared_task
@correct_model_lock
def correct_model(dataset_id: int):
    base_log_content = f"celery task: task -> correct_model, dataset_id -> {dataset_id}"
    logger.info(f"{base_log_content} begin")

    redis_conn = redis.get_redis_conn()

    correct_info = dict(redis_conn.hgetall(f"{redis.REDIS_KEY_PREFIX.WEB_CACHE}:dataset:{dataset_id}:correct"))
    correct_info = {k: int(v) for k, v in correct_info.items()}

    logger.info(f"correct_info -> {correct_info}")

    if not (correct_info and sum(correct_info.values()) > constants.CORRECT_MODEL_THRESHOLD):
        logger.info(
            f"{base_log_content}, error num -> {sum(correct_info.values())}, "
            f"threshold ->（{constants.CORRECT_MODEL_THRESHOLD}"
        )
        return

    logger.info(f"{base_log_content} 模型调优开始（错误键数 > {constants.CORRECT_MODEL_THRESHOLD})")

    # 取前20个样本进行训练
    train_info, train_data = get_model_data(dataset_id, constants.LabelType.TRAIN, 1, 20)

    labels = list(train_info.keys())
    shape = train_data[labels[0]][0].shape

    logger.info(f"{base_log_content}, train_labels -> {labels}, shape -> {shape} " f"train_info -> \n {train_info}")

    blstm_model = tf_models.BLstmModel(max_label_num=len(constants.ALL_CLASS_LABELS), input_shape=shape)

    blstm_model.train(train_data)

    with transaction.atomic():
        model_inst = models.AlgorithmModelInst.objects.create(
            dataset_id=dataset_id, train_info=train_info, algorithm=blstm_model.MODEL_NAME
        )
        model_inst.save_model(blstm_model)

    # 把更新部分置0
    redis_conn.hmset(
        f"{redis.REDIS_KEY_PREFIX.WEB_CACHE}:dataset:{dataset_id}:correct",
        {label: 0 for label, error_num in correct_info.items() if error_num},
    )

    logger.info(f"{base_log_content} finished")
