# -*- coding: utf-8 -*-
import logging
import uuid
from typing import Any, Dict, List, Union

import numpy as np
from django.core.cache import cache

from apps.keyboard import constants, exceptions, memory_cache, models, tasks
from apps.keyboard.core.format import get_mfcc
from apps.keyboard.core.vad import vad
from djangocli.constants import LogModule, TimeUnit
from djangocli.utils import redis

logger = logging.getLogger(LogModule.APPS)


class ModelInstHandler:

    redis_conn = redis.get_redis_conn()

    @classmethod
    def predict(cls, inst_id: int, signal: List[int]) -> Dict[str, Any]:

        model_inst = models.AlgorithmModelInst.objects.get(id=inst_id)
        dataset = models.Dataset.objects.get(id=model_inst.dataset_id)

        tf_model = memory_cache.TF_MODEL_CACHE.get(model_inst.save_path)
        if not tf_model:
            logger.info(f"save_path -> {model_inst.save_path}")

            tf_model = model_inst.load_model()
            memory_cache.TF_MODEL_CACHE[model_inst.save_path] = tf_model

        signal_np = np.asarray(signal, dtype=np.float64) / constants.TRANSFER_INT

        left, _ = vad(signal_np)

        mfcc_feature = get_mfcc(vec=signal_np[: dataset.length + 1], fs=dataset.fs)

        # 短时间缓存，便于重试拿出
        cache_key = str(uuid.uuid4())
        cache.set(cache_key, mfcc_feature, 5 * TimeUnit.MINUTE)

        scores = tf_model.model.predict(np.expand_dims(mfcc_feature, 0))
        return {"label": constants.ID_LABEL_MAP[np.argmax(scores)], "cache_key": cache_key}

    @classmethod
    def debug(cls, inst_id: int, signal: List[int], label: str) -> None:
        if signal[0] == 0 and signal[-1] == 3999:
            logger.info("假数据，不记录到DB")
            return
        models.DatasetOriginalData.objects.create(
            dataset_id=inst_id, original_data=signal, label_type=constants.LabelType.TRAIN, label=f"DEBUG-{label}"
        )

    @classmethod
    def correct(cls, dataset_id: int, cache_key: str, expect_label: str) -> Dict[str, Any]:
        mfcc_feature: Union[np.ndarray, None] = cache.get(cache_key)
        if mfcc_feature is None:
            raise exceptions.ModelInstFeatureCacheNotFoundExc({"dataset_id": dataset_id, "cache_key": cache_key})

        dataset_mfcc_feature_obj = models.DatasetMfccFeature(
            dataset_id=dataset_id,
            mfcc_feature=mfcc_feature.tolist(),
            label=expect_label,
            label_type=constants.LabelType.TRAIN,
        )
        dataset_mfcc_feature_obj.save()

        lock_name = f":dataset:{dataset_id}:correct"

        with redis.RedisLock(lock_name=lock_name, lock_expire=5 * TimeUnit.SECOND) as identifier:
            correct_hash_key = f"{redis.REDIS_KEY_PREFIX.WEB_CACHE}:dataset:{dataset_id}:correct"
            correct_hash_tmp_key = f"{redis.REDIS_KEY_PREFIX.WEB_CACHE}:dataset:{dataset_id}:correct:tmp"

            if identifier is None:
                # 模型正在重新训练，抢不到锁，先把数据怼到备份hash中
                # 错误个数 +1
                cls.redis_conn.hincrby(correct_hash_tmp_key, expect_label)
                logger.info(f"模型正在重新训练，抢不到锁，先把数据怼到备份hash -> {correct_hash_tmp_key} 中")
            else:
                correct_info_tmp = {
                    k: int(v) for k, v in dict(cls.redis_conn.hgetall(correct_hash_tmp_key)).items() if int(v)
                }
                # 先把这部分数据通过pipeline原子操作减除
                redis_pipeline = cls.redis_conn.pipeline()
                for label, err_num in correct_info_tmp:
                    redis_pipeline.hincrby(correct_hash_tmp_key, label, -err_num)

                # 错误个数 +1
                redis_pipeline.hincrby(correct_hash_key, expect_label)

                # 如果抢到锁，把备份数据怼到实时数据中
                for label, err_num in correct_info_tmp:
                    redis_pipeline.hincrby(correct_hash_key, label, err_num)

                response = redis_pipeline.execute()

                # 反查出临时数据，查看是否准确
                correct_info_tmp_r = {
                    k: int(v) for k, v in dict(cls.redis_conn.hgetall(correct_hash_tmp_key)).items() if int(v)
                }
                logger.info(
                    f"correct_info_tmp redis_pipeline response -> {response}, "
                    f"correct_info_tmp_r -> {correct_info_tmp_r}"
                )
                logger.info(f"暂无模型重新训练任务，label -> {expect_label} 错误个数 +1 -> {correct_hash_key}")

        # 异步调度
        tasks.correct_model.delay(dataset_id=dataset_id)

        return {"dataset_mfcc_feature_id": dataset_mfcc_feature_obj.id}
