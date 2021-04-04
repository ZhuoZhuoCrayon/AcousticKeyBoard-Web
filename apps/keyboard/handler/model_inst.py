# -*- coding: utf-8 -*-
import logging
import uuid
from typing import Any, Dict, List

import numpy as np
from django.core.cache import cache

from apps.keyboard import constants, exceptions, models
from apps.keyboard.core.format import get_mfcc
from apps.keyboard.core.vad import vad
from djangocli.constants import LogModule, TimeUnit

logger = logging.getLogger(LogModule.APPS)


TF_MODEL_CACHE = {}


class ModelInstHandler:
    @classmethod
    def predict(cls, inst_id: int, signal: List[int]) -> Dict[str, Any]:

        model_inst = models.AlgorithmModelInst.objects.get(id=inst_id)
        dataset = models.Dataset.objects.get(id=model_inst.dataset_id)

        tf_model = TF_MODEL_CACHE.get(model_inst.save_path)
        if not tf_model:
            tf_model = model_inst.load_model()
            TF_MODEL_CACHE[model_inst.save_path] = tf_model

        signal_np = np.asarray(signal, dtype=np.float64) / constants.TRANSFER_INT

        left, _ = vad(signal_np)

        mfcc_feature = get_mfcc(vec=signal_np[: dataset.length + 1], fs=dataset.fs)

        # 短时间缓存，便于重试拿出
        cache_key = uuid.uuid4()
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
        mfcc_feature = cache.get(cache_key)
        if not mfcc_feature:
            raise exceptions.ModelInstFeatureCacheNotFoundExc({"dataset_id": dataset_id, "cache_key": cache_key})

        dataset_mfcc_feature_obj = models.DatasetMfccFeature(
            dataset_id=dataset_id, mfcc_feature=mfcc_feature, label=expect_label, label_type=constants.LabelType.TRAIN
        )
        dataset_mfcc_feature_obj.save()

        return {"dataset_mfcc_feature_id": dataset_mfcc_feature_obj.id}
