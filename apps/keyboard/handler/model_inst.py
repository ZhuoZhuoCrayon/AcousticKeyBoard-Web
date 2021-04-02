# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List

import numpy as np

from apps.keyboard import constants, models
from apps.keyboard.core.format import get_mfcc
from apps.keyboard.core.vad import vad
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


TF_MODEL_CACHE = {}


class ModelInstHandler:
    @classmethod
    def predict(cls, inst_id: int, signal: List[int]) -> Dict[str, Any]:

        model_inst = models.AlgorithmModelInst.objects.get(id=inst_id)
        dataset = models.Dataset.objects.get(id=model_inst.dataset_id)

        tf_model = TF_MODEL_CACHE.get(model_inst.id)
        if not tf_model:
            tf_model = model_inst.load_model()
            TF_MODEL_CACHE[model_inst.id] = tf_model

        signal_np = np.asarray(signal, dtype=np.float64) / constants.TRANSFER_INT

        left, _ = vad(signal_np)

        signal_treated = signal_np[: dataset.length + 1]

        mfcc_feature = get_mfcc(vec=signal_treated, fs=dataset.fs)

        mfcc_feature = np.expand_dims(mfcc_feature, 0)
        scores = tf_model.model.predict(mfcc_feature)
        return {"label": constants.ID_LABEL_MAP[np.argmax(scores)]}

    @classmethod
    def debug(cls, inst_id: int, signal: List[int]) -> None:
        models.DatasetOriginalData.objects.create(
            dataset_id=inst_id, original_data=signal, label_type=constants.LabelType.TRAIN, label="DEBUG"
        )
