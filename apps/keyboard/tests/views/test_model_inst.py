# -*- coding: utf-8 -*-
import os

import numpy as np
from django.conf import settings

from apps.keyboard import constants, models
from apps.keyboard.core import load
from djangocli.utils.unittest.testcase import DjangoCliAPITestCase


class ModelInstTestView(DjangoCliAPITestCase):
    dataset_obj = None
    model_inst_obj = None
    dataset_unit = None

    @classmethod
    def setUpTestData(cls):
        """Hook in testcase.__call__ , before setUpClass"""

        dataset_create_data = {
            "verbose_name": "0307-全键盘-0号麦克风-2000个样本点",
            "dataset_name": "20210307",
            "data_type": "all-0micro",
            "project_type": "single",
            "description": "cxx_0307_1__0306_123",
            "description_more": "0307-30键训练集，其余为测试集，0号麦克风，2000个样本点",
            "length": 2000,
            "fs": 48000,
        }

        cls.dataset_obj = models.Dataset(**dataset_create_data)
        cls.dataset_obj.save()

        model_inst_create_data = {
            "dataset_id": cls.dataset_obj.id,
            "algorithm": "blstm",
            "save_path": os.path.join(settings.MODEL_INST_ROOT, "dataset_id:10:algorithm:blstm/inst_id:7"),
            "extra_info": {"input_shape_list": [4, 14], "max_label_num": 29},
            "is_ready": True,
            "is_latest": True,
        }
        cls.model_inst_obj = models.AlgorithmModelInst(**model_inst_create_data)
        cls.model_inst_obj.save()

        cls.dataset_unit = load.load_dataset_unit(
            save_path=os.path.join(settings.LIB_ROOT, "preprocessed_data/2250.pickle")
        )

        super().setUpTestData()

    def test_predict(self):
        total_count = 0
        total_correct_count = 0
        back_signal_count = 0
        for label in self.dataset_unit.test_data:
            count = 0
            correct_count = 0
            label_data = self.dataset_unit.test_data[label]
            for signal in label_data:
                signal = (signal * constants.TRANSFER_INT).astype(np.int)
                if np.max(signal) < 0:
                    back_signal_count = back_signal_count + 1
                    continue
                pred_label = self.client.post(
                    path="/api/v1/keyboard/model_inst/predict/",
                    data={
                        "dataset_id": self.dataset_obj.id,
                        "algorithm": constants.AlgorithmModel.BLSTM,
                        "signal": signal.tolist(),
                    },
                )["data"]["label"]
                if pred_label == label:
                    correct_count = correct_count + 1
                count = count + 1
            total_count += count
            total_correct_count += correct_count
            print(
                f"label -> {label}, count -> {count}, correct_count -> {correct_count}, acc -> {correct_count / count}"
            )
        print(
            f"total_count -> {total_count}, total_correct_count -> {total_correct_count}, "
            f"average -> {total_correct_count / total_count}, back_signal_count -> {back_signal_count}"
        )

    def test_get_signal_threshold(self):
        signal_max_list = []
        back_signal_count = 0
        for label in self.dataset_unit.test_data:
            label_data = self.dataset_unit.test_data[label]
            for signal in label_data:
                signal = (signal * constants.TRANSFER_INT).astype(np.int)
                signal_max = np.max(signal)
                if signal_max < 10000:
                    back_signal_count = back_signal_count + 1
                    continue
                signal_max_list.append(signal_max)
        print(min(signal_max_list), max(signal_max_list), back_signal_count)


# 8000
# label -> O, count -> 61, correct_count -> 52, acc -> 0.8524590163934426
# label -> S, count -> 64, correct_count -> 17, acc -> 0.265625
# label -> B, count -> 40, correct_count -> 35, acc -> 0.875
# label -> L_SHIFT, count -> 63, correct_count -> 59, acc -> 0.9365079365079365
# label -> R, count -> 62, correct_count -> 11, acc -> 0.1774193548387097
# label -> V, count -> 61, correct_count -> 25, acc -> 0.4098360655737705
# label -> G, count -> 61, correct_count -> 46, acc -> 0.7540983606557377
# label -> F, count -> 61, correct_count -> 8, acc -> 0.13114754098360656
# label -> Q, count -> 63, correct_count -> 36, acc -> 0.5714285714285714
# label -> BACKSPACE, count -> 62, correct_count -> 60, acc -> 0.967741935483871
# label -> C, count -> 61, correct_count -> 61, acc -> 1.0
# label -> Y, count -> 60, correct_count -> 48, acc -> 0.8
# label -> H, count -> 65, correct_count -> 29, acc -> 0.4461538461538462
# label -> X, count -> 51, correct_count -> 50, acc -> 0.9803921568627451
# label -> D, count -> 65, correct_count -> 63, acc -> 0.9692307692307692
# label -> W, count -> 64, correct_count -> 52, acc -> 0.8125
# label -> P, count -> 63, correct_count -> 60, acc -> 0.9523809523809523
# label -> N, count -> 57, correct_count -> 48, acc -> 0.8421052631578947
# label -> K, count -> 64, correct_count -> 54, acc -> 0.84375
# label -> ENTER, count -> 60, correct_count -> 59, acc -> 0.9833333333333333
# label -> I, count -> 63, correct_count -> 60, acc -> 0.9523809523809523
# label -> A, count -> 37, correct_count -> 29, acc -> 0.7837837837837838
# label -> R_CTRL, count -> 64, correct_count -> 52, acc -> 0.8125
# label -> T, count -> 64, correct_count -> 30, acc -> 0.46875
# label -> L, count -> 63, correct_count -> 58, acc -> 0.9206349206349206
# label -> U, count -> 65, correct_count -> 61, acc -> 0.9384615384615385
# label -> M, count -> 48, correct_count -> 39, acc -> 0.8125
# label -> E, count -> 65, correct_count -> 62, acc -> 0.9538461538461539
# label -> J, count -> 62, correct_count -> 56, acc -> 0.9032258064516129
# total_count -> 1739, total_correct_count -> 1320, average -> 0.7590569292696953, back_signal_count -> 146
