# -*- coding: utf-8 -*-
import os
from typing import List, Tuple

from django.conf import settings
from django.utils.translation import ugettext_lazy as _


class ProjectType:
    SINGLE = "single"
    COMBINE = "combine"

    @staticmethod
    def get_choices() -> Tuple:
        return (
            (ProjectType.SINGLE, _("单键")),
            (ProjectType.SINGLE, _("组合键")),
        )


class DataType:
    ALL_1MICRO = "all-1micro"
    ALL_0MICRO = "all-0micro"
    ALL_1MICRO_FAR12 = "all-1micro-far12"
    ALL_0MICRO_FAR12 = "all-0micro-far12"
    ALL_1MICRO_NEAR12 = "all-1micro-near12"
    ALL_0MICRO_NEAR12 = "all-0micro-near12"

    @staticmethod
    def get_choices() -> Tuple:
        return (
            (DataType.ALL_1MICRO, _("1号麦克风-30个键位")),
            (DataType.ALL_0MICRO, _("0号麦克风-30个键位")),
            (DataType.ALL_1MICRO_FAR12, _("1号麦克风-远间隔12个键位")),
            (DataType.ALL_0MICRO_FAR12, _("0号麦克风-远间隔12个键位")),
            (DataType.ALL_1MICRO_NEAR12, _("1号麦克风-近间隔12个键位")),
            (DataType.ALL_0MICRO_NEAR12, _("0号麦克风-近间隔12个键位")),
        )


class LabelType:
    TEST = "test"
    TRAIN = "train"

    @staticmethod
    def get_choices() -> Tuple:
        return (LabelType.TRAIN, _("训练数据")), (LabelType.TEST, _("测试数据"))


class FileType:
    PICKLE = "pickle"

    @staticmethod
    def get_names() -> List:
        return [FileType.PICKLE]


class AlgorithmModel:
    RNN = "rnn"
    BLSTM = "blstm"
    LSTM = "lstm"

    @staticmethod
    def get_choices() -> Tuple:
        return (
            (AlgorithmModel.RNN, _("RNN")),
            (AlgorithmModel.BLSTM, _("BiLSTM")),
            (AlgorithmModel.LSTM, _("LSTM")),
        )


DATA_ROOT = os.path.join(settings.AK_ROOT, "data")

PREPROCESSED_DATA_ROOT = os.path.join(settings.AK_ROOT, "preprocessed_data")

RESULT_ROOT = os.path.join(PREPROCESSED_DATA_ROOT, "result")


ALL_CLASS_LABELS = [chr(alpha_ord) for alpha_ord in range(ord("A"), ord("Z") + 1)] + [
    "BACKSPACE",
    "ENTER",
    "L_SHIFT",
    "R_CTRL",
]

LABEL_ID_MAP = {label: index for index, label in enumerate(ALL_CLASS_LABELS)}

ID_LABEL_MAP = {index: label for label, index in LABEL_ID_MAP.items()}

TRANSFER_INT = 32767

CORRECT_MODEL_THRESHOLD = 20
