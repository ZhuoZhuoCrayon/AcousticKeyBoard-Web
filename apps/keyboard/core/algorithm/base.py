# -*- coding: utf-8 -*-
import abc
import logging
import math
import random
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import six
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


class ModelDataType:
    TRAIN = 0
    TEST = 1


class BaseModel(metaclass=abc.ABCMeta):
    MODEL_NAME = "base-model"

    def __init__(
        self,
        labels: List[str],
        input_shape: Tuple[int],
        select_policy: Tuple[float, float] = (1.0, 1.0),
        *args,
        **kwargs,
    ):
        self.labels = sorted(labels)
        self.input_shape = input_shape
        self.select_policy = select_policy

        self.model = self.get_model(labels, input_shape, *args, **kwargs)

    @abc.abstractmethod
    def get_model(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def data_format(self, original_data: Dict[str, List[np.ndarray]]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, train_data: Dict[str, List[np.ndarray]], per_train_rate: float = None, *args, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, test_data: Dict[str, List[np.ndarray]], per_test_rate: float = None, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def get_data_sample_num(data: Dict[str, List[np.ndarray]]) -> int:
        return sum([len(data[label]) for label in data])

    def data_select(
        self, original_data: Dict[str, List[np.ndarray]], per_action_rate: float
    ) -> Dict[str, List[np.ndarray]]:
        original_data = self.copy_data_layer(original_data)
        for label in self.labels:
            if not original_data.get(label):
                logger.warning(f"label -> {label} not in data.")
                continue
            per_label_select_num = math.ceil(len(original_data[label]) * per_action_rate)
            random.shuffle(original_data[label])
            original_data[label] = original_data[label][:per_label_select_num]
        return original_data

    def copy_data_layer(self, data: Any) -> Any:
        """
        对原数据外层进行拷贝，防止修改原数据集
        :param data:
        :return:
        """
        if isinstance(data, dict):
            return {key: self.copy_data_layer(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.copy_data_layer(value) for value in data]
        else:
            return data

    def get_confusion_matrix(
        self, pred_labels: np.ndarray, expect_labels: np.ndarray, save_path: str = None, is_show: bool = False, **kwargs
    ) -> np.ndarray:
        confusion_mtx = tf.math.confusion_matrix(expect_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=self.labels, yticklabels=self.labels, annot=True, fmt="g")
        plt.xlabel("Prediction")
        plt.ylabel("Label")
        if save_path:
            plt.savefig(save_path)
        if is_show:
            plt.show()
        confusion_mtx_np = confusion_mtx.numpy()
        if not kwargs.get("confusion_mtx_save_path"):
            return confusion_mtx_np
        confusion_mtx_pd = pd.DataFrame(data=confusion_mtx_np, index=self.labels, columns=self.labels)
        confusion_mtx_pd.to_csv(kwargs["confusion_mtx_save_path"], encoding="utf-8", index=True)


class TfBaseModel(six.with_metaclass(abc.ABCMeta, BaseModel)):
    MODEL_NAME = "tf-base-model"

    def __init__(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs):
        super().__init__(labels, input_shape, *args, **kwargs)
        self.model: tf.keras.Sequential = self.model

        self.model.summary()

    @abc.abstractmethod
    def get_model(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        raise NotImplementedError

    def data_format(
        self, original_data: Dict[str, List[np.ndarray]]
    ) -> Tuple[Union[DatasetV1, DatasetV2], np.ndarray, np.ndarray]:

        begin = 0
        label_features_list = []
        labels_format = np.zeros(self.get_data_sample_num(original_data))
        for index, label in enumerate(self.labels):
            label_features = original_data.get(label, [])

            label_sample_num = len(label_features)
            label_features_list.append(np.vstack(label_features).reshape(label_sample_num, *self.input_shape))
            labels_format[begin : begin + label_sample_num] = index

            begin = begin + label_sample_num

        data_format = np.vstack(label_features_list)

        dataset = tf.data.Dataset.from_tensors((data_format, labels_format))

        # dataset = dataset.shuffle(self.get_data_sample_num(original_data))

        return dataset, data_format, labels_format

    def train(self, train_data: Dict[str, List[np.ndarray]], per_train_rate: float = None, *args, **kwargs) -> None:

        per_train_rate = per_train_rate or self.select_policy[ModelDataType.TRAIN]

        train_dataset, _, labels_format = self.data_format(original_data=self.data_select(train_data, per_train_rate))

        logger.info(
            f"train num -> {labels_format.size}, total -> {self.get_data_sample_num(train_data)}, "
            f"per_train_rate -> {per_train_rate}",
        )
        self.model.fit(train_dataset, epochs=100, batch_size=int(labels_format.size / len(self.labels)), verbose=0)

    def test(
        self, test_data: Dict[str, List[np.ndarray]], per_test_rate: float = None, *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, float]:

        per_test_rate = per_test_rate or self.select_policy[ModelDataType.TRAIN]

        _, test_data_format, except_labels = self.data_format(self.data_select(test_data, per_test_rate))
        pred_labels = np.argmax(self.model.predict(test_data_format), axis=1)

        test_acc = sum(pred_labels == except_labels) / len(except_labels)

        logger.info(
            f"test num -> {except_labels.size}, total -> {self.get_data_sample_num(test_data)}, "
            f"per_test_rate -> {per_test_rate}, accuracy -> {test_acc}",
        )
        return pred_labels, except_labels, round(test_acc, 2)
