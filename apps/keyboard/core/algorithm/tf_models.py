# -*- coding: utf-8 -*-
import logging
from typing import Tuple

import tensorflow as tf

from apps.keyboard import constants
from apps.keyboard.core.algorithm.base import TfBaseModel

# https://stackoverflow.com/questions/55318626/module-tensorflow-has-no-attribute-logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger("matplotlib.font_manager").disabled = True


class BLstmModel(TfBaseModel):
    MODEL_NAME = constants.AlgorithmModel.BLSTM

    def __init__(self, max_label_num: int, input_shape: Tuple[int], *args, **kwargs):
        super().__init__(max_label_num, input_shape, *args, **kwargs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def get_model(self, max_label_num: int, input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units=100, return_sequences=False), input_shape=input_shape
                ),
                # tf.keras.layers.Dropout(0.6),
                # tf.keras.layers.Bidirectional(
                #     tf.keras.layers.LSTM(units=32, return_sequences=False)
                # ),
                # tf.keras.layers.Dense(64, activation="relu"),
                # tf.keras.layers.Dropout(0.8),
                tf.keras.layers.Dense(units=max_label_num, activation="softmax"),
            ]
        )


class LstmModel(BLstmModel):
    MODEL_NAME = constants.AlgorithmModel.LSTM

    def get_model(self, max_label_num: int, input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(units=100, return_sequences=False, input_shape=input_shape),
                tf.keras.layers.Dense(units=max_label_num, activation="softmax"),
            ]
        )


class RnnModel(BLstmModel):
    MODEL_NAME = constants.AlgorithmModel.RNN

    def get_model(self, max_label_num: int, input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.SimpleRNN(units=100, input_shape=input_shape),
                tf.keras.layers.Dense(units=max_label_num, activation="softmax"),
            ]
        )
