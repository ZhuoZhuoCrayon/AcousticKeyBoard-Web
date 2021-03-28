# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import tensorflow as tf

from apps.keyboard.core.algorithm.base import TfBaseModel

# https://stackoverflow.com/questions/55318626/module-tensorflow-has-no-attribute-logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
logging.getLogger("matplotlib.font_manager").disabled = True


class BLstmModel(TfBaseModel):
    MODEL_NAME = "blstm"

    def __init__(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs):
        super().__init__(labels, input_shape, *args, **kwargs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def get_model(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
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
                tf.keras.layers.Dense(units=len(labels), activation="softmax"),
            ]
        )


class LstmModel(BLstmModel):
    MODEL_NAME = "lstm"

    def get_model(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.LSTM(units=100, return_sequences=False, input_shape=input_shape),
                tf.keras.layers.Dense(units=len(labels), activation="softmax"),
            ]
        )


class RnnModel(BLstmModel):
    MODEL_NAME = "rnn"

    def get_model(self, labels: List[str], input_shape: Tuple[int], *args, **kwargs) -> tf.keras.Sequential:
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.SimpleRNN(units=100, input_shape=input_shape),
                tf.keras.layers.Dense(units=len(labels), activation="softmax"),
            ]
        )
