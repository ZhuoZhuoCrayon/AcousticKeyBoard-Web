# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

import django
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.extend([BASE_DIR])

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "settings")

django.setup()


def plot_data(data):
    plt.figure()
    plt.plot(data)
    plt.show()


if __name__ == "__main__":
    from apps.keyboard import constants, models

    for ori_data_obj in models.DatasetOriginalData.objects.filter(label__contains="DEBUG"):
        original_data = np.asarray(ori_data_obj.original_data, dtype=np.float64)
        print(np.max(original_data))
        signal = original_data / constants.TRANSFER_INT
        plot_data(signal)
