# -*- coding: utf-8 -*-
import math
import random

from apps.keyboard import constants
from djangocli.utils.unittest.base import ApiMockData

API_DATASET_IMPORT_DATASET = ApiMockData(
    request_data={"verbose_name": "测试数据库", "description_more": "更多关于数据库的描述"},
    response_data={"task_id": "377ef8e1-395b-4a94-9ee5-d084e4b20567"},
)

API_DATASET_TRAIN = ApiMockData(
    request_data={"dataset_id": 1, "per_train_rate": 0.8},
    response_data={"task_id": "377ef8e1-395b-4a94-9ee5-d084e4b20567"},
)


API_COMMON_BATCH_CELERY_RESULTS_DELAY = ApiMockData(
    request_data={"task_ids": ["55022997-19cf-4314-88c1-f1dfff282649", "7258bbf7-853f-4a2b-adb6-222ae23cd4af"]},
    response_data=[
        {
            "date_done": "2021-03-25T03:57:30.471903",
            "task_id": "55022997-19cf-4314-88c1-f1dfff282649",
            "result": 3,
            "status": "SUCCESS",
            "state": "SUCCESS",
        },
        {
            "date_done": "2021-03-25T03:57:13.090748",
            "task_id": "7258bbf7-853f-4a2b-adb6-222ae23cd4af",
            "result": 3,
            "status": "SUCCESS",
            "state": "SUCCESS",
        },
    ],
)


API_MODEL_INST_PREDICT = ApiMockData(
    request_data={
        "dataset_id": 1,
        "algorithm": constants.AlgorithmModel.BLSTM,
        "signal": [random.randint(-int(math.pow(2, 15)), int(math.pow(2, 15))) for i in range(2002)],
    },
    response_data={"predict": "A"},
)
