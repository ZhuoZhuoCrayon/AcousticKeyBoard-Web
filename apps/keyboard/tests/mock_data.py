# -*- coding: utf-8 -*-

from djangocli.utils.unittest.base import ApiMockData

API_DATASET_IMPORT_DATASET = ApiMockData(
    request_data={"verbose_name": "测试数据库", "description_more": "更多关于数据库的描述"},
    response_data={"task_id": "377ef8e1-395b-4a94-9ee5-d084e4b20567"},
)
