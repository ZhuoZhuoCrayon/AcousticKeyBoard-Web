# -*- coding: utf-8 -*-
import os
from typing import List


def check_and_create_log_file(log_file_paths: List[str]):
    for log_file_path in log_file_paths:
        if os.path.exists(log_file_path):
            return
        log_file_root = log_file_path.rsplit("/", 1)[0]
        if not os.path.exists(log_file_root):
            os.makedirs(log_file_root)
        with open(file=log_file_path, mode="w+") as _:
            pass
