# -*- coding: utf-8 -*-
import copy
import math
import random
from typing import Dict, List


def data_select(original_data: Dict[str, List], per_action_rate: float, sort_key=lambda x: x) -> Dict[str, List]:
    original_data = copy.deepcopy(original_data)
    for label in original_data:
        if not original_data[label]:
            continue
        per_label_select_num = math.ceil(len(original_data[label]) * per_action_rate)
        random.shuffle(original_data[label])
        original_data[label] = sorted(original_data[label][:per_label_select_num], key=sort_key)
    return original_data
