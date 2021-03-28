# -*- coding: utf-8 -*-
import os
from typing import List


def dir_checker(dir_roots: List[str]):
    for dir_root in dir_roots:
        if os.path.exists(dir_root):
            continue
        os.makedirs(dir_root)
