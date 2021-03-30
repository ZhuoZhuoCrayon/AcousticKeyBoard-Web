# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List

from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


class ModelInstHandler:
    @classmethod
    def predict(cls, inst_id: int, signal: List[int]) -> Dict[str, Any]:
        return {"label": "A"}
