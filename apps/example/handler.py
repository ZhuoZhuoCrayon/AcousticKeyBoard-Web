# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List

from celery.result import AsyncResult
from django.utils.translation import ugettext_lazy as _

from apps.example import task
from djangocli.constants import LogModule

logger = logging.getLogger(LogModule.APPS)


class CommonHandler:
    @staticmethod
    def celery_delay(left_val: float, right_val: float, operate: str) -> str:
        task_id = getattr(task, operate).delay(left_val, right_val).task_id
        logger.info(
            _("left_val -> {left_val}, right_val -> {right_val}, task_id -> {task_id}").format(
                left_val=left_val, right_val=right_val, task_id=task_id
            )
        )
        return task_id

    @staticmethod
    def batch_celery_results(task_ids: str) -> List[Dict[str, Any]]:
        celery_results = []
        for task_id in task_ids:
            task_info_obj = AsyncResult(task_id)
            celery_result = {
                "date_done": task_info_obj.date_done,
                "task_id": task_info_obj.task_id,
                "result": task_info_obj.result,
                "status": task_info_obj.status,
                "state": task_info_obj.state,
            }
            celery_results.append(celery_result)
            logger.info(
                _("task_id -> {task_id}, celery_result -> {celery_result}").format(
                    task_id=task_id, celery_result=celery_result
                )
            )
        return celery_results
