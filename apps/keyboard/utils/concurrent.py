# -*- coding: utf-8 -*-
import sys
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from multiprocessing import cpu_count, get_context
from typing import Dict, List

from django.conf import settings


def batch_call(func, params_list: List[Dict], get_data=lambda x: x["info"], expand_result: bool = False) -> List:
    """
    并发请求接口，每次按不同参数请求最后叠加请求结果
    :param func: 请求方法
    :param params_list: 参数列表
    :param get_data: 获取数据函数
    :param expand_result: 是否展开结果
    :return: 请求结果累计
    """

    result = []
    with ThreadPoolExecutor(max_workers=settings.CONCURRENT_NUMBER) as ex:
        tasks = [ex.submit(func, **params) for params in params_list]
    for future in as_completed(tasks):
        if expand_result:
            result.extend(get_data(future.result()))
        else:
            result.append(get_data(future.result()))
    return result


def batch_call_multi_proc(
    func, params_list: List[Dict], get_data=lambda x: x["info"], expand_result: bool = False
) -> List:
    """多进程执行函数"""
    if sys.platform in ["win32", "cygwim", "msys"]:
        return batch_call(func, params_list, get_data, expand_result)
    else:
        ctx = get_context("fork")

    result = []

    pool = ctx.Pool(processes=cpu_count())
    futures = [pool.apply_async(func=func, kwds=params) for params in params_list]

    pool.close()
    pool.join()

    # 取值
    for future in futures:
        if expand_result:
            result.extend(get_data(future.get()))
        else:
            result.append(get_data(future.get()))
    return result
