# -*- coding: utf-8 -*-


def class_member_cache(name: str):
    """
    类成员缓存
    :param name:
    :return:
    """
    cache_field = f"_{name}"

    def class_member_cache_inner(class_func):
        def wrapper(self, *args, **kwargs):
            cache_member = getattr(self, cache_field, None)
            if cache_member:
                return cache_member
            cache_member = class_func(self, *args, **kwargs)
            setattr(self, cache_field, cache_member)
            return cache_member

        return wrapper

    return class_member_cache_inner
