# -*- coding: utf-8 -*-
import math
import threading
import uuid

from django.conf import settings
from django_redis import get_redis_connection
from redis import StrictRedis

from djangocli.constants import TimeUnit

DEFAULT_TIMEOUT = 5 * TimeUnit.MINUTE


class REDIS_KEY_PREFIX:
    BASE_REDIS_KEY_PREFIX = f"{settings.APP_NAME.lower()}:{settings.APP_VERSION.lower()}"

    WEB_CACHE = f"{BASE_REDIS_KEY_PREFIX}:web-cache"
    LOCK = f"{BASE_REDIS_KEY_PREFIX}:lock"


def django_cache_key_maker(key: str, key_prefix: str, version: int):
    """
    组装django cache key
    :param key:
    :param key_prefix:
    :param version: 弃用该字段，改用后台的版本号
    :return:
    """
    return f"{REDIS_KEY_PREFIX.WEB_CACHE}:{key_prefix or 'default'}:{key}"


class RedisInstSingleTon:
    _inst_lock = threading.Lock()
    _inst_name = "redis_inst"

    @classmethod
    def get_inst(cls) -> StrictRedis:
        if hasattr(RedisInstSingleTon, RedisInstSingleTon._inst_name):
            return getattr(RedisInstSingleTon, RedisInstSingleTon._inst_name)
        with RedisInstSingleTon._inst_lock:
            setattr(RedisInstSingleTon, RedisInstSingleTon._inst_name, get_redis_connection())
        return getattr(RedisInstSingleTon, RedisInstSingleTon._inst_name)


# shortcut
def get_redis_conn():
    return RedisInstSingleTon.get_inst()


class RedisLock:
    redis_inst = RedisInstSingleTon.get_inst()

    def __init__(self, lock_name: str = None, lock_expire: int = DEFAULT_TIMEOUT):
        self.lock_name = f"{REDIS_KEY_PREFIX.LOCK}:{lock_name}"
        self.lock_expire = lock_expire
        self.identifier = None

    def __enter__(self):
        if self.lock_name is None:
            return None
        self.identifier = self.acquire_lock_with_timeout()
        return self.identifier

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.identifier is None:
            return False
        self.release_lock()

    def acquire_lock_with_timeout(self):
        """
        获得锁
        :return: success: identifier failed: None
        """
        identifier = str(uuid.uuid4())
        lock_timeout = int(math.ceil(self.lock_expire))
        # 如果不存在这个锁则加锁并设置过期时间，避免死锁
        if self.redis_inst.set(self.lock_name, identifier, ex=lock_timeout, nx=True):
            return identifier
        # 锁已存在并被持有，此时放弃排队竞争，直接返回None
        return None

    def release_lock(self):
        """
        释放锁
        :return:
        """
        unlock_script = """
        if redis.call("get",KEYS[1]) == ARGV[1] then
            return redis.call("del",KEYS[1])
        else
            return 0
        end
        """
        unlock = self.redis_inst.register_script(unlock_script)
        result = unlock(keys=[self.lock_name], args=[self.identifier])
        return result
