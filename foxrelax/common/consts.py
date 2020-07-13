# -*- coding:utf-8 -*-
import threading
import sys
from enum import Enum, unique

API_URL = 'http://api.foxrelax.com'
API_VERSION = '1.0'
IS_PY3 = (sys.version_info.major >= 3)
THREAD_LOCAL = threading.local()
DEFAULT_TIMEOUT = 15  # seconds


@unique
class SignType(Enum):
    MD5 = 'md5'


@unique
class Charset(Enum):
    UTF8 = 'utf-8'


@unique
class Lang(Enum):
    EN = 'en'  # 英文
