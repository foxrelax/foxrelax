import os
import time
import json
from datetime import datetime
from uuid import uuid4
import hashlib
import base64


def root_path():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def version():
    fpath = os.path.join(root_path(), 'foxrelax', 'version.txt')
    with open(fpath, 'r') as f:
        f = open(fpath, 'r')
        line = f.readline()
        ver = line.strip()

    return ver


def install_requires():
    fpath = os.path.join(root_path(), 'requirements.txt')
    requires = []

    with open(fpath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                requires.append(line)

    return requires


def md5(s, mode='hex'):
    """
    计算MD5

    Parameters
    ----------
    s : str
    mode : str
        hex | base64

    Returns
    -------
    str

    """

    m = hashlib.md5()
    if isinstance(s, str):
        m.update(s.encode('utf-8'))
    elif isinstance(s, bytes):
        m.update(s)

    if mode == 'hex':
        result = m.hexdigest()
    elif mode == 'base64':
        result = base64.b64encode(m.digest()).decode('utf-8')

    return result


def gen_timestamp(is_millisecond=True, is_str=True):
    """
    获取当前时间戳

    Parameters
    ----------
    is_millisecond : bool
    is_str : bool

    Examples
    --------
    >>> gen_timestamp(is_millisecond=False, is_str=False)
    1571671574

    >>> gen_timestamp(is_millisecond=True, is_str=False)
    1571671574966

    >>> gen_timestamp(is_millisecond=False, is_str=True)
    2019-10-21 23:26:14

    >>> gen_timestamp(is_millisecond=True, is_str=True)
    2019-10-21 23:26:14.966

    """

    if is_str:
        if is_millisecond:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            ts = ts[:-3]
        else:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        ts = time.time()
        if is_millisecond:
            ts = int(time.time() * 1000)
        else:
            ts = int(time.time())

    return ts


def gen_uuid():
    """
    返回一个长度为32的UUID

    Parameters
    ----------
    None

    Returns
    -------
    str

    """

    return str(uuid4()).replace('-', '')


def gen_nonce_str():
    """
    返回一个长度为32的nonce str

    Parameters
    ----------
    None

    Returns
    -------
    str

    """

    return gen_uuid()


def sign(req_dict, app_secret, to_md5=True):
    params = []
    for (k, v) in sorted(req_dict.items()):
        if k == 'sign':
            continue

        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)

        params.append('{0}={1}'.format(k, v))

    # 在最后拼接app_secret
    params.append('app_secret={0}'.format(app_secret))

    # 用&连接各k-v对
    if to_md5:
        return md5('&'.join(params))
    else:
        return '&'.join(params)
