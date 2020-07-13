# -*- coding:utf-8 -*-
from functools import partial
import requests
from foxrelax.config import (auth, Config)
from foxrelax.common.request import Request
from foxrelax.common.response import Response


def api_client(app_id=None, app_secret=None, api_url=None):
    auth(app_id, app_secret, api_url)
    return ApiClient()


class ApiClient:
    """
    HTTP Client
    """
    def __init__(self):
        self._config = Config()
        self.__headers = {
            'Content-Type':
            'application/json; charset={charset}'.format(
                charset=self._config.charset),
            'User-Agent':
            'foxrelax-client-{version}'.format(
                version=self._config.client_version)
        }

    @property
    def config(self):
        return self._config

    def query(self, method, **kwargs):
        trace = kwargs.pop('trace', False)

        req = Request(method, kwargs)
        res = requests.post(self.config.api_url,
                            json=req.to_dict(),
                            timeout=self.config.timeout)
        resp = Response(res.text)

        if trace:
            print('REQ=>{0}'.format(req))
            print('RESP=>{0}'.format(resp))

        return resp

    def __getattr__(self, name):
        return partial(self.query, name)
