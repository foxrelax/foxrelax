# -*- coding:utf-8 -*-
import json
from foxrelax.config import Config
from foxrelax.common.util import (sign, gen_nonce_str, gen_timestamp)


class Request:
    def __init__(self, method, params=None):
        self._config = Config()
        self._method = method
        self._params = params if params else {}
        self._app_id = self._config.app_id
        self._app_secret = self._config.app_secret
        self._sign_type = self._config.sign_type
        self._charset = self._config.charset
        self._version = self._config.version
        self._nonce_str = gen_nonce_str()
        self._timestamp = gen_timestamp()
        self._lang = self._config.lang

    @property
    def config(self):
        return self._config

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @property
    def app_id(self):
        return self._app_id

    @property
    def app_secret(self):
        return self._app_secret

    @property
    def sign_type(self):
        return self._sign_type

    @property
    def charset(self):
        return self._charset

    @property
    def version(self):
        return self._version

    @property
    def nonce_str(self):
        return self._nonce_str

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def lang(self):
        return self._lang

    def to_dict(self):
        req_dict = {
            'app_id': self.app_id,
            'method': self.method,
            'params': self.params,
            'sign_type': self.sign_type,
            'charset': self.charset,
            'version': self.version,
            'nonce_str': self.nonce_str,
            'timestamp': self.timestamp,
            'lang': self.lang
        }
        req_dict['sign'] = sign(req_dict, self.app_secret)
        return req_dict

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return json.dumps(self.to_dict())
