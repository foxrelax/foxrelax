# -*- coding:utf-8 -*-

from foxrelax.common.consts import (API_URL, API_VERSION, SignType, Charset,
                                    Lang, DEFAULT_TIMEOUT)
from foxrelax.common.util import version


def auth(app_id=None, app_secret=None, api_url=None):
    """
    用户认证
    """

    config = Config()

    if app_id:
        config.app_id = app_id

    if app_secret:
        config.app_secret = app_secret

    if api_url:
        config.api_url = api_url


class Config:
    """
    单例模式的全局配置
    """
    def __new__(cls, *args, **kwargs):
        return cls.init(*args, **kwargs)

    @classmethod
    def init(cls):
        """
        初始化全局配置
        """

        if not hasattr(cls, '_instance'):
            cls._instance = super(Config, cls).__new__(cls)

            Config._api_url = API_URL
            Config._app_id = ''
            Config._app_secret = ''
            Config._sign_type = SignType.MD5
            Config._charset = Charset.UTF8
            Config._version = API_VERSION
            Config._client_version = version()
            Config._lang = Lang.EN
            Config._timeout = DEFAULT_TIMEOUT

        return cls._instance

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)

        return cls._instance

    @property
    def api_url(self):
        return Config._api_url

    @api_url.setter
    def api_url(self, value):
        Config._api_url = value if value else API_URL

    @property
    def app_id(self):
        return Config._app_id

    @app_id.setter
    def app_id(self, value):
        Config._app_id = value if value else ''

    @property
    def app_secret(self):
        return Config._app_secret

    @app_secret.setter
    def app_secret(self, value):
        Config._app_secret = value if value else ''

    @property
    def sign_type(self):
        return Config._sign_type.value

    @property
    def charset(self):
        return Config._charset.value

    @property
    def version(self):
        return Config._version

    @property
    def client_version(self):
        return Config._client_version

    @property
    def lang(self):
        return Config._lang.value

    @property
    def timeout(self):
        return Config._timeout

    @timeout.setter
    def timeout(self, value):
        Config._timeout = value
