# -*- coding:utf-8 -*-


class BaseError(Exception):
    pass


class RequestError(BaseError):
    pass


class ResponseError(BaseError):
    pass
