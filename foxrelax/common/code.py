# -*- coding:utf-8 -*-
from enum import Enum, unique


def code_to_message(code):
    """
    {
        "code": 0,
        "message": "success"
    }
    """

    if isinstance(code, int):
        code = Code(code)

    return Message[code]


@unique
class Code(Enum):
    SUCCESS = 0
    SERVER_ERROR = 1
    CLIENT_ERROR = 2
    NETWORK_TIMEOUT = 3

    COMMON_ERROR = 10000
    INVALID_HTTP_METHOD_ERROR = 10001
    NOT_AUTH = 10002
    INVALID_APP_ID_ERROR = 10003
    INVALID_APP_SECRET_ERROR = 10004
    SIGN_ERROR = 10005
    NEED_PRIVILEGE = 10006
    MAX_LIMIT = 10007
    INVALID_METHOD_ERROR = 10008
    INVALID_PARAMS_ERROR = 10009
    REQUIRED_PARAMS_MISSING = 10010
    INVALID_FIELDS_ERROR = 10011
    NOT_FOUND = 10012
    NOT_VIP = 10013


Message = {
    Code.SUCCESS: 'success',
    Code.SERVER_ERROR: 'server error',
    Code.CLIENT_ERROR: 'client error',
    Code.NETWORK_TIMEOUT: 'network timeout',
    Code.COMMON_ERROR: 'common error',
    Code.INVALID_HTTP_METHOD_ERROR: 'invalid http method',
    Code.NOT_AUTH: 'not auth',
    Code.INVALID_APP_ID_ERROR: 'invalid app id',
    Code.INVALID_APP_SECRET_ERROR: 'invalid app secret',
    Code.SIGN_ERROR: 'sign error',
    Code.NEED_PRIVILEGE: 'need privilege',
    Code.MAX_LIMIT: 'max limit',
    Code.INVALID_METHOD_ERROR: 'invalid method',
    Code.INVALID_PARAMS_ERROR: 'invalid params error',
    Code.REQUIRED_PARAMS_MISSING: 'required params missing',
    Code.INVALID_FIELDS_ERROR: 'invalid fields error',
    Code.NOT_FOUND: 'not found',
    Code.NOT_VIP: 'not vip'
}
