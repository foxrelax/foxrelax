# -*- coding:utf-8 -*-

import json
import pandas as pd
from foxrelax.common.code import Code


class Response:
    def __init__(self, response):
        if isinstance(response, str):
            self._response = response
            response = json.loads(response)

        if 'code' in response:
            self._code = Code(response['code'])
        if 'message' in response:
            self._message = response['message']
        if self._code == Code.SUCCESS and 'data' in response:
            data = response['data']
            fields = data['fields']
            items = data['items']
            df = pd.DataFrame(items, columns=fields)
            self._data = data
            self._result = df
        else:
            self._data = None
            self._result = None
        if 'timestamp' in response:
            self._timestamp = response['timestamp']

    @property
    def code(self):
        return self._code.value

    @property
    def message(self):
        return self._message

    @property
    def data(self):
        return self._data

    @property
    def result(self):
        return self._result

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def is_success(self):
        return self._code == Code.SUCCESS

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self._response
