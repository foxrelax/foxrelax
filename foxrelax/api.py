# -*- coding:utf-8 -*-
"""
Data APIs
"""

from foxrelax.client import api_client


def echo(text=None, fields=None):
    """
    echo test
    """

    client = api_client()

    return client.query('echo', text=text, fields=fields)


def trade_calendar(start_date=None, end_date=None, fields=None):
    """
    交易日历
    """

    client = api_client()

    return client.query('trade_calendar',
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def exchange_info(fields=None):
    """
    交易所信息
    """

    client = api_client()

    return client.query('exchange_info', fields=fields)


def stock_info(symbol=None,
               exchange=None,
               market=None,
               list_status=None,
               fields=None):
    """
    股票基本信息
    """

    client = api_client()

    return client.query('stock_info',
                        symbol=symbol,
                        exchange=exchange,
                        market=market,
                        list_status=list_status,
                        fields=fields)


def normalize_symbol(symbols, fields=None):
    """
    格式化symbol
    """

    client = api_client()

    return client.query('normalize_symbol', symbols=symbols, fields=fields)


def stock_daily(symbol=None,
                trade_date=None,
                start_date=None,
                end_date=None,
                fields=None):
    """
    股票日线行情
    """

    client = api_client()

    return client.query('stock_daily',
                        symbol=symbol,
                        trade_date=trade_date,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def adj_factor(symbol=None, start_date=None, end_date=None, fields=None):
    """
    股票复权信息
    """

    client = api_client()

    return client.query('adj_factor',
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def stock_suspend(symbol=None,
                  suspend_date=None,
                  resume_date=None,
                  fields=None):
    """
    股票停复牌信息
    """

    client = api_client()

    return client.query('stock_suspend',
                        symbol=symbol,
                        suspend_date=suspend_date,
                        resume_date=resume_date,
                        fields=fields)


def deposit_rate(start_date=None, end_date=None, fields=None):
    """
    存款利率
    """

    client = api_client()

    return client.query('deposit_rate',
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def loan_rate(start_date=None, end_date=None, fields=None):
    """
    贷款利率
    """

    client = api_client()

    return client.query('loan_rate',
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def cpi(start_month=None, end_month=None, fields=None):
    """
    居民消费价格指数 - 总指数
    """

    client = api_client()

    return client.query('cpi',
                        start_month=start_month,
                        end_month=end_month,
                        fields=fields)


def cpi_area(area=None, start_month=None, end_month=None, fields=None):
    """
    居民消费价格指数 - 区域
    """

    client = api_client()

    return client.query('cpi_area',
                        area=area,
                        start_month=start_month,
                        end_month=end_month,
                        fields=fields)
