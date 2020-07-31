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


def stock_weekly(symbol=None, start_date=None, end_date=None, fields=None):
    """
    股票周线行情
    """

    client = api_client()

    return client.query('stock_weekly',
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def stock_monthly(symbol=None, start_date=None, end_date=None, fields=None):
    """
    股票月线行情
    """

    client = api_client()

    return client.query('stock_monthly',
                        symbol=symbol,
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


def stock_bar(symbol=None,
              freq=None,
              adjust=None,
              trade_date=None,
              start_date=None,
              end_date=None,
              fields=None):
    """
    股票通用行情
    """

    client = api_client()

    return client.query('stock_bar',
                        symbol=symbol,
                        freq=freq,
                        adjust=adjust,
                        trade_date=trade_date,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def balance_sheet(symbol=None,
                  report_type=None,
                  report_date=None,
                  start_date=None,
                  end_date=None,
                  fields=None):
    """
    股票资产负债表
    """

    client = api_client()

    return client.query('balance_sheet',
                        symbol=symbol,
                        report_type=report_type,
                        report_date=report_date,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def income(symbol=None,
           report_type=None,
           report_date=None,
           start_date=None,
           end_date=None,
           fields=None):
    """
    股票利润表
    """

    client = api_client()

    return client.query('income',
                        symbol=symbol,
                        report_type=report_type,
                        report_date=report_date,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def cash_flow(symbol=None,
              report_type=None,
              report_date=None,
              start_date=None,
              end_date=None,
              fields=None):
    """
    股票现金流量表
    """

    client = api_client()

    return client.query('cash_flow',
                        symbol=symbol,
                        report_type=report_type,
                        report_date=report_date,
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def industry_industry(fields=None):
    """
    行业板块
    """

    client = api_client()

    return client.query('industry_industry', fields=fields)


def industry_concept(fields=None):
    """
    概念板块
    """

    client = api_client()

    return client.query('industry_concept', fields=fields)


def industry_region(fields=None):
    """
    地域板块
    """

    client = api_client()

    return client.query('industry_region', fields=fields)


def industry(industry_type=None, fields=None):
    """
    板块(通用)
    """

    client = api_client()

    return client.query('industry', industry_type=industry_type, fields=fields)


def industry_stock_industry(industry_symbol=None, fields=None):
    """
    行业板块成分股
    """

    client = api_client()

    return client.query('industry_stock_industry',
                        industry_symbol=industry_symbol,
                        fields=fields)


def industry_stock_concept(industry_symbol=None, fields=None):
    """
    概念板块成分股
    """

    client = api_client()

    return client.query('industry_stock_concept',
                        industry_symbol=industry_symbol,
                        fields=fields)


def industry_stock_region(industry_symbol=None, fields=None):
    """
    地域板块成分股
    """

    client = api_client()

    return client.query('industry_stock_region',
                        industry_symbol=industry_symbol,
                        fields=fields)


def industry_stock(industry_symbol=None, fields=None):
    """
    板块成分股(通用)
    """

    client = api_client()

    return client.query('industry_stock',
                        industry_symbol=industry_symbol,
                        fields=fields)


def stock_industry_industry(symbol=None, fields=None):
    """
    股票所属行业板块
    """

    client = api_client()

    return client.query('stock_industry_industry',
                        symbol=symbol,
                        fields=fields)


def stock_industry_concept(symbol=None, fields=None):
    """
    股票所属概念板块
    """

    client = api_client()

    return client.query('stock_industry_concept', symbol=symbol, fields=fields)


def stock_industry_region(symbol=None, fields=None):
    """
    股票所属地域板块
    """

    client = api_client()

    return client.query('stock_industry_region', symbol=symbol, fields=fields)


def stock_industry(symbol=None, industry_type=None, fields=None):
    """
    股票所属板块(通用)
    """

    client = api_client()

    return client.query('stock_industry',
                        symbol=symbol,
                        industry_type=industry_type,
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


def money_supply(start_date=None, end_date=None, fields=None):
    """
    货币供应量
    """

    client = api_client()

    return client.query('money_supply',
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def reserve_ratio(start_date=None, end_date=None, fields=None):
    """
    存款准备金率
    """

    client = api_client()

    return client.query('reserve_ratio',
                        start_date=start_date,
                        end_date=end_date,
                        fields=fields)


def gold_foreign_reserve(start_date=None, end_date=None, fields=None):
    """
    央行黄金和外汇储备
    """

    client = api_client()

    return client.query('gold_foreign_reserve',
                        start_date=start_date,
                        end_date=end_date,
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
