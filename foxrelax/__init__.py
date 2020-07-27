# -*- coding:utf-8 -*-

from foxrelax.common.code import (Code, code_to_message)
from foxrelax.common.exceptions import (RequestError, ResponseError)
from foxrelax.common.util import (root_path, version, install_requires, md5,
                                  gen_timestamp, gen_nonce_str, gen_uuid, sign)

from foxrelax.client import api_client
from foxrelax.config import auth

from foxrelax.api import (
    echo, trade_calendar, exchange_info, stock_info, adj_factor,
    normalize_symbol, stock_daily, stock_weekly, stock_monthly, stock_bar,
    balance_sheet, income, cash_flow, industry_industry, industry_concept,
    industry_region, industry, industry_stock_industry, industry_stock_concept,
    industry_stock_region, industry_stock, stock_industry_industry,
    stock_industry_concept, stock_industry_region, stock_industry,
    stock_suspend, deposit_rate, loan_rate, cpi, cpi_area)
