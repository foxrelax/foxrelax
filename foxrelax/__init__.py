# -*- coding:utf-8 -*-

from foxrelax.common.code import (Code, code_to_message)
from foxrelax.common.exceptions import (RequestError, ResponseError)
from foxrelax.common.util import (root_path, version, install_requires, md5,
                                  gen_timestamp, gen_nonce_str, gen_uuid, sign)

from foxrelax.client import api_client
from foxrelax.config import auth

from foxrelax.api import (
    echo, trade_calendar, area, exchange_info, stock_info, adj_factor, bar,
    index_info, index_daily, index_weekly, index_monthly, index_bar,
    money_flow_hk, money_flow_hk_stat, rise_fall_stat, limit_rise_fall_stat,
    normalize_symbol, normalize_stock_symbol, normalize_index_symbol,
    news_flash, stock_daily, stock_weekly, stock_monthly, stock_bar,
    balance_sheet, income, cash_flow, industry_industry, industry_concept,
    industry_region, industry, industry_stock_industry, industry_stock_concept,
    industry_stock_region, industry_stock, stock_industry_industry,
    stock_industry_concept, stock_industry_region, stock_industry,
    money_supply, reserve_ratio, gold_foreign_reserve, stock_suspend,
    deposit_rate, loan_rate, cpi, cpi_item)
