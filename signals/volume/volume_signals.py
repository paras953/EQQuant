import pandas as pd
import os
import numpy as np
from datetime import datetime
import talib as ta
from utils.pandas_utils import custom_rolling_mean, custom_rolling_sum
from analytics.risk_return import get_returns
from data.NSEDataAccess import NSEMasterDataAccess
from utils.decorators import timer
from utils.config import YFINANCE_PRICES_PATH
from utils.config import Columns
from typing import Tuple
from utils.signal_helper import _get_adj_prices_helper


def TALIB_ADChaikin(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['TALIB_ADChaikin'] = ta.AD(high=prices[Columns.ADJ_HIGH.value],
                                      low=prices[Columns.ADJ_LOW.value],
                                      close=prices[Columns.ADJ_CLOSE.value],
                                      volume=prices[Columns.VOLUME.value])
    return prices[['TALIB_ADChaikin']], 'TALIB_ADChaikin'


def TALIB_ADOSCChaikin(prices: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Uses the TA-LIB implementation just to cross check if the values calculated are correct or not
    :param prices: prices df
    :param fastperiod: period for fast ema of ADL
    :param slowperiod: period for slow ema of ADL
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices[f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'] = ta.ADOSC(high=prices[Columns.ADJ_HIGH.value],
                                                                        low=prices[Columns.ADJ_LOW.value],
                                                                        close=prices[Columns.ADJ_CLOSE.value],
                                                                        volume=prices[Columns.VOLUME.value],
                                                                        fastperiod=fastperiod,
                                                                        slowperiod=slowperiod)
    return prices[
        [f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}']], f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'

def TALIB_OBV(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['OBV'] = ta.OBV(prices[Columns.ADJ_CLOSE.value],
                           prices[Columns.VOLUME.value])
    return prices[
        ['OBV']], 'OBV'
