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

@timer
def TALIB_ADChaikin(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    CLV = ((Close - Low) - (High - Close)) / (High - Low)
    - CLV ranges from -1 to 1:
    - +1 when Close is at High
    - -1 when Close is at Low
    - 0 when Close is in the middle of the range
    AD_today = AD_previous + (CLV * Volume)
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['TALIB_ADChaikin'] = ta.AD(high=prices[Columns.ADJ_HIGH.value],
                                      low=prices[Columns.ADJ_LOW.value],
                                      close=prices[Columns.ADJ_CLOSE.value],
                                      volume=prices[Columns.VOLUME.value])

    prices['TALIB_ADChaikin_10EWM'] = prices['TALIB_ADChaikin'].ewm(span=10, min_periods=10).mean()
    prices['TALIB_ADChaikin_20EWM'] = prices['TALIB_ADChaikin'].ewm(span=20, min_periods=10).mean()
    prices['TALIB_ADChaikin_50EWM'] = prices['TALIB_ADChaikin'].ewm(span=50, min_periods=10).mean()
    prices['TALIB_ADChaikin_100EWM'] = prices['TALIB_ADChaikin'].ewm(span=100, min_periods=10).mean()

    prices.loc[(prices["TALIB_ADChaikin_10EWM"] > prices['TALIB_ADChaikin_50EWM']) & (prices["TALIB_ADChaikin_20EWM"] > prices['TALIB_ADChaikin_100EWM']),"TALIB_ADChaikin_Signal"] = 1
    prices.loc[(prices["TALIB_ADChaikin_10EWM"] < prices['TALIB_ADChaikin_50EWM']) & (prices["TALIB_ADChaikin_20EWM"] < prices['TALIB_ADChaikin_100EWM']),"TALIB_ADChaikin_Signal"] = -1
    prices["TALIB_ADChaikin_Signal"] = prices["TALIB_ADChaikin_Signal"].fillna(0)

    return prices[['TALIB_ADChaikin_Signal']], 'TALIB_ADChaikin_Signal'

@timer
def TALIB_ADOSCChaikin(prices: pd.DataFrame, fastperiod: int = 3, slowperiod: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    The Chaikin Oscillator is the difference between the 3-day EMA and the 10-day EMA of the Chaikin AD line:
    ADOSC = EMA_3(AD) - EMA_10(AD)
    - EMA_3(AD): 3-day Exponential Moving Average of the Chaikin AD line
    - EMA_10(AD): 10-day Exponential Moving Average of the Chaikin AD line
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

    prices['TALIB_ADOSCChaikin_10EWM'] = prices[f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'].ewm(span=10, min_periods=10).mean()
    prices['TALIB_ADOSCChaikin_20EWM'] = prices[f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'].ewm(span=20, min_periods=10).mean()
    prices['TALIB_ADOSCChaikin_50EWM'] = prices[f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'].ewm(span=50, min_periods=10).mean()
    prices['TALIB_ADOSCChaikin_100EWM'] = prices[f'ADOSCChaikin_fast_{fastperiod}_slow_{slowperiod}'].ewm(span=100, min_periods=10).mean()

    prices.loc[(prices["TALIB_ADOSCChaikin_10EWM"] > prices['TALIB_ADOSCChaikin_50EWM']) & (prices["TALIB_ADOSCChaikin_20EWM"] > prices['TALIB_ADOSCChaikin_100EWM']),"TALIB_ADOSCChaikin_Signal"] = 1
    prices.loc[(prices["TALIB_ADOSCChaikin_10EWM"] < prices['TALIB_ADOSCChaikin_50EWM']) & (prices["TALIB_ADOSCChaikin_20EWM"] < prices['TALIB_ADOSCChaikin_100EWM']),"TALIB_ADOSCChaikin_Signal"] = -1
    prices["TALIB_ADOSCChaikin_Signal"] = prices["TALIB_ADOSCChaikin_Signal"].fillna(0)

    return prices[
        ["TALIB_ADOSCChaikin_Signal"]], "TALIB_ADOSCChaikin_Signal"
@timer
def TALIB_OBV(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['OBV'] = ta.OBV(prices[Columns.ADJ_CLOSE.value],
                           prices[Columns.VOLUME.value])

    prices['OBV_10EWM'] = prices['OBV'].ewm(span=10, min_periods=10).mean()
    prices['OBV_20EWM'] = prices['OBV'].ewm(span=20, min_periods=10).mean()
    prices['OBV_50EWM'] = prices['OBV'].ewm(span=50, min_periods=10).mean()
    prices['OBV_100EWM'] = prices['OBV'].ewm(span=100, min_periods=10).mean()

    # prices.loc[(prices["OBV_10EWM"] > prices['OBV_50EWM']) & (prices["OBV_10EWM"].shift() < prices['OBV'].shift()),"OBV_10Signal"] = 1
    # prices.loc[(prices["OBV_10EWM"] < prices['OBV']) & (prices["OBV_10EWM"].shift() > prices['OBV'].shift()),"OBV_10Signal"] = -1
    # prices["OBV_10Signal"] = prices["OBV_10Signal"].fillna(0)
    #
    # prices.loc[(prices["OBV_20EWM"] > prices['OBV']) & (prices["OBV_20EWM"].shift() < prices['OBV'].shift()),"OBV_20Signal"] = 1
    # prices.loc[(prices["OBV_20EWM"] < prices['OBV']) & (prices["OBV_20EWM"].shift() > prices['OBV'].shift()),"OBV_20Signal"] = -1
    # prices["OBV_20Signal"] = prices["OBV_20Signal"].fillna(0)
    #
    # prices.loc[(prices["OBV_50EWM"] > prices['OBV']) & (prices["OBV_50EWM"].shift() < prices['OBV'].shift()),"OBV_50Signal"] = 1
    # prices.loc[(prices["OBV_50EWM"] < prices['OBV']) & (prices["OBV_50EWM"].shift() > prices['OBV'].shift()),"OBV_50Signal"] = -1
    # prices["OBV_50Signal"] = prices["OBV_50Signal"].fillna(0)
    #
    # prices.loc[(prices["OBV_100EWM"] > prices['OBV']) & (prices["OBV_100EWM"].shift() < prices['OBV'].shift()),"OBV_100Signal"] = 1
    # prices.loc[(prices["OBV_100EWM"] < prices['OBV']) & (prices["OBV_100EWM"].shift() > prices['OBV'].shift()),"OBV_100Signal"] = -1
    # prices["OBV_100Signal"] = prices["OBV_100Signal"].fillna(0)
    #
    #
    # prices["OBV_Signal"] = prices[["OBV_100Signal", "OBV_50Signal", "OBV_20Signal", "OBV_10Signal"]].mode(axis=1).apply(
    #     lambda x: x[0] if len(x.dropna()) == 1 else 0, axis=1)

    prices.loc[(prices["OBV_10EWM"] > prices['OBV_50EWM']) & (prices["OBV_20EWM"] > prices['OBV_100EWM']),"OBV_Signal"] = 1
    prices.loc[(prices["OBV_10EWM"] < prices['OBV_50EWM']) & (prices["OBV_20EWM"] < prices['OBV_100EWM']),"OBV_Signal"] = -1
    prices["OBV_Signal"] = prices["OBV_Signal"].fillna(0)

    return prices[["OBV_Signal"]], 'OBV_Signal'
