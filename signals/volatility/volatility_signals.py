from utils.config import Columns
from typing import Tuple
from utils.signal_helper import _get_adj_prices_helper
import pandas as pd
import talib as ta

def TALIB_ATR(prices: pd.DataFrame, average_window: int = 14) -> Tuple[pd.DataFrame, str]:
    """
    :param average_window: lookback of the ATR signal
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['TALIB_ATR'] = ta.ATR(high=prices[Columns.ADJ_HIGH.value],
                                      low=prices[Columns.ADJ_LOW.value],
                                      close=prices[Columns.ADJ_CLOSE.value],
                                      timeperiod=average_window)
    return prices[['TALIB_ATR']], 'TALIB_ATR'


def TALIB_NATR(prices: pd.DataFrame, average_window: int = 14) -> Tuple[pd.DataFrame, str]:
    """
    :param average_window: lookback of the NATR signal
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['TALIB_NATR'] = ta.NATR(high=prices[Columns.ADJ_HIGH.value],
                                      low=prices[Columns.ADJ_LOW.value],
                                      close=prices[Columns.ADJ_CLOSE.value],
                                      timeperiod=average_window)
    return prices[['TALIB_NATR']], 'TALIB_NATR'


def TALIB_TRANGE(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices['TALIB_TRANGE'] = ta.TRANGE(high=prices[Columns.ADJ_HIGH.value],
                                      low=prices[Columns.ADJ_LOW.value],
                                      close=prices[Columns.ADJ_CLOSE.value])
    return prices[['TALIB_TRANGE']], 'TALIB_TRANGE'