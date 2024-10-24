import pandas as pd
import os
import numpy as np
from datetime import datetime
from data.NSEDataAccess import NSEMasterDataAccess
from utils.decorators import timer
from utils.config import  YFINANCE_PRICES_PATH
from utils.config import Columns
from typing import Tuple


def true_range(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a prices df having columns AdjHigh, AdjLow, AdjClose
    :return: a df having having true range values
    """
    prices[f'prev_{Columns.ADJ_CLOSE.value}'] = prices[Columns.ADJ_CLOSE.value].shift()
    prices['true_range'] = prices.apply(lambda x: max((x[Columns.ADJ_HIGH.value] - x[Columns.ADJ_LOW.value]),
                                                      abs(x[Columns.ADJ_HIGH.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}']),
                                                      abs(x[Columns.ADJ_LOW.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}'])), axis=1)
    return prices[['true_range']], 'true_range'


# WIP
def directional_movement(prices: pd.DataFrame, window: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a prices df having AdjHigh, AdjLow, AdjClose
    :return: a df having both positive and negative directional movement
    """
    prices[f'prev_{Columns.ADJ_HIGH.value}'] = prices[Columns.ADJ_HIGH.value].shift()
    prices[f'prev_{Columns.ADJ_LOW.value}'] = prices[Columns.ADJ_LOW.value].shift()
    prices['positive_direction'] = prices.apply(
        lambda x: x[Columns.ADJ_HIGH.value] - x[f'prev_{Columns.ADJ_HIGH.value}']
        if (x[Columns.ADJ_HIGH.value] - x[f'prev_{Columns.ADJ_HIGH.value}']) >
           (x[f'prev_{Columns.ADJ_LOW.value}'] - x[Columns.ADJ_LOW.value]) else 0)

    prices['negative_direction'] = prices.apply(
        lambda x: x[Columns.ADJ_HIGH.value] - x[f'prev_{Columns.ADJ_HIGH.value}']
        if (x[f'prev_{Columns.ADJ_LOW.value}'] - x[Columns.ADJ_LOW.value]) >
           (x[Columns.ADJ_HIGH.value] - x[f'prev_{Columns.ADJ_HIGH.value}']) else 0)

@timer
def moving_average_crossover(prices: pd.DataFrame, column_name: str, slow_window: int, fast_window: int,
                             average_type: str = 'exponential')->Tuple[pd.DataFrame,str]:
    """
    :param prices: a pandas df
    :param column_name: the column you want to calculate the signal on
    :param slow_window: slow window length
    :param fast_window: fast window length
    :param average_type: one of exponential/simple
    :return: The df having signal and the column name
    """
    prices = prices.sort_index()
    if average_type not in ['simple','exponential']:
        raise ValueError(f'average_type should be simple/exponential you passed {average_type}')
    if average_type == 'exponential':
        prices[f'{column_name}_slow'] = prices[column_name].ewm(span=slow_window,min_periods=slow_window).mean()
        prices[f'{column_name}_fast'] = prices[column_name].ewm(span=fast_window, min_periods=fast_window).mean()
    else:
        prices[f'{column_name}_slow'] = prices[column_name].rolling(window=slow_window, min_periods=slow_window).mean()
        prices[f'{column_name}_fast'] = prices[column_name].rolling(window=fast_window, min_periods=fast_window).mean()

    prices[f'MACD_{fast_window}_{slow_window}'] = (prices[f'{column_name}_fast'] - prices[f'{column_name}_slow'])/prices[f'{column_name}_slow']
    return prices[[f'{column_name}_slow',f'{column_name}_fast',f'MACD_{fast_window}_{slow_window}']],f'MACD_{fast_window}_{slow_window}'
@timer
def timeseries_momentum(prices: pd.DataFrame, column_name: str, lookback_window: int)->Tuple[pd.DataFrame,str]:
    """
    :param prices: a dataframe having prices
    :param column_name: which column you want to calculate the factor on
    :param lookback_window: lookback for time series momentum (in days)
    :return: df having factor and column name that has the factor values
    """
    prices[f'{column_name}_lagged'] = prices[column_name].shift(lookback_window)
    prices[f'ts_momentum_{lookback_window}'] = -1 + prices[column_name]/prices[f'{column_name}_lagged']
    return prices[[f'ts_momentum_{lookback_window}']],f'ts_momentum_{lookback_window}'


if __name__ =='__main__':
    symbol = 'RELIANCE'
    period = (datetime(2002,1,1),datetime(2024,10,10))
    data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    prices = data_access.get_prices(symbol=symbol,start_date=period[0],end_date=period[-1])
    ts_mom,_ = timeseries_momentum(prices=prices,column_name=Columns.ADJ_CLOSE.value,lookback_window=66)
    # macd_signal,_ = moving_average_crossover(prices=prices,column_name=Columns.ADJ_CLOSE.value,slow_window=64,fast_window=16)
    # prices = prices.join(macd_signal,how='left')
    print('hello')


