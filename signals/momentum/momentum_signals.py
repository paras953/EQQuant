import pandas as pd
import os
import numpy as np
from datetime import datetime
import talib as ta
from utils.pandas_utils import custom_rolling_mean,custom_rolling_sum
from analytics.risk_return import get_returns
from data.NSEDataAccess import NSEMasterDataAccess
from utils.decorators import timer
from utils.config import YFINANCE_PRICES_PATH
from utils.config import Columns
from typing import Tuple
from utils.signal_helper import _get_adj_prices_helper




def average_true_range(prices: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a prices df having columns AdjHigh, AdjLow, AdjClose
    :param Lookback for calculating the average true range
    :return: a df having having true range values
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'prev_{Columns.ADJ_CLOSE.value}'] = prices[Columns.ADJ_CLOSE.value].shift()
    prices['true_range'] = prices.apply(lambda x: max((x[Columns.ADJ_HIGH.value] - x[Columns.ADJ_LOW.value]),
                                                      abs(x[Columns.ADJ_HIGH.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}']),
                                                      abs(x[Columns.ADJ_LOW.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}'])), axis=1)
    prices[f'average_true_range_{lookback}'] = custom_rolling_sum(series=prices['true_range'],lookback=lookback)

    return prices[[f'average_true_range_{lookback}']], f'average_true_range_{lookback}'


# WIP
def directional_movement(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a prices df having AdjHigh, AdjLow, AdjClose
    :return: a df having both positive and negative directional movement
    src https://stackoverflow.com/questions/63020750/how-to-find-average-directional-movement-for-stocks-using-pandas
    """
    prices = _get_adj_prices_helper(prices)
    prices['high_minus_prev_high'] = prices[Columns.ADJ_HIGH.value].diff()
    prices['prev_low_minus_low'] = prices[Columns.ADJ_LOW.value].diff() * -1
    prices['positive_direction'] = np.where(
        (prices['high_minus_prev_high'] > prices['prev_low_minus_low']) & (prices['high_minus_prev_high'] > 0),
        prices['high_minus_prev_high'], 0)

    prices['negative_direction'] = np.where(
        (prices['high_minus_prev_high'] < prices['prev_low_minus_low']) & (prices['prev_low_minus_low'] > 0),
        prices['prev_low_minus_low'], 0)

    return prices[['positive_direction', 'negative_direction']], 'positive_direction|negative_direction'


def ADX(prices: pd.DataFrame, average_window: int):
    """
    TA lib implementation of ADX will try to modify later
    :param prices: prices df
    :param average_window:  lookback for taking averages
    :return: The signal and the column name
    """


    directional_df, directional_col = directional_movement(prices=prices)
    positive_column, negative_column = directional_col.split('|')[0], directional_col.split('|')[-1]
    average_true_range_df , average_true_range_column = average_true_range(prices=prices,lookback=average_window)
    directional_df[f'smoothed_{positive_column}'] = custom_rolling_sum(series=directional_df[positive_column],lookback=average_window)
    directional_df[f'smoothed_{negative_column}'] = custom_rolling_sum(series=directional_df[negative_column],lookback=average_window)

    directional_df = pd.concat([directional_df[[f'smoothed_{positive_column}', f'smoothed_{negative_column}']],
                                average_true_range_df], axis=1)
    directional_df['normalized_positive_direction'] = directional_df[f'smoothed_{positive_column}'] / directional_df[
        average_true_range_column]
    directional_df['normalized_negative_direction'] = directional_df[f'smoothed_{negative_column}'] / directional_df[
        average_true_range_column]
    directional_df['DX'] = abs(
        (directional_df['normalized_positive_direction'] - directional_df['normalized_negative_direction'])) / (
                                   directional_df['normalized_positive_direction'] + directional_df[
                               'normalized_negative_direction'])
    directional_df[f'ADX_{average_window}'] = custom_rolling_mean(series=directional_df['DX'],lookback=average_window)
    directional_df[f'ADX_{average_window}'] = directional_df[f'ADX_{average_window}']*100
    first_valid_index = directional_df[f'ADX_{average_window}'].first_valid_index()
    directional_df = directional_df.truncate(first_valid_index)
    print('Min ADX - ', directional_df[f'ADX_{average_window}'].min())
    print('Min ADX - ', directional_df[f'ADX_{average_window}'].max())


    return directional_df[[f'ADX_{average_window}']], f'ADX_{average_window}'


@timer
def TALIB_ADX(prices: pd.DataFrame, average_window: int) -> Tuple[pd.DataFrame, str]:
    """
    Uses the TA-LIB implementation just to cross check if the values calculated are correct or not
    :param prices: prices df
    :param average_window: lookback of the ADX signal
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()
    prices[f'TALIB_ADX_{average_window}'] = ta.ADX(high=prices[Columns.ADJ_HIGH.value],
                                                   low=prices[Columns.ADJ_LOW.value],
                                                   close=prices[Columns.ADJ_CLOSE.value], timeperiod=average_window)
    return prices[[f'TALIB_ADX_{average_window}']], f'TALIB_ADX_{average_window}'


@timer
def moving_average_crossover(prices: pd.DataFrame, column_name: str, slow_window: int, fast_window: int,
                             average_type: str = 'exponential') -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a pandas df
    :param column_name: the column you want to calculate the signal on
    :param slow_window: slow window length
    :param fast_window: fast window length
    :param average_type: one of exponential/simple
    :return: The df having signal and the column name
    """
    prices = prices.sort_index()
    if average_type not in ['simple', 'exponential']:
        raise ValueError(f'average_type should be simple/exponential you passed {average_type}')
    if average_type == 'exponential':
        prices[f'{column_name}_slow'] = prices[column_name].ewm(span=slow_window, min_periods=slow_window).mean()
        prices[f'{column_name}_fast'] = prices[column_name].ewm(span=fast_window, min_periods=fast_window).mean()
    else:
        prices[f'{column_name}_slow'] = prices[column_name].rolling(window=slow_window, min_periods=slow_window).mean()
        prices[f'{column_name}_fast'] = prices[column_name].rolling(window=fast_window, min_periods=fast_window).mean()

    prices[f'MACD_{fast_window}_{slow_window}'] = (prices[f'{column_name}_fast'] - prices[f'{column_name}_slow']) / \
                                                  prices[f'{column_name}_slow']
    first_valid_index = prices[f'MACD_{fast_window}_{slow_window}'].first_valid_index()
    prices = prices.truncate(first_valid_index)
    return prices[[f'{column_name}_slow', f'{column_name}_fast',
                   f'MACD_{fast_window}_{slow_window}']], f'MACD_{fast_window}_{slow_window}'


@timer
def timeseries_momentum(prices: pd.DataFrame, column_name: str, lookback_window: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a dataframe having prices
    :param column_name: which column you want to calculate the factor on
    :param lookback_window: lookback for time series momentum (in days)
    :return: df having factor and column name that has the factor values
    """
    prices[f'{column_name}_lagged'] = prices[column_name].shift(lookback_window)
    prices[f'ts_momentum_{lookback_window}'] = -1 + (prices[column_name] / prices[f'{column_name}_lagged'])
    return prices[[f'ts_momentum_{lookback_window}']], f'ts_momentum_{lookback_window}'


@timer
def RSI(prices: pd.DataFrame, lookback_window: int, column_name: str = Columns.ADJ_CLOSE.value,
        average_type: str = 'simple', calculate_rsi_on: str = 'price_change') -> Tuple[
    pd.DataFrame, str]:
    """

    :param prices: prices df
    :param lookback_window: lookback of the signal
    :param column_name: which prices to use to calculate the signal
    :param average_type: simple for simple average and exponential for exponential moving average
    :param calculate_rsi_on: price_change : just use the price difference to calculate the RSI/ 'returns' use returns to calculate RSI
    :return: the signal and the column name of the signal
    """
    if calculate_rsi_on == 'price_change':
        prices['diff'] = prices[column_name].diff()
    else:
        prices['diff'] = prices[column_name].pct_change()

    prices['up_change'] = np.where(prices['diff'] > 0, prices['diff'], 0)
    prices['down_change'] = np.where(prices['diff'] < 0, prices['diff'], 0)
    if average_type == 'simple':
        prices['smooth_up_change'] = prices['up_change'].rolling(window=lookback_window,
                                                                 min_periods=lookback_window).mean()
        prices['smooth_down_change'] = prices['down_change'].rolling(window=lookback_window,
                                                                     min_periods=lookback_window).mean()
    else:
        prices['smooth_up_change'] = prices['up_change'].ewm(span=lookback_window,
                                                             min_periods=lookback_window).mean()
        prices['smooth_down_change'] = prices['down_change'].ewm(span=lookback_window,
                                                                 min_periods=lookback_window).mean().abs()

    first_valid_index = prices['smooth_up_change'].first_valid_index()
    prices[f'RSI_{lookback_window}'] = prices['smooth_up_change'] / (
            prices['smooth_up_change'] + prices['smooth_down_change'])
    prices[f'RSI_{lookback_window}'] = 100 * prices[f'RSI_{lookback_window}']
    prices = prices.truncate(first_valid_index)

    return prices[[f'RSI_{lookback_window}']], f'RSI_{lookback_window}'


if __name__ == '__main__':
    symbol = 'RELIANCE'
    period = (datetime(2002, 1, 1), datetime(2024, 10, 10))
    data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    prices = data_access.get_prices(symbol=symbol, start_date=period[0], end_date=period[-1])
    # rsi, _ = RSI(prices=prices, lookback_window=14, average_type='exponential', calculate_rsi_on='returns')
    adx, adx_signal = ADX(prices=prices, average_window=14)
    talib_adx, talib_adx_signal = TALIB_ADX(prices=prices, average_window=14)
    # ts_mom, _ = timeseries_momentum(prices=prices, column_name=Columns.ADJ_CLOSE.value, lookback_window=66)
    # macd_signal,_ = moving_average_crossover(prices=prices,column_name=Columns.ADJ_CLOSE.value,slow_window=64,fast_window=16)
    # prices = prices.join(macd_signal,how='left')
    print('hello')
