import pandas as pd
import os
import numpy as np
from datetime import datetime
from data.NSEDataAccess import NSEMasterDataAccess
from utils.decorators import timer
from utils.config import YFINANCE_PRICES_PATH
from utils.config import Columns
from typing import Tuple


def _get_adj_prices_helper(prices: pd.DataFrame) -> pd.DataFrame:
    """
    :param prices: having Close and AdjClose
    :return:  prices having AdjHigh,AdjLow,AdjOpen
    """
    if Columns.ADJ_CLOSE.value not in prices.columns:
        raise ValueError(f"Adj Close price column missing!")
    if Columns.CLOSE.value not in prices.columns:
        raise ValueError(f"Close price column missing!")

    prices['adjustment_factor'] = prices[Columns.ADJ_CLOSE.value] / prices[Columns.CLOSE.value]
    prices[Columns.ADJ_HIGH.value] = prices[Columns.HIGH.value] * prices['adjustment_factor']
    prices[Columns.ADJ_LOW.value] = prices[Columns.LOW.value] * prices['adjustment_factor']
    prices[Columns.ADJ_OPEN.value] = prices[Columns.OPEN.value] * prices['adjustment_factor']
    return prices


def true_range(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: a prices df having columns AdjHigh, AdjLow, AdjClose
    :return: a df having having true range values
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'prev_{Columns.ADJ_CLOSE.value}'] = prices[Columns.ADJ_CLOSE.value].shift()
    prices['true_range'] = prices.apply(lambda x: max((x[Columns.ADJ_HIGH.value] - x[Columns.ADJ_LOW.value]),
                                                      abs(x[Columns.ADJ_HIGH.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}']),
                                                      abs(x[Columns.ADJ_LOW.value] - x[
                                                          f'prev_{Columns.ADJ_CLOSE.value}'])), axis=1)
    return prices[['true_range']], 'true_range'


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


def ADX(prices: pd.DataFrame, average_window: int, average_type: str = 'simple'):
    if average_type not in ['simple', 'exponential']:
        raise ValueError(f"Can only pass average_type as simple/exponential you passed {average_type}")

    true_range_df, true_range_col = true_range(prices)
    directional_df, directional_col = directional_movement(prices=prices)
    positive_column, negative_column = directional_col.split('|')[0], directional_col.split('|')[-1]
    if average_type == 'simple':
        true_range_df[f'smoothed_{true_range_col}'] = true_range_df[true_range_col].rolling(window=average_window,
                                                                                         min_periods=average_window).mean()
        directional_df[f'smoothed_{positive_column}'] = directional_df[positive_column].rolling(window=average_window,
                                                                                                min_periods=average_window).mean()
        directional_df[f'smoothed_{negative_column}'] = directional_df[negative_column].rolling(window=average_window,
                                                                                                min_periods=average_window).mean()
    else:
        true_range_df[f'smoothed_{true_range_col}'] = true_range[true_range_col].ewm(span=average_window,
                                                                                     min_periods=average_window).mean()
        directional_df[f'smoothed_{positive_column}'] = directional_df[positive_column].ewm(span=average_window,
                                                                                            min_periods=average_window).mean()
        directional_df[f'smoothed_{negative_column}'] = directional_df[negative_column].ewm(span=average_window,
                                                                                            min_periods=average_window).mean()
    directional_df = pd.concat([directional_df[[f'smoothed_{positive_column}', f'smoothed_{negative_column}']],
                                true_range_df[[f'smoothed_{true_range_col}']]], axis=1)
    directional_df['normalized_positive_direction'] = directional_df[f'smoothed_{positive_column}'] / directional_df[
        f'smoothed_{true_range_col}']
    directional_df['normalized_negative_direction'] = directional_df[f'smoothed_{negative_column}'] / directional_df[
        f'smoothed_{true_range_col}']
    directional_df['DX'] = abs(
        (directional_df['normalized_positive_direction'] - directional_df['normalized_negative_direction'])) / (
                directional_df['normalized_positive_direction'] + directional_df['normalized_negative_direction'])
    if average_type == 'simple':
        directional_df[f'ADX_{average_window}'] = 100 * directional_df['DX'].rolling(window=average_window,
                                                                                     min_periods=average_window).mean()
    else:
        directional_df[f'ADX_{average_window}'] = 100 * directional_df['DX'].ewm(span=average_window,
                                                                                 min_periods=average_window).mean()

    return directional_df[[f'ADX_{average_window}']], f'ADX_{average_window}'


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
    prices[f'ts_momentum_{lookback_window}'] = -1 + prices[column_name] / prices[f'{column_name}_lagged']
    return prices[[f'ts_momentum_{lookback_window}']], f'ts_momentum_{lookback_window}'


if __name__ == '__main__':
    symbol = 'RELIANCE'
    period = (datetime(2002, 1, 1), datetime(2024, 10, 10))
    data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    prices = data_access.get_prices(symbol=symbol, start_date=period[0], end_date=period[-1])
    adx, _ = ADX(prices=prices, average_window=14)
    # ts_mom, _ = timeseries_momentum(prices=prices, column_name=Columns.ADJ_CLOSE.value, lookback_window=66)
    # macd_signal,_ = moving_average_crossover(prices=prices,column_name=Columns.ADJ_CLOSE.value,slow_window=64,fast_window=16)
    # prices = prices.join(macd_signal,how='left')
    print('hello')
