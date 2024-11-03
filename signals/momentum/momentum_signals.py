from signal import signal

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
    prices[f'average_true_range_{lookback}'] = custom_rolling_sum(series=prices['true_range'], lookback=lookback)

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
    average_true_range_df, average_true_range_column = average_true_range(prices=prices, lookback=average_window)
    directional_df[f'smoothed_{positive_column}'] = custom_rolling_sum(series=directional_df[positive_column],
                                                                       lookback=average_window)
    directional_df[f'smoothed_{negative_column}'] = custom_rolling_sum(series=directional_df[negative_column],
                                                                       lookback=average_window)

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
    directional_df[f'ADX_{average_window}'] = custom_rolling_mean(series=directional_df['DX'], lookback=average_window)
    directional_df[f'ADX_{average_window}'] = directional_df[f'ADX_{average_window}'] * 100
    first_valid_index = directional_df[f'ADX_{average_window}'].first_valid_index()
    directional_df = directional_df.truncate(first_valid_index)
    print('Min ADX - ', directional_df[f'ADX_{average_window}'].min())
    print('Min ADX - ', directional_df[f'ADX_{average_window}'].max())

    return directional_df[[f'ADX_{average_window}']], f'ADX_{average_window}'


@timer
def TALIB_ADX(prices: pd.DataFrame, average_window: int) -> Tuple[pd.DataFrame, str]:
    """
    Cannot directly use as a signal we can use it as a complement to other momentum signal
    TA LIB implementation of ADX
    :param prices: prices df
    :param average_window: lookback of the ADX signal
    :return: the signal and name of the column
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()  # if the prices are null TA LIB messes up
    prices[f'TALIB_ADX_{average_window}'] = ta.ADX(high=prices[Columns.ADJ_HIGH.value],
                                                   low=prices[Columns.ADJ_LOW.value],
                                                   close=prices[Columns.ADJ_CLOSE.value], timeperiod=average_window)
    return prices[[f'TALIB_ADX_{average_window}']], f'TALIB_ADX_{average_window}'


@timer
def TALIB_ADXR(prices: pd.DataFrame, average_window: int) -> Tuple[pd.DataFrame, str]:
    """
    Cannot directly use as a signal we can use it as a complement to other momentum signal
    See TALIB_ADX
    :param prices:
    :param average_window:
    :return: ADXR implementation of TA LIB
    """
    prices = _get_adj_prices_helper(prices)
    prices = prices.ffill()  # if the prices are null TA LIB messes up
    prices[f'TALIB_ADXR_{average_window}'] = ta.ADXR(high=prices[Columns.ADJ_HIGH.value],
                                                     low=prices[Columns.ADJ_LOW.value],
                                                     close=prices[Columns.ADJ_CLOSE.value],
                                                     timeperiod=average_window)
    return prices[[f'TALIB_ADXR_{average_window}']], f'TALIB_ADXR_{average_window}'


@timer
def TALIB_APO(prices: pd.DataFrame, slow: int, fast: int, average_type: str = 'exponential') -> Tuple[
    pd.DataFrame, str]:
    """
    APO is just the Fast - SLow avg of prices
    :param prices: prices df
    :param slow:  slow avg window length
    :param fast: fast avg window length
    :param average_type: exponential/simple
    :return:
    """
    ma_type_dict = {'simple': 0,
                    'exponential': 1}  # https://github.com/twopirllc/pandas-ta/blob/main/pandas_ta/utils/_core.py#L86
    prices[f'TALIB_APO_{fast}_{slow}'] = ta.APO(real=prices[Columns.ADJ_CLOSE.value], slowperiod=slow, fastperiod=fast,
                                                matype=ma_type_dict[average_type])
    return prices[[f'TALIB_APO_{fast}_{slow}']], f'TALIB_APO_{fast}_{slow}'


@timer
def TALIB_AROONOSC(prices: pd.DataFrame, lookback: int):
    """
    :param prices:
    :param lookback: lookback for the signal
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'TALIB_AROONOSC_{lookback}'] = ta.AROONOSC(high=prices[Columns.ADJ_HIGH.value],
                                                       low=prices[Columns.ADJ_LOW.value],
                                                       timeperiod=lookback)
    return prices[[f'TALIB_AROONOSC_{lookback}']], f'TALIB_AROONOSC_{lookback}'


@timer
def TALIB_AROONUPDOWN(prices: pd.DataFrame, lookback: int, signal_type: str = 'UP'):
    """

    :param prices:
    :param lookback: lookback for the signal
    :param signal_type : if 'UP' then AROONUP values are returned else AROONDOWN values are returned
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    aroondown, aroonup = ta.AROON(high=prices[Columns.ADJ_HIGH.value],
                                  low=prices[Columns.ADJ_LOW.value],
                                  timeperiod=lookback)
    prices[f'TALIB_AROONUP_{lookback}'] = aroonup
    prices[f'TALIB_AROONDOWN_{lookback}'] = aroondown
    if signal_type == 'UP':
        return prices[[f'TALIB_AROONUP_{lookback}']], f'TALIB_AROONUP_{lookback}'
    else:
        return prices[[f'TALIB_AROONDOWN_{lookback}']], f'TALIB_AROONDOWN_{lookback}'


@timer
def TALIB_BOP(prices: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Balance of Power ta lib implementation
    :param prices: prices df
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices['TALIB_BOP'] = ta.BOP(open=prices[Columns.ADJ_OPEN.value], high=prices[Columns.ADJ_HIGH.value],
                                 low=prices[Columns.ADJ_LOW.value], close=prices[Columns.ADJ_CLOSE.value])
    return prices[['TALIB_BOP']], 'TALIB_BOP'


@timer
def TALIB_CCI(prices: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :param lookback:
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices['TALIB_CCI'] = ta.CCI(high=prices[Columns.ADJ_HIGH.value],
                                 low=prices[Columns.ADJ_LOW.value], close=prices[Columns.ADJ_CLOSE.value],
                                 timeperiod=lookback)
    return prices[['TALIB_CCI']], 'TALIB_CCI'


@timer
def TALIB_CMO(prices: pd.DataFrame, lookback: int, column: str = Columns.ADJ_CLOSE.value) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :param lookback: lookback of the signal
    :param column, which prices you want to use to calculate the signal
    :return:
    """
    prices[f'TALIB_CMO_{lookback}'] = ta.CMO(real=prices[column], timeperiod=lookback)
    return prices[[f'TALIB_CMO_{lookback}']], f'TALIB_CMO_{lookback}'


@timer
def TALIB_MFI(prices: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices: prices df
    :param lookback: lookback of the signal
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'TALIB_MFI_{lookback}'] = ta.MFI(high=prices[Columns.ADJ_HIGH.value],
                                             low=prices[Columns.ADJ_LOW.value], close=prices[Columns.ADJ_CLOSE.value],
                                             volume=prices[Columns.VOLUME.value], timeperiod=lookback)
    return prices[[f'TALIB_MFI_{lookback}']], f'TALIB_MFI_{lookback}'


@timer
def TALIB_PPO(prices: pd.DataFrame, slow: int, fast: int, column: str = Columns.ADJ_CLOSE.value) -> Tuple[
    pd.DataFrame, str]:
    """
    PPO is actually (f-s/s)
    :param prices: prices df
    :return:
    """
    # matype = 1 is exponential moving average
    prices[f'TALIB_PPO_{fast}_{slow}'] = ta.PPO(real=prices[column], slowperiod=slow, fastperiod=fast, matype=1)
    return prices[[f'TALIB_PPO_{fast}_{slow}']], f'TALIB_PPO_{fast}_{slow}'


@timer
def TALIB_RSI(prices: pd.DataFrame, lookback: int, column: str = Columns.ADJ_CLOSE.value) -> Tuple[pd.DataFrame, str]:
    """

    :param prices: prices df
    :param lookback: lookback of the signal
    :param column:
    :return:
    """
    prices[f'TALIB_RSI_{lookback}'] = ta.RSI(real=prices[column], timeperiod=lookback)
    return prices[[f'TALIB_RSI_{lookback}']], f'TALIB_RSI_{lookback}'


@timer
def TALIB_STOCH(prices: pd.DataFrame, fastk_period: int, slowk_period: int, slowd_period: int,
                signal_type: str = 'combined') -> \
        Tuple[pd.DataFrame, str]:
    """
    How to interpret these values?
    :param prices: prices df
    :param fastk: fast k lookback
    :param slowk: slow k lookback
    :param slowd: slow d lookback
    :return:
    """
    if signal_type not in ['slowk', 'slowd', 'combined']:
        raise ValueError('Signal type has to be one of slowk, slowd or combined')

    signal_name = f'TALIB_RSI_{signal_type}_{fastk_period}_{slowk_period}_{slowd_period}'
    prices = _get_adj_prices_helper(prices)
    slowk, slowd = ta.STOCH(high=prices[Columns.ADJ_HIGH.value], low=prices[Columns.ADJ_LOW.value],
                            close=prices[Columns.ADJ_CLOSE.value], fastk_period=fastk_period,
                            slowk_period=slowk_period, slowd_period=slowd_period, slowk_matype=1, slowd_matype=1)
    if signal_type == 'slowk':
        prices[signal_name] = slowk
    elif signal_type == 'slowd':
        prices[signal_name] = slowd
    else:
        prices[signal_name] = (slowk - slowd) / slowk

    return prices[[signal_name]], signal_name


@timer
def TALIB_STOCHF(prices: pd.DataFrame, fastk_period: int, fastd_period: int, signal_type: str = 'combined') -> Tuple[
    pd.DataFrame, str]:
    """
    :param prices:
    :param fastk:
    :param fastd:
    :param signal_type:
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    fastk, fastd = ta.STOCHF(high=prices[Columns.ADJ_HIGH.value], low=prices[Columns.ADJ_LOW.value],
                             close=prices[Columns.ADJ_CLOSE.value], fastk_period=fastk_period,
                             fastd_period=fastd_period, fastd_matype=1)
    signal_name = f'TALIB_STOCHF_{signal_type}_{fastk_period}_{fastd_period}'
    if signal_type not in ['fastk', 'fastd', 'combined']:
        raise ValueError(f'signal_type should be one of fastk,fastd or combined you passed {signal_type}')
    if signal_type == 'fastk':
        prices[signal_name] = fastk
    elif signal_type == 'fastd':
        prices[signal_name] = fastd
    else:
        prices[signal_name] = (fastk - fastd) / fastk

    return prices[[signal_name]], signal_name


# fastk gives some zeros, check before using it
@timer
def TALIB_STOCHRSI(prices: pd.DataFrame, lookback: int, fastk_period: int, fastd_period: int,
                   signal_type: str = 'combined', column: str = Columns.ADJ_CLOSE.value) -> Tuple[
    pd.DataFrame, str]:
    """
    :param prices:
    :param fastk: lookback for fastk
    :param fastd: lookback for fastd
    :param  lookback : lookback to calculate RSI
    :param column : which prices you want to create signal on
    :param signal_type:
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    fastk, fastd = ta.STOCHRSI(real=prices[column], timeperiod=lookback, fastk_period=fastk_period,
                               fastd_period=fastd_period, fastd_matype=1)
    signal_name = f'TALIB_STOCHRSI_{signal_type}_{fastk_period}_{fastd_period}'
    if signal_type not in ['fastk', 'fastd', 'combined']:
        raise ValueError(f'signal_type should be one of fastk,fastd or combined you passed {signal_type}')
    if signal_type == 'fastk':
        prices[signal_name] = fastk
    elif signal_type == 'fastd':
        prices[signal_name] = fastd
    else:
        prices[signal_name] = (fastk - fastd) / fastk

    return prices[[signal_name]], signal_name


@timer
def TALIB_ULTOSC(prices: pd.DataFrame, lookback_1: int, lookback_2: int, lookback_3: int) -> Tuple[pd.DataFrame, str]:
    """
    :param prices:
    :param lookback_1: lookback 1
    :param lookback_2: lookback 2
    :param lookback_3: lookback 3
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'TALIB_ULTOSC_{lookback_1}_{lookback_2}_{lookback_3}'] = ta.ULTOSC(high=prices[Columns.ADJ_HIGH.value],
                                                                               low=prices[Columns.ADJ_LOW.value],
                                                                               close=prices[Columns.ADJ_CLOSE.value],
                                                                               timeperiod1=lookback_1,
                                                                               timeperiod2=lookback_2,
                                                                               timeperiod3=lookback_3)
    return prices[[
        f'TALIB_ULTOSC_{lookback_1}_{lookback_2}_{lookback_3}']], f'TALIB_ULTOSC_{lookback_1}_{lookback_2}_{lookback_3}'


@timer
def TALIB_WILLR(prices:pd.DataFrame,lookback:int):
    """
    :param prices: prices df
    :param lookback: lookback of the signal
    :return:
    """
    prices = _get_adj_prices_helper(prices)
    prices[f'TALIB_WILLR_{lookback}'] = ta.WILLR(high=prices[Columns.ADJ_HIGH.value],
                                                 low=prices[Columns.ADJ_LOW.value],
                                                 close=prices[Columns.ADJ_CLOSE.value],
                                                 timeperiod=lookback)
    return prices[[f'TALIB_WILLR_{lookback}']],f'TALIB_WILLR_{lookback}'

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
    # adx, adx_signal = TALIB_ADX(prices=prices, average_window=14)
    # talib_adx, talib_adx_signal = TALIB_ADXR(prices=prices, average_window=14)
    # ts_mom, _ = timeseries_momentum(prices=prices, column_name=Columns.ADJ_CLOSE.value, lookback_window=66)
    # macd_signal,_ = moving_average_crossover(prices=prices,column_name=Columns.ADJ_CLOSE.value,slow_window=64,fast_window=16)
    # prices = prices.join(macd_signal,how='left')
    # apo, apo_col = TALIB_APO(prices=prices, slow=5, fast=2, average_type='simple')
    # apo2, apo_col2 = TALIB_APO(prices=prices, slow=5, fast=2, average_type='exponential')
    # aroonosc, _ = TALIB_AROONOSC(prices=prices, lookback=5)
    # aroonup, _ = TALIB_AROONUPDOWN(prices=prices, lookback=5, signal_type='UP')
    # aroondown, _ = TALIB_AROONUPDOWN(prices=prices, lookback=5, signal_type='DOWN')
    # cci, _ = TALIB_CCI(prices=prices, lookback=20)
    # cmo, _ = TALIB_CMO(prices=prices, lookback=20)
    # mfi, _ = TALIB_MFI(prices=prices, lookback=14)
    # ppo,_ = TALIB_PPO(prices=prices,slow=64,fast=16)
    # rsi, _ = TALIB_RSI(prices=prices, lookback=20)
    # combined, _ = TALIB_STOCH(prices=prices, fastk_period=25, slowk_period=5, slowd_period=5, signal_type='combined')
    # combined, _ = TALIB_STOCHF(prices=prices, fastk_period=20, fastd_period=10)
    # combined,_ = TALIB_STOCHRSI(prices=prices, lookback=25, fastk_period=14, fastd_period=3, signal_type='fastk')
    # ultosc,_ = TALIB_ULTOSC(prices=prices,lookback_1=22,lookback_2=66,lookback_3=99)
    willr,_ = TALIB_WILLR(prices=prices,lookback=25)

    print('hello')

