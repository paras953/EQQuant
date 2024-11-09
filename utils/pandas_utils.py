import numpy as np
import pandas as pd
import os
from utils.config import Columns


def clean_prices(df: pd.DataFrame, column: str, window_size: int = 11, threshold_zscore: int = 3) -> pd.DataFrame:
    """
    :param df: a pandas dataframe that has the prices
    :param column: which column you want to consider for z scoring
    :param window_size: size of the window if 7 it will consider 7 values before, current value and 7 values after
    :param threshold_zscore: what is the threshold zscore you want to delete the data
    :return: returns the dataframe where rows having zscore>4 are removed
    """
    print(f'Removing outliers for {column}')
    stats_dict = {}
    df['rolling_mean'] = df[column].rolling(window=window_size, min_periods=1, center=True).mean()
    df['std_dev'] = df[column].rolling(window=window_size, min_periods=1, center=True).std()
    df['zscore'] = abs(df[column] - df['rolling_mean']) / df['std_dev']
    mask = df['zscore'] > threshold_zscore
    df.loc[mask, :] = np.nan
    stats_dict['OUTLIER_PERCENT'] = mask.mean() * 100

    # remove rows where open/close>High, open/cl
    mask_open_gt_high = df[Columns.OPEN.value] > df[Columns.HIGH.value]
    mask_close_gt_high = df[Columns.CLOSE.value] > df[Columns.HIGH.value]
    mask_open_lt_low = df[Columns.OPEN.value] < df[Columns.LOW.value]
    mask_close_lt_low = df[Columns.CLOSE.value] < df[Columns.LOW.value]
    df.loc[mask_open_gt_high, :] = np.nan
    df.loc[mask_close_gt_high, :] = np.nan
    df.loc[mask_open_lt_low, :] = np.nan
    df.loc[mask_close_lt_low, :] = np.nan

    mask_volume_lt_eq_zero = df[Columns.VOLUME.value] <= 0
    df.loc[mask_volume_lt_eq_zero,:] = np.nan

    stats_dict['OPEN_GT_HIGH'] = mask_open_gt_high.mean() * 100
    stats_dict['CLOSE_GT_HIGH'] = mask_close_gt_high.mean() * 100
    stats_dict['OPEN_LT_LOW'] = mask_open_lt_low.mean() * 100
    stats_dict['CLOSE_LT_LOW'] = mask_close_lt_low.mean() * 100
    stats_dict['VOLUME_LT_EQ_ZERO'] = mask_volume_lt_eq_zero.mean() * 100

    print("Stats dict, all figures are mentioned in percentage terms")
    print(stats_dict)
    df[Columns.ADJ_CLOSE.value] = df[Columns.ADJ_CLOSE.value].ffill()
    df[Columns.CLOSE.value] = df[Columns.CLOSE.value] .ffill()
    df[Columns.OPEN.value] = df[Columns.OPEN.value].combine_first(df[Columns.CLOSE.value])
    df[Columns.HIGH.value] = df[Columns.HIGH.value].combine_first(df[Columns.CLOSE.value])
    df[Columns.LOW.value] = df[Columns.LOW.value].combine_first(df[Columns.CLOSE.value])
    df = df.drop(columns=['rolling_mean', 'std_dev', 'zscore'])
    return df


def custom_rolling_mean(series: pd.Series, lookback: int) -> pd.Series:
    """
    :param series: any pandas series
    :param lookback: lookback to calculate the average
    :return:
    """
    # Initialize an array to store results, starting with NaNs
    result = np.full(len(series), np.nan)

    # Compute the first value as the simple rolling mean with a full window of `x`
    if len(series) >= lookback:
        result[lookback - 1] = series[:lookback].mean()  # Only set the mean after we have `x` elements

    # Compute the rest using the custom formula
    for i in range(lookback, len(series)):
        result[i] = (result[i - 1] * (lookback - 1) + series[i]) / lookback

    return pd.Series(result, index=series.index)


def custom_rolling_sum(series: pd.Series, lookback: int) -> pd.Series:
    """
    Used to calculate TA Lib implementation of Average True Range, Smooth Positive Directional Movement
    Smoothed Negative Directional Movement
    :param series: any pandas series
    :param lookback: lookback to calculate the average
    :return: the smoothed value as used by TA lib PLUS_DI and MINUS_DI indicators
    """
    # Initialize an array to store results, starting with NaNs
    result = np.full(len(series), np.nan)

    # Compute the first value as the simple rolling mean with a full window of `x`
    if len(series) >= lookback:
        result[lookback - 1] = series[:lookback].sum()  # Only set the mean after we have `x` elements

    # Compute the rest using the custom formula
    for i in range(lookback, len(series)):
        prev_value = result[i-1]
        rolling_avg_prev_value = pd.Series(series[:i]).rolling(window=lookback,min_periods=lookback).mean().iloc[-1]
        result[i] = prev_value - rolling_avg_prev_value + series[i]

    return pd.Series(result, index=series.index)


