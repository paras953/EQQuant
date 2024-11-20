import numpy as np
import pandas as pd
import os
from typing import Dict
from utils.config import Columns


def clean_prices(df: pd.DataFrame, column: str, window_size: int = 11, threshold_zscore: int = 3,
                 ignore_volume:bool=False) -> pd.DataFrame:
    """
    :param df: a pandas dataframe that has the prices
    :param column: which column you want to consider for z scoring
    :param window_size: size of the window if 7 it will consider 7 values before, current value and 7 values after
    :param threshold_zscore: what is the threshold zscore you want to delete the data
    :param ignore_volume : We dont have volume data
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

    if not ignore_volume:
        print('Not ignoring Volume data!')
        mask_volume_lt_eq_zero = df[Columns.VOLUME.value] <= 0
        df.loc[mask_volume_lt_eq_zero, :] = np.nan

    stats_dict['OPEN_GT_HIGH'] = mask_open_gt_high.mean() * 100
    stats_dict['CLOSE_GT_HIGH'] = mask_close_gt_high.mean() * 100
    stats_dict['OPEN_LT_LOW'] = mask_open_lt_low.mean() * 100
    stats_dict['CLOSE_LT_LOW'] = mask_close_lt_low.mean() * 100
    if not ignore_volume:
        stats_dict['VOLUME_LT_EQ_ZERO'] = mask_volume_lt_eq_zero.mean() * 100

    print("Stats dict, all figures are mentioned in percentage terms")
    print(stats_dict)
    df[Columns.ADJ_CLOSE.value] = df[Columns.ADJ_CLOSE.value].ffill()
    df[Columns.CLOSE.value] = df[Columns.CLOSE.value].ffill()
    df[Columns.OPEN.value] = df[Columns.OPEN.value].combine_first(df[Columns.CLOSE.value])
    df[Columns.HIGH.value] = df[Columns.HIGH.value].combine_first(df[Columns.CLOSE.value])
    df[Columns.LOW.value] = df[Columns.LOW.value].combine_first(df[Columns.CLOSE.value])
    df = df.drop(columns=['rolling_mean', 'std_dev', 'zscore'])
    df = df.asfreq('B')
    df = df.dropna(subset=[Columns.CLOSE.value])  # automatically removes holidays that fall between mon-fri
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
        prev_value = result[i - 1]
        rolling_avg_prev_value = pd.Series(series[:i]).rolling(window=lookback, min_periods=lookback).mean().iloc[-1]
        result[i] = prev_value - rolling_avg_prev_value + series[i]

    return pd.Series(result, index=series.index)


def df_to_excel(df_dict: Dict[str, pd.DataFrame], output_path: str, file_name: str) -> None:
    """
    :param df_dict: dict having values as df, the key will be the sheet name
    :param output_path: directory where the excel gets dumped
    :param file_name: name of the file
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Specify the output Excel file
    output_file = f"{output_path}/{file_name}.xlsx"

    # Write the dictionary of DataFrames to an Excel file
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)  # Write each DataFrame to a sheet

    print(f'Excel file saved in {output_file}')
    return None
