import numpy as np
import pandas as pd
import os


def remove_outliers(df:pd.DataFrame,column:str,window_size:int=11,threshold_zscore:int=3)->pd.DataFrame:
    """
    :param df: a pandas dataframe that has the prices
    :param column: which column you want to consider for z scoring
    :param window_size: size of the window if 7 it will consider 7 values before, current value and 7 values after
    :param threshold_zscore: what is the threshold zscore you want to delete the data
    :return: returns the dataframe where rows having zscore>4 are removed
    """
    print(f'Removing outliers for {column}')
    df['rolling_mean'] = df[column].rolling(window=window_size,min_periods=1,center=True).mean()
    df['std_dev'] = df[column].rolling(window=window_size,min_periods=1,center=True).std()
    df['zscore'] = abs(df[column]-df['rolling_mean'])/df['std_dev']
    mask = df['zscore']>threshold_zscore
    df.loc[mask,:] = np.nan
    percent_removed = mask.mean()*100
    print(f'Removed {percent_removed:.2f}%')
    df = df.ffill()
    return df
