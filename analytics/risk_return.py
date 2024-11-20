import pandas as pd
import numpy as np
import os
from utils.config import Columns
from typing import Dict


def get_returns(prices: pd.DataFrame, price_column: str = Columns.ADJ_CLOSE.value) -> pd.DataFrame:
    """
    :param prices: a df having prices
    :param price_column: which prices you want to use to calculate the returns?
    :return: a df having the returns
    """
    prices['returns'] = prices[price_column].pct_change()
    return prices[['returns']]


def get_volatility(prices: pd.DataFrame, price_column: str = Columns.ADJ_CLOSE.value) -> pd.DataFrame:
    """
    :param prices: a df having prices
    :param price_column: which prices you want to use to calculate the std?
    :return: a df having the volatility
    """
    prices['returns'] = prices[price_column].pct_change()
    prices['volatility'] = prices['returns'].ewm(span=33, min_periods=33).std(bias=True)

    first_valid_index = prices['volatility'].first_valid_index()
    prices = prices.truncate(first_valid_index)
    return prices[['volatility']]


def get_asset_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: where columns are the symbols and values are prices (can be open or close or vwap anything)
    :return: df having returns
    """
    df = df.pct_change()
    return df

def get_asset_volatility(prices:pd.DataFrame):
    """
    :param df:  where columns are the symbols and values are prices (can be open or close or vwap anything)
    :return: df having vol for each asset
    """
    df_list = []
    for columns in prices.columns:
        vol_df = get_volatility(prices=prices[[columns]],price_column=columns)
        vol_df = vol_df.rename(columns={'volatility':columns})
        df_list.append(vol_df)
    vol_df = pd.concat(df_list,axis=1)
    return vol_df



def get_intraday_returns(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    :param open_prices: open prices per symbol
    :param close_prices: close prices per symbol
    :return: dict having OpenToClose and PrevDay CloseToOpen returns
    """
    open_to_close_returns = (-1 + (close_prices / open_prices))
    prev_close_to_open_returns = (-1 + (open_prices / close_prices.shift()))
    close_to_close_returns = get_asset_returns(close_prices)
    return {'open_to_close_returns': open_to_close_returns,
            'prev_day_close_to_open_returns': prev_close_to_open_returns,
            'close_to_close_returns' : close_to_close_returns
            }
