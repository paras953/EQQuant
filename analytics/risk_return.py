import pandas as pd
import numpy as np
import os
from utils.config import Columns

def get_returns(prices:pd.DataFrame,price_column:str=Columns.ADJ_CLOSE.value)->pd.DataFrame:
    """
    :param prices: a df having prices
    :param price_column: which prices you want to use to calculate the returns?
    :return: a df having the returns
    """
    prices['returns'] = prices[price_column].pct_change()
    return prices[['returns']]


def get_volatility(prices:pd.DataFrame,price_column:str=Columns.ADJ_CLOSE.value)->pd.DataFrame:
    """
    :param prices: a df having prices
    :param price_column: which prices you want to use to calculate the std?
    :return: a df having the volatility
    """
    prices['returns'] = prices[price_column].pct_change()
    prices['volatility'] = prices['returns'].ewm(span=33,min_periods=33).std(bias=True)
    return prices[['volatility']]