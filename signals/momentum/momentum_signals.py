import pandas as pd
import os
import numpy as np
from utils.decorators import timer
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
def directional_movement(prices: pd.DataFrame,window:int) -> Tuple[pd.DataFrame, str]:
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

    