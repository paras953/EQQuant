from utils.config import Columns
import pandas as pd

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