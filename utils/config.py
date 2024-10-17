from enum import Enum
from datetime import datetime
import os

base_path = 'C:/Users/paras/PycharmProjects/EQQuant'

PRICES_PKL_PATH = f'{base_path}/additional_data/'
TICKER_METADATA_PATH = f'{base_path}/additional_data/ticker_metadata_2024-10-08.csv'

# just make sure to update these files everyday
# TODO : try to auto download these files from NSE website
# src : https://www.nseindia.com/companies-listing/corporate-filings-actions
BONUS_PATH_ALL = f'{base_path}/additional_data/BONUS-01-01-2002-to-17-10-2024.csv'
DIVIDEND_PATH_ALL = f'{base_path}/additional_data/DIVIDENDS-01-01-2002-to-17-10-2024.csv'
SPLIT_PATH_ALL = f'{base_path}/additional_data/SPLIT_PART1-01-01-2002-to-17-10-2024.csv|C:/Users/paras/PycharmProjects/EQQuant/additional_data/SPLIT_PART2-01-01-2002-to-17-10-2024.csv'

# Corporate actions for nifty 50 stocks manually extracted the bonus, split and dividend data
# has nifty50 stocks as of 2024-10-17
BONUS_NIFTY50 = f'{base_path}/additional_data/nifty50_corporate_actions/bonus_nifty50_stocks.csv'
SPLIT_NIFTY50 = f'{base_path}/additional_data/nifty50_corporate_actions/split_nifty50_stocks.csv'
DIVIDEND_NIFTY50 = f'{base_path}/additional_data/nifty50_corporate_actions/dividend_nifty50_stocks_processed.csv'

GOOD_DATE_MAP = {
    'ADANIENT': datetime(2002, 1, 1)
}


class Columns(Enum):
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    LTP = 'LastTradedPrice'
    VWAP = 'VWAP'
    ADJ_CLOSE = 'AdjClose'
    ADJ_HIGH = 'AdjHigh'
    ADJ_LOW = 'AdjLow'
    ADJ_OPEN = 'AdjOpen'
    ADJ_LTP = 'AdjLastTradedPrice'
    ADJ_VWAP = 'AdjVWAP'

    VOLUME = 'Volume'
    TRADES = 'Trades'
