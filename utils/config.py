from enum import Enum
from datetime import datetime
import os

base_path = 'C:/Users/paras/PycharmProjects/EQQuant'
PRICES_PKL_PATH = '../additional_data/'
TICKER_METADATA_PATH = '../additional_data/ticker_metadata_2024-10-08.csv'

# just make sure to update these files everyday
# TODO : try to auto download these files from NSE website
# src : https://www.nseindia.com/companies-listing/corporate-filings-actions
CORPORATE_ACTIONS_PATH = '../additional_data/NSE_CORPORATE_ACTIONS-01-01-2002-to-25-09-2024.csv'
DIVIDEND_PATH = '../additional_data/DIVIDENDS-01-01-2002-to-07-10-2024.csv'



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
