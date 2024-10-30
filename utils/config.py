from enum import Enum
from datetime import datetime
import os


repo_root = '../..'


NSEPYTHON_PRICES_PATH = f'{repo_root}/additional_data/'
TICKER_METADATA_PATH = f'{repo_root}/additional_data/ticker_metadata_2024-10-08.csv'
YFINANCE_PRICES_PATH = f'{repo_root}/additional_data/yfinance_data/'
# just make sure to update these files everyday
# TODO : try to auto download these files from NSE website
# src : https://www.nseindia.com/companies-listing/corporate-filings-actions
CORPORATE_ACTIONS_PATH = f'{repo_root}/additional_data/NSE_CORPORATE_ACTIONS-01-01-2002-to-25-09-2024.csv'
DIVIDEND_PATH = f'{repo_root}/additional_data/DIVIDENDS-01-01-2002-to-07-10-2024.csv'

GOOD_DATE_MAP = {'ADANIENT': datetime(2007, 1, 1),
                 'ADANIPORTS': datetime(2008, 1, 1),
                 'APOLLOHOSP': datetime(2003, 1, 1),
                 'ASIANPAINT': datetime(2003, 1, 1),
                 'AXISBANK': datetime(2003, 1, 1),
                 'BAJAJ-AUTO': datetime(2009, 1, 1),  # de-merger in march 2008 weird drop so set this as good date.
                 'BAJFINANCE': datetime(2003, 1, 1),
                 'BAJAJFINSV': datetime(2009, 1, 1),
                 'BEL': datetime(2003, 1, 1),
                 'BPCL': datetime(2003, 1, 1),
                 'BHARTIARTL': datetime(2003, 1, 1),
                 'BRITANNIA': datetime(2003, 1, 1),
                 'CIPLA': datetime(2005, 1, 1),  # pre 2005 the zscoring failed to recognize outliers will check later,
                 'COALINDIA': datetime(2011, 1, 1),  # listed in 2010 mid
                 'DRREDDY': datetime(2003, 1, 1),
                 'EICHERMOT': datetime(2003, 1, 1),
                 'GRASIM': datetime(2003, 1, 1),
                 'HCLTECH': datetime(2003, 1, 1),
                 'HDFCBANK': datetime(2003, 1, 1),
                 'HDFCLIFE': datetime(2018, 1, 1),  # listed in nov 2017
                 'HEROMOTOCO': datetime(2003, 1, 1),
                 'HINDALCO': datetime(2003, 1, 1),
                 'HINDUNILVR': datetime(2003, 1, 1),
                 'ICICIBANK': datetime(2003, 1, 1),
                 'INDUSINDBK': datetime(2003, 1, 1),
'INFY': datetime(2003, 1, 1), #2003 - biggest fall ,2017 - CEO Resigned - 9% fall,2019 - 16.2% fall ,https://x.com/SumitResearch/status/1780938157413797988
                 'ITC': datetime(2003, 1, 1),
                 'JSWSTEEL': datetime(2004, 1, 1), #IPO in may 2003
                 'KOTAKBANK': datetime(2003, 1, 1),
                 'LT': datetime(2005, 1, 1), # relisted in 2004
                 'M&M': datetime(2004, 1, 1),
                 'MARUTI': datetime(2004, 1, 1),
                 'NESTLEIND': datetime(2010, 1, 1), # Prices missing from 2004 (duplicated prices)
                 'NTPC': datetime(2005, 1, 1),
                 'ONGC': datetime(2003, 1, 1),
                 'POWERGRID': datetime(2008, 1, 1), # listed in 2007
                 'RELIANCE': datetime(2003, 1, 1),
                 'SBILIFE': datetime(2018, 1, 1), # listed in 2017
                 'SBIN': datetime(2003, 1, 1),
                 'SHRIRAMFIN': datetime(2003, 1, 1),
                 'SUNPHARMA': datetime(2003, 1, 1),
                 'TATACONSUM': datetime(2003, 1, 1),
                 'TATAMOTORS': datetime(2003, 1, 1),
                 'TATASTEEL': datetime(2003, 1, 1),
                 'TCS': datetime(2005, 1, 1), # listed in 2004
                 'TECHM': datetime(2007, 1, 1), # listed in 2007
                 'TITAN': datetime(2003, 1, 1),
                 'TRENT': datetime(2003, 1, 1),
                 'ULTRACEMCO': datetime(2005, 1, 1),# listed in 2004
                 'WIPRO': datetime(2003, 1, 1),
                 }

NSE_INDEX_MASTER = {
    "Broad Market Indices": [
        "NIFTY 50",
        "NIFTY NEXT 50",
        "NIFTY MIDCAP 50",
        "NIFTY MIDCAP 100",
        "NIFTY MIDCAP 150",
        "NIFTY SMALLCAP 50",
        "NIFTY SMALLCAP 100",
        "NIFTY SMALLCAP 250",
        "NIFTY MIDSMALLCAP 400",
        "NIFTY 100",
        "NIFTY 200",
        "NIFTY500 MULTICAP 50:25:25",
        "NIFTY LARGEMIDCAP 250",
        "NIFTY MIDCAP SELECT",
        "NIFTY TOTAL MARKET",
        "NIFTY MICROCAP 250",
        "NIFTY 500"
    ],
    "Sectoral Indices": [
        "NIFTY AUTO",
        "NIFTY BANK",
        "NIFTY ENERGY",
        "NIFTY FINANCIAL SERVICES",
        "NIFTY FINANCIAL SERVICES 25/50",
        "NIFTY FMCG",
        "NIFTY IT",
        "NIFTY MEDIA",
        "NIFTY METAL",
        "NIFTY PHARMA",
        "NIFTY PSU BANK",
        "NIFTY REALTY",
        "NIFTY PRIVATE BANK",
        "NIFTY HEALTHCARE INDEX",
        "NIFTY CONSUMER DURABLES",
        "NIFTY OIL & GAS",
        "NIFTY MIDSMALL HEALTHCARE"
    ],
    "Thematic Indices": [
        "NIFTY COMMODITIES",
        "NIFTY INDIA CONSUMPTION",
        "NIFTY CPSE",
        "NIFTY INFRASTRUCTURE",
        "NIFTY MNC",
        "NIFTY GROWTH SECTORS 15",
        "NIFTY PSE",
        "NIFTY SERVICES SECTOR",
        "NIFTY100 LIQUID 15",
        "NIFTY MIDCAP LIQUID 15",
        "NIFTY INDIA DIGITAL",
        "NIFTY100 ESG",
        "NIFTY INDIA MANUFACTURING",
        "NIFTY INDIA CORPORATE GROUP INDEX - TATA GROUP 25% CAP",
        "NIFTY500 MULTICAP INDIA MANUFACTURING 50:30:20",
        "NIFTY500 MULTICAP INFRASTRUCTURE 50:30:20"
    ],
    "Strategy Indices": [
        "NIFTY DIVIDEND OPPORTUNITIES 50",
        "NIFTY50 VALUE 20",
        "NIFTY100 QUALITY 30",
        "NIFTY50 EQUAL WEIGHT",
        "NIFTY100 EQUAL WEIGHT",
        "NIFTY100 LOW VOLATILITY 30",
        "NIFTY ALPHA 50",
        "NIFTY200 QUALITY 30",
        "NIFTY ALPHA LOW-VOLATILITY 30",
        "NIFTY200 MOMENTUM 30",
        "NIFTY MIDCAP150 QUALITY 50",
        "NIFTY200 ALPHA 30",
        "NIFTY MIDCAP150 MOMENTUM 50"
    ],
    "Others": [
        "Securities in F&O",
        "Permitted to Trade"
    ]
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
