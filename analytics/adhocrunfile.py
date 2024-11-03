import pandas as pd
import yfinance as yf
import os
import numpy as np
from yfinance.utils import auto_adjust
from signals.volume.volume_signals import TALIB_OBV, TALIB_ADChaikin, TALIB_ADOSCChaikin
from data.NSEDataAccess import NSEMasterDataAccess
from utils.config import YFINANCE_PRICES_PATH,TICKER_METADATA_PATH
from datetime import datetime
from signals.volatility.volatility_signals import TALIB_ATR,TALIB_NATR,TALIB_TRANGE



if __name__ == '__main__':
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Example data
    start_date = datetime(2002, 1, 1)
    end_date = datetime(2024, 12, 31)
    nse_data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    df = nse_data_access.get_prices(symbol="WIPRO", start_date=start_date, end_date=end_date)


    data_rolling = TALIB_ATR(df)
    data_rolling = TALIB_NATR(df)
    data_rolling = TALIB_TRANGE(df)

    print(data_rolling)
