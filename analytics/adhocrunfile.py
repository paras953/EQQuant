import pandas as pd
import yfinance as yf
import os

from yfinance.utils import auto_adjust

if __name__ == '__main__':
    symbol = 'SBIN'
    yf_ticker = symbol + ".NS"
    ticker = yf.Ticker(yf_ticker)
    hist_prices = ticker.history(start='2014-11-01',end='2014-11-30',auto_adjust=False,period='1d',back_adjust=False)
    print('hello')
