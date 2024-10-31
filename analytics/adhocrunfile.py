import pandas as pd
import yfinance as yf
import os
import numpy as np
from yfinance.utils import auto_adjust




if __name__ == '__main__':
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Example data
    x = 3  # Window size

    data_rolling = custom_rolling_mean(data, x)
    print(data_rolling)
