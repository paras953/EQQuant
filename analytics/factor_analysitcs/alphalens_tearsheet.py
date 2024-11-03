from data.NSEDataAccess import NSEMasterDataAccess
from signals.momentum.momentum_signals import moving_average_crossover
import alphalens
from datetime import datetime
import pandas as pd

from utils.config import YFINANCE_PRICES_PATH, Columns

if __name__ == '__main__':
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    nse_data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    symbol_list = nse_data_access.get_index_constituents(index_name='NIFTY 50')
    symbol_list = ['TECHM']
    all_prices = []
    factor_list = []
    slow = 64
    fast = 16
    signal_name = ''
    for symbol in symbol_list:
        prices = nse_data_access.get_prices(symbol=symbol, start_date=start_date, end_date=end_date,
                                            )
        factor_score, column = moving_average_crossover(prices=prices, slow_window=slow, fast_window=fast,
                                                        average_type='exponential',column_name=Columns.ADJ_CLOSE.value)
        signal_name = column
        prices = prices[[Columns.ADJ_CLOSE.value]]
        prices.columns = [symbol]
        prices.index.names = ['Date']
        all_prices.append(prices)
        # factor_score[signal_name] = factor_score[signal_name].shift()  # lagging the factor score
        factor_score['asset'] = symbol
        factor_score.index.names = ['Date']
        factor_score = factor_score.dropna()
        factor_list.append(factor_score)

    all_prices = pd.concat(all_prices, axis=1)
    factor_score = pd.concat(factor_list)
    first = factor_score[signal_name].first_valid_index()
    factor_score = factor_score.reset_index().set_index(['Date', 'asset'])
    all_prices = all_prices.truncate(first)
    alphalens_df = alphalens.utils.get_clean_factor_and_forward_returns(factor=factor_score[signal_name],
                                                                        prices=all_prices, quantiles=1,periods=[1,3,5])
    alphalens.tears.create_returns_tear_sheet(factor_data=alphalens_df)
