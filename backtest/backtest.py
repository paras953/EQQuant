import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
from typing import Dict

from analytics.risk_return import get_intraday_returns
from data.NSEDataAccess import NSEMasterDataAccess
from utils.config import YFINANCE_PRICES_PATH, Columns
from utils.decorators import timer


@timer
def backtest(weights: pd.DataFrame):
    """
    :param weights: same day portfolio weights  i.e executed at prev
    :return: daily gross (will build net return later) return executed @ prev,next day open and close
    """
    period = (weights.index.min() - relativedelta(years=1), weights.index.max())
    nse_data_access = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
    prices_dict = nse_data_access.get_prices_multiple_assets(
        symbol_list=list(weights.columns), period=period)

    # returns @ prev close
    intraday_returns_dict = get_intraday_returns(open_prices=prices_dict[Columns.OPEN.value],
                                                 close_prices=prices_dict[Columns.CLOSE.value])

    open_to_close_returns = intraday_returns_dict['open_to_close_returns'].copy()
    close_to_close_returns = intraday_returns_dict['close_to_close_returns'].copy()

    portfolio_return_contribution_at_prev_close = weights.shift() * close_to_close_returns
    portfolio_return_at_prev_close = pd.DataFrame(portfolio_return_contribution_at_prev_close.sum(axis=1))
    portfolio_return_at_prev_close.columns = ['portfolio_return']

    ppw_weights_dict = get_post_performance_weights(weights=weights, intraday_returns_dict=intraday_returns_dict)
    next_day_bod_weights = ppw_weights_dict['next_day_bod'].copy()
    next_day_eod_weights = ppw_weights_dict['next_day_eod'].copy()

    # returns at open
    trades_executed_at_open = next_day_bod_weights - next_day_eod_weights.shift()
    portfolio_return_contribution_at_open = next_day_eod_weights.shift() * close_to_close_returns + trades_executed_at_open * open_to_close_returns
    portfolio_return_at_open = pd.DataFrame(portfolio_return_contribution_at_open.sum(axis=1))
    portfolio_return_at_open.columns = ['portfolio_return']

    # returns at next day close
    trades_executed_at_close = next_day_eod_weights.diff()
    portfolio_return_contribution_at_close = next_day_eod_weights.shift() * close_to_close_returns
    portfolio_return_at_close = pd.DataFrame(portfolio_return_contribution_at_close.sum(axis=1))
    portfolio_return_at_close.columns = ['portfolio_return']

    new_period = (weights.index.min(), weights.index.max())

    portfolio_return_contribution_at_prev_close = portfolio_return_contribution_at_prev_close.truncate(new_period[0],
                                                                                                       new_period[-1])
    portfolio_return_at_prev_close = portfolio_return_at_prev_close.truncate(new_period[0], new_period[-1])

    portfolio_return_contribution_at_open = portfolio_return_contribution_at_open.truncate(new_period[0],
                                                                                           new_period[-1])
    portfolio_return_at_open = portfolio_return_at_open.truncate(new_period[0], new_period[-1])

    portfolio_return_contribution_at_close = portfolio_return_contribution_at_close.truncate(new_period[0],
                                                                                             new_period[-1])
    portfolio_return_at_close = portfolio_return_at_close.truncate(new_period[0], new_period[-1])

    return {'gross': {
        'prev': {'portfolio_return_contribution': portfolio_return_contribution_at_prev_close,
                 'portfolio_return': portfolio_return_at_prev_close},
        'open': {'portfolio_return_contribution': portfolio_return_contribution_at_open,
                 'portfolio_return': portfolio_return_at_open},
        'close': {'portfolio_return_contribution': portfolio_return_contribution_at_close,
                  'portfolio_return': portfolio_return_at_close},
    }
    }


@timer
def get_post_performance_weights(weights: pd.DataFrame, intraday_returns_dict: Dict[str, pd.DataFrame]) -> Dict[
    str, pd.DataFrame]:
    """
    :param weights: same day EOD weights which will be executed in prev
    :param intraday_returns_dict: has OpenToClose and CloseToOpen returns for each asset
    :return: ppw weights dataframe
    """
    prev_close_to_open_returns = intraday_returns_dict['prev_day_close_to_open_returns'].copy()
    open_to_close_returns = intraday_returns_dict['open_to_close_returns'].copy()

    portfolio_return_contribution_close_to_open = weights.shift() * prev_close_to_open_returns
    portfolio_returns_close_to_open = pd.DataFrame(portfolio_return_contribution_close_to_open.sum(axis=1))
    portfolio_returns_close_to_open.columns = ['portfolio_return']
    next_day_bod_weights = weights.shift() * (1 + portfolio_return_contribution_close_to_open)
    next_day_bod_weights = next_day_bod_weights.div(1 + portfolio_returns_close_to_open['portfolio_return'], axis=0)

    portfolio_return_contribution_open_to_close = next_day_bod_weights * open_to_close_returns
    portfolio_returns_open_to_close = pd.DataFrame(portfolio_return_contribution_open_to_close.sum(axis=1))
    portfolio_returns_open_to_close.columns = ['portfolio_return']
    next_day_eod_weights = next_day_bod_weights * (1 + portfolio_return_contribution_open_to_close)
    next_day_eod_weights = next_day_eod_weights.div(1 + portfolio_returns_open_to_close['portfolio_return'], axis=0)

    return {'next_day_bod': next_day_bod_weights, 'next_day_eod': next_day_eod_weights}


if __name__ == '__main__':
    period = (datetime(2015, 1, 1), datetime(2023, 12, 31))
    np.random.seed(42)
    nse_data = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
    symbol_list = nse_data.get_index_constituents('NIFTY 50')[0:2]
    prices_dict = nse_data.get_prices_multiple_assets(symbol_list=symbol_list, period=period)
    df = pd.DataFrame(np.random.random_sample(size=(len(prices_dict['Open'].index), len(symbol_list))),
                      index=prices_dict['Open'].index, columns=symbol_list)
    df = df.div(df.sum(axis=1), axis=0)
    ret_dict = backtest(weights=df)
