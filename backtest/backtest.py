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
def backtest(weights: pd.DataFrame, rebalance_frequency: str = 'M'):
    """
    :param weights: same day portfolio weights  i.e executed at prev
    :param rebalance_frequency: one of D/W/BW/M/Q
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
    if rebalance_frequency == 'D':
        ppw_weights_dict = get_post_performance_weights_daily_rebalance(weights=weights,
                                                                        intraday_returns_dict=intraday_returns_dict)
    else:
        ppw_weights_dict = get_post_performance_weights_by_rebalance_frequency(weights=weights,
                                                                               intraday_returns_dict=intraday_returns_dict,
                                                                               rebalance_frequency=rebalance_frequency
                                                                               )

    same_day_eod_weights = ppw_weights_dict['same_day_eod'].copy()
    next_day_bod_weights = ppw_weights_dict['next_day_bod'].copy()
    next_day_eod_weights = ppw_weights_dict['next_day_eod'].copy()

    portfolio_return_contribution_at_prev_close = same_day_eod_weights.shift() * close_to_close_returns
    portfolio_return_at_prev_close = pd.DataFrame(portfolio_return_contribution_at_prev_close.sum(axis=1))
    portfolio_return_at_prev_close.columns = ['portfolio_return']

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
def get_post_performance_weights_daily_rebalance(weights: pd.DataFrame,
                                                 intraday_returns_dict: Dict[str, pd.DataFrame]) -> Dict[
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

    return {'next_day_bod': next_day_bod_weights, 'next_day_eod': next_day_eod_weights,
            'same_day_eod': weights,
            'trades_at_next_day_open': next_day_bod_weights.diff(),
            'trades_at_next_day_close': next_day_eod_weights.diff(),
            'trades_at_same_day_close': weights.diff()
            }


@timer
def get_post_performance_weights_by_rebalance_frequency(weights: pd.DataFrame,
                                                        intraday_returns_dict: Dict[str, pd.DataFrame],
                                                        rebalance_frequency: str = 'M') -> Dict[str, pd.DataFrame]:
    """
    :param weights: daily same day weights output of portfolio_manager
    :param intraday_returns_dict: has OpenToClose and CloseToOpen returns for each asset
    :param rebalance_frequency: one of Weekly (W), Bi Weekly (BW), Monthly (M), Quarterly (Q)
    :return: the dict having the daily post performance weights at beginning of day and end of day
    """
    universe = list(weights.columns)
    if rebalance_frequency not in ['W', 'BW', 'M', 'Q']:
        raise ValueError(f'rebalance_frequency should be one of W/BW/M/Q you passed {rebalance_frequency}')

    start_date = weights.index.min()
    weights['days_since_start'] = (weights.index - start_date).days
    weights['day_of_week'] = (weights.index.dayofweek % 7) + 1  # 1 is Monday etc.
    weights['day'] = weights.index.day
    weights['week'] = weights.index.isocalendar().week
    weights['month'] = weights.index.month
    weights['quarter'] = weights.index.quarter
    weights['year'] = weights.index.year

    temp = weights.groupby(['year', 'week'])['day_of_week'].min().to_frame().rename(
        columns={'day_of_week': 'first_day_of_week'})
    weights = weights.join(temp[['first_day_of_week']], on=['year', 'week'])
    weights['first_day_of_week'] = (weights['day_of_week'] == weights['first_day_of_week']).astype(int)

    weights['first_day_of_fortnight'] = (weights['days_since_start'] % 14 == 0).astype(int)
    temp = weights.groupby(['year', 'month'])['day'].min().to_frame().rename(
        columns={'day': 'first_day_of_month'})
    weights = weights.join(temp[['first_day_of_month']], on=['year', 'month'])
    weights['first_day_of_month'] = (weights['first_day_of_month'] == weights['day']).astype(int)
    temp = weights.groupby(['year', 'quarter'])['days_since_start'].min().to_frame().rename(
        columns={'days_since_start': 'first_day_of_quarter'})
    weights = weights.join(temp[['first_day_of_quarter']], on=['year', 'quarter'])
    weights['first_day_of_quarter'] = (weights['first_day_of_quarter'] == weights['days_since_start']).astype(int)

    rebalance_day_map = {'W': 'first_day_of_week', 'BW': 'first_day_of_fortnight',
                         'M': 'first_day_of_month', 'Q': 'first_day_of_quarter'}

    rebalance_dates = weights.loc[weights[rebalance_day_map[rebalance_frequency]] == 1].index
    next_day_bod_weights = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    next_day_eod_weights = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    same_day_eod_weights = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    trades_at_next_day_open = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    trades_at_next_day_close = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    trades_at_same_day_close = pd.DataFrame(np.nan, index=weights.index, columns=universe)
    prev_date = None

    prev_close_to_open_returns = intraday_returns_dict['prev_day_close_to_open_returns'].copy()
    open_to_close_returns = intraday_returns_dict['open_to_close_returns'].copy()
    close_to_close_returns = intraday_returns_dict['close_to_close_returns'].copy()

    for idx, dates in enumerate(weights.index):

        if prev_date is None:
            prev_date = dates
            continue
        if (dates in rebalance_dates) or (idx == 1 and prev_date in rebalance_dates):
            # for the first day in the weights df make an exception
            prev_day_weight_vector = weights.loc[prev_date, universe]
            same_day_eod_weights.loc[prev_date, universe] = prev_day_weight_vector
            prev_day_weight_vector_executed_at_same_day_close = prev_day_weight_vector
        else:
            prev_day_weight_vector = next_day_eod_weights.loc[prev_date, universe]
            prev_day_weight_vector_executed_at_same_day_close = same_day_eod_weights.loc[prev_date, universe]

        prev_close_to_open_returns_vector = prev_close_to_open_returns.loc[dates]
        open_to_close_returns_vector = open_to_close_returns.loc[dates]
        close_to_close_returns_vector = close_to_close_returns.loc[dates]

        portfolio_return_contribution_prev_close_to_open = prev_day_weight_vector * prev_close_to_open_returns_vector
        portfolio_returns_prev_close_to_open = portfolio_return_contribution_prev_close_to_open.sum()
        next_day_bod_weight_vector = prev_day_weight_vector * (1 + portfolio_return_contribution_prev_close_to_open) / (
                1 + portfolio_returns_prev_close_to_open)

        portfolio_return_contribution_open_to_close = next_day_bod_weight_vector * open_to_close_returns_vector
        portfolio_returns_open_to_close = portfolio_return_contribution_open_to_close.sum()
        next_day_eod_weight_vector = next_day_bod_weight_vector * (1 + portfolio_return_contribution_open_to_close) / (
                1 + portfolio_returns_open_to_close)

        portfolio_return_contribution_prev_close_to_close = prev_day_weight_vector_executed_at_same_day_close * close_to_close_returns_vector
        portfolio_returns_prev_close_to_close = portfolio_return_contribution_prev_close_to_close.sum()
        same_day_eod_weight_vector = prev_day_weight_vector_executed_at_same_day_close * (
                1 + portfolio_return_contribution_prev_close_to_close) / (1 + portfolio_returns_prev_close_to_close)

        same_day_eod_weights.loc[dates, universe] = same_day_eod_weight_vector
        next_day_bod_weights.loc[dates, universe] = next_day_bod_weight_vector
        next_day_eod_weights.loc[dates, universe] = next_day_eod_weight_vector

        trades_at_next_day_open.loc[dates, universe] = next_day_bod_weight_vector - prev_day_weight_vector
        trades_at_next_day_close.loc[dates, universe] = next_day_eod_weight_vector - prev_day_weight_vector
        trades_at_same_day_close.loc[
            prev_date, universe] = weights.loc[prev_date, universe] - same_day_eod_weight_vector #will be only true on rebalance dates

        prev_date = dates

    rebalance_flag = weights[rebalance_day_map[rebalance_frequency]]
    trades_at_next_day_close = trades_at_next_day_close.mul(rebalance_flag, axis=0)
    trades_at_next_day_open = trades_at_next_day_open.mul(rebalance_flag, axis=0)
    trades_at_same_day_close = trades_at_same_day_close.mul(rebalance_flag.shift(-1),
                                                            axis=0)  # day before is rebal day

    return {'next_day_bod': next_day_bod_weights, 'next_day_eod': next_day_eod_weights,
            'same_day_eod': same_day_eod_weights,
            'trades_at_next_day_open': trades_at_next_day_open, 'trades_at_next_day_close': trades_at_next_day_close,
            'trades_at_same_day_close': trades_at_same_day_close}


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
