import pandas as pd
import os
import numpy as np
import calendar
from typing import Dict, List

from dateutil.relativedelta import relativedelta


def get_sharpe_ratio(portfolio_returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 256,
                     annualized: bool = True, years_lookback: int = None):
    """
    :param portfolio_returns: a series of portfolio returns
    :param risk_free_rate : risk free rate
    :param annualized : return the annualized version or not
    :param : years_lookback: returns the sharpe ratio of the last x years
    :return: the sharpe ratio of the portfolio returns (annualized)
    """
    if years_lookback:
        end_date = portfolio_returns.index.max()
        start_date = end_date - relativedelta(years=years_lookback)
        portfolio_returns = portfolio_returns.truncate(start_date, end_date)

    excess_returns = (portfolio_returns - risk_free_rate)
    avg_return = excess_returns.mean()
    std_dev = excess_returns.std()
    if annualized:
        return (avg_return / std_dev) * np.sqrt(periods_per_year)

    return avg_return / std_dev


def get_sharpe_tstat(portfolio_returns: pd.Series, risk_free_rate: float = 0.0, years_lookback: int = None):
    """
    src https://quant.stackexchange.com/questions/54921/how-to-test-signifcance-of-a-sharpe-ratio
    :param portfolio_returns: series of portfolio returns
    :param risk_free_rate: risk free rate
    :param : years_lookback: returns the sharpe ratio of the last x years
    :return: the sharpe tstat
    """
    sharpe_ratio = get_sharpe_ratio(portfolio_returns=portfolio_returns, risk_free_rate=risk_free_rate,
                                    annualized=False,years_lookback=years_lookback)
    std_error = np.sqrt((1 + (sharpe_ratio ** 2)*0.5) / len(portfolio_returns))
    sharpe_tstat = sharpe_ratio / std_error
    return sharpe_tstat


def get_portfolio_volatility(portfolio_returns: pd.Series, periods_per_year: int = 256):
    """
    :param portfolio_returns: a series of portfolio returns
    :param periods_per_year:  Number of days to be considered in a year
    :return: the annualized portfolio volatility
    """
    std_dev = portfolio_returns.std() * np.sqrt(periods_per_year)
    return std_dev*100


def get_time_in_market(portfolio_returns: pd.Series):
    """
    :param portfolio_returns: a series of portfolio returns
    :return: % of days where the strategy was active
    """
    time_in_market = (portfolio_returns != 0).sum() / len(portfolio_returns)
    return time_in_market*100


def get_hit_ratio(portfolio_returns: pd.Series):
    """
    :param portfolio_returns: a series of portfolio returns
    :return: out of the days where the strategy was active how many days the strategy was correct
    """
    active_ratio = get_time_in_market(portfolio_returns=portfolio_returns)
    portfolio_success_rate = ((portfolio_returns > 0).sum() / len(portfolio_returns))*100
    hit_ratio = portfolio_success_rate / active_ratio
    return hit_ratio*100


def get_return_index(portfolio_returns: pd.Series):
    """
    :param portfolio_returns: a series of portfolio returns
    :return: a series that has the daily return index
    """
    return_index_series = (1 + portfolio_returns).cumprod()
    return return_index_series


def get_cagr(portfolio_returns: pd.Series, periods_per_year: int = 252):
    """
    :param portfolio_returns: a series of portfolio returns
    :param periods_per_year : No of days in a year
    :return: CAGR of the portfolio in %
    """
    no_of_years = len(portfolio_returns) / periods_per_year
    cagr = ((1 + portfolio_returns).prod() ** (1 / no_of_years)) - 1
    return cagr * 100


def get_sortino_ratio(portfolio_returns: pd.Series, periods_per_year: int = 256, risk_free_rate: float = 0.0):
    """

    :param portfolio_returns:  a series of portfolio returns
    :param periods_per_year: No of days in a year
    :param risk_free_rate: risk free rate
    :return:
    """
    mean = (portfolio_returns.mean() - risk_free_rate) * periods_per_year
    std_neg = portfolio_returns[portfolio_returns < risk_free_rate].std() * np.sqrt(periods_per_year)
    return mean / std_neg


def get_omega_ratio(portfolio_returns: pd.Series, threshold: float = 0.0):
    """
    :param portfolio_returns: a series of portfolio returns
    :param threshold: a daily threshold return metrics to compare
    :return:
    """
    excess_returns = portfolio_returns - threshold
    omega_ratio = excess_returns[excess_returns > 0].sum() / (-1 * excess_returns[excess_returns < 0].sum())
    return omega_ratio


def get_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series):
    """
    :param portfolio_returns: portfolio returns
    :param benchmark_returns: benchmark returns
    :return: information ratio value
    """
    benchmark_returns.name = 'benchmark_return'
    returns_df = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
    information_ratio = (returns_df['portfolio_return'].mean() - returns_df['benchmark_return'].mean()) / (
        (returns_df['portfolio_return'] - returns_df['benchmark_return']).std())
    return information_ratio


def get_drawdown(portfolio_returns: pd.Series, max: bool = True):
    """
    :param portfolio_returns:
    :return:
    """
    cumulative_return = (1 + portfolio_returns).cumprod()
    peak_value = cumulative_return.cummax()
    drawdown = -1 + (cumulative_return / peak_value)
    if max:
        return drawdown.min() * 100

    return drawdown * 100


def get_latest_return(portfolio_return: pd.Series):
    """
    :param portfolio_return: portfolio return series
    :return: returns 1M,3M, 1Y,3Y,YTD,MTD,QTD returns
    """
    max_date = portfolio_return.index.max()
    log_returns = np.log(1 + portfolio_return)
    last_1M_returns = np.exp(log_returns.loc[max_date - pd.DateOffset(months=1):max_date].sum()) - 1
    last_3M_returns = np.exp(log_returns.loc[max_date - pd.DateOffset(months=3):max_date].sum()) - 1
    last_6M_returns = np.exp(log_returns.loc[max_date - pd.DateOffset(months=6):max_date].sum()) - 1
    last_1Y_returns = np.exp(log_returns.loc[max_date - pd.DateOffset(years=1):max_date].sum()) - 1
    last_3Y_returns = np.exp(log_returns.loc[max_date - pd.DateOffset(years=3):max_date].sum()) - 1

    mtd_returns = np.exp(log_returns.loc[max_date.replace(day=1):max_date].sum()) - 1
    ytd_returns = np.exp(log_returns.loc[max_date.replace(day=1, month=1):max_date].sum()) - 1
    qtd_returns = np.exp(
        log_returns.loc[max_date - pd.tseries.offsets.QuarterBegin(startingMonth=1):max_date].sum()) - 1

    return {'1M': last_1M_returns * 100,
            '3M': last_3M_returns * 100,
            '6M': last_6M_returns * 100,
            '1Y': last_1Y_returns * 100,
            '3Y': last_3Y_returns * 100,
            'MTD': mtd_returns * 100,
            'QTD': qtd_returns,
            'YTD': ytd_returns * 100
            }


def get_AM_GM(portfolio_return: pd.Series):
    """
    :param portfolio_return: portfolio return series
    :return: returns AM and GM of the returns
    """
    am = portfolio_return.mean()
    gm = ((1 + portfolio_return).prod() ** (1 / len(portfolio_return))) - 1
    return {"AM": am * 100, "GM": gm * 100}


def get_var_cvar(portfolio_return: pd.Series, level: float = 0.05):
    """
    :param portfolio_return: a series of portfolio return
    :param level : to calculate the Var,CVaR
    :return: returns VaR, CVaR
    """
    var = np.quantile(portfolio_return, level)
    cvar = portfolio_return.loc[portfolio_return < var].mean()
    return {'var': var * 100, 'cvar': cvar * 100}


def get_returns_by_year_month(portfolio_return: pd.Series):
    """
    :param portfolio_return: portfolio return series
    :return: returns grouped by year and month
    """
    returns_df = pd.DataFrame(portfolio_return)
    returns_df['month'] = returns_df.index.month
    returns_df['year'] = returns_df.index.year
    returns_df['log_return'] = np.log(1 + returns_df['portfolio_return'])
    grouped_df = returns_df.groupby(['year', 'month']).agg({'log_return': 'sum'}).reset_index()
    grouped_df = grouped_df.pivot_table(values='log_return', index='year', columns='month')
    grouped_df['Total Return'] = grouped_df.sum(axis=1)
    grouped_df = (np.exp(grouped_df) - 1) * 100
    month_name_dict = {i: name for i, name in enumerate(calendar.month_name) if name}
    grouped_df = grouped_df.rename(columns=month_name_dict)
    grouped_df = grouped_df[list(month_name_dict.values()) + ['Total Return']]
    grouped_df = grouped_df.sort_index(ascending=False)
    return grouped_df

if __name__ == "__main__":
    print('Hello')