import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from functools import partial

from quantstats.stats import information_ratio
from scipy.stats import skew, kurtosis

from pypfopt.objective_functions import portfolio_return

from backtest.backtest import backtest
from data.NSEDataAccess import NSEMasterDataAccess

from collections import defaultdict
from portfolio_manager.portfolio_manager import PortfolioManager
import quantstats as qs
from utils.config import YFINANCE_PRICES_PATH
from utils.pandas_utils import df_to_excel
from utils.plotters import line_plot, combine_plot, multi_bar_plot, plot_histogram
from utils.portfolio_tearsheet_metrics import get_sharpe_ratio, get_sharpe_tstat, get_portfolio_volatility, \
    get_time_in_market, get_hit_ratio, get_cagr, get_sortino_ratio, get_drawdown, get_omega_ratio, get_var_cvar, \
    get_AM_GM, get_latest_return, get_returns_by_year_month, get_return_index, get_information_ratio

# extend pandas functionality with metrics, etc.
qs.extend_pandas()


def generate_performance_tearsheet(portfolio_returns_dict: Dict, output_path: str, file_name: str,
                                   benchmark_strategy: str):
    """
    :param portfolio_returns_dict: key is the strat name and value is the
                                   returns dict returned by the function
                                   backtest()
    :param output_path: directory where the excel will be dumped
    :param file_name : name of the excel sheet
    :param benchmark_strategy : name of strategy to be used as benchmark (should be one of the keys in portfolio_returns_dict)
    :return: An excel dumped in the output dir
    """
    output_dict = defaultdict(list)
    plotting_dict = defaultdict(list)

    func_dict = {
        'Sharpe (Ann.)': get_sharpe_ratio,
        'Sharpe Tstat': get_sharpe_tstat,
        'Sharpe 5Y (Ann.)': partial(get_sharpe_ratio, years_lookback=5),
        'Sharpe Tstat 5Y': partial(get_sharpe_tstat, years_lookback=5),
        'Volatility % (Ann.)': get_portfolio_volatility,
        '%Active': get_time_in_market,
        '%Hit ratio': get_hit_ratio,
        'CAGR': get_cagr,
        'Sortino Ratio (Ann.)': get_sortino_ratio,
        'Max Drawdown': partial(get_drawdown, max=True),
        'Profit Factor': get_omega_ratio,

    }

    for strat, strat_dict in portfolio_returns_dict.items():
        for tc, tc_dict in strat_dict.items():
            for exec, return_dict in tc_dict.items():
                return_df = return_dict['portfolio_return'].copy()
                benchmark_return_df = portfolio_returns_dict[benchmark_strategy][tc][exec]['portfolio_return'].copy()

                pop_stats = return_df['portfolio_return'].aggregate(func_dict)
                var_dict = get_var_cvar(portfolio_return=return_df['portfolio_return'].copy())
                am_gm_dict = get_AM_GM(portfolio_return=return_df['portfolio_return'].copy())
                return_over_diff_periods_dict = get_latest_return(portfolio_return=return_df['portfolio_return'].copy())
                var_dict.update(am_gm_dict)
                var_dict.update(return_over_diff_periods_dict)

                information_ratio = get_information_ratio(portfolio_returns=return_df['portfolio_return'].copy(),
                                                          benchmark_returns=benchmark_return_df[
                                                              'portfolio_return'].copy())
                var_dict.update({'Information Ratio': information_ratio,
                                 'Skew': return_df['portfolio_return'].skew(),
                                 'Kurtosis': return_df['portfolio_return'].kurt()
                                 })

                for keys, values in var_dict.items():
                    pop_stats[keys] = values

                pop_stats = pd.DataFrame(pop_stats)
                pop_stats.columns = [strat]
                output_dict[f'{exec}_{tc}'].append(pop_stats)

                calendar_month_year_returns = get_returns_by_year_month(
                    portfolio_return=return_df['portfolio_return'].copy())

                calendar_month_year_returns.columns = pd.MultiIndex.from_product(
                    [[strat], calendar_month_year_returns.columns])
                calendar_month_year_returns.columns.names = ['Strategy', 'Month']
                output_dict[f'{exec}_{tc}_calendar_returns'].append(calendar_month_year_returns)

                # data for plotting
                if exec == 'close':
                    return_index = get_return_index(portfolio_returns=return_df['portfolio_return'].copy())
                    return_index = pd.DataFrame(return_index)
                    return_index.columns = [strat]
                    plotting_dict[f'Return Index at {exec} {tc}'].append(return_index)

                    drawdown = get_drawdown(portfolio_returns=return_df['portfolio_return'].copy(), max=False)
                    drawdown = pd.DataFrame(drawdown)
                    drawdown.columns = [strat]
                    plotting_dict[f'Drawdown at {exec} {tc}'].append(drawdown)

                    yearly_return = calendar_month_year_returns[[(strat, 'Total Return')]]
                    yearly_return.columns = [strat]
                    yearly_return = yearly_return.sort_index()
                    plotting_dict[f'Yearly Return at {exec} {tc}'].append(yearly_return)

                    plotting_dict[f'Histogram at {exec} {tc} {strat}'].append(return_df[['portfolio_return']])

    pop_stats_excel_dict = {}
    plot_data = {}
    for keys, pop_stats_df_list in output_dict.items():
        pop_stats_excel_dict[keys] = pd.concat(pop_stats_df_list, axis=1).round(2)

    for keys, plot_df_list in plotting_dict.items():
        plot_data[keys] = pd.concat(plot_df_list, axis=1)

    fig_list = []
    for keys, plot_df in plot_data.items():
        if keys.startswith('Yearly'):
            plot_df = plot_df.reset_index()
            columns = set(list(plot_df.columns)) - {'year'}
            fig = multi_bar_plot(df=plot_df, x_column='year', y_column_list=list(columns), data_type='%Returns',
                                 title=keys)
        elif keys.startswith('Histogram'):
            fig = plot_histogram(df=plot_df, column_name='portfolio_return', title=keys)
        else:
            fig = line_plot(df=plot_df, column_list=list(plot_df.columns),
                            title=keys, data_type=keys, figsize=(10, 10))
        fig_list.append(fig)

    df_to_excel(df_dict=pop_stats_excel_dict, output_path=output_path, file_name=file_name)
    combine_plot(fig_list=fig_list, output_path=output_path, file_name=file_name)


if __name__ == '__main__':
    period = (datetime(2012, 1, 1), datetime(2023, 12, 31))
    nse_data = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
    universe_1 = nse_data.get_index_constituents('NIFTY 50')
    universe_benchmark = ['NSEI']  # NIFTY
    output_path = r'C:/Users/paras/PycharmProjects/EQQuant/res/performance/factor_performance_v2/'
    file_name = 'CCI'
    execution = 'executed_at_close'

    CCI_1 = {
        'universe': universe_1,
        'signals': {
            'TALIB_CCI_1': {'signal_function': 'TALIB_CCI', 'lookback': 22, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': False},
        'label': "TALIB_CCI_1"
    }

    CCI_2 = {
        'universe': universe_1,
        'signals': {
            'TALIB_CCI_2': {'signal_function': 'TALIB_CCI', 'lookback': 66, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': False},
        'label': "TALIB_CCI_2"
    }

    CCI_3 = {
        'universe': universe_1,
        'signals': {
            'TALIB_CCI_3': {'signal_function': 'TALIB_CCI', 'lookback': 128, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': False},
        'label': "TALIB_CCI_3"
    }

    AROONOSC_1 = {
        'universe': universe_1,
        'signals': {
            'TALIB_AROONOSC_1': {'signal_function': 'TALIB_AROONOSC', 'lookback': 22, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': False},
        'label': "TALIB_AROONOSC_1"
    }

    BOP_1 = {
        'universe': universe_1,
        'signals': {
            'TALIB_BOP_1': {'signal_function': 'TALIB_BOP', 'lookback': 22, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': True},
        'label': "TALIB_BOP_1"
    }

    NIFTY_BAH = {
        'universe': universe_benchmark,
        'signals': {
            'HOLD_NIFTY': {'signal_function': 'HOLD', 'position': 1, 'budget': 1},
        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.01, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01, 'optimize': False},
        'label': "HOLD_NIFTY"
    }

    config_list = [BOP_1]
    returns_dict = {}
    for config in config_list:
        pm = PortfolioManager(config=config)
        weights_dict = pm.run_portfolio()
        returns_dict[config['label']] = backtest(weights=weights_dict['pre_optimization_weight'],
                                                 rebalance_frequency='M')

    generate_performance_tearsheet(portfolio_returns_dict=returns_dict,
                                   benchmark_strategy="HOLD_NIFTY",
                                   output_path=output_path,
                                   file_name=file_name)
    prev = returns_dict['TALIB_CCI_1']['gross']['prev']['portfolio_return'].rename(
        columns={'portfolio_return': 'executed_at_prev'})
    open = returns_dict['TALIB_CCI_1']['gross']['open']['portfolio_return'].rename(
        columns={'portfolio_return': 'executed_at_open'})
    close = returns_dict['TALIB_CCI_1']['gross']['close']['portfolio_return'].rename(
        columns={'portfolio_return': 'executed_at_close'})

    benchmark = returns_dict['HOLD_NIFTY']['gross']['close']['portfolio_return'].rename(
        columns={'portfolio_return': 'executed_at_close_benchmark'})
    #
    output_path = f'C:/Users/paras/PycharmProjects/EQQuant/res/performance/factor_performance/CCI1_{execution}_performance_v2.html'
    report = qs.reports.html(returns=close['executed_at_close'], benchmark=benchmark['executed_at_close_benchmark'],
                             output=output_path)
