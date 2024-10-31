import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from typing import Dict

from datetime import datetime
from dateutil.relativedelta import relativedelta
from numpy.ma.extras import average

from analytics.risk_return import get_returns, get_volatility
from utils.config import YFINANCE_PRICES_PATH, Columns, ALL_RESULTS_PATH
from utils.plotters import multi_bar_plot, combine_plot
from data.NSEDataAccess import NSEMasterDataAccess
from signals.momentum.momentum_signals import moving_average_crossover,timeseries_momentum,ADX,RSI



def run_linear_regression(df: pd.DataFrame, x_column: str, y_column: str) -> Dict:
    """
    :param df: pandas df having the x and y columns
    :param x_column: x column name
    :param y_column: y column name
    :return:
    """
    x = df[x_column]
    y = df[y_column]
    x = sm.add_constant(x)
    nw_lag = int(np.floor(4.0 * (len(df) / 100) ** (2 / 9)))
    model = sm.OLS(y, x)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': nw_lag}, use_t=True)
    print(results.summary())
    coeff_dict = dict(results.params)
    tstats_dict = dict(results.tvalues)
    rsq = results.rsquared
    rsq_adj = results.rsquared_adj
    output_dict = {'coeff': coeff_dict, 'tstats': tstats_dict, 'rsq': rsq, 'rsq_adj': rsq_adj}
    return output_dict



def plot_linear_regression_stats(data_dict: Dict[str, pd.DataFrame], x_column: str, y_column: str,
                                 output_path: str, file_name: str, **kwargs) -> pd.DataFrame:

    """
    :param data_dict: dict having key as stock symbol and
    :param x_column: independent variable of the regression
    :param y_column: dependant column of the regression
    :param output_path: directory where plots will be dumped
    :return: None
    """

    tstat_list = []
    period_list = []
    symbol_list = []
    rsq_list = []
    rsq_adj_list = []
    coeff_list = []

    for symbol, df in data_dict.items():
        for period in ['full_history', '5Y', '3Y']:
            end_date = df.index.max()
            if period == '5Y':
                start_date = end_date - relativedelta(years=5)
            elif period == '3Y':
                start_date = end_date - relativedelta(years=3)
            else:
                start_date = df.index.min()
            regression_df = df.truncate(start_date, end_date)
            regression_stats = run_linear_regression(df=regression_df, x_column=x_column, y_column=y_column)
            tstat_list.append(regression_stats['tstats'][x_column])
            period_list.append(period)
            symbol_list.append(symbol)
            rsq_list.append(regression_stats['rsq'])
            rsq_adj_list.append(regression_stats['rsq_adj'])
            coeff_list.append(regression_stats['coeff'][x_column])

    df = pd.DataFrame({'symbol': symbol_list, 'period': period_list, 'tstat': tstat_list, 'coeff': coeff_list,
                       'rsq_adj': rsq_adj_list, 'rsq': rsq_list})
    if kwargs.get('plot', True):

        pivot_df = pd.pivot_table(data=df, values='tstat', columns='period', index='symbol').sort_index()
        symbol_list = sorted(data_dict.keys())
        fig_list = []
        for i in range(0, len(data_dict), 10):
            list_chunk = symbol_list[i:i + 10]
            subset_df = pivot_df[pivot_df.index.isin(list_chunk)].sort_index().reset_index()
            fig = multi_bar_plot(df=subset_df, x_column='symbol', y_column_list=['full_history','5Y','3Y'], data_type='tstat',
                                 title=f'{x_column} ~ {y_column}')
            fig_list.append(fig)
        combine_plot(fig_list=fig_list, output_path=output_path, file_name=file_name)

    return df


if __name__ == '__main__':

    nse_data = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    symbol_list = sorted(nse_data.get_index_constituents(index_name='NIFTY 50'))
    signal_name = ''
    output_path = f'{ALL_RESULTS_PATH}/regression_results/'
    data_dict = {}
    slow = 32
    fast = 8
    adx_lookback = 14
    vol_adjusted = True
    use_adx = True
    low_adx_cut_off = 0
    high_adx_cut_off = 25
    adx_stats = []
    tsmom_lookback = 10
    pooled = True
    for symbol in symbol_list:
        prices = nse_data.get_prices(symbol=symbol, start_date=datetime(2002, 1, 1), end_date=datetime(2024, 12, 31))
        returns = get_returns(prices=prices)
        vol = get_volatility(prices=prices)
        data = pd.concat([returns, vol], axis=1)
        data['vol_adjusted_returns'] = data['returns'] / data['volatility']
        signal_df, signal_name = moving_average_crossover(prices=prices, slow_window=slow, fast_window=fast,
                                             column_name=Columns.ADJ_CLOSE.value)

        # signal_df, signal_name = timeseries_momentum(prices=prices,column_name=Columns.ADJ_CLOSE.value,lookback_window=tsmom_lookback)
        # signal_df ,signal_name = ADX(prices=prices,average_type='simple',average_window=adx_lookback)
        # signal_df, signal_name = RSI(prices=prices, average_type='exponential', lookback_window=adx_lookback,calculate_rsi_on='returns')

        data = pd.concat([data, signal_df], axis=1, join='inner')
        if vol_adjusted:
            data[f'vol_adjusted_{signal_name}'] = data[signal_name] / data['volatility']
            data[f'vol_adjusted_{signal_name}'] = data[f'vol_adjusted_{signal_name}'].shift()
            signal_name = f'vol_adjusted_{signal_name}'
        else:
            data[signal_name] = data[signal_name].shift()

        if use_adx:
            adx_df,adx = ADX(prices=prices,average_window=adx_lookback,average_type='exponential')
            adx_df[adx] = adx_df[adx].shift() # use previous days adx for predictions
            data = pd.concat([data,adx_df],axis=1,join='inner')
            data = data.dropna()
            adx_mask = ((data[adx]>=low_adx_cut_off) & (data[adx]<high_adx_cut_off))
            data = data.loc[adx_mask]
            adx_stats.append(adx_mask.mean())

        data = data.dropna(subset=signal_name)
        data_dict[symbol] = data

    if pooled:
        print('Pooling all stocks data')
        all_pooled =  pd.concat(list(data_dict.values())).dropna()
        all_pooled = all_pooled.sort_index()
        data_dict = {'pooled':all_pooled}

    plot_linear_regression_stats(data_dict=data_dict, x_column=signal_name, y_column='vol_adjusted_returns',
                                 output_path=output_path, file_name=f'{signal_name}_results_nifty_50_adx_{use_adx}_{low_adx_cut_off}_{high_adx_cut_off}_pooled_{pooled}')
    print(f'Mean ADX for given scenario {np.array(adx_stats).mean()}')

