import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
from typing import Dict

from dateutil.relativedelta import relativedelta
from utils.plotters import multi_bar_plot, combine_plot


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


def plot_linear_regression_stats(data_dict: Dict[pd.DataFrame], x_column: str, y_column: str,
                                 output_path: str, **kwargs) -> pd.DataFrame:
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
        pivot_df = pd.pivot_table(data=df, values='tstat', columns='period', index='symbol').reset_index()
        fig = multi_bar_plot(df=pivot_df, x_column='symbol', y_column_list=list(pivot_df.columns), data_type='tstat',
                             title=f'{x_column} ~ {y_column}')
        combine_plot(fig_list=[fig], output_path=output_path, file_name='tstat_across_time')
    return df


if __name__ == '__main__':
    pass
