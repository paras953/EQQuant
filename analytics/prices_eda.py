import os

import pandas as pd
import numpy as np
from data.NSEDataAccess import NSEMasterDataAccess
from utils.plotters import line_plot, combine_plot
from datetime import datetime
from typing import Tuple,List
from utils.config import YFINANCE_PRICES_PATH,TICKER_METADATA_PATH
import os


def plot_prices(symbol: str, output_path: str, period: Tuple[datetime, datetime]) -> None:
    """
    :param symbol: stock symbol you want to plot the prices for
    :param output_path: directory where the plots will be dumped
    :param period: start and end date in a tuple
    :return: None
    """
    nse_data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    attrs_df = pd.read_csv(TICKER_METADATA_PATH)
    df = nse_data_access.get_prices(symbol=symbol, start_date=period[0], end_date=period[-1])
    df['year'] = df.index.year
    years = sorted(list(set(df['year'])))
    fig_list = []
    company_name = attrs_df.loc[attrs_df['symbol']==symbol,'companyName'].unique()[0]
    df['price_adjustment_factor'] = df['AdjClose']/df['Close']
    for yr in years:
        yearly_prices = df.loc[df['year'] == yr]
        yearly_prices['returns'] = yearly_prices['AdjClose'].pct_change()
        std_dev = yearly_prices['returns'].std()
        mean = yearly_prices['returns'].mean()
        yearly_prices['return_zcore'] = abs((yearly_prices['returns']-mean)/std_dev)
        fig_1 = line_plot(df=yearly_prices, column_list=['Open', 'Close', 'High', 'Low'],
                          data_type='Prices',title= f'{company_name}_{yr}',figsize=(10,10))
        fig_2 = line_plot(df=yearly_prices, column_list=['Close','AdjClose'],
                          data_type='Prices', title=f'{company_name}_{yr}', figsize=(10, 10))
        fig_3 = line_plot(df=yearly_prices, column_list=['Volume'],
                          data_type='Volume', title=f'{company_name}_{yr}', figsize=(10, 10))
        fig_4 = line_plot(df=yearly_prices, column_list=['price_adjustment_factor'],
                          data_type='price_adjustment_factor', title=f'{company_name}_{yr}', figsize=(10, 10))
        fig_5 = line_plot(df=yearly_prices, column_list=['return_zcore'],
                          data_type='return_zcore', title=f'{company_name}_{yr}', figsize=(10, 10))

        fig_list.append(fig_1)
        fig_list.append(fig_2)
        fig_list.append(fig_3)
        fig_list.append(fig_4)
        fig_list.append(fig_5)

    combine_plot(fig_list=fig_list,output_path=output_path,file_name=f'{company_name}_eda_plots')
    return None

if __name__=='__main__':
    output_path = '../EDA'
    file_list = sorted(os.listdir(f'{YFINANCE_PRICES_PATH}/prices/'))
    file_list = file_list[0:25]
    file_list = ['RELIANCE']
    period = (datetime(2002,1,1),datetime(2024,10,17))
    failed_symbol = []
    for file in file_list:
        symbol = file.split("_")[0]
        try:
            plot_prices(symbol=symbol, output_path=output_path, period=period)
        except Exception as e:
            print(e)
            failed_symbol.append(symbol)

    print(f"Symbols for which the code failed are {failed_symbol}")







