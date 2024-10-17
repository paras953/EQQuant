import os

import pandas as pd
import numpy as np
from data.NSEDataAccess import NSEMasterDataAccess
from utils.plotters import line_plot, combine_plot
from datetime import datetime
from typing import Tuple
from utils.config import PRICES_PKL_PATH,TICKER_METADATA_PATH,DIVIDEND_PATH_ALL,BONUS_PATH_ALL,SPLIT_PATH_ALL
from typing import List

def plot_prices(symbol: str, output_path: str, period: Tuple[datetime, datetime]) -> None:
    """

    :param symbol: stock symbol you want to plot the prices for
    :param output_path: directory where the plots will be dumped
    :param period: start and end date in a tuple
    :return: None
    """
    nse_data_access = NSEMasterDataAccess(output_path=PRICES_PKL_PATH)
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
        fig_1 = line_plot(df=yearly_prices, column_list=['Open', 'Close', 'High', 'Low', 'VWAP', 'LastTradedPrice'],
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


def filter_corporate_actions_data(symbol_list:List[str], action_type:str)->pd.DataFrame:
    """
    :param symbol: list of stock symbols
    :param action_type: one of dividend, bonus, split
    :return: returns the historical corporate action of th stock symbol
    """
    if action_type not in ['dividend','bonus','split']:
        raise ValueError(f'invalid action type, it should be one of [dividend,bonus,split] you passed {action_type}')

    path_dict = {'bonus':BONUS_PATH_ALL,
                 'split':SPLIT_PATH_ALL,
                 'dividend':DIVIDEND_PATH_ALL
                 }
    df_list = []
    for paths in path_dict[action_type].split('|'):
        df = pd.read_csv(paths)
        df_list.append(df)
    df = pd.concat(df_list)
    actions_df_list = []

    for symbol in symbol_list:
        action_df = df.loc[df['SYMBOL']==symbol]
        actions_df_list.append(action_df)
    action_df = pd.concat(actions_df_list)
    return action_df

if __name__=='__main__':
    output_path = r'C:/Users/paras/NSE_DATA/eda_plots/'
    prices_path = f'{PRICES_PKL_PATH}/prices'
    symbol_list = sorted([a.split('_')[0] for a in os.listdir(prices_path)])
    symbol_list = symbol_list[0:25]
    # action_type = 'split'
    # bonus_df = filter_corporate_actions_data(symbol_list=symbol_list,action_type=action_type)
    # bonus_df.to_csv(f'{output_path}/{action_type}_nifty50_stocks.csv')
    period = (datetime(2002,1,1),datetime(2024,12,31))
    for symbol  in symbol_list:
        plot_prices(symbol=symbol,output_path=output_path,period=period)





