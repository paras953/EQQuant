from nsepython import *
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict, List
from urllib.parse import urlparse, quote
from datetime import datetime
import yfinance as yf
import os
from pandas.tseries.offsets import BDay
from utils.decorators import timer
from utils.config import NSEPYTHON_PRICES_PATH, CORPORATE_ACTIONS_PATH, DIVIDEND_PATH, Columns, GOOD_DATE_MAP, \
    YFINANCE_PRICES_PATH, NSE_INDEX_MASTER
from utils.pandas_utils import remove_outliers


# TODO : add basic price cleaning function
class NSEMasterDataAccess():
    def __init__(self, output_path: str):
        self.yfinance_prices_path = f'{YFINANCE_PRICES_PATH}/prices/'
        self.nsepython_prices_path = f'{NSEPYTHON_PRICES_PATH}/prices/'
        self.output_path = output_path

        if not os.path.exists(self.yfinance_prices_path):
            os.makedirs(self.yfinance_prices_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        if not os.path.exists(self.nsepython_prices_path):
            os.makedirs(self.nsepython_prices_path)

        self.today = datetime.today()

    def getTickerMetaData(self, index_input: str) -> pd.DataFrame:

        index_encode = quote(index_input.upper())
        print(f'Fetching Data for {index_input} - {index_encode}')
        positions = nsefetch(f'https://www.nseindia.com/api/equity-stockIndices?index={index_encode}')
        symbol_df = pd.DataFrame(positions['data'])
        if 'meta' in symbol_df.columns:
            symbol_df = symbol_df.dropna(subset=['meta'])
        return symbol_df

    @timer
    def extractTickerMasterData(self, index_name: str = None, save_metadata: bool = False) -> pd.DataFrame:

        index_data = NSE_INDEX_MASTER  # checkout 'https://www.nseindia.com/api/equity-master'
        data_dict = defaultdict(list)

        for keys, index_list in index_data.items():
            for idx in index_list:
                if idx == 'Permitted to Trade':
                    continue
                if index_name:
                    if index_name != idx:
                        continue

                symbol_df = self.getTickerMetaData(index_input=idx)
                symbol_metadata = list(symbol_df['meta'])
                fields = ['symbol', 'companyName', 'industry', 'isFNOSec', 'isETFSec', 'isDelisted', 'isin', 'index']
                for i in symbol_metadata:
                    for field in fields:
                        if field != 'index':
                            data_dict[field].append(i[field])
                        else:
                            data_dict[field].append(idx)

        master_data = pd.DataFrame(data_dict)
        # master_data = master_data.drop_duplicates(subset='symbol')
        if save_metadata:
            today = self.today.strftime('%Y-%m-%d')
            master_data.to_csv(f'{self.output_path}/ticker_metadata_{today}.csv')
            print('Fetched data for ', master_data['symbol'].nunique(), 'Tickers')

        return master_data

    @timer
    def download_historical_prices(self, symbol: str, start_date: datetime, end_date: datetime,
                                   use_yfinance: bool = True) -> pd.DataFrame:
        """
        This fn downloads the data and dumps the raw data into pkls
        :param symbol: symbol of the stock
        :param start_date: start date of the data download
        :param end_date: end date of data download
        :param use_yfinance: True means it uses the prices downloaded from yfinance
        :return: raw data download from the nse api
        """
        print(f'Fetching Data for {symbol}, using Yahoo finance? -> {use_yfinance}')

        if not use_yfinance:
            start_date_str = start_date.strftime('%d-%m-%Y')
            end_date_str = end_date.strftime('%d-%m-%Y')
            prices_pkl_path = f'{self.nsepython_prices_path}/{symbol}_prices.pkl'
        else:
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            prices_pkl_path = f'{self.yfinance_prices_path}/{symbol}_prices.pkl'

        if os.path.exists(prices_pkl_path):
            old_prices = pd.read_pickle(prices_pkl_path)
            start_date = old_prices.index.max() + BDay(1)
            if not use_yfinance:
                start_date_str = start_date.strftime('%d-%m-%Y')
            else:
                start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            old_prices = None

        if start_date < end_date:
            if not use_yfinance:
                prices_data = equity_history(symbol, 'EQ', start_date_str, end_date_str)
            else:
                ticker = yf.Ticker(f'{symbol}.NS')
                prices_data = ticker.history(start=start_date_str, end=end_date_str, auto_adjust=False,
                                             period='1d', back_adjust=False)

            if len(prices_data) == 0:
                print(f'No data found for {symbol} between {start_date_str} and {end_date_str}')
                # just return an empty dataframe
                return pd.DataFrame()

            if not use_yfinance:
                prices_data = prices_data.rename(columns={'CH_TIMESTAMP': 'Date'})
                prices_data['Date'] = pd.to_datetime(prices_data['Date'])
                prices_data = prices_data.set_index('Date')

            prices_data.index = prices_data.index.tz_localize(None)
            prices_data = prices_data.sort_index()

            if old_prices is not None:
                print('Appending Latest Data to Old data')
                prices_data = pd.concat([old_prices, prices_data])
                if not use_yfinance:
                    prices_data = prices_data.drop_duplicates(keep='first', subset=['_id'])

            prices_data.to_pickle(prices_pkl_path)
            return prices_data
        else:
            return old_prices

    def get_prices(self, symbol: str, start_date: datetime, end_date: datetime,
                   use_yfinance: bool = True) -> pd.DataFrame:
        """
        This fn reads the prices downloaded using the download_historical_prices function and then
        processes it and returns a processed price df

        :param symbol: symbol of the stock
        :param start_date: start date of the prices
        :param end_date: end date of the prices
        :param use_yfinance: True means it uses the prices downloaded from yfinance
        :return: processed prices dataframe
        """
        print(f'Fetching prices for {symbol}')
        if not use_yfinance:
            prices_pkl_path = f'{self.nsepython_prices_path}/{symbol}_prices.pkl'
        else:
            prices_pkl_path = f'{self.yfinance_prices_path}/{symbol}_prices.pkl'

        prices_data = pd.read_pickle(prices_pkl_path)

        ## Handling price update by sorting it on updatedAt with date and then selecting the last one
        if not use_yfinance:
            prices_data["updatedAt"] = pd.to_datetime(prices_data["updatedAt"])
            prices_data = prices_data.sort_values(["Date", "updatedAt"])
            prices_data = prices_data[~prices_data.index.duplicated(keep='last')]
            rename_dict = {'CH_TIMESTAMP': 'Date', 'CH_TRADE_HIGH_PRICE': Columns.HIGH.value,
                           'CH_TRADE_LOW_PRICE': Columns.LOW.value,
                           'CH_CLOSING_PRICE': Columns.CLOSE.value,
                           'CH_OPENING_PRICE': Columns.OPEN.value, 'CH_LAST_TRADED_PRICE': Columns.LTP.value,
                           'CH_TOTAL_TRADES': Columns.TRADES.value, 'CH_TOT_TRADED_QTY': Columns.VOLUME.value}
            prices_data = prices_data.rename(columns=rename_dict)

            prices_data = self.adjust_prices_for_corporate_actions(action_type='split', prices=prices_data, symbol=symbol,
                                                                   prices_columns=[Columns.OPEN.value, Columns.HIGH.value,
                                                                                   Columns.LOW.value, Columns.CLOSE.value,
                                                                                   Columns.LTP.value, Columns.VWAP.value])

            prices_data = self.adjust_prices_for_corporate_actions(action_type='dividend', prices=prices_data,
                                                                   symbol=symbol,
                                                                   prices_columns=[Columns.OPEN.value, Columns.HIGH.value,
                                                                                   Columns.LOW.value, Columns.CLOSE.value,
                                                                                   Columns.LTP.value, Columns.VWAP.value])
        else:
            rename_dict = {'Open':Columns.OPEN.value,'Close':Columns.CLOSE.value,
                           'High':Columns.HIGH.value,'Low':Columns.LOW.value,'Adj Close':Columns.ADJ_CLOSE.value}
            prices_data = prices_data.rename(columns=rename_dict)

        prices_data = remove_outliers(df=prices_data,column=Columns.ADJ_CLOSE.value)
        good_date = GOOD_DATE_MAP.get(symbol, datetime(2002, 1, 1))
        start_date = max(good_date, start_date)
        prices_data = prices_data.truncate(start_date, end_date)

        return prices_data

    def _find_corporate_action(self, symbol: str, action_type: str, prices: pd.DataFrame, prices_columns: List[str]):
        if action_type not in ['split', 'dividend']:
            # split and bonus data is included in splits itself
            raise NotImplementedError(
                f"Only implemented for action_type -> split or dividend you passed {action_type}")

        if action_type == 'split':
            split_info = yf.Ticker(symbol + ".NS").splits
            stock_split_df = pd.DataFrame(data=split_info)
            stock_split_df.columns = ["price_multiplier"]
            stock_split_df["price_multiplier"] = 1 / stock_split_df["price_multiplier"]
            stock_split_df.index = stock_split_df.index.date.astype("datetime64[ns]")
            stock_split_df.index.name = 'EX-DATE'
        else:
            split_info = yf.Ticker(symbol + ".NS").dividends
            stock_split_df = pd.DataFrame(data=split_info)
            stock_split_df.columns = ["dividend_amount"]
            stock_split_df.index = stock_split_df.index.date.astype("datetime64[ns]")
            stock_split_df.index.name = 'EX-DATE'
            stock_split_df = stock_split_df.groupby("EX-DATE").sum()

            prices = prices.join(stock_split_df[['dividend_amount']], how='left')
            prices['dividend_amount'] = prices['dividend_amount'].shift(-1)
            prices_before_ex_date = prices[~prices['dividend_amount'].isna()]
            prices_before_ex_date['price_multiplier'] = 1 - (
                    prices_before_ex_date['dividend_amount'] / prices_before_ex_date['Close'])
            prices_before_ex_date = prices_before_ex_date.sort_index(ascending=False)
            prices_before_ex_date['price_multiplier'] = prices_before_ex_date['price_multiplier'].cumprod()
            prices_before_ex_date = prices_before_ex_date.sort_index()
            prices = prices.join(prices_before_ex_date[['price_multiplier']], how='left')
            prices['price_multiplier'] = prices['price_multiplier'].bfill().fillna(1)
            stock_split_df = prices[['price_multiplier']]

        return stock_split_df[['price_multiplier']]

    def adjust_prices_for_corporate_actions(self, action_type: str, symbol: str, prices: pd.DataFrame,
                                            prices_columns: List[str], volume: str = 'Volume') -> pd.DataFrame:
        """
        get the stock split data from file NSE_CORPORATE_ACTIONS-01-01-2002-to-25-09-2024.csv,
        stored in the colab folder of drive, the data is only until 2024-09-25,
        TODO : I downloaded it from the NSE website, but just dont know the exact link
        find that link
        Ideally it has to be downloaded everyday, to check for new split actions
        """
        print(f'Adjusting prices for {symbol} - {action_type}')

        split_data = self._find_corporate_action(symbol=symbol, action_type=action_type, prices=prices,
                                                 prices_columns=prices_columns)
        if action_type in ['bonus', 'split']:
            for ex_date, row in split_data.iterrows():
                mask = (prices.index < ex_date)
                if mask.sum() > 0:
                    prices.loc[mask, prices_columns] *= row['price_multiplier']
                    prices.loc[mask, volume] /= row['price_multiplier']
        else:
            for cols in prices_columns:
                print(f"Currently Adjusting for {cols}")
                prices[f'Adj{cols}'] = prices[cols] * split_data['price_multiplier']
        return prices

    def download_historical_prices_yfinance(self, symbol: str, start_date: datetime, end_date: datetime) -> None:
        """
        This function will download and dump the data in pkl format
        :param symbol: pass the NSE symbol only i.e. ADANIENT and not ADANIENT.NS -> this is what yfinance needs actually
        :param start_date: start date of the data download
        :param end_date: end date of data download
        :return: None
        """
        ticker = yf.Ticker(f'{symbol}.NS')
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        prices = ticker.history(start=start_str, end=end_str, auto_adjust=False, period='1d', back_adjust=False)
        prices = prices.sort_index()
        prices.to_pickle(f'{self.yfinance_prices_path}/{symbol}_prices.pkl')
        return None

    def get_index_constituents(self, index_name: str) -> List[str]:
        """
        :param index_name: eg NIFTY 50
        :return: returns a list of underlying stocks in that index
        """
        ticker_metadata = self.extractTickerMasterData(index_name=index_name)
        return list(set(ticker_metadata['symbol']))


if __name__ == '__main__':
    nse_data_access = NSEMasterDataAccess(output_path=YFINANCE_PRICES_PATH)
    ticker_list = nse_data_access.get_index_constituents(index_name='NIFTY 50')  # downloading nifty 50 stocks
    ticker_list = sorted(ticker_list)
    # ticker_list = ['RELIANCE']
    year_list = [i for i in range(2007, 2025)]
    for ticker in ticker_list:
        for year in year_list:
            prices = nse_data_access.download_historical_prices(symbol=ticker, start_date=datetime(year, 1, 1),
                                                                end_date=datetime(year, 12, 31))

    # for ticker in ticker_list:
    #     prices = nse_data_access.get_prices(symbol=ticker, start_date=datetime(2002, 1, 1),
    #                                         end_date=datetime(2024, 12, 31))
    #     print("hello")
