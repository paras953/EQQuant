import numpy as np
import pandas as pd
from typing import Dict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from numpy.core.defchararray import upper
from pypfopt import EfficientFrontier, expected_returns, risk_models, HRPOpt

from analytics.risk_return import get_asset_volatility, get_asset_returns
from data.NSEDataAccess import NSEMasterDataAccess
from signals.momentum.momentum_signals import TALIB_AROONOSC, TALIB_BOP, TALIB_CCI,HOLD
from utils.config import YFINANCE_PRICES_PATH, Columns, Constants
from utils.decorators import timer
from utils.risk_parity import risk_parity_with_target_vol
from utils.constraints_helper import sum_product_constraint

class PortfolioManager():
    def __init__(self, config: Dict):
        self.config = config

    @timer
    def prices_to_signals(self) -> pd.DataFrame:
        print('In method prices_to_signals()')
        signal_dict = {}
        for signal, signal_config in self.config['signals'].items():
            signal_dict[signal] = self._create_signal_data(signal_type=signal_config['signal_function'],
                                                           signal_config=signal_config)

        multi_index_signal_df_list = []
        for keys, signal_df in signal_dict.items():
            signal_df.columns = pd.MultiIndex.from_product([[keys], signal_df.columns])
            multi_index_signal_df_list.append(signal_df)
        all_signal_df = pd.concat(multi_index_signal_df_list, axis=1)
        all_signal_df.columns.names = ['signal', 'symbol']
        return all_signal_df

    def _create_signal_data(self, signal_type: str, signal_config: Dict) -> pd.DataFrame:
        print('In method _create_signal_data()')
        nse_data_access = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
        period = self.config['period']
        start_date = period[0] - relativedelta(years=5)
        df_list = []

        for symbol in self.config['universe']:
            prices = nse_data_access.get_prices(symbol=symbol,
                                                start_date=start_date,
                                                end_date=period[-1])

            if signal_type == 'TALIB_AROONOSC':
                signal, signal_name = TALIB_AROONOSC(prices=prices, lookback=signal_config['lookback'])
            elif signal_type == 'TALIB_BOP':
                signal, signal_name = TALIB_BOP(prices=prices)
            elif signal_type == 'TALIB_CCI':
                signal, signal_name = TALIB_CCI(prices=prices, lookback=signal_config['lookback'])
            elif signal_type == 'HOLD':
                signal, signal_name = HOLD(prices=prices, position=signal_config['position'])
            else:
                raise NotImplementedError("other signals not implemented yet!")

            signal = signal.rename(columns={signal_name: symbol})
            df_list.append(signal)

        signal_df = pd.concat(df_list, axis=1)
        return signal_df

    @timer
    def get_composite_signal(self, all_signal_df: pd.DataFrame) -> pd.DataFrame:
        print('In method get_composite_signal()')
        weight_dict = {}

        for symbol in self.config['universe']:
            asset_level_df = all_signal_df.xs(key=symbol, level='symbol', axis=1)
            first_valid_indices = {signal: asset_level_df[signal].first_valid_index() for signal in
                                   asset_level_df.columns}
            budget_df = pd.DataFrame(0, index=asset_level_df.index, columns=asset_level_df.columns, dtype=float)

            for signal, start_idx in first_valid_indices.items():
                if start_idx is not None:  # Ensure the signal has at least one non-NaN value
                    budget_df.loc[start_idx:, signal] = self.config['signals'][signal]['budget']
            row_sums = budget_df.sum(axis=1)
            budget_df = budget_df.div(row_sums, axis=0)
            weight_dict[symbol] = budget_df

        multi_index_budget_df_list = []
        for symbol, weight_df in weight_dict.items():
            columns = pd.MultiIndex.from_product([weight_df.columns, [symbol]])
            weight_df.columns = columns
            multi_index_budget_df_list.append(weight_df)

        all_weight_df = pd.concat(multi_index_budget_df_list, axis=1)
        all_weight_df.columns.names = ['signal', 'symbol']
        composite_signal_df = all_signal_df.mul(all_weight_df).sum(axis=1, level='symbol')
        return composite_signal_df

    @timer
    def get_positions(self, composite_signal: pd.DataFrame) -> pd.DataFrame:
        print('In method get_positions()')
        nse_data = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
        start_date = self.config['period'][0] - relativedelta(years=5)
        end_date = self.config['period'][-1]
        if self.config['position']['position_type'] == 'vol_adjusted':
            prices_dict = nse_data.get_prices_multiple_assets(symbol_list=self.config['universe'],
                                                              period=(start_date, end_date))
            prices = prices_dict[Columns.CLOSE.value]
            vol_df = get_asset_volatility(prices=prices)
            risk_adjustment_factor_1 = self.config['position']['target_vol'] / vol_df
            positions_df = composite_signal.mul(risk_adjustment_factor_1)
            if self.config['position']['direction'] == 'long_only':
                print('Filtering only Long only positions')
                positions_df = positions_df.clip(lower=0)
                positions_df = positions_df.div(positions_df.sum(axis=1) + Constants.NOISE.value, axis=0)
            else:
                raise NotImplementedError('Can only go long!!')

        else:
            raise NotImplementedError("Other position sizing algos not implemented!")

        return positions_df

    @timer
    def allocate(self, positions: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        print('In method allocate()')
        nse_data_access = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
        period = (self.config['period'][0] - relativedelta(years=5), self.config['period'][-1])
        prices_dict = nse_data_access.get_prices_multiple_assets(symbol_list=self.config['universe'],
                                                                 period=period)
        prices = prices_dict[Columns.CLOSE.value]
        long_only_returns = get_asset_returns(prices)

        # optimization on this return series
        pre_optimized_portfolio_return_contribution = positions.shift() * long_only_returns
        optimization_lookback = self.config['allocation']['lookback']
        start_date = self.config['period'][0] - relativedelta(days=optimization_lookback)
        post_op_weight_list = []

        for dates, _ in pre_optimized_portfolio_return_contribution.loc[start_date:].iterrows():
            print(dates)
            return_series = pre_optimized_portfolio_return_contribution[
                            dates - relativedelta(days=optimization_lookback):dates]
            mean_return = expected_returns.mean_historical_return(prices=return_series, returns_data=True)
            var_cov_matrix = risk_models.sample_cov(prices=return_series, returns_data=True)
            var_cov_matrix_ledoit_wolf = risk_models.CovarianceShrinkage(return_series, returns_data=True).ledoit_wolf()

            pre_op_weight_vector = positions.loc[dates]
            weight_bounds_list = []
            non_zero_assets = (pre_op_weight_vector != 0).sum()
            for symbol in positions.columns:
                upper_bound = 1 / (
                        (non_zero_assets * abs(pre_op_weight_vector[symbol])) + Constants.NOISE.value)
                weight_bounds_list.append((0, upper_bound))

            ef = EfficientFrontier(expected_returns=mean_return, cov_matrix=var_cov_matrix_ledoit_wolf,
                                   weight_bounds = (0,np.inf))
            # ef.add_objective(sum_product_constraint,pre_op_weight_vector=pre_op_weight_vector.values)

            # total_weight = pre_op_weight_vector.sum()
            ef.add_constraint(lambda w : np.dot(w,pre_op_weight_vector))
            # for symbol in positions.columns:
            #     ticker_index = ef.tickers.index(symbol)
            #     limit = 1 / (
            #             (non_zero_assets * pre_op_weight_vector[symbol]) + Constants.NOISE.value)
            #     ef.add_constraint(lambda w: w[ticker_index]<=limit)


            if self.config['allocation']['type'] == 'max_sharpe':
                post_op_weight_vector = ef.max_sharpe(risk_free_rate=0.0)
                ef.portfolio_performance(verbose=True, risk_free_rate=0.0)
            elif self.config['allocation']['type'] == 'min_vol':
                # only this works
                post_op_weight_vector = ef.min_volatility()
                ef.portfolio_performance(verbose=True, risk_free_rate=0.0)
            elif self.config['allocation']['type'] == 'efficient_risk':
                post_op_weight_vector = ef.efficient_risk(
                    target_volatility=self.config['allocation']['portfolio_vol_target'])
            elif self.config['allocation']['type'] == 'efficient_return':
                post_op_weight_vector = ef.efficient_return(
                    target_return=self.config['allocation']['portfolio_return_target'])

            elif self.config['allocation']['type'] == 'risk_parity':
                # hrp = HRPOpt(return_series)
                # hrp.optimize()
                # post_op_weight_vector = hrp.clean_weights()
                # hrp.portfolio_performance(verbose=True,risk_free_rate=0.0)
                post_op_weight = risk_parity_with_target_vol(pre_opt_weights=pre_op_weight_vector.values,
                                                             cov_matrix=var_cov_matrix,
                                                             target_vol=self.config['allocation']['target_vol'])
                post_op_weight_vector = dict(zip(positions.columns, post_op_weight))
            elif self.config['allocation']['type'] ==  'max_utility':
                post_op_weight_vector = ef.max_quadratic_utility()
                ef.portfolio_performance(verbose=True, risk_free_rate=0.0)
            else:
                raise NotImplementedError('Other optimization techniques not implemented yet!')
            post_op_weight_vector["Date"] = dates
            post_op_weight_list.append(post_op_weight_vector)

        post_optimization_df = pd.DataFrame(post_op_weight_list).set_index('Date')
        post_optimization_df = post_optimization_df.truncate(self.config['period'][0], self.config['period'][-1])
        positions = positions.truncate(self.config['period'][0], self.config['period'][-1])
        final_weight = positions.mul(post_optimization_df)
        return {'dollar_weight': final_weight,
                'optimization_weight': post_optimization_df,
                'pre_optimization_weight': positions}

    @timer
    def run_portfolio(self) -> Dict[str, pd.DataFrame]:
        portfolio_period = self.config['period']
        signal_df = self.prices_to_signals()
        composite_signal = self.get_composite_signal(all_signal_df=signal_df)
        position_df = self.get_positions(composite_signal=composite_signal)


        if self.config['allocation']['optimize']:
            weights_dict = self.allocate(positions=position_df)
        else:
            weights_dict = {'dollar_weight': position_df,
                'optimization_weight': position_df,
                'pre_optimization_weight': position_df}

        for keys,values in weights_dict.items():
            weights_dict[keys] = values.copy().truncate(portfolio_period[0],portfolio_period[-1])

        return weights_dict


if __name__ == '__main__':
    period = (datetime(2022, 6, 1), datetime(2023, 12, 31))
    nse_data = NSEMasterDataAccess(YFINANCE_PRICES_PATH)
    universe = nse_data.get_index_constituents('NIFTY 50')
    # universe = universe[0:2]
    # universe = ['SUNPHARMA', 'HCLTECH']
    pf_config = {
        'universe': universe,
        'signals': {
            # 'TALIB_AROONOSC_1': {'signal_function': 'TALIB_AROONOSC', 'lookback': 33, 'budget': 0.25},
            # 'TALIB_AROONOSC_2': {'signal_function': 'TALIB_AROONOSC', 'lookback': 66, 'budget': 0.25},

            # 'TALIB_BOP_1': {'signal_function': 'TALIB_BOP', 'budget': 0.5},
            'TALIB_CCI_1': {'signal_function': 'TALIB_CCI', 'budget': 0.5, 'lookback': 22},
            'TALIB_CCI_2': {'signal_function': 'TALIB_CCI', 'budget': 0.5, 'lookback': 66},

        },
        'period': period,
        'position': {'position_type': 'vol_adjusted', 'target_vol': 0.005, 'direction': 'long_only'},
        'allocation': {'lookback': 365, 'type': 'max_utility', 'target_vol': 0.01,'optimize':False}
    }

    pm = PortfolioManager(config=pf_config)
    weights = pm.run_portfolio()
