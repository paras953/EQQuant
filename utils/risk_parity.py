import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize

from utils.config import Constants


def risk_parity_with_target_vol(pre_opt_weights,cov_matrix, target_vol):
    def risk_parity_objective(weights, cov_matrix,pre_opt_weights):
        portfolio_vol = weights @ cov_matrix @ weights.T
        marginal_risk_contrib = cov_matrix @ weights.T/portfolio_vol
        risk_contrib = weights * marginal_risk_contrib / portfolio_vol
        loss_1 = np.sum((risk_contrib - risk_contrib.mean()) ** 2)
        return loss_1

    num_assets = len(cov_matrix)
    init_weights = np.ones(num_assets) / num_assets
    bounds_list = []
    for i in range(num_assets):
        lower_limit = 0
        upper_limit = 1/((num_assets*pre_opt_weights[i]) + Constants.NOISE.value)
        bounds_list.append((lower_limit,1))

    result = minimize(
        risk_parity_objective,
        init_weights,
        args=(cov_matrix,pre_opt_weights),

        method='SLSQP'
    )

    opt_weights = result.x
    portfolio_vol = np.sqrt(opt_weights @ cov_matrix @ opt_weights.T)
    scaling_factor = target_vol/portfolio_vol
    opt_weights = opt_weights * scaling_factor

    return opt_weights

