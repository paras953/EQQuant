import numpy as np

from utils.config import Constants


def sum_product_constraint(w, pre_op_weight_vector):
    return np.dot(pre_op_weight_vector, w) - 1


# Step 2: Define Custom Objective Function
def min_sharpe(w, mu, cov_matrix, risk_free_rate=0.0):
    # Calculate portfolio return and volatility
    portfolio_return = np.dot(w, mu)
    portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio
