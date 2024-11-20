import numpy as np

def sum_product_constraint(w,pre_op_weight_vector):
    return np.dot(pre_op_weight_vector, w) - 1


# Step 2: Define Custom Objective Function
def custom_objective(w, mu, cov_matrix, pre_weights, risk_free_rate=0.02, penalty=100):
    # Calculate portfolio return and volatility
    portfolio_return = np.dot(w, mu)
    portfolio_volatility = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    # Add penalty for deviation from sum-product constraint
    penalty_term = penalty * (np.dot(pre_weights, w) - 1) ** 2

    # Maximizing Sharpe ratio -> Minimizing negative Sharpe with penalty
    return -sharpe_ratio + penalty_term