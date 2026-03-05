import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==========================================
# 1. SYNTHETIC DATA GENERATION
# ==========================================
def generate_portfolio_data(n_assets=5):
    """Generates synthetic asset data with varying risk profiles."""
    np.random.seed(42)
    assets = [f'Asset_{i+1}' for i in range(n_assets)]
    
    # Randomly assign volatilities (risk) and expected returns
    volatilities = np.random.uniform(0.05, 0.25, n_assets)
    expected_returns = np.random.uniform(0.02, 0.10, n_assets)
    
    # Initial equal weights
    weights = np.array([1/n_assets] * n_assets)
    
    return pd.DataFrame({
        'Asset': assets,
        'Weight': weights,
        'Volatility': volatilities,
        'Return': expected_returns
    })

# ==========================================
# 2. MONTE CARLO STRESS TESTING ENGINE
# ==========================================
def monte_carlo_simulation(weights, returns, vols, iterations=10000, days=252):
    """Simulates portfolio outcomes over a year to calculate Value at Risk (VaR)."""
    portfolio_sims = np.zeros(iterations)
    
    for i in range(iterations):
        # Simulate daily returns using Gaussian distribution
        daily_returns = np.random.normal(returns/days, vols/np.sqrt(days), (days, len(weights)))
        portfolio_daily_returns = np.dot(daily_returns, weights)
        # Cumulative product to get end-of-year value
        portfolio_sims[i] = np.prod(1 + portfolio_daily_returns)
    
    # Calculate 95% VaR (The 5th percentile of outcomes)
    var_95 = np.percentile(portfolio_sims, 5) - 1
    return var_95, portfolio_sims

# ==========================================
# 3. ALGORITHMIC WEIGHT REBALANCING
# ==========================================
def algorithmic_rebalancing(df):
    """
    Reduces VaR by reallocating weights from high-volatility to 
    low-volatility assets (Inverse-Variance Optimization).
    """
    inv_vol = 1.0 / df['Volatility']
    optimized_weights = inv_vol / inv_vol.sum()
    return optimized_weights

# ==========================================
# 4. EXECUTION & VALIDATION
# ==========================================
# Load Data
df = generate_portfolio_data()
print("--- Initial Portfolio ---\n", df[['Asset', 'Weight', 'Volatility']])

# Run Baseline Stress Test
baseline_var, baseline_sims = monte_carlo_simulation(
    df['Weight'].values, df['Return'].values, df['Volatility'].values
)

# Apply Algorithmic Rebalancing (The Optimization Step)
df['Optimized_Weight'] = algorithmic_rebalancing(df)

# Run Optimized Stress Test
optimized_var, optimized_sims = monte_carlo_simulation(
    df['Optimized_Weight'].values, df['Return'].values, df['Volatility'].values
)

# Calculate Reduction
reduction = (baseline_var - optimized_var) / baseline_var
print(f"\n--- Results ---")
print(f"Baseline VaR (95%): {baseline_var:.2%}")
print(f"Optimized VaR (95%): {optimized_var:.2%}")
print(f"Calculated VaR Reduction: {reduction:.2%}")

# ==========================================
# 5. VISUALIZATION FOR PORTFOLIO
# ==========================================
plt.figure(figsize=(10, 6))
plt.hist(baseline_sims, bins=50, alpha=0.5, label='Baseline Portfolio', color='blue')
plt.hist(optimized_sims, bins=50, alpha=0.5, label='Optimized Portfolio', color='green')
plt.axvline(1 + baseline_var, color='blue', linestyle='--', label=f'Baseline VaR: {baseline_var:.2%}')
plt.axvline(1 + optimized_var, color='green', linestyle='--', label=f'Optimized VaR: {optimized_var:.2%}')
plt.title("Portfolio Stress Test: VaR Reduction Analysis")
plt.xlabel("End-of-Year Portfolio Value (Initial = 1.0)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
