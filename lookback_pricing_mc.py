import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# --- Configuration & Parameters ---
S0_val = 100
r_val = 0.05
sigma_val = 0.2
T_val = 1.0
steps_val = 10000
sims_val = 100000  # High number for better convergence

def lookback_put_analytical(S, J, r, sigma, T):
    """
    Analytical formula for a Floating Strike Lookback Put.
    Reference: Floating Strike Lookback Options - Thesis Page 19.
    """
    if T == 0: return max(J - S, 0)
    
    tau = T 
    k = 2 * r / (sigma**2)
    
    # Standard components d5, d6, d7
    d5 = (np.log(J/S) + (0.5 * sigma**2 - r) * tau) / (sigma * np.sqrt(tau))
    d6 = (np.log(S/J) + (0.5 * sigma**2 - r) * tau) / (sigma * np.sqrt(tau))
    d7 = (np.log(S/J) + (0.5 * sigma**2 + r) * tau) / (sigma * np.sqrt(tau))
    
    # Formula components
    term1 = J * np.exp(-r * tau) * norm.cdf(d5)
    term2 = J * np.exp(-r * tau) * (1/k) * (S/J)**(1-k) * norm.cdf(d6)
    term3 = S * ((1 + 1/k) * norm.cdf(d7) - 1)
    
    return term1 - term2 + term3

def simulate_lookback_batches(S0, r, sigma, T, steps, n_simulations, batch_size=2000):
    """
    Monte Carlo Simulation using Batch Vectorization to optimize RAM usage.
    Handles the bias by including S0 in the path (t=0).
    """
    dt = T / steps
    discount = np.exp(-r * T)
    total_payoffs = []
    
    for i in range(0, n_simulations, batch_size):
        current_batch = min(batch_size, n_simulations - i)
        
        # Vectorized generation of random returns per batch
        z = np.random.standard_normal((current_batch, steps))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        cum_log_returns = np.cumsum(log_returns, axis=1)
        
        # Price reconstruction including S0 at t=0
        paths = S0 * np.exp(np.hstack([np.zeros((current_batch, 1)), cum_log_returns]))
        
        # Calculate payoff for the floating strike lookback put: Max - Final
        batch_payoffs = (np.max(paths, axis=1) - paths[:, -1]) * discount
        total_payoffs.append(batch_payoffs)
        
    all_payoffs = np.concatenate(total_payoffs)
    return np.mean(all_payoffs)

def plot_convergence(S0, r, sigma, T, steps, total_sims, analytical, batch_size=2000):
    """
    Visualizes how Monte Carlo price converges to the analytical value as simulations increase.
    """
    iterations = np.arange(batch_size, total_sims + 1, batch_size)
    prices = []
    running_sum = 0
    
    print("Computing convergence plot data...")
    for i, _ in enumerate(iterations):
        # We simulate batch by batch to track the moving average
        p_batch = simulate_lookback_batches(S0, r, sigma, T, steps, batch_size)
        running_sum += p_batch
        prices.append(running_sum / (i + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, prices, label='Monte Carlo (Running Average)', color='#1f77b4', lw=2)
    plt.axhline(y=analytical, color='#d62728', linestyle='--', label=f'Analytical Price ({analytical:.4f})')
    
    plt.title("Monte Carlo Convergence vs. Analytical Pricing (Lookback Put)", fontsize=14)
    plt.xlabel("Number of Simulations", fontsize=12)
    plt.ylabel("Option Price ($)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Execution
print("Starting Valuation...")
start_time = time.time()

analytical_price = lookback_put_analytical(S0_val, S0_val, r_val, sigma_val, T_val)
mc_price = simulate_lookback_batches(S0_val, r_val, sigma_val, T_val, steps_val, sims_val)

execution_time = time.time() - start_time

print(f"\nFinal Results (Time: {execution_time:.2f}s)")
print("-" * 35)
print(f"Analytical Price:    {analytical_price:.4f}")
print(f"Monte Carlo Price:   {mc_price:.4f}")
print(f"Absolute Difference: {abs(analytical_price - mc_price):.4f}")
print("-" * 35)

# Visual Analysis
plot_convergence(S0_val, r_val, sigma_val, T_val, steps_val, sims_val, analytical_price)

# Delta Calculation (Risk Management)
epsilon = 0.01 * S0_val  # 1% price variation
print(f"\nCalculating Delta with +/- {epsilon}% variation...")

# Price with S0 + epsilon
price_up = simulate_lookback_batches(S0_val + epsilon, r_val, sigma_val, T_val, steps_val, sims_val)
# Price with S0 - epsilon
price_down = simulate_lookback_batches(S0_val - epsilon, r_val, sigma_val, T_val, steps_val, sims_val)

delta = (price_up - price_down) / (2 * epsilon)

print(f"Option Delta: {delta:.4f}")
print(f"Interpretation: If S0 increases by $1, the option price changes by approximation ${delta:.4f}")