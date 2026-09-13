import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env_bs import SimplifiedEnvConfig, make_simplified_env

# Suppress GUI backend warnings
import matplotlib
matplotlib.use('Agg')

def twap_action(env, step):
    total_steps = env.config.simulation_planning_horizon
    elapsed = step - env.data.start_index
    remaining = total_steps - elapsed
    if remaining <= 0:
        ratio = 1.0
    else:
        qty_per_step = env.quantity / remaining
        ratio = min(1.0, qty_per_step / max(env.quantity, 1e-8))
    return np.array([0.0, ratio])  # no limit order, only market execution

def run_episode(env, policy):
    market_state, private_state = env.reset()
    step = env.data.current_index
    done = False
    while not done:
        action = policy(env, step)
        market_state, private_state, reward, done, info = env.step(action)
        step = env.data.current_index
    metrics = env.get_trading_metrics()
    return metrics['price_performance_bp'], metrics['cancel_rate']

def main():
    config = SimplifiedEnvConfig()
    env = make_simplified_env(config)

    bp_list = []
    cancel_list = []
    count = 0

    print("Collecting TWAP metrics over all (code, date)...")
    for code, date in env.valid_code_date_list:
        for _ in range(10):
            try:
                bp, cancel = run_episode(env, twap_action)
                bp_list.append(-bp)
                cancel_list.append(cancel)
                count += 1
                if count % 100 == 0:
                    print(f"Collected {count} episodes")
            except Exception as e:
                continue  # skip invalid episodes

    print(f"Total valid episodes: {len(bp_list)}")

    # Plot 1: price_performance_bp
    plt.figure(figsize=(8, 5))
    sns.histplot(bp_list, bins=50, kde=True, color='steelblue')
    plt.axvline(np.mean(bp_list), color='red', linestyle='--', 
                label=f'Mean = {np.mean(bp_list):.2f} bp')
    plt.xlabel('Price Performance (bp)')
    plt.ylabel('Frequency')
    plt.title('TWAP Price Performance Distribution (Sell Order)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('twap_bp_distribution.png', dpi=300)
    plt.close()

    # Plot 2: cancel_rate
    plt.figure(figsize=(8, 5))
    sns.histplot(cancel_list, bins=20, kde=False, color='tomato')
    plt.xlabel('Cancel Rate')
    plt.ylabel('Frequency')
    plt.title('TWAP Cancel Rate Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('twap_cancel_distribution.png', dpi=300)
    plt.close()

    # Print summary
    print("\n=== TWAP Performance Summary ===")
    print(f"price_performance_bp: {np.mean(bp_list):.2f} ± {np.std(bp_list):.2f} bp")
    print(f"cancel_rate: {np.mean(cancel_list):.4f} ({np.mean(cancel_list)*100:.2f}%)")
    print(f"Saved plots: twap_bp_distribution.png, twap_cancel_distribution.png")

if __name__ == "__main__":
    main()
