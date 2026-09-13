import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

# twap_defensive_compare_v2.py
"""
Enhanced TWAP strategy comparison with full intelligent execution engine.

Includes:
- Parameter sweep for limit_part
- Volatility-aware adaptive TWAP
- Smart timing strategy
- Dynamic tail risk management
- ✅ Enhanced multi-scale TWAP with microstructure signals + signal logging
- VWAP benchmark
- Cost model integration
- Market regime classification

Outputs:
    ./twap_outputs/policy_compare_results.csv
    ./twap_outputs/policy_compare_summary.csv
    ./twap_outputs/twap_bp_distribution.png
    ./twap_outputs/limit_part_sweep_summary.csv
    ./twap_outputs/limit_part_tradeoff.png
    ./twap_outputs/policy_performance.png
    ./twap_outputs/signal_history.pkl  <-- For XGBoost training
"""

import os
import sys
import time
import math
import logging
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from env_bs import SimplifiedEnvConfig, make_simplified_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


class StrategySafetyMonitor:
    def __init__(self):
        self.completion_rates = {}
        self.execution_metrics = {}
    
    def record_episode(self, policy_name, completed, total_steps, final_inventory):
        if policy_name not in self.completion_rates:
            self.completion_rates[policy_name] = []
            self.execution_metrics[policy_name] = []
        
        self.completion_rates[policy_name].append(completed)
        self.execution_metrics[policy_name].append({
            'total_steps': total_steps,
            'final_inventory': final_inventory,
            'completion_rate': 1.0 - final_inventory if hasattr(final_inventory, '__float__') else 0.0
        })
    
    def get_completion_summary(self):
        summary = {}
        for policy, completions in self.completion_rates.items():
            completion_rate = np.mean(completions) if completions else 0.0
            summary[policy] = {
                'completion_rate': completion_rate,
                'total_episodes': len(completions),
                'successful_episodes': sum(completions)
            }
        return summary


def safe_reset(env, code=None, date=None, start_index=None):
    try:
        return env.reset(code=code, date=date, start_index=start_index)
    except TypeError:
        try:
            return env.reset(code, date, start_index)
        except TypeError:
            return env.reset()


def steps_left(env):
    try:
        return max(1, env.data.end_index - env.data.current_index + 1)
    except Exception:
        return max(1, getattr(env.data, "end_index", 1) - getattr(env.data, "current_index", 0) + 1)


def get_action_threshold(env):
    try:
        return getattr(env, "action_threshold", 
                      env.config.action_threshold if hasattr(env.config, "action_threshold") else 0.2)
    except Exception:
        return 0.2


# ======================
# COST MODEL HELPERS
# ======================

def estimate_limit_fill_prob(env, place_level):
    try:
        action_threshold = get_action_threshold(env)
        if (1.0 - action_threshold) < 1e-8:
            rel_level = 0.0
        else:
            rel_level = (place_level - action_threshold) / (1.0 - action_threshold)
        level_idx = max(1, min(5, int(rel_level * 5) + 1))
        depth_ahead = sum(env.data.obtain_level("askVolume", i) for i in range(1, level_idx))
        own_size = getattr(env, "current_inventory", 1.0) * 0.1
        fill_prob = min(1.0, own_size / (depth_ahead + 1e-6))
        return fill_prob
    except:
        return 0.5


def estimate_market_impact(env, exec_frac):
    try:
        depth = sum(env.data.obtain_level("askVolume", i) for i in range(1, 6))
        impact_bp = exec_frac * 0.5 * (1000 / (depth + 1))  # heuristic
        return impact_bp
    except:
        return exec_frac * 0.1


# ======================
# STRATEGIES
# ======================

def market_twap_policy(env, step, params=None):
    if params is None:
        params = {}
    action_threshold = get_action_threshold(env)
    left = steps_left(env)
    desired_frac_of_remaining = 1.0 / left
    exec_int = action_threshold + desired_frac_of_remaining * (1.0 - action_threshold)
    return np.array([0.0, float(min(1.0, exec_int))], dtype=np.float32)


def mixed_twap_policy(env, step, params=None):
    if params is None:
        params = {}
    ideal_limit_part = float(params.get("limit_part", 0.35))
    action_threshold = get_action_threshold(env)
    left = steps_left(env)

    if left <= 1:
        return np.array([0.0, 1.0], dtype=np.float32)

    try:
        depth = sum(env.data.obtain_level("bidVolume", i) + env.data.obtain_level("askVolume", i) for i in range(1, 6))
        liquidity_score = min(1.0, depth / 5000.0)
    except:
        liquidity_score = 0.5

    current_inventory = getattr(env, "current_inventory", 1.0)
    initial_inventory = getattr(env, "initial_inventory", 1.0)
    inv_ratio = current_inventory / initial_inventory if initial_inventory > 0 else 1.0

    tail_risk = (1.0 - liquidity_score) * 0.5 + (1.0 if inv_ratio > 0.3 else 0.0) * 0.5
    tail_risk = np.clip(tail_risk, 0, 1)

    if left <= 5:
        decay_factor = (left - 1) / 5.0
        adjusted_limit_part = ideal_limit_part * decay_factor * (1.0 - tail_risk)
    else:
        adjusted_limit_part = ideal_limit_part

    max_limit_tail = 0.3 * (1.0 - tail_risk)
    adjusted_limit_part = min(adjusted_limit_part, max_limit_tail)

    baseline_market_per_step = 1.0 / float(left)
    max_allowed_limit = max(0.0, 1.0 - baseline_market_per_step)
    adjusted_limit_part = min(adjusted_limit_part, max_allowed_limit)

    market_total_frac = 1.0 - adjusted_limit_part
    per_step_market_frac = market_total_frac / max(1, left)
    if per_step_market_frac < baseline_market_per_step:
        per_step_market_frac = baseline_market_per_step
        adjusted_limit_part = max(0.0, 1.0 - per_step_market_frac * left)

    exec_int = action_threshold + per_step_market_frac * (1.0 - action_threshold)
    place_level = action_threshold + adjusted_limit_part * (1.0 - action_threshold)

    return np.array([np.clip(place_level, 0, 1), np.clip(exec_int, 0, 1)], dtype=np.float32)


def adaptive_twap_policy(env, step, params=None):
    if params is None:
        params = {}
    base_limit = float(params.get("base_limit", 0.35))
    max_limit = float(params.get("max_limit", 0.75))
    spread_tick_thresh = float(params.get("spread_tick_thresh", 1e-6))
    imbalance_thresh = float(params.get("imbalance_thresh", 0.1))
    vol_window = int(params.get("vol_window", 10))

    left = steps_left(env)
    if left <= 1:
        return np.array([0.0, 1.0], dtype=np.float32)

    try:
        bid = env.data.obtain_level("bidPrice", 1)
        ask = env.data.obtain_level("askPrice", 1)
        bid_vol = env.data.obtain_level("bidVolume", 1)
        ask_vol = env.data.obtain_level("askVolume", 1)
        mid = (ask + bid) / 2.0 if (ask and bid) else np.nan

        history = []
        idx = env.data.current_index
        for i in range(max(0, idx - vol_window), idx + 1):
            try:
                b = env.data.raw_data.loc[i, "bidPrice"]
                a = env.data.raw_data.loc[i, "askPrice"]
                if b and a:
                    history.append((a + b) / 2.0)
            except Exception:
                continue
        volatility = np.std(history) / mid if history and mid else 0.0
    except Exception:
        volatility = 0.0
        bid = ask = bid_vol = ask_vol = None

    imbalance = 0.0
    if bid_vol and ask_vol and (bid_vol + ask_vol) > 0:
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)

    limit_part = base_limit

    if bid and ask and (ask - bid) >= spread_tick_thresh:
        limit_part += 0.12
    if imbalance >= imbalance_thresh:
        limit_part += 0.12
    if imbalance <= -imbalance_thresh:
        limit_part -= 0.25

    vol_thresh = 0.001
    if volatility > vol_thresh:
        vol_penalty = min(0.3, (volatility - vol_thresh) / vol_thresh * 0.2)
        limit_part -= vol_penalty
    elif volatility < vol_thresh * 0.5 and volatility > 0:
        limit_part += 0.08

    limit_part = np.clip(limit_part, 0.05, max_limit)

    place_level_candidate = get_action_threshold(env) + limit_part * (1.0 - get_action_threshold(env))
    fill_prob = estimate_limit_fill_prob(env, place_level_candidate)
    impact_cost = estimate_market_impact(env, (1.0 - limit_part) / left)
    if fill_prob < 0.3 and impact_cost < 0.5:
        limit_part *= 0.7

    baseline_market_per_step = 1.0 / float(left)
    limit_part = min(limit_part, 1.0 - baseline_market_per_step)
    market_frac = 1.0 - limit_part
    per_step_market_frac = market_frac / left
    if per_step_market_frac < baseline_market_per_step:
        per_step_market_frac = baseline_market_per_step
        limit_part = max(0.0, 1.0 - per_step_market_frac * left)

    action_threshold = get_action_threshold(env)
    place_level = action_threshold + limit_part * (1.0 - action_threshold)
    exec_int = action_threshold + per_step_market_frac * (1.0 - action_threshold)

    return np.array([np.clip(place_level, 0, 1), np.clip(exec_int, 0, 1)], dtype=np.float32)


def smart_timing_twap_policy(env, step, params=None):
    if params is None:
        params = {}
    base_limit = float(params.get("base_limit", 0.4))
    action_threshold = get_action_threshold(env)
    left = steps_left(env)
    if left <= 1:
        return np.array([0.0, 1.0], dtype=np.float32)

    try:
        current_time = getattr(env.data, 'current_time', None)
        if current_time:
            hour = int(current_time.split(':')[0])
            is_open_close = hour in [9, 10, 14, 15]
        else:
            is_open_close = False
    except:
        is_open_close = False

    try:
        bid1 = env.data.obtain_level("bidPrice", 1)
        ask1 = env.data.obtain_level("askPrice", 1)
        bid_depth = sum(env.data.obtain_level("bidVolume", i) for i in range(1, 6))
        ask_depth = sum(env.data.obtain_level("askVolume", i) for i in range(1, 6))
        spread = (ask1 - bid1) if (ask1 and bid1) else np.inf
        depth = bid_depth + ask_depth
    except:
        spread = np.inf
        depth = 0

    favorable = (spread < 1e-4) and (depth > 1000) and (not is_open_close)
    if favorable:
        limit_part = min(0.8, base_limit + 0.2)
    else:
        limit_part = max(0.1, base_limit - 0.2)

    baseline_market_per_step = 1.0 / float(left)
    limit_part = min(limit_part, 1.0 - baseline_market_per_step)
    market_frac = 1.0 - limit_part
    per_step_market_frac = market_frac / left
    if per_step_market_frac < baseline_market_per_step:
        per_step_market_frac = baseline_market_per_step
        limit_part = max(0.0, 1.0 - per_step_market_frac * left)

    place_level = action_threshold + limit_part * (1.0 - action_threshold)
    exec_int = action_threshold + per_step_market_frac * (1.0 - action_threshold)

    return np.array([np.clip(place_level, 0, 1), np.clip(exec_int, 0, 1)], dtype=np.float32)


# ======================
# ENHANCED MULTI-SCALE INTELLIGENT EXECUTION ENGINE
# ======================

def enhanced_multi_scale_twap_policy(env, step, params=None):
    """
    智能多尺度执行引擎 v2.0
    """
    if params is None:
        params = {}
    
    base_limit = float(params.get("base_limit", 0.5))
    signal_sensitivity = float(params.get("signal_sensitivity", 0.4))
    max_signal_risk_budget = float(params.get("max_signal_risk_budget", 0.2))
    left = steps_left(env)
    
    if left <= 1:
        return np.array([0.0, 1.0], dtype=np.float32)

    # === 1. 多信号采集 ===
    signals = _extract_micro_signals(env, step)
    
    # === 2. 市场状态检测 ===
    market_state = _detect_market_state(env)
    
    # === 3. 手动信号融合（XGBoost will replace this later）===
    signal_weights = {
        'ofi': 0.4,
        'momentum': 0.25,
        'liquidity': 0.2,
        'consumption': 0.15
    }
    composite_signal = sum(signals.get(sig, 0) * weight for sig, weight in signal_weights.items())
    composite_signal = np.tanh(composite_signal)  # saturate

    # === 4. 自适应参数调整 ===
    sensitivity = signal_sensitivity
    if market_state['high_volatility']:
        sensitivity *= 0.7
    if market_state['low_liquidity']:
        sensitivity *= 0.8
    if market_state['opening'] or market_state['closing']:
        sensitivity *= 0.6
    sensitivity = np.clip(sensitivity, 0.1, 0.6)

    time_decay = min(1.0, left / 10.0)
    effective_sensitivity = sensitivity * time_decay

    # === 5. 执行路径监控 ===
    current_inventory = getattr(env, "current_inventory", 1.0)
    initial_inventory = getattr(env, "initial_inventory", 1.0)
    executed_frac = 1.0 - (current_inventory / initial_inventory) if initial_inventory > 0 else 0.0
    planned_executed_frac = 1.0 - (left / getattr(env, "total_steps", left))
    deviation = executed_frac - planned_executed_frac
    max_deviation = 0.15

    if deviation > max_deviation:
        effective_sensitivity *= 0.3
    elif deviation < -max_deviation:
        effective_sensitivity *= 0.7

    # === 6. 风险预算分配 ===
    planned_exec_frac = 1.0 / float(left)
    base_market_frac = 0.8 * planned_exec_frac
    signal_adjustment = np.clip(composite_signal * effective_sensitivity, -max_signal_risk_budget, max_signal_risk_budget)
    total_market_frac = base_market_frac + signal_adjustment * planned_exec_frac
    total_market_frac = np.clip(total_market_frac, 0.0, 1.0)
    
    limit_part = 1.0 - total_market_frac * left
    limit_part = np.clip(limit_part, 0.1, 0.8)

    # === 7. 完成保障 ===
    baseline_market_per_step = 1.0 / float(left)
    if total_market_frac < baseline_market_per_step:
        total_market_frac = baseline_market_per_step
        limit_part = max(0.0, 1.0 - total_market_frac * left)

    # === 8. 映射到动作空间 ===
    action_threshold = get_action_threshold(env)
    place_level = action_threshold + limit_part * (1.0 - action_threshold)
    exec_int = action_threshold + total_market_frac * (1.0 - action_threshold)

    # === 9. 记录信号（供 XGBoost 训练）===
    if not hasattr(env, '_signal_log'):
        env._signal_log = []
    env._signal_log.append({
        'signals': signals.copy(),
        'market_state': market_state.copy(),
        'steps_left': left,
        'inventory_ratio': current_inventory / initial_inventory if initial_inventory > 0 else 1.0,
        'composite_signal': composite_signal,
        'total_market_frac': total_market_frac
    })

    return np.array([np.clip(place_level, 0, 1), np.clip(exec_int, 0, 1)], dtype=np.float32)


def _extract_micro_signals(env, step: int) -> dict:
    signals = {}
    try:
        bid_vol_3 = sum(env.data.obtain_level("bidVolume", i) for i in range(1, 4))
        ask_vol_3 = sum(env.data.obtain_level("askVolume", i) for i in range(1, 4))
        ofi = (bid_vol_3 - ask_vol_3) / (bid_vol_3 + ask_vol_3 + 1e-6) if (bid_vol_3 + ask_vol_3) > 0 else 0.0
        signals['ofi'] = np.clip(ofi, -1.0, 1.0)

        mids = []
        current_idx = env.data.current_index
        for i in range(max(0, current_idx - 5), current_idx + 1):
            try:
                b = env.data.raw_data.loc[i, "bidPrice"]
                a = env.data.raw_data.loc[i, "askPrice"]
                if b and a:
                    mids.append((a + b) / 2.0)
            except:
                continue
        if len(mids) >= 2:
            returns = np.diff(np.log(mids))
            momentum = np.mean(returns[-3:]) if len(returns) >= 3 else np.mean(returns)
            signals['momentum'] = np.clip(momentum * 1000, -1.0, 1.0)
        else:
            signals['momentum'] = 0.0

        spread = env.data.obtain_level("askPrice", 1) - env.data.obtain_level("bidPrice", 1)
        depth = bid_vol_3 + ask_vol_3
        avg_spread = getattr(env, 'rolling_avg_spread', spread)
        norm_spread = spread / (avg_spread + 1e-8)
        liquidity_score = 1.0 / (1.0 + norm_spread) * min(1.0, depth / 1000.0)
        signals['liquidity'] = np.clip(liquidity_score, 0.0, 1.0)

        last_trade_size = getattr(env.data, 'last_trade_size', 0)
        avg_volume = getattr(env, 'rolling_avg_volume', 100)
        consumption_event = 1.0 if last_trade_size > 3 * avg_volume else 0.0
        signals['consumption'] = consumption_event

    except Exception as e:
        signals = {'ofi': 0.0, 'momentum': 0.0, 'liquidity': 0.5, 'consumption': 0.0}
    return signals


def _detect_market_state(env) -> dict:
    state = {
        'high_volatility': False,
        'low_liquidity': False,
        'trending': False,
        'opening': False,
        'closing': False
    }
    try:
        history = []
        idx = env.data.current_index
        for i in range(max(0, idx - 20), idx + 1):
            try:
                mid = (env.data.raw_data.loc[i, "askPrice"] + env.data.raw_data.loc[i, "bidPrice"]) / 2
                history.append(mid)
            except:
                continue
        if len(history) > 5:
            vol = np.std(np.diff(np.log(history))) * np.sqrt(252)
            state['high_volatility'] = vol > 0.02

        depth = sum(env.data.obtain_level("bidVolume", i) + env.data.obtain_level("askVolume", i) for i in range(1, 6))
        state['low_liquidity'] = depth < 500

        try:
            current_time = getattr(env.data, 'current_time', "12:00:00")
            hour = int(current_time.split(':')[0])
            state['opening'] = hour in [9, 10]
            state['closing'] = hour in [14, 15]
        except:
            pass

        if len(history) > 10:
            trend = (history[-1] / history[0] - 1)
            state['trending'] = abs(trend) > 0.01

    except:
        pass
    return state


# ======================
# EPISODE & METRICS
# ======================

def run_one_episode(env, policy_fn, policy_params=None, code=None, date=None, safety_monitor=None):
    obs = safe_reset(env, code=code, date=date)
    step = env.data.current_index
    done = False
    
    # Clear signal log for new episode
    if hasattr(env, '_signal_log'):
        delattr(env, '_signal_log')
    
    while not done:
        action = policy_fn(env, step, policy_params)
        market_state, private_state, reward, done, info = env.step(action)
        step = env.data.current_index
    
    metrics = env.get_trading_metrics()
    
    # Add VWAP benchmark
    try:
        executed = getattr(env, 'execution_log', [])
        if executed and "vwap_benchmark" not in metrics:
            total_vol = sum(v for _, v in executed)
            if total_vol > 0:
                strategy_vwap = sum(p * v for p, v in executed) / total_vol
                start_idx = env.data.start_index
                end_idx = env.data.current_index
                market_data = env.data.raw_data.iloc[start_idx:end_idx+1]
                if 'volume' in market_data.columns and 'price' in market_data.columns:
                    market_vwap = (market_data["price"] * market_data["volume"]).sum() / market_data["volume"].sum()
                    metrics["vwap_slippage_bp"] = (strategy_vwap / market_vwap - 1) * 1e4
    except Exception as e:
        logging.debug(f"VWAP calc failed: {e}")

    # Save signal history for enhanced strategy
    if (hasattr(env, '_signal_log') and 
        policy_fn.__name__ == 'enhanced_multi_scale_twap_policy'):
        signal_history = getattr(env, '_full_signal_history', {})
        key = (getattr(env, 'current_code', 'unknown'), getattr(env, 'current_date', 'unknown'))
        if key not in signal_history:
            signal_history[key] = []
        signal_history[key].append(env._signal_log)
        env._full_signal_history = signal_history

    if safety_monitor is not None:
        final_inventory = getattr(env, "current_inventory", 0.0)
        completed = final_inventory <= 0.01
        safety_monitor.record_episode(
            policy_fn.__name__, completed, step, final_inventory
        )
    
    return metrics


def compare_policies(env, policies, samples_per_pair=3, output_dir="twap_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    safety_monitor = StrategySafetyMonitor()
    
    code_date_list = getattr(env, "valid_code_date_list", None)
    if code_date_list is None or len(code_date_list) == 0:
        raise RuntimeError("env.valid_code_date_list is empty.")
    
    rows = []
    start_time = time.time()
    
    for code, date in tqdm(code_date_list, desc="Trading pairs"):
        for s in range(samples_per_pair):
            for name, fn, params in policies:
                try:
                    metrics = run_one_episode(
                        env, fn, params, code=code, date=date, 
                        safety_monitor=safety_monitor
                    )
                    
                    bp = -float(metrics.get("price_performance_bp", np.nan))
                    cancel_rate = float(metrics.get("cancel_rate", np.nan))
                    exec_vol = float(metrics.get("total_executed", np.nan))
                    vwap_slippage = float(metrics.get("vwap_slippage_bp", np.nan))
                    
                    rows.append({
                        "code": code,
                        "date": date,
                        "sample": s,
                        "policy": name,
                        "price_performance_bp": bp,
                        "vwap_slippage_bp": vwap_slippage,
                        "cancel_rate": cancel_rate,
                        "total_executed": exec_vol,
                        "error": None
                    })
                    
                except Exception as e:
                    logging.warning(f"Error running {name} on {code}-{date}: {str(e)}")
                    rows.append({
                        "code": code,
                        "date": date,
                        "sample": s,
                        "policy": name,
                        "price_performance_bp": np.nan,
                        "vwap_slippage_bp": np.nan,
                        "cancel_rate": np.nan,
                        "total_executed": np.nan,
                        "error": str(e)[:200]
                    })
    
    df = pd.DataFrame(rows)
    execution_time = time.time() - start_time
    
    csv_path = os.path.join(output_dir, "policy_compare_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Save signal history if exists
    if hasattr(env, '_full_signal_history'):
        signal_path = os.path.join(output_dir, "signal_history.pkl")
        with open(signal_path, "wb") as f:
            pickle.dump(env._full_signal_history, f)
        logging.info(f"Signal history saved to {signal_path}")
    
    summary = df.groupby("policy").agg(
        mean_bp=("price_performance_bp", "mean"),
        std_bp=("price_performance_bp", "std"),
        median_bp=("price_performance_bp", "median"),
        mean_vwap_slippage=("vwap_slippage_bp", "mean"),
        mean_cancel=("cancel_rate", "mean"),
        completion_rate=("total_executed", lambda x: (x >= 0.95).mean()),
        count=("price_performance_bp", "count"),
        success_count=("price_performance_bp", lambda x: x.notna().sum())
    ).reset_index()
    
    summary_path = os.path.join(output_dir, "policy_compare_summary.csv")
    summary.to_csv(summary_path, index=False)
    
    generate_visualizations(df, summary, output_dir)
    
    logging.info(f"Comparison completed in {execution_time:.2f} seconds")
    return df, summary


def generate_visualizations(df, summary, output_dir):
    valid_data = df[df["price_performance_bp"].notna()]
    if len(valid_data) == 0:
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(data=valid_data, x="price_performance_bp", bins=30, alpha=0.7, kde=True)
    mean_val = valid_data["price_performance_bp"].mean()
    median_val = valid_data["price_performance_bp"].median()
    plt.axvline(mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}")
    plt.axvline(median_val, color="green", linestyle="--", label=f"Median: {median_val:.2f}")
    plt.xlabel("Price Performance (bp) - Positive is Better")
    plt.ylabel("Frequency")
    plt.title("TWAP Strategy Performance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "twap_bp_distribution.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=valid_data, x="policy", y="price_performance_bp")
    policy_means = valid_data.groupby("policy")["price_performance_bp"].mean()
    for i, (policy, mean_val) in enumerate(policy_means.items()):
        plt.text(i, mean_val, f'{mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=45)
    plt.title("Performance by Policy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "policy_performance.png"), dpi=300)
    plt.close()


def run_limit_part_sweep(env, limit_parts, samples_per_pair=2, output_dir="twap_outputs"):
    policies = [
        (f"mixed_twap_lp_{lp:.1f}", mixed_twap_policy, {"limit_part": lp})
        for lp in limit_parts
    ]
    df, summary = compare_policies(env, policies, samples_per_pair, output_dir)
    
    sweep_summary = summary[["policy", "mean_bp", "completion_rate"]].copy()
    sweep_summary["limit_part"] = sweep_summary["policy"].str.extract(r"lp_(\d\.\d)")[0].astype(float)
    sweep_summary.to_csv(os.path.join(output_dir, "limit_part_sweep_summary.csv"), index=False)

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.plot(sweep_summary["limit_part"], sweep_summary["mean_bp"], 'o-', color='blue', label="Mean BP (bp)")
    ax1.set_xlabel("limit_part")
    ax1.set_ylabel("Mean BP", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(sweep_summary["limit_part"], sweep_summary["completion_rate"], 's--', color='orange', label="Completion Rate")
    ax2.set_ylabel("Completion Rate", color='orange')
    plt.title("Trade-off: Price Performance vs Completion Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "limit_part_tradeoff.png"), dpi=300)
    plt.close()
    
    return df, sweep_summary


def classify_market_regime(env, code, date):
    try:
        data = env.data.raw_data
        if 'price' not in data.columns:
            return "unknown"
        log_ret = np.log(data["price"]).diff().dropna()
        vol = log_ret.std() * np.sqrt(252)
        volume = data["volume"].mean() if "volume" in data.columns else 1.0
        trend = (data["price"].iloc[-1] / data["price"].iloc[0] - 1) if len(data) > 1 else 0.0
        
        if vol > 0.02:
            regime = "high_vol"
        elif vol < 0.005:
            regime = "low_vol"
        else:
            regime = "normal"
        
        if abs(trend) > 0.02:
            regime += "_trending"
        else:
            regime += "_mean_reverting"
        return regime
    except:
        return "unknown"


def main():
    logging.info("Starting Enhanced TWAP Strategy Comparison (v2)...")
    
    config = SimplifiedEnvConfig()
    env = make_simplified_env(config)
    
    # Full policy suite - NOTE: enhanced strategy name MUST contain "enhanced"
    policies = [
        ("market_twap", market_twap_policy, None),
        ("mixed_twap_conservative", mixed_twap_policy, {"limit_part": 0.35}),
        ("mixed_twap_aggressive", mixed_twap_policy, {"limit_part": 0.6}),
        ("adaptive_twap", adaptive_twap_policy, {"base_limit": 0.5, "max_limit": 0.9}),
        ("smart_timing_twap", smart_timing_twap_policy, {"base_limit": 0.4}),
        ("enhanced_multi_scale_twap", enhanced_multi_scale_twap_policy, {
            "base_limit": 0.5,
            "signal_sensitivity": 0.4,
            "max_signal_risk_budget": 0.2
        }),
    ]
    
    samples_per_pair = 2
    df, summary = compare_policies(
        env, policies, 
        samples_per_pair=samples_per_pair, 
        output_dir="twap_outputs"
    )
    
    print("\n" + "="*60)
    print("ENHANCED TWAP COMPARISON RESULTS")
    print("="*60)
    print(f"\nTotal episodes: {len(df)}")
    print(f"Successful episodes: {df['price_performance_bp'].notna().sum()}")
    
    cols = ["policy", "mean_bp", "std_bp", "median_bp", "mean_vwap_slippage", "mean_cancel", "completion_rate"]
    print("\nPerformance Summary (bp) - Positive is Better:")
    print(summary[cols].to_string(index=False))
    
    print("\nDetailed results saved to: twap_outputs/")
    
    if not summary.empty:
        best_idx = summary['mean_bp'].idxmax()
        best_policy = summary.loc[best_idx, 'policy']
        best_bp = summary.loc[best_idx, 'mean_bp']
        print(f"\n🏆 Best strategy: {best_policy} ({best_bp:.2f} bp)")

    # Run parameter sweep
    print("\nRunning limit_part parameter sweep...")
    limit_parts = [0.2, 0.3, 0.4, 0.5, 0.6]
    run_limit_part_sweep(env, limit_parts, samples_per_pair=2)


if __name__ == "__main__":
    main()
