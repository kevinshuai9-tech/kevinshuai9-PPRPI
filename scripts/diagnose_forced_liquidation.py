import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
诊断脚本：统计 PPO checkpoint 在真实环境里触发强制清算(_force_liquidation)的频率，
用来确认 training_curves.png 里 Reward 曲线的深谷（-150~-215）是不是主要由强平惩罚
(forced_pen，最多 -100/episode) 造成，而不是订单量 Q 没被归一化。

必须在能访问 config/const_path_ob.py 里 path_pkl_data / path_orders 真实行情数据的机器上跑
（本地 Mac 上没有这份数据，无法直接跑）。

用法：
    python scripts/diagnose_forced_liquidation.py \
        --checkpoints results/ppo/model_episode_1000.pth results/ppo/model_episode_4000.pth \
        --episodes 200
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

from ppo_bs import PPOTrainer, EnvironmentUtils
from env_bs import SimplifiedEnvConfig


def probe_checkpoint(ckpt_path, config, device, n_episodes, max_steps):
    trainer = PPOTrainer(config, os.path.join('results', '_forced_liq_probe'), device)
    if not trainer.load_model(ckpt_path):
        return None

    n_forced = 0
    forced_fracs = []
    rewards_total = []

    for _ in range(n_episodes):
        reset_ret = trainer.env.reset()
        market_state, private_state = reset_ret[0], reset_ret[1]
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            action, _, _ = trainer.agent.select_action(market_state, private_state)
            step_ret = trainer.env.step(action)
            market_state, private_state, reward, done, info = EnvironmentUtils.parse_step_return(
                step_ret, len(private_state)
            )
            ep_reward += float(reward)
            steps += 1

        trades = getattr(trainer.env, 'trade_history', [])
        liquidation_trades = [t for t in trades if t.get('status') == 'liquidation']
        total_q = getattr(trainer.env, 'total_quantity', 0)

        if liquidation_trades:
            n_forced += 1
            forced_fracs.append(liquidation_trades[0].get('quantity', 0) / max(1, total_q))

        rewards_total.append(ep_reward)

    rewards_total = np.array(rewards_total)
    return {
        'forced_rate': n_forced / n_episodes,
        'n_forced': n_forced,
        'n_episodes': n_episodes,
        'avg_forced_frac': float(np.mean(forced_fracs)) if forced_fracs else 0.0,
        'reward_mean': float(rewards_total.mean()),
        'reward_std': float(rewards_total.std()),
        'reward_min': float(rewards_total.min()),
        'reward_p10': float(np.percentile(rewards_total, 10)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True,
                         help='一个或多个 .pth checkpoint 路径')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=400)
    args = parser.parse_args()

    device = torch.device('cpu')
    config = SimplifiedEnvConfig()

    print(f"{'checkpoint':<35}{'forced_rate':>12}{'avg_forced_frac':>18}"
          f"{'reward_mean':>14}{'reward_std':>12}{'reward_p10':>12}{'reward_min':>12}")
    for ckpt in args.checkpoints:
        stats = probe_checkpoint(ckpt, config, device, args.episodes, args.max_steps)
        if stats is None:
            print(f"{os.path.basename(ckpt):<35} load failed")
            continue
        print(f"{os.path.basename(ckpt):<35}{stats['forced_rate']:>11.1%} "
              f"{stats['avg_forced_frac']:>17.1%} "
              f"{stats['reward_mean']:>14.2f}{stats['reward_std']:>12.2f}"
              f"{stats['reward_p10']:>12.2f}{stats['reward_min']:>12.2f}")


if __name__ == '__main__':
    main()
