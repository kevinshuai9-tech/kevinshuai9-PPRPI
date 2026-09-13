import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
七算法对比实验脚本（论文完整对比框架）

基线（无需训练）
─────────────────────────────────────────────────────────────────
TWAP           : 等时间均匀执行，标准行业基线
Adaptive TWAP  : 基于微结构信号自适应调整，更强的规则基线

RL 方法
─────────────────────────────────────────────────────────────────
PPO            : on-policy，连续动作，策略梯度
DQN            : off-policy，离散动作，Q-learning
QR-DQN (mean)  : 分布式 Q，风险中性（期望 CVaR）
QR-DQN (CVaR)  : 分布式 Q，CVaR 近似（下尾分位数均值），风险规避
FZ-DQN (CVaR)  : 【论文方法】FZ0 联合损失，VaR+CVaR 双头直接估计，严格一致

论文叙述层次
─────────────────────────────────────────────────────────────────
TWAP < Adaptive TWAP < DQN ≈ PPO < QR-DQN(mean) < QR-DQN(CVaR) < FZ-DQN
在 price_performance_bp 上不一定最高，但在尾部风险（CVaR）和稳健性上最优

用法：
    # 训练全部 RL 算法并统一对比（含 TWAP 基线）
    python scripts/compare_agents.py --mode train_all

    # 只加载已有模型做评估对比（跳过 RL 训练，仍跑 TWAP 评估）
    python scripts/compare_agents.py --mode eval_only \
        --ppo_model    results/ppo_xxx/ppo_model_final.pth \
        --dqn_model    results/dqn_xxx/dqn_final.pth \
        --qrdqn_model  results/qrdqn_mean_xxx/qrdqn_final.pth \
        --qrcvar_model results/qrdqn_cvar0.1_xxx/qrdqn_final.pth \
        --fzdqn_model  results/fzdqn_cvar0.1_xxx/fzdqn_final.pth
"""

import argparse
import numpy as np
import torch
import os
import json
import csv
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_bs    import SimplifiedEnvConfig, SimplifiedExecutionEnv
from ppo_bs    import PPOTrainer
from dqn       import DQNTrainer, idx_to_action
from qr_dqn    import QRDQNTrainer
from fz_rl     import FZDQNTrainer
from twap      import twap_action
from twap_plus import enhanced_multi_scale_twap_policy


# ─── 单个 agent 评估 ──────────────────────────────────────────────────────────
def evaluate_agent(env, run_episode_fn, n_episodes: int = 20) -> dict:
    """
    通用评估接口：接受一个 run_episode_fn(env) → (reward, metrics)
    收集 n_episodes 次结果，返回汇总统计。
    """
    rewards, bps, cancel_rates, passive_rates = [], [], [], []

    for _ in range(n_episodes):
        reward, metrics = run_episode_fn(env)
        rewards.append(reward)
        bps.append(metrics.get('price_performance_bp', 0.0))
        cancel_rates.append(metrics.get('cancel_rate', 0.0))
        passive_rates.append(metrics.get('passive_rate', 0.0))

    return {
        'reward_mean':       float(np.mean(rewards)),
        'reward_std':        float(np.std(rewards)),
        'bp_mean':           float(np.mean(bps)),
        'bp_std':            float(np.std(bps)),
        'cancel_rate_mean':  float(np.mean(cancel_rates)),
        'passive_rate_mean': float(np.mean(passive_rates)),
        'n_episodes':        n_episodes,
    }


# ─── TWAP 基线（无需训练）────────────────────────────────────────────────────
def make_twap_episode_fn():
    """标准 TWAP：等时间均匀执行，place_level=0（市价单），intensity=1/remaining"""
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        step = env.data.current_index
        done = False
        total_reward = 0.0
        while not done:
            action = twap_action(env, step)
            ms, ps, reward, done, _ = env.step(action)
            step = env.data.current_index
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


def make_enhanced_twap_episode_fn():
    """
    Enhanced Multi-Scale TWAP：OFI/动量/流动性/成交消耗 四路信号融合
    是 twap_plus.py 中最强的规则基线，对应论文 Section 5 的 Enhanced TWAP baseline
    """
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        step = env.data.current_index
        done = False
        total_reward = 0.0
        params = {
            'base_limit':            0.5,
            'signal_sensitivity':    0.4,
            'max_signal_risk_budget': 0.2,
        }
        while not done:
            action = enhanced_multi_scale_twap_policy(env, step, params)
            ms, ps, reward, done, _ = env.step(action)
            step = env.data.current_index
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


# ─── PPO 评估包装 ─────────────────────────────────────────────────────────────
def make_ppo_episode_fn(agent, device):
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        done = False
        total_reward = 0.0
        while not done:
            action, _, _ = agent.select_action(ms, ps)
            ret = env.step(action)
            ms, ps, reward, done, _ = ret
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


# ─── DQN 评估包装 ─────────────────────────────────────────────────────────────
def make_dqn_episode_fn(agent, device):
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                ms_t = torch.FloatTensor(ms).unsqueeze(0).to(device)
                ps_t = torch.FloatTensor(ps).unsqueeze(0).to(device)
                act_idx = int(agent.online_net(ms_t, ps_t).argmax(1).item())
            action = idx_to_action(act_idx)
            ms, ps, reward, done, _ = env.step(action)
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


# ─── FZ-DQN 评估包装（论文方法：直接用 e 头 CVaR 选动作）─────────────────────
def make_fzdqn_episode_fn(agent, device):
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                ms_t = torch.FloatTensor(ms).unsqueeze(0).to(device)
                ps_t = torch.FloatTensor(ps).unsqueeze(0).to(device)
                _, e = agent.online_net(ms_t, ps_t)
                act_idx = int(agent._cvar_scores(e).argmax(1).item())
            action = idx_to_action(act_idx)
            ms, ps, reward, done, _ = env.step(action)
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


# ─── QR-DQN 评估包装 ──────────────────────────────────────────────────────────
def make_qrdqn_episode_fn(agent, device):
    def run(env):
        reset_ret = env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                ms_t = torch.FloatTensor(ms).unsqueeze(0).to(device)
                ps_t = torch.FloatTensor(ps).unsqueeze(0).to(device)
                z = agent.online_net(ms_t, ps_t)
                act_idx = int(agent._action_scores(z).argmax(1).item())
            action = idx_to_action(act_idx)
            ms, ps, reward, done, _ = env.step(action)
            total_reward += float(reward)
        return total_reward, env.get_trading_metrics()
    return run


# ─── 对比报告输出 ─────────────────────────────────────────────────────────────
def print_comparison_table(results: dict):
    header = f"{'算法':<20} {'bp均值':>10} {'bp标准差':>10} {'累计奖励':>10} {'撤单率':>10} {'被动成交率':>12}"
    print("\n" + "=" * 75)
    print("对比框架：TWAP → Enhanced TWAP → RL 方法 → FZ-DQN（论文方法）")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<20} {r['bp_mean']:>+10.2f} {r['bp_std']:>10.2f} "
              f"{r['reward_mean']:>+10.4f} {r['cancel_rate_mean']:>10.3f} "
              f"{r['passive_rate_mean']:>12.3f}")
    print("=" * 75)
    print("bp > 0 表示优于市场 VWAP（对卖方有利）\n")


def save_comparison_charts(results: dict, result_dir: str):
    names = list(results.keys())
    bp_means = [results[n]['bp_mean'] for n in names]
    bp_stds  = [results[n]['bp_std']  for n in names]
    palette  = ['#607D8B', '#9E9E9E',  # TWAP 灰色系（规则基线）
                '#2196F3', '#4CAF50',  # DQN 蓝、PPO 绿
                '#FF9800', '#E91E63',  # QR-DQN 橙、QR-CVaR 粉
                '#9C27B0']             # FZ-DQN 紫（论文方法）
    colors   = palette[:len(names)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 价格表现 (bp)
    bars = axes[0].bar(names, bp_means, yerr=bp_stds, capsize=5,
                       color=colors, alpha=0.8)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.6, label='VWAP baseline')
    axes[0].set_title('Price Performance (bp)\n高于 0 优于 VWAP')
    axes[0].set_ylabel('bp')
    for bar, v in zip(bars, bp_means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{v:+.1f}', ha='center', va='bottom', fontsize=9)

    # 撤单率
    cancel = [results[n]['cancel_rate_mean'] for n in names]
    axes[1].bar(names, cancel, color=colors, alpha=0.8)
    axes[1].set_title('Cancel Rate\n撤单率（越低越好）')
    axes[1].set_ylim(0, 1.1)

    # 被动成交率
    passive = [results[n]['passive_rate_mean'] for n in names]
    axes[2].bar(names, passive, color=colors, alpha=0.8)
    axes[2].set_title('Passive Fill Rate\n被动成交率（越高越省成本）')
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    path = os.path.join(result_dir, 'comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"对比图已保存: {path}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train_all', 'train_one', 'eval_only'], default='train_all')
    parser.add_argument('--algo', choices=['ppo', 'dqn', 'qrdqn_mean', 'qrdqn_cvar', 'fzdqn'],
                        default=None, help='train_one 模式下指定单个算法')
    parser.add_argument('--episodes',    type=int,   default=5000)
    parser.add_argument('--eval_n',      type=int,   default=20,
                        help='评估时每个算法跑的 episode 数')
    parser.add_argument('--cvar_alpha',  type=float, default=0.1,
                        help='QR-DQN 风险规避版本的 CVaR 水平')
    # eval_only 模式需要提供已训练模型路径
    parser.add_argument('--ppo_model',    type=str, default=None)
    parser.add_argument('--dqn_model',    type=str, default=None)
    parser.add_argument('--qrdqn_model',  type=str, default=None)
    parser.add_argument('--qrcvar_model', type=str, default=None)
    parser.add_argument('--fzdqn_model',  type=str, default=None,
                        help='FZ-DQN 已训练模型路径（eval_only 模式）')
    args = parser.parse_args()

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    config    = SimplifiedEnvConfig()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'./results/comparison_{timestamp}'
    os.makedirs(result_dir, exist_ok=True)

    print(f"对比实验开始 | device={device} | mode={args.mode}")
    print(f"结果目录: {result_dir}\n")

    # ── 阶段一：训练（train_all 模式）────────────────────────────────────────
    ppo_trainer    = None
    dqn_trainer    = None
    qrdqn_trainer  = None
    qrcvar_trainer = None
    fzdqn_trainer  = None

    if args.mode == 'train_one':
        if args.algo is None:
            raise ValueError("--mode train_one 必须同时指定 --algo")
        algo = args.algo
        if algo == 'ppo':
            ppo_trainer = PPOTrainer(config, result_dir=os.path.join(result_dir, 'ppo'), device=device)
            ppo_trainer.train(max_episodes=args.episodes)
        elif algo == 'dqn':
            dqn_trainer = DQNTrainer(config, result_dir=os.path.join(result_dir, 'dqn'), device=device)
            dqn_trainer.train(max_episodes=args.episodes)
        elif algo == 'qrdqn_mean':
            qrdqn_trainer = QRDQNTrainer(config, result_dir=os.path.join(result_dir, 'qrdqn_mean'),
                                         device=device, cvar_alpha=1.0)
            qrdqn_trainer.train(max_episodes=args.episodes)
        elif algo == 'qrdqn_cvar':
            qrcvar_trainer = QRDQNTrainer(config,
                                          result_dir=os.path.join(result_dir, f'qrdqn_cvar{args.cvar_alpha}'),
                                          device=device, cvar_alpha=args.cvar_alpha)
            qrcvar_trainer.train(max_episodes=args.episodes)
        elif algo == 'fzdqn':
            fzdqn_trainer = FZDQNTrainer(config,
                                         result_dir=os.path.join(result_dir, f'fzdqn_cvar{args.cvar_alpha}'),
                                         device=device, cvar_alpha=args.cvar_alpha)
            fzdqn_trainer.train(max_episodes=args.episodes)
        print(f"\n{algo} 训练完成，结果保存在 {result_dir}")
        return

    if args.mode == 'train_all':
        print("=" * 50)
        print("训练 PPO")
        print("=" * 50)
        ppo_trainer = PPOTrainer(config,
                                 result_dir=os.path.join(result_dir, 'ppo'),
                                 device=device)
        ppo_trainer.train(max_episodes=args.episodes)

        print("\n" + "=" * 50)
        print("训练 DQN")
        print("=" * 50)
        dqn_trainer = DQNTrainer(config,
                                 result_dir=os.path.join(result_dir, 'dqn'),
                                 device=device)
        dqn_trainer.train(max_episodes=args.episodes)

        print("\n" + "=" * 50)
        print("训练 QR-DQN (mean, 风险中性)")
        print("=" * 50)
        qrdqn_trainer = QRDQNTrainer(config,
                                     result_dir=os.path.join(result_dir, 'qrdqn_mean'),
                                     device=device, cvar_alpha=1.0)
        qrdqn_trainer.train(max_episodes=args.episodes)

        print("\n" + "=" * 50)
        print(f"训练 QR-DQN (CVaR α={args.cvar_alpha}, 风险规避)")
        print("=" * 50)
        qrcvar_trainer = QRDQNTrainer(config,
                                      result_dir=os.path.join(result_dir, f'qrdqn_cvar{args.cvar_alpha}'),
                                      device=device, cvar_alpha=args.cvar_alpha)
        qrcvar_trainer.train(max_episodes=args.episodes)

        print("\n" + "=" * 50)
        print(f"训练 FZ-DQN (CVaR α={args.cvar_alpha}, 论文方法：FZ0 联合损失)")
        print("=" * 50)
        fzdqn_trainer = FZDQNTrainer(config,
                                     result_dir=os.path.join(result_dir, f'fzdqn_cvar{args.cvar_alpha}'),
                                     device=device, cvar_alpha=args.cvar_alpha)
        fzdqn_trainer.train(max_episodes=args.episodes)

    elif args.mode == 'eval_only':
        # 加载已有模型
        if args.ppo_model:
            ppo_trainer = PPOTrainer(config,
                                     result_dir=os.path.join(result_dir, 'ppo'),
                                     device=device)
            ppo_trainer.load_model(args.ppo_model)

        if args.dqn_model:
            dqn_trainer = DQNTrainer(config,
                                     result_dir=os.path.join(result_dir, 'dqn'),
                                     device=device)
            dqn_trainer.load_model(args.dqn_model)

        if args.qrdqn_model:
            qrdqn_trainer = QRDQNTrainer(config,
                                         result_dir=os.path.join(result_dir, 'qrdqn_mean'),
                                         device=device, cvar_alpha=1.0)
            qrdqn_trainer.load_model(args.qrdqn_model)

        if args.qrcvar_model:
            qrcvar_trainer = QRDQNTrainer(config,
                                          result_dir=os.path.join(result_dir, f'qrdqn_cvar'),
                                          device=device, cvar_alpha=args.cvar_alpha)
            qrcvar_trainer.load_model(args.qrcvar_model)

        if args.fzdqn_model:
            fzdqn_trainer = FZDQNTrainer(config,
                                         result_dir=os.path.join(result_dir, 'fzdqn_cvar'),
                                         device=device, cvar_alpha=args.cvar_alpha)
            fzdqn_trainer.load_model(args.fzdqn_model)

    # ── 阶段二：统一评估 ──────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"开始评估（每算法 {args.eval_n} episodes）")
    print("=" * 50)

    eval_env = SimplifiedExecutionEnv(config)
    comparison_results = {}

    # TWAP 基线（无需训练，始终评估）
    print("评估 TWAP 基线（标准均匀执行）...")
    comparison_results['TWAP'] = evaluate_agent(eval_env, make_twap_episode_fn(), args.eval_n)

    print("评估 Enhanced TWAP 基线（OFI/动量/流动性四路信号融合）...")
    comparison_results['Enhanced TWAP'] = evaluate_agent(eval_env, make_enhanced_twap_episode_fn(), args.eval_n)

    if ppo_trainer is not None:
        print("评估 PPO...")
        fn = make_ppo_episode_fn(ppo_trainer.agent, device)
        comparison_results['PPO'] = evaluate_agent(eval_env, fn, args.eval_n)

    if dqn_trainer is not None:
        print("评估 DQN...")
        fn = make_dqn_episode_fn(dqn_trainer.agent, device)
        comparison_results['DQN'] = evaluate_agent(eval_env, fn, args.eval_n)

    if qrdqn_trainer is not None:
        print("评估 QR-DQN (mean)...")
        fn = make_qrdqn_episode_fn(qrdqn_trainer.agent, device)
        comparison_results['QR-DQN (mean)'] = evaluate_agent(eval_env, fn, args.eval_n)

    if qrcvar_trainer is not None:
        print(f"评估 QR-DQN (CVaR α={args.cvar_alpha})...")
        fn = make_qrdqn_episode_fn(qrcvar_trainer.agent, device)
        comparison_results[f'QR-DQN (CVaR {args.cvar_alpha})'] = \
            evaluate_agent(eval_env, fn, args.eval_n)

    if fzdqn_trainer is not None:
        print(f"评估 FZ-DQN (CVaR α={args.cvar_alpha}, 论文方法)...")
        fn = make_fzdqn_episode_fn(fzdqn_trainer.agent, device)
        comparison_results[f'FZ-DQN (CVaR {args.cvar_alpha})'] = \
            evaluate_agent(eval_env, fn, args.eval_n)

    # ── 阶段三：输出报告 ──────────────────────────────────────────────────────
    if comparison_results:
        print_comparison_table(comparison_results)
        save_comparison_charts(comparison_results, result_dir)

        # 保存 JSON 汇总
        summary_path = os.path.join(result_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)

        # 保存 CSV 汇总
        csv_path = os.path.join(result_dir, 'comparison_summary.csv')
        with open(csv_path, 'w', newline='') as f:
            fieldnames = ['algorithm', 'bp_mean', 'bp_std', 'reward_mean',
                          'reward_std', 'cancel_rate_mean', 'passive_rate_mean']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, r in comparison_results.items():
                writer.writerow({'algorithm': name, **{k: r[k] for k in fieldnames[1:]}})

        print(f"\n汇总报告已保存:")
        print(f"  JSON: {summary_path}")
        print(f"  CSV:  {csv_path}")
        print(f"  图表: {result_dir}/comparison.png")
    else:
        print("没有可用的评估结果，请检查模型路径或训练是否完成。")


if __name__ == '__main__':
    main()
