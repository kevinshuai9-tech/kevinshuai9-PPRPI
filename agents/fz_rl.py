import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
FZ-DQN: 基于 Fissler-Ziegel 联合损失的分布式 RL（论文核心方法）

与 QR-DQN 的本质区别
─────────────────────────────────────────────────────────────────
QR-DQN:  只估计 VaR（分位数 q_i），CVaR 靠对 q_i 取下尾均值"近似"
FZ-DQN:  同时估计 VaR（q_i 头）和 CVaR（e_i 头），用 FZ0 联合损失
         保证两者共同严格一致（Fissler & Ziegel 2016，Thm 5）

FZ0 损失（e > 0 参数化版本）
─────────────────────────────────────────────────────────────────
S₀(y, q, e; τ) = -1 + q/e + (y-q)·1{y≤q}/(τ·e) - log(e)

最小化器：q* = VaR_τ(Y)，e* = -CVaR_τ(Y)

由于我们的奖励 Y 通常 < 0（执行低于 VWAP），CVaR_τ(Y) < 0，
所以 e* = -CVaR_τ > 0，可用 softplus 参数化确保 e > 0。

风险敏感策略
─────────────────────────────────────────────────────────────────
argmax_a CVaR_τ(Z^π(s,a))
  = argmax_a (-e_τ(s,a))     ← 直接用 e 头，不需要近似
  = argmin_a  e_τ(s,a)

论文联系
─────────────────────────────────────────────────────────────────
本文件实现论文 Section 4 的 FZ Loss 驱动的分布式 RL 方法。
QR-DQN (qr_dqn.py) 是 Section 5 的 baseline 对比。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import os
import csv
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env_bs import SimplifiedExecutionEnv, SimplifiedEnvConfig
from dqn import DISCRETE_ACTIONS, N_ACTIONS, idx_to_action, ReplayBuffer

N_TAU = 32   # 分位数水平数量（与 QR-DQN 对齐，便于公平对比）


# ─── FZ 双头网络 ──────────────────────────────────────────────────────────────
class FZNetwork(nn.Module):
    """
    输出两个头：
      q_head → [batch, n_actions, N_TAU]  VaR 估计（无约束）
      e_head → [batch, n_actions, N_TAU]  -CVaR 估计（softplus 保证 > 0）

    骨架与 PPO / DQN / QR-DQN 完全相同，只有输出头不同。
    """
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int = N_ACTIONS, n_tau: int = N_TAU,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.n_tau     = n_tau

        # 与其他 agent 相同的双路编码器
        self.market_net = nn.Sequential(
            nn.Linear(market_state_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.private_net = nn.Sequential(
            nn.Linear(private_state_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attention  = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim)

        # 双输出头（共用 backbone）
        self.q_head   = nn.Linear(hidden_dim, n_actions * n_tau)  # VaR
        self.e_head   = nn.Linear(hidden_dim, n_actions * n_tau)  # -CVaR（raw）

    def forward(self, market_state: torch.Tensor,
                private_state: torch.Tensor):
        """
        返回 (q, e)：
          q: [batch, n_actions, N_TAU]  — VaR_τi
          e: [batch, n_actions, N_TAU]  — -CVaR_τi  (e > 0，softplus 保证)
        """
        mf = self.market_net(market_state)
        pf = self.private_net(private_state)

        mf_attn = mf.unsqueeze(1) if mf.dim() == 2 else mf
        attn_out, _ = self.attention(mf_attn, mf_attn, mf_attn)
        attn_out = attn_out.squeeze(1) if mf.dim() == 2 else attn_out

        combined = torch.cat([attn_out, pf], dim=-1)
        features = self.fusion_net(combined) + self.residual(torch.cat([mf, pf], dim=-1))

        q = self.q_head(features).view(-1, self.n_actions, self.n_tau)
        e = F.softplus(self.e_head(features)).view(-1, self.n_actions, self.n_tau) + 1e-3
        return q, e


# ─── FZ0 联合损失 ─────────────────────────────────────────────────────────────
def fz0_loss(q_pred: torch.Tensor, e_pred: torch.Tensor,
             y_targets: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """
    FZ0 联合严格一致损失（Fissler & Ziegel 2016, e > 0 版本）

    S₀(y, q, e; τ) = -1 + q/e + (y-q)·1{y≤q}/(τ·e) - log(e)

    最小化器：q* = VaR_τ(Y)，e* = -CVaR_τ(Y)

    Args:
        q_pred:    [batch, N_TAU]      — VaR 预测
        e_pred:    [batch, N_TAU]      — -CVaR 预测（> 0）
        y_targets: [batch, N_targets]  — 分布式 Bellman 目标样本
        taus:      [N_TAU]             — 分位数水平

    Returns:
        标量 loss
    """
    # 扩展维度做两两计算
    q   = q_pred.unsqueeze(2)      # [batch, N_TAU, 1]
    e   = e_pred.unsqueeze(2)      # [batch, N_TAU, 1]
    y   = y_targets.unsqueeze(1)   # [batch, 1, N_targets]
    tau = taus.view(1, -1, 1)      # [1, N_TAU, 1]

    indicator = (y <= q).float()   # [batch, N_TAU, N_targets]  1{y ≤ q}

    # S₀ = -1 + q/e + (y-q)·1{y≤q}/(τ·e) - log(e)
    # 归一化 q 和 y 以防止 q/e 项在奖励量级较大时爆炸
    scale = (y.detach().abs().mean().clamp(min=1.0))
    q_n   = q / scale
    y_n   = y / scale

    s0 = (-1.0
          + q_n / e
          + (y_n - q_n) * indicator / (tau * e)
          - torch.log(e))          # [batch, N_TAU, N_targets]

    # clamp 防止极端数据点主导梯度
    s0 = torch.clamp(s0, -200.0, 200.0)

    return s0.mean()


# ─── FZ-DQN Agent ─────────────────────────────────────────────────────────────
class FZDQNAgent:
    """
    基于 FZ0 联合损失的分布式 DQN。

    cvar_alpha：CVaR 水平（使用 τ_i ≤ cvar_alpha 的 e 头均值做动作选取）
      1.0 → 所有分位数 e 均值（近似期望）
      0.1 → 最差 10% 情景下的 CVaR（强风险规避）
    """
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int = N_ACTIONS, n_tau: int = N_TAU,
                 hidden_dim: int = 256, lr: float = 1e-4, gamma: float = 0.99,
                 buffer_capacity: int = 100_000, batch_size: int = 256,
                 target_update_freq: int = 500,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 5000,
                 cvar_alpha: float = 0.1,
                 device: str = 'cpu'):

        self.n_actions          = n_actions
        self.n_tau              = n_tau
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = device
        self.cvar_alpha         = cvar_alpha

        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        # 分位数水平 τ_i = (i+0.5)/N_TAU
        self.taus = torch.FloatTensor(
            [(i + 0.5) / n_tau for i in range(n_tau)]
        ).to(device)

        # 在线网络 + 目标网络（均为双头）
        self.online_net = FZNetwork(market_state_dim, private_state_dim,
                                    n_actions, n_tau, hidden_dim).to(device)
        self.target_net = FZNetwork(market_state_dim, private_state_dim,
                                    n_actions, n_tau, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.AdamW(self.online_net.parameters(),
                                        lr=lr, weight_decay=1e-4)
        self.buffer       = ReplayBuffer(buffer_capacity)
        self.update_count = 0

    # ── CVaR 动作评分 ─────────────────────────────────────────────────────────
    def _cvar_scores(self, e: torch.Tensor) -> torch.Tensor:
        """
        e: [batch, n_actions, N_TAU]  (-CVaR 预测，e > 0)

        CVaR_τi(Z) = -e_i(s,a)
        argmax_a CVaR = argmin_a e

        cvar_alpha 控制用哪个 τ 水平的 CVaR：
          取 τ_i ≤ cvar_alpha 的 e 头均值 → 对应该水平的 CVaR 动作选取
        """
        n_tail = max(1, int(self.n_tau * self.cvar_alpha))
        # taus 从小到大排列，前 n_tail 个是最悲观分位数
        # e 越小 → CVaR 越大 → 动作越好
        return -e[:, :, :n_tail].mean(dim=2)   # [batch, n_actions]，越大越好

    # ── 动作选取 ──────────────────────────────────────────────────────────────
    def select_action(self, market_state, private_state) -> int:
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            ms = torch.FloatTensor(market_state).unsqueeze(0).to(self.device)
            ps = torch.FloatTensor(private_state).unsqueeze(0).to(self.device)
            _, e = self.online_net(ms, ps)          # [1, n_actions, N_TAU]
            scores = self._cvar_scores(e)            # [1, n_actions]
            return int(scores.argmax(dim=1).item())

    def current_epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    def store(self, ms, ps, action_idx, reward, nms, nps, done):
        self.buffer.push(ms, ps, action_idx, reward, nms, nps, done)

    # ── 网络更新 ──────────────────────────────────────────────────────────────
    def update(self):
        if len(self.buffer) < self.batch_size:
            return None  # 区别于 FZ0 loss=0 的合法情况

        ms, ps, acts, rews, nms, nps, dones = self.buffer.sample(self.batch_size)

        ms    = torch.FloatTensor(ms).to(self.device)
        ps    = torch.FloatTensor(ps).to(self.device)
        acts  = torch.LongTensor(acts).to(self.device)
        rews  = torch.FloatTensor(rews).to(self.device)
        nms   = torch.FloatTensor(nms).to(self.device)
        nps   = torch.FloatTensor(nps).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        batch_idx = torch.arange(self.batch_size, device=self.device)

        # ── 目标：用在线网络选动作（Double DQN），用目标网络的 VaR 头做 Bellman ──
        with torch.no_grad():
            _, e_next_online = self.online_net(nms, nps)          # CVaR 选动作
            next_acts = self._cvar_scores(e_next_online).argmax(1)  # [batch]

            q_next_target, _ = self.target_net(nms, nps)          # VaR 头 Bellman 目标
            # 取 next_acts 对应的 VaR 分位数向量作为分布样本
            next_acts_exp = next_acts.view(-1, 1, 1).expand(-1, 1, self.n_tau)
            q_target_sel  = q_next_target.gather(1, next_acts_exp).squeeze(1)  # [batch, N_TAU]

            # 分布式 Bellman：N_TAU 个样本
            y_targets = (rews.unsqueeze(1)
                         + self.gamma * q_target_sel
                         * (1.0 - dones.unsqueeze(1)))              # [batch, N_TAU]

        # ── 在线网络预测（选定动作的双头输出）────────────────────────────────
        q_all, e_all = self.online_net(ms, ps)
        acts_exp = acts.view(-1, 1, 1).expand(-1, 1, self.n_tau)

        q_pred = q_all.gather(1, acts_exp).squeeze(1)  # [batch, N_TAU]
        e_pred = e_all.gather(1, acts_exp).squeeze(1)  # [batch, N_TAU]

        # ── FZ0 联合损失 ──────────────────────────────────────────────────────
        loss = fz0_loss(q_pred, e_pred, y_targets, self.taus)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


# ─── FZ-DQN Trainer ───────────────────────────────────────────────────────────
class FZDQNTrainer:
    def __init__(self, config, result_dir: str = './results/fz_dqn',
                 device: str = 'cpu', cvar_alpha: float = 0.1):
        self.config     = config
        self.result_dir = result_dir
        self.device     = device
        os.makedirs(result_dir, exist_ok=True)

        self.env = SimplifiedExecutionEnv(config)
        market_dim  = self.env.market_state_dim
        private_dim = self.env.private_state_dim

        self.agent = FZDQNAgent(
            market_state_dim  = market_dim,
            private_state_dim = private_dim,
            n_actions         = N_ACTIONS,
            n_tau             = N_TAU,
            cvar_alpha        = cvar_alpha,
            device            = device,
        )

        print(f"FZ-DQN 初始化完成 — market_dim={market_dim}, private_dim={private_dim}, "
              f"n_actions={N_ACTIONS}, n_tau={N_TAU}, "
              f"cvar_alpha={cvar_alpha}, device={device}")
        print(f"  论文方法：VaR + CVaR 双头，FZ0 联合损失")

    def _run_episode(self, eval_mode: bool = False):
        reset_ret = self.env.reset()
        ms, ps    = reset_ret[0], reset_ret[1]
        done  = False
        total_reward = 0.0
        losses = []

        while not done:
            if eval_mode:
                with torch.no_grad():
                    ms_t = torch.FloatTensor(ms).unsqueeze(0).to(self.device)
                    ps_t = torch.FloatTensor(ps).unsqueeze(0).to(self.device)
                    _, e = self.agent.online_net(ms_t, ps_t)
                    act_idx = int(self.agent._cvar_scores(e).argmax(1).item())
            else:
                act_idx = self.agent.select_action(ms, ps)

            action = idx_to_action(act_idx)
            nms, nps, reward, done, _ = self.env.step(action)

            if not eval_mode:
                self.agent.store(ms, ps, act_idx, reward, nms, nps, done)
                loss = self.agent.update()
                if loss is not None:   # FZ0 loss 可以为负，不能用 > 0 过滤
                    losses.append(loss)

            total_reward += float(reward)
            ms, ps = nms, nps

        metrics = self.env.get_trading_metrics()
        return total_reward, np.mean(losses) if losses else float('nan'), metrics

    def train(self, max_episodes: int = 5000, eval_every: int = 1000,
              eval_episodes: int = 5):

        rewards, losses, price_perfs, eval_log = [], [], [], []

        for ep in range(1, max_episodes + 1):
            ep_reward, ep_loss, metrics = self._run_episode(eval_mode=False)
            rewards.append(ep_reward)
            losses.append(ep_loss)
            price_perfs.append(metrics.get('price_performance_bp', 0.0))

            if ep % 10 == 0:
                avg_bp = np.mean(price_perfs[-10:])
                loss_str = f"{ep_loss:.5f}" if not np.isnan(ep_loss) else "warming"
                print(f"Episode {ep:5d} | reward {ep_reward:+8.4f} | "
                      f"fz_loss {loss_str} | bp {metrics.get('price_performance_bp', 0):+.2f} "
                      f"| avg_bp(10) {avg_bp:+.2f} | eps {self.agent.current_epsilon():.3f}")

            if ep % eval_every == 0:
                eval_rewards, eval_bps = [], []
                for _ in range(eval_episodes):
                    r, _, m = self._run_episode(eval_mode=True)
                    eval_rewards.append(r)
                    eval_bps.append(m.get('price_performance_bp', 0.0))
                print(f"\n[EVAL ep={ep}] reward {np.mean(eval_rewards):+.4f}±{np.std(eval_rewards):.4f} | "
                      f"bp {np.mean(eval_bps):+.2f}±{np.std(eval_bps):.2f}\n")
                eval_log.append({'episode': ep,
                                 'eval_reward_mean': float(np.mean(eval_rewards)),
                                 'eval_bp_mean':     float(np.mean(eval_bps)),
                                 'eval_bp_std':      float(np.std(eval_bps))})
                self._save_model(f"fzdqn_ep{ep}.pth")

        self._save_model("fzdqn_final.pth")
        self._save_curves(rewards, losses, price_perfs, eval_log)
        return rewards, price_perfs, eval_log

    def _save_model(self, name: str):
        torch.save({
            'online_net':   self.agent.online_net.state_dict(),
            'target_net':   self.agent.target_net.state_dict(),
            'optimizer':    self.agent.optimizer.state_dict(),
            'steps_done':   self.agent.steps_done,
            'update_count': self.agent.update_count,
            'cvar_alpha':   self.agent.cvar_alpha,
            'n_tau':        self.agent.n_tau,
        }, os.path.join(self.result_dir, name))

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.online_net.load_state_dict(ckpt['online_net'])
        self.agent.target_net.load_state_dict(ckpt['target_net'])
        self.agent.optimizer.load_state_dict(ckpt['optimizer'])
        self.agent.steps_done   = ckpt.get('steps_done', 0)
        self.agent.update_count = ckpt.get('update_count', 0)
        print(f"模型已加载: {path}")

    def _save_curves(self, rewards, losses, price_perfs, eval_log):
        window = min(50, len(rewards))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(rewards, alpha=0.3, label='raw')
        if len(rewards) >= window:
            sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), sm, label=f'MA{window}')
        axes[0].set_title('Episode Reward')
        axes[0].legend()

        axes[1].plot(losses, alpha=0.4)
        axes[1].set_title('FZ0 Loss')

        axes[2].plot(price_perfs, alpha=0.3, label='raw bp')
        if len(price_perfs) >= window:
            sm_bp = np.convolve(price_perfs, np.ones(window)/window, mode='valid')
            axes[2].plot(range(window-1, len(price_perfs)), sm_bp, label=f'MA{window}')
        axes[2].axhline(0, color='red', linestyle='--', alpha=0.5, label='VWAP')
        axes[2].set_title('Price Performance (bp)')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'fzdqn_training_curves.png'), dpi=150)
        plt.close()

        with open(os.path.join(self.result_dir, 'fzdqn_training_log.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'reward', 'fz_loss', 'price_perf_bp'])
            for i, (r, l, b) in enumerate(zip(rewards, losses, price_perfs), 1):
                w.writerow([i, r, l, b])

        with open(os.path.join(self.result_dir, 'fzdqn_eval_log.json'), 'w') as f:
            json.dump(eval_log, f, indent=2)

        print(f"训练曲线已保存: {self.result_dir}/fzdqn_training_curves.png")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvar_alpha', type=float, default=0.1,
                        help='CVaR 水平（0.1=关注最差10%情景）')
    parser.add_argument('--episodes', type=int, default=5000)
    args = parser.parse_args()

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    config     = SimplifiedEnvConfig()
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'./results/fzdqn_cvar{args.cvar_alpha}_{timestamp}'

    print(f"FZ-DQN 训练开始 | device={device} | cvar_alpha={args.cvar_alpha}")
    print(f"论文方法：FZ0 联合损失，直接估计 VaR + CVaR\n")

    trainer = FZDQNTrainer(config, result_dir=result_dir,
                           device=device, cvar_alpha=args.cvar_alpha)
    trainer.train(max_episodes=args.episodes, eval_every=1000, eval_episodes=5)
    print(f"\n训练完成，结果保存到: {result_dir}")


if __name__ == '__main__':
    main()
