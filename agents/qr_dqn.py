import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
QR-DQN: Quantile Regression DQN (Dabney et al. 2018)

与 DQN 的核心区别：
  - Q(s,a) 是标量  →  Z(s,a) 是 N 个分位数值的向量（建模回报分布）
  - MSE/Huber 损失  →  分位数 Huber 损失（非对称）
  - argmax E[Z]     →  支持 mean / CVaR 两种风险偏好的动作选取

与论文的联系：
  Z(s,a) 的分布估计是 FZ Loss / CVaR 策略优化的基础数据结构。
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

N_QUANTILES = 32   # 分位数个数（原论文用 200，32 训练更快）


# ─── 分位数网络（ZNetwork）────────────────────────────────────────────────────
class ZNetwork(nn.Module):
    """
    输出 [n_actions × n_quantiles] 的回报分布估计。
    骨架与 DQN/PPO 完全相同，只有输出头不同。
    """
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int = N_ACTIONS, n_quantiles: int = N_QUANTILES,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles

        # 与 DQN/PPO 相同的双路编码器
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
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual = nn.Linear(hidden_dim * 2, hidden_dim)

        # 输出头：每个动作对应 n_quantiles 个分位数值
        self.z_head = nn.Linear(hidden_dim, n_actions * n_quantiles)

    def forward(self, market_state: torch.Tensor,
                private_state: torch.Tensor) -> torch.Tensor:
        """返回 [batch, n_actions, n_quantiles]"""
        mf = self.market_net(market_state)
        pf = self.private_net(private_state)

        mf_attn = mf.unsqueeze(1) if mf.dim() == 2 else mf
        attn_out, _ = self.attention(mf_attn, mf_attn, mf_attn)
        attn_out = attn_out.squeeze(1) if mf.dim() == 2 else attn_out

        combined = torch.cat([attn_out, pf], dim=-1)
        features = self.fusion_net(combined) + self.residual(torch.cat([mf, pf], dim=-1))

        z = self.z_head(features)                              # [batch, n_actions*n_quantiles]
        return z.view(-1, self.n_actions, self.n_quantiles)    # [batch, n_actions, n_quantiles]


# ─── 分位数 Huber 损失 ────────────────────────────────────────────────────────
def quantile_huber_loss(predicted: torch.Tensor, target: torch.Tensor,
                         taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """
    QR-DQN 的核心损失函数（Dabney et al. 2018, Eq. 10）

    Args:
        predicted: [batch, N]   — 在选定动作上预测的 N 个分位数值
        target:    [batch, N']  — TD 目标分位数（来自目标网络）
        taus:      [N]          — 分位数水平 τ_i = (i-0.5)/N
        kappa:     Huber 阈值（默认 1.0）

    Returns:
        标量 loss
    """
    # TD 误差矩阵 u_{ij} = target_j - predicted_i，shape [batch, N, N']
    # target: [batch, N'] → [batch, 1, N']
    # predicted: [batch, N] → [batch, N, 1]
    u = target.unsqueeze(1) - predicted.unsqueeze(2)   # [batch, N, N']

    # Huber 损失 L_κ(u)
    huber = torch.where(
        u.abs() <= kappa,
        0.5 * u.pow(2) / kappa,
        u.abs() - 0.5 * kappa,
    )

    # 非对称分位数权重 |τ_i − 1(u < 0)|，shape [1, N, 1]
    taus_exp = taus.view(1, -1, 1)
    weights  = (taus_exp - (u < 0).float()).abs()

    # 对 N' 目标分位数取均值，对 N 预测分位数求和，对 batch 取均值
    loss = (weights * huber).mean(dim=2).sum(dim=1).mean()
    # clamp 防止极端市场数据造成单次大幅梯度更新
    return torch.clamp(loss, max=20.0)


# ─── QR-DQN Agent ─────────────────────────────────────────────────────────────
class QRDQNAgent:
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int = N_ACTIONS, n_quantiles: int = N_QUANTILES,
                 hidden_dim: int = 256, lr: float = 1e-4, gamma: float = 0.99,
                 buffer_capacity: int = 100_000, batch_size: int = 256,
                 target_update_freq: int = 500,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 5000,
                 cvar_alpha: float = 1.0,   # 1.0 = mean（风险中性），< 1.0 = CVaR（风险规避）
                 device: str = 'cpu'):

        self.n_actions          = n_actions
        self.n_quantiles        = n_quantiles
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = device
        self.cvar_alpha         = cvar_alpha   # 控制风险偏好

        # epsilon-greedy 参数
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        # 分位数水平 τ_i = (i - 0.5) / N，shape [N]
        self.taus = torch.FloatTensor(
            [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
        ).to(device)

        # 在线网络 + 目标网络
        self.online_net = ZNetwork(market_state_dim, private_state_dim,
                                   n_actions, n_quantiles, hidden_dim).to(device)
        self.target_net = ZNetwork(market_state_dim, private_state_dim,
                                   n_actions, n_quantiles, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer    = optim.AdamW(self.online_net.parameters(),
                                        lr=lr, weight_decay=1e-4)
        self.buffer       = ReplayBuffer(buffer_capacity)
        self.update_count = 0

    # ── 风险偏好动作评分 ──────────────────────────────────────────────────────
    def _action_scores(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [batch, n_actions, n_quantiles]
        返回每个动作的评分 [batch, n_actions]

        cvar_alpha=1.0  →  E[Z] = 所有分位数均值（等同于 DQN）
        cvar_alpha<1.0  →  CVaR_α = α 分位数以下的均值（下尾期望，风险规避）
        """
        if self.cvar_alpha >= 1.0:
            return z.mean(dim=2)   # [batch, n_actions]

        # 取 τ_i ≤ α 的分位数，计算其均值
        n_tail = max(1, int(self.n_quantiles * self.cvar_alpha))
        # taus 已排序（从小到大），前 n_tail 个是下尾
        return z[:, :, :n_tail].mean(dim=2)   # [batch, n_actions]

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
            z  = self.online_net(ms, ps)               # [1, n_actions, n_quantiles]
            scores = self._action_scores(z)             # [1, n_actions]
            return int(scores.argmax(dim=1).item())

    def current_epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    def store(self, ms, ps, action_idx, reward, nms, nps, done):
        self.buffer.push(ms, ps, action_idx, reward, nms, nps, done)

    # ── 网络更新 ──────────────────────────────────────────────────────────────
    def update(self) -> float:
        if len(self.buffer) < self.batch_size:
            return 0.0

        ms, ps, acts, rews, nms, nps, dones = self.buffer.sample(self.batch_size)

        ms    = torch.FloatTensor(ms).to(self.device)
        ps    = torch.FloatTensor(ps).to(self.device)
        acts  = torch.LongTensor(acts).to(self.device)
        rews  = torch.FloatTensor(rews).to(self.device)
        nms   = torch.FloatTensor(nms).to(self.device)
        nps   = torch.FloatTensor(nps).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前分布 Z(s, a_selected)，shape [batch, N]
        z_all    = self.online_net(ms, ps)                         # [batch, n_actions, N]
        acts_exp = acts.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)
        z_pred   = z_all.gather(1, acts_exp).squeeze(1)            # [batch, N]

        # 目标分布：用目标网络，动作由在线网络选择（Double DQN）
        with torch.no_grad():
            z_next_online  = self.online_net(nms, nps)             # [batch, n_actions, N]
            next_acts      = self._action_scores(z_next_online).argmax(dim=1)  # [batch]
            z_next_target  = self.target_net(nms, nps)             # [batch, n_actions, N]
            next_acts_exp  = next_acts.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)
            z_next         = z_next_target.gather(1, next_acts_exp).squeeze(1)  # [batch, N]

            # Bellman 目标：r + γ Z(s', a*) · (1 - done)
            z_target = rews.unsqueeze(1) + self.gamma * z_next * \
                       (1.0 - dones.unsqueeze(1))                  # [batch, N]

        loss = quantile_huber_loss(z_pred, z_target, self.taus)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ── 分布可视化（调试用）──────────────────────────────────────────────────
    def get_return_distribution(self, market_state, private_state) -> np.ndarray:
        """返回当前最优动作的分位数分布 [N_QUANTILES]，用于分析"""
        with torch.no_grad():
            ms = torch.FloatTensor(market_state).unsqueeze(0).to(self.device)
            ps = torch.FloatTensor(private_state).unsqueeze(0).to(self.device)
            z  = self.online_net(ms, ps)
            scores  = self._action_scores(z)
            best_a  = int(scores.argmax(dim=1).item())
            return z[0, best_a].cpu().numpy()


# ─── QR-DQN Trainer ───────────────────────────────────────────────────────────
class QRDQNTrainer:
    def __init__(self, config, result_dir: str = './results/qr_dqn',
                 device: str = 'cpu', cvar_alpha: float = 1.0):
        self.config     = config
        self.result_dir = result_dir
        self.device     = device
        os.makedirs(result_dir, exist_ok=True)

        self.env = SimplifiedExecutionEnv(config)
        market_dim  = self.env.market_state_dim
        private_dim = self.env.private_state_dim

        self.agent = QRDQNAgent(
            market_state_dim  = market_dim,
            private_state_dim = private_dim,
            n_actions         = N_ACTIONS,
            n_quantiles       = N_QUANTILES,
            cvar_alpha        = cvar_alpha,
            device            = device,
        )

        risk_label = f"CVaR(α={cvar_alpha})" if cvar_alpha < 1.0 else "Mean (risk-neutral)"
        print(f"QR-DQN 初始化完成 — market_dim={market_dim}, private_dim={private_dim}, "
              f"n_actions={N_ACTIONS}, n_quantiles={N_QUANTILES}, "
              f"risk_mode={risk_label}, device={device}")

    def _run_episode(self, eval_mode: bool = False):
        reset_ret = self.env.reset()
        ms, ps = reset_ret[0], reset_ret[1]
        done = False
        total_reward = 0.0
        losses = []

        while not done:
            if eval_mode:
                with torch.no_grad():
                    ms_t = torch.FloatTensor(ms).unsqueeze(0).to(self.device)
                    ps_t = torch.FloatTensor(ps).unsqueeze(0).to(self.device)
                    z    = self.agent.online_net(ms_t, ps_t)
                    act_idx = int(self.agent._action_scores(z).argmax(1).item())
            else:
                act_idx = self.agent.select_action(ms, ps)

            action  = idx_to_action(act_idx)
            nms, nps, reward, done, _ = self.env.step(action)

            if not eval_mode:
                self.agent.store(ms, ps, act_idx, reward, nms, nps, done)
                loss = self.agent.update()
                if loss > 0:
                    losses.append(loss)

            total_reward += float(reward)
            ms, ps = nms, nps

        metrics = self.env.get_trading_metrics()
        return total_reward, np.mean(losses) if losses else 0.0, metrics

    def train(self, max_episodes: int = 5000, eval_every: int = 1000,
              eval_episodes: int = 5):

        rewards     = []
        losses      = []
        price_perfs = []
        eval_log    = []

        for ep in range(1, max_episodes + 1):
            ep_reward, ep_loss, metrics = self._run_episode(eval_mode=False)
            rewards.append(ep_reward)
            losses.append(ep_loss)
            price_perfs.append(metrics.get('price_performance_bp', 0.0))

            if ep % 10 == 0:
                avg_bp = np.mean(price_perfs[-10:])
                print(f"Episode {ep:5d} | reward {ep_reward:+8.4f} | "
                      f"loss {ep_loss:.5f} | bp {metrics.get('price_performance_bp', 0):+.2f} "
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
                self._save_model(f"qrdqn_ep{ep}.pth")

        self._save_model("qrdqn_final.pth")
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
            'n_quantiles':  self.agent.n_quantiles,
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        window = min(50, len(rewards))
        axes[0].plot(rewards, alpha=0.3, label='raw')
        if len(rewards) >= window:
            sm = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), sm, label=f'MA{window}')
        axes[0].set_title('Episode Reward')
        axes[0].legend()

        axes[1].plot(losses, alpha=0.4)
        axes[1].set_title('Quantile Huber Loss')

        axes[2].plot(price_perfs, alpha=0.3, label='raw bp')
        if len(price_perfs) >= window:
            sm_bp = np.convolve(price_perfs, np.ones(window)/window, mode='valid')
            axes[2].plot(range(window-1, len(price_perfs)), sm_bp, label=f'MA{window}')
        axes[2].axhline(0, color='red', linestyle='--', alpha=0.5, label='VWAP')
        axes[2].set_title('Price Performance (bp)')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'qrdqn_training_curves.png'), dpi=150)
        plt.close()

        csv_path = os.path.join(self.result_dir, 'qrdqn_training_log.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'reward', 'loss', 'price_perf_bp'])
            for i, (r, l, b) in enumerate(zip(rewards, losses, price_perfs), 1):
                w.writerow([i, r, l, b])

        with open(os.path.join(self.result_dir, 'qrdqn_eval_log.json'), 'w') as f:
            json.dump(eval_log, f, indent=2)

        print(f"训练曲线已保存: {self.result_dir}/qrdqn_training_curves.png")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cvar_alpha', type=float, default=1.0,
                        help='CVaR 水平（1.0=风险中性，0.1=强风险规避）')
    parser.add_argument('--episodes', type=int, default=5000)
    args = parser.parse_args()

    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    config    = SimplifiedEnvConfig()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag       = f"cvar{args.cvar_alpha}" if args.cvar_alpha < 1.0 else "mean"
    result_dir = f'./results/qrdqn_{tag}_{timestamp}'

    print(f"QR-DQN 训练开始 | device={device} | cvar_alpha={args.cvar_alpha} | result_dir={result_dir}")
    trainer = QRDQNTrainer(config, result_dir=result_dir,
                           device=device, cvar_alpha=args.cvar_alpha)
    trainer.train(max_episodes=args.episodes, eval_every=1000, eval_episodes=5)
    print(f"\n训练完成，结果保存到: {result_dir}")


if __name__ == '__main__':
    main()
