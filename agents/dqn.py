import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

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

# ─── 动作空间离散化 ────────────────────────────────────────────────────────────
# 将连续的 [place_level, intensity] ∈ [0,1]² 离散为 N_ACTIONS 个动作
_PLACE_LEVELS   = [0.0, 0.33, 0.67, 1.0]   # 4个挂单档位
_INTENSITIES    = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5个执行强度
DISCRETE_ACTIONS = [(p, i) for p in _PLACE_LEVELS for i in _INTENSITIES]
N_ACTIONS = len(DISCRETE_ACTIONS)  # 4×5 = 20


def idx_to_action(idx: int) -> np.ndarray:
    """离散动作索引 → 连续动作向量"""
    return np.array(DISCRETE_ACTIONS[idx], dtype=np.float32)


# ─── Q 网络（与 PPO PolicyNetwork 同款骨架，输出改为 Q 值向量）─────────────────
class QNetwork(nn.Module):
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.market_net = nn.Sequential(
            nn.Linear(market_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.private_net = nn.Sequential(
            nn.Linear(private_state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.residual   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.q_head     = nn.Linear(hidden_dim, n_actions)

    def forward(self, market_state: torch.Tensor,
                private_state: torch.Tensor) -> torch.Tensor:
        mf = self.market_net(market_state)
        pf = self.private_net(private_state)

        mf_attn = mf.unsqueeze(1) if mf.dim() == 2 else mf
        attn_out, _ = self.attention(mf_attn, mf_attn, mf_attn)
        attn_out = attn_out.squeeze(1) if mf.dim() == 2 else attn_out

        combined = torch.cat([attn_out, pf], dim=-1)
        features = self.fusion_net(combined)
        features = features + self.residual(torch.cat([mf, pf], dim=-1))
        return self.q_head(features)


# ─── 经验回放缓冲区（off-policy）──────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, market_state, private_state, action_idx,
             reward, next_market_state, next_private_state, done):
        self.buffer.append((
            np.array(market_state,      dtype=np.float32),
            np.array(private_state,     dtype=np.float32),
            int(action_idx),
            float(reward),
            np.array(next_market_state, dtype=np.float32),
            np.array(next_private_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        ms, ps, acts, rews, nms, nps, dones = zip(*batch)
        return (
            np.stack(ms),
            np.stack(ps),
            np.array(acts,  dtype=np.int64),
            np.array(rews,  dtype=np.float32),
            np.stack(nms),
            np.stack(nps),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, market_state_dim: int, private_state_dim: int,
                 n_actions: int = N_ACTIONS, hidden_dim: int = 256,
                 lr: float = 1e-4, gamma: float = 0.99,
                 buffer_capacity: int = 50_000, batch_size: int = 256,
                 target_update_freq: int = 500,
                 eps_start: float = 1.0, eps_end: float = 0.05,
                 eps_decay: int = 5000, device: str = 'cpu'):

        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = device

        # epsilon-greedy 参数
        self.eps_start  = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self.steps_done = 0

        # 在线网络 + 目标网络
        self.online_net = QNetwork(market_state_dim, private_state_dim,
                                   n_actions, hidden_dim).to(device)
        self.target_net = QNetwork(market_state_dim, private_state_dim,
                                   n_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.AdamW(self.online_net.parameters(), lr=lr,
                                      weight_decay=1e-4)
        self.scheduler  = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)
        self.buffer     = ReplayBuffer(buffer_capacity)
        self.update_count = 0

    # ── 动作选取 ──────────────────────────────────────────────────────────────
    def select_action(self, market_state, private_state) -> int:
        """epsilon-greedy，返回离散动作索引"""
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            ms = torch.FloatTensor(market_state).unsqueeze(0).to(self.device)
            ps = torch.FloatTensor(private_state).unsqueeze(0).to(self.device)
            q_vals = self.online_net(ms, ps)
            return int(q_vals.argmax(dim=1).item())

    def current_epsilon(self) -> float:
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-self.steps_done / self.eps_decay)

    # ── 经验存储 ──────────────────────────────────────────────────────────────
    def store(self, ms, ps, action_idx, reward, nms, nps, done):
        self.buffer.push(ms, ps, action_idx, reward, nms, nps, done)

    # ── 网络更新 ──────────────────────────────────────────────────────────────
    def update(self) -> float:
        """从 replay buffer 采样，执行一步 TD 更新，返回 loss"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        ms, ps, acts, rews, nms, nps, dones = self.buffer.sample(self.batch_size)

        ms   = torch.FloatTensor(ms).to(self.device)
        ps   = torch.FloatTensor(ps).to(self.device)
        acts = torch.LongTensor(acts).to(self.device)
        rews = torch.FloatTensor(rews).to(self.device)
        nms  = torch.FloatTensor(nms).to(self.device)
        nps  = torch.FloatTensor(nps).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 当前 Q(s, a)
        q_values = self.online_net(ms, ps).gather(1, acts.unsqueeze(1)).squeeze(1)

        # 目标：r + γ · max_a' Q_target(s', a') · (1 - done)
        with torch.no_grad():
            next_q = self.target_net(nms, nps).max(dim=1)[0]
            target = rews + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1

        # 定期更新目标网络
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


# ─── DQN Trainer ──────────────────────────────────────────────────────────────
class DQNTrainer:
    def __init__(self, config, result_dir: str = './results/dqn',
                 device: str = 'cpu'):
        self.config     = config
        self.result_dir = result_dir
        self.device     = device
        os.makedirs(result_dir, exist_ok=True)

        self.env = SimplifiedExecutionEnv(config)
        market_dim  = self.env.market_state_dim
        private_dim = self.env.private_state_dim

        self.agent = DQNAgent(
            market_state_dim  = market_dim,
            private_state_dim = private_dim,
            n_actions         = N_ACTIONS,
            device            = device,
        )

        print(f"DQN 初始化完成 — market_dim={market_dim}, "
              f"private_dim={private_dim}, n_actions={N_ACTIONS}, device={device}")

    # ── 单回合评估 ────────────────────────────────────────────────────────────
    def _run_episode(self, eval_mode: bool = False):
        reset_ret = self.env.reset()
        ms, ps    = reset_ret[0], reset_ret[1]
        done  = False
        total_reward = 0.0
        losses = []

        while not done:
            if eval_mode:
                # 评估时不探索
                with torch.no_grad():
                    ms_t = torch.FloatTensor(ms).unsqueeze(0).to(self.device)
                    ps_t = torch.FloatTensor(ps).unsqueeze(0).to(self.device)
                    act_idx = int(self.agent.online_net(ms_t, ps_t).argmax(1).item())
            else:
                act_idx = self.agent.select_action(ms, ps)

            action = idx_to_action(act_idx)
            step_ret = self.env.step(action)
            nms, nps, reward, done, _ = step_ret

            if not eval_mode:
                self.agent.store(ms, ps, act_idx, reward, nms, nps, done)
                loss = self.agent.update()
                if loss > 0:
                    losses.append(loss)

            total_reward += float(reward)
            ms, ps = nms, nps

        metrics = self.env.get_trading_metrics()
        return total_reward, np.mean(losses) if losses else 0.0, metrics

    # ── 训练主循环 ────────────────────────────────────────────────────────────
    def train(self, max_episodes: int = 5000, eval_every: int = 1000,
              eval_episodes: int = 5):

        rewards          = []
        losses           = []
        price_perfs      = []
        eval_log         = []

        for ep in range(1, max_episodes + 1):
            ep_reward, ep_loss, metrics = self._run_episode(eval_mode=False)
            self.agent.scheduler.step()
            rewards.append(ep_reward)
            losses.append(ep_loss)
            price_perfs.append(metrics.get('price_performance_bp', 0.0))

            if ep % 10 == 0:
                avg_bp = np.mean(price_perfs[-10:])
                print(f"Episode {ep:5d} | reward {ep_reward:+8.4f} | "
                      f"loss {ep_loss:.4f} | bp {metrics.get('price_performance_bp', 0):+.2f} "
                      f"| avg_bp(10) {avg_bp:+.2f} | eps {self.agent.current_epsilon():.3f}")

            if ep % eval_every == 0:
                eval_rewards = []
                eval_bps     = []
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
                self._save_model(f"dqn_ep{ep}.pth")

        self._save_model("dqn_final.pth")
        self._save_curves(rewards, losses, price_perfs, eval_log)
        return rewards, price_perfs, eval_log

    # ── 工具方法 ──────────────────────────────────────────────────────────────
    def _save_model(self, name: str):
        torch.save({
            'online_net':  self.agent.online_net.state_dict(),
            'target_net':  self.agent.target_net.state_dict(),
            'optimizer':   self.agent.optimizer.state_dict(),
            'steps_done':  self.agent.steps_done,
            'update_count': self.agent.update_count,
        }, os.path.join(self.result_dir, name))

    def load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.online_net.load_state_dict(ckpt['online_net'])
        self.agent.target_net.load_state_dict(ckpt['target_net'])
        self.agent.optimizer.load_state_dict(ckpt['optimizer'])
        self.agent.steps_done  = ckpt.get('steps_done', 0)
        self.agent.update_count = ckpt.get('update_count', 0)
        print(f"模型已加载: {path}")

    def _save_curves(self, rewards, losses, price_perfs, eval_log):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(rewards, alpha=0.4, label='raw')
        window = min(50, len(rewards))
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(rewards)), smoothed, label=f'MA{window}')
        axes[0].set_title('Episode Reward')
        axes[0].legend()

        axes[1].plot(losses, alpha=0.5)
        axes[1].set_title('TD Loss')

        axes[2].plot(price_perfs, alpha=0.4, label='raw bp')
        if len(price_perfs) >= window:
            smoothed_bp = np.convolve(price_perfs, np.ones(window)/window, mode='valid')
            axes[2].plot(range(window-1, len(price_perfs)), smoothed_bp, label=f'MA{window}')
        axes[2].axhline(0, color='red', linestyle='--', alpha=0.5, label='VWAP baseline')
        axes[2].set_title('Price Performance (bp)')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.result_dir, 'dqn_training_curves.png'), dpi=150)
        plt.close()

        # 保存 CSV
        csv_path = os.path.join(self.result_dir, 'dqn_training_log.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'loss', 'price_perf_bp'])
            for i, (r, l, b) in enumerate(zip(rewards, losses, price_perfs), 1):
                writer.writerow([i, r, l, b])

        # 保存 eval log
        json_path = os.path.join(self.result_dir, 'dqn_eval_log.json')
        with open(json_path, 'w') as f:
            json.dump(eval_log, f, indent=2)

        print(f"训练曲线已保存: {self.result_dir}/dqn_training_curves.png")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = SimplifiedEnvConfig()

    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f'./results/dqn_{timestamp}'

    print(f"DQN 训练开始 | device={device} | result_dir={result_dir}")
    trainer = DQNTrainer(config, result_dir=result_dir, device=device)
    trainer.train(max_episodes=5000, eval_every=1000, eval_episodes=5)
    print(f"\n训练完成，结果保存到: {result_dir}")


if __name__ == '__main__':
    main()
