# From Value Distributions to Policy Risk
## An Axiomatic Approach for Risk-Sensitive Reinforcement Learning

> ICLR 2025 Paper Submission project

## Project description

The core idea is to extend classical distributed reinforcement learning (Distributional RL) to risk metrics in the policy space; propose an axiomatic framework centered on the Fissler-Ziegel joint loss (FZ0 Loss); and validate the approach through systematic comparisons with TWAP, DQN, and QR-DQN in the context of high-frequency limit order book (LOB) matching and execution on Chinese stock market.

**Key Contributions of the Paper: FZ-DQN**
- Uses the FZ0 joint loss to directly estimate both VaR and CVaR simultaneously (strictly consistent, no approximation)
- Uses CVaR as the policy optimization objective to achieve risk-sensitive execution
- Fundamental difference from QR-DQN: QR-DQN’s CVaR is an a posteriori approximation, whereas FZ-DQN’s CVaR is a strictly joint-estimated minimizer

## Documents
```
kevinshuai9-tech.github.io/
├── paper/
│   └── iclr2025_conference.tex        # 论文主文件（ICLR 2025 投稿）
│
├── config/                            # 全局配置层
│   ├── const_path_ob.py               # 服务器数据路径（path_snapshots / path_pkl_data）
│   ├── const_simple.py                # 股票代码 & 日期列表（44训练 + 16验证）
│   └── const_base.py                  # 基础超参数
│
├── data/                              # 数据层
│   ├── prepare.py                     # 离线预处理：原始LOB快照 → bar_data pkl（只跑一次）
│   ├── loader.py                      # 训练时数据加载：读取pkl，管理时间步游标
│   ├── orderbook.py                   # 从ClickHouse下载原始LOB快照
│   └── samples/                       # 本地样例数据
│       ├── md.csv                     # 行情快照样本（600000，2024-12-02）
│       ├── trader.csv                 # 成交数据样本
│       └── long_order.csv             # 委托流水样本
│
├── env/                               # 交易执行环境层
│   ├── env_bs.py                      # 主环境：SimplifiedEnvConfig + SimplifiedExecutionEnv
│   └── env_new.py                     # 下一版环境（分层动作空间，开发中）
│
├── agents/                            # RL 智能体层（已实现 4 种）
│   ├── ppo_bs.py                      # PPO：on-policy，连续动作 [place_level, intensity]∈[0,1]²
│   ├── dqn.py                         # DQN：off-policy，离散动作（4×5=20种）
│   ├── qr_dqn.py                      # QR-DQN：分布式Q，N=32分位数，CVaR近似（baseline）
│   └── fz_rl.py                       # FZ-DQN：FZ0联合损失，双头直接估计VaR+CVaR（论文方法）
│
├── baselines/                         # 规则基线层
│   ├── twap.py                        # TWAP：标准均匀拆单（学术标准基线）
│   ├── twap_plus.py                   # 增强TWAP：OFI/动量/流动性四路信号融合（最强规则基线）
│   └── twap_xgb.py                    # XGBoost信号融合（辅助分析，非主对比）
│
├── survival/                          # 生存分析层（Fill Time 建模）
│   ├── cox_estimation.py              # Cox比例风险模型
│   ├── deep_survival.py               # DeepHit神经生存模型
│   └── survival_order.py              # 生存分析数据集构建
│
├── scripts/                           # 运行脚本
│   ├── compare_agents.py              # 七算法统一对比（论文实验主脚本）
│   ├── run_sample.py                  # 本地样例测试（无需服务器）
│   └── check_env.py                   # 服务器环境接口验证（6关检查）
│
├── archive/                           # 已归档的旧版文件
│   ├── env_v.py / env_v_copy.py / env_v_copy_ppo.py   # 旧版5档LOB环境
│   ├── policy_tuned_ppo.py / ppo0.py / simple_ppo.py  # 旧版PPO（依赖env_v）
│   ├── policy_tuned_dqn.py / storage.py               # 旧版DQN
│   ├── policy_cash_work.py / policy_cash_new.py        # CASH算法（IJCAI 2023复现）
│   └── Env Analysis.py / diagnose_env_vwap.py / plot1.py
│
└── README.md
```

### Algorithm level Comparison
```
TWAP   →     Enhanced TWAP → PPO / DQN → QR-DQN (mean) → QR-DQN (CVaR) → FZ-DQN ★
  ↑                ↑              ↑               ↑                ↑            ↑
not learning
