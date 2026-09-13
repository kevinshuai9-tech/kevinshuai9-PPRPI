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
│   └── iclr2025_conference.tex        
│
├── config/                            # Configuration Layer
│   ├── const_path_ob.py               # Server data paths
│   ├── const_simple.py                # List of Stock Tickers & Dates (44 Training + 16 Validation)
│   └── const_base.py                  
│
├── data/                              # Data Layer
│   ├── prepare.py                     # Original LOB snapshot → bar_data pkl (run once only)
│   ├── loader.py                      # Read the PKL and manage the time-step cursor
│   ├── orderbook.py                   # download data from ClickHouse
│   └── samples/                       # Download raw LOB snapshots
│       ├── md.csv                     # Sample Market Snapshot（600000，2024-12-02）
│       ├── trader.csv                 # Sample Transaction Data
│       └── long_order.csv             # Sample of Entrusted Transaction Records
│
├── env/                               # Trade Execution Environment Layer
│   ├── env_bs.py                      # main environment：SimplifiedEnvConfig + SimplifiedExecutionEnv
│   └── env_new.py                     # advanced environment（Under development）
│
├── agents/                            # RL agents（4）
│   ├── ppo_bs.py                      # PPO：on-policy，continuous [place_level, intensity]∈[0,1]²
│   ├── dqn.py                         # DQN：off-policy，discrete（4×5=20）
│   ├── qr_dqn.py                      # QR-DQN
│   └── fz_rl.py                       # FZ-DQN
│
├── baselines/                         # Rule Baseline Layer
│   ├── twap.py                        
│   ├── twap_plus.py                   
│   └── twap_xgb.py                    
│
├── survival/                          # Survival Analysis Layer（Fill Time Modelling）
│   ├── cox_estimation.py              # Cox model
│   ├── deep_survival.py               
│   └── survival_order.py              
│
├── scripts/                           
│   ├── compare_agents.py              
│   ├── run_sample.py                  
│   └── check_env.py                   
│
├── archive/                           # old documents
│   ├── env_v.py / env_v_copy.py / env_v_copy_ppo.py   
│   ├── policy_tuned_ppo.py / ppo0.py / simple_ppo.py  
│   ├── policy_tuned_dqn.py / storage.py               
│   ├── policy_cash_work.py / policy_cash_new.py        
│   └── Env Analysis.py / diagnose_env_vwap.py / plot1.py
│
└── README.md
```

### Algorithm level Comparison
```
TWAP     →     Enhanced TWAP   →   PPO / DQN   →   QR-DQN (mean)   →   QR-DQN (CVaR)   →     FZ-DQN ★
  ↑                ↑                  ↑               ↑                      ↑                  ↑
not learning    Rules + Signals       RL        Distributional Q    CVaR approximation    Direct Estimation of CVaR
```
FZ-DQN may not necessarily have the highest score on price_performance_bp, but it is the most robust in terms of tail risk (CVaR, worst-case execution price).
