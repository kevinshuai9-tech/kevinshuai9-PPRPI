import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
用本地 md.csv / long_order.csv 样例数据跑通完整 RL 训练循环。

绕过了：
  - DataPrepare 自动下载 / 预处理
  - 服务器路径依赖
  - const_path_ob.py 中的硬编码路径

验证的内容：
  - 数据处理管道（CSV → bar_data）
  - env reset / step 接口
  - PPO 网络前向 + 梯度更新
  - get_trading_metrics 输出
"""

import os, sys, warnings, pickle, tempfile, types
import numpy as np
import pandas as pd
import torch
warnings.filterwarnings('ignore')

HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)           # Sonny RL/
SAMPLES  = os.path.join(ROOT, "data", "samples")

# ──────────────────────────────────────────────────────────
# Step 0: 伪造 const_path_ob，指向临时目录
# ──────────────────────────────────────────────────────────
TMP_DIR   = tempfile.mkdtemp(prefix="sonny_sample_")
path_orders    = os.path.join(TMP_DIR, "orders")
path_snapshots = os.path.join(TMP_DIR, "snapshots")
path_pkl_data  = os.path.join(TMP_DIR, "pkl")

for p in [path_orders, path_snapshots, path_pkl_data]:
    os.makedirs(p, exist_ok=True)

# 注入假模块，在 import 之前覆盖
fake_path_mod = types.ModuleType("const_path_ob")
fake_path_mod.path_orders    = path_orders
fake_path_mod.path_snapshots = path_snapshots
fake_path_mod.path_pkl_data  = path_pkl_data
sys.modules["const_path_ob"] = fake_path_mod

# Mock clickhouse_driver + orderbook（服务器专用库，本地没有）
fake_ch = types.ModuleType("clickhouse_driver")
fake_ch.Client = object
sys.modules["clickhouse_driver"] = fake_ch

fake_ob = types.ModuleType("orderbook")
fake_ob.download_snapshots = lambda *a, **kw: None
sys.modules["orderbook"] = fake_ob

# gym → gymnasium 兼容层（只用到 spaces.Box）
try:
    import gym
except ModuleNotFoundError:
    import gymnasium
    fake_gym = types.ModuleType("gym")
    fake_gym.spaces = gymnasium.spaces
    sys.modules["gym"] = fake_gym

# ──────────────────────────────────────────────────────────
# Step 1: 读取样例 md.csv，处理成 bar_data
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 1: 处理样例 md.csv → bar_data")
print("=" * 55)

MD_CSV    = os.path.join(SAMPLES, "md.csv")
ORDER_CSV = os.path.join(SAMPLES, "long_order.csv")

assert os.path.exists(MD_CSV),    f"找不到 {MD_CSV}"
assert os.path.exists(ORDER_CSV), f"找不到 {ORDER_CSV}"

raw = pd.read_csv(MD_CSV)
print(f"  原始行数: {len(raw)}, 列数: {len(raw.columns)}")

# 列名重命名（与 DataPrepare 相同的映射）
rename = {
    'TradVolume': 'volume', 'LastPrice': 'lastPrice',
    'PreCloPrice': 'prevClosePrice', 'Turnover': 'value',
    'OpenPrice': 'openPrice', 'UpdateTime': 'time',
}
for i in range(1, 11):
    rename[f'BidPrice{i}']   = f'bidPrice{i}'
    rename[f'AskPrice{i}']   = f'askPrice{i}'
    rename[f'BidVolume{i}']  = f'bidVolume{i}'
    rename[f'AskVolume{i}']  = f'askVolume{i}'

raw.rename(columns=rename, inplace=True)

# 从 SecurityID 推断日期（long_order.csv 的 OrderTime 字段包含日期）
orders_raw = pd.read_csv(ORDER_CSV)
trade_date = str(pd.to_datetime(orders_raw['OrderTime'].iloc[0]).date())
print(f"  推断交易日期: {trade_date}")
code = str(int(orders_raw['SecurityID'].iloc[0]))
print(f"  股票代码: {code}")

# 设置时间索引（使用 LocalTime 字段作为 intraday 时间）
raw['time'] = pd.to_datetime(trade_date + ' ' + raw['time'].astype(str))
raw.index   = pd.DatetimeIndex(raw['time'])

# 过滤非交易时段
raw = raw[raw['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
raw = raw[~raw['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]

if len(raw) == 0:
    print("  ⚠️  过滤后无数据，尝试不过滤午休...")
    raw = pd.read_csv(MD_CSV)
    raw.rename(columns=rename, inplace=True)
    raw['time'] = pd.to_datetime(trade_date + ' ' + raw['time'].astype(str))
    raw.index   = pd.DatetimeIndex(raw['time'])
    raw = raw[raw['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]

print(f"  过滤后行数: {len(raw)}")

# 计算增量成交量 / 金额
raw['volume_dt'] = (raw['volume'] - raw['volume'].shift(1)).fillna(0).clip(lower=0)
raw['value_dt']  = (raw['value']  - raw['value'].shift(1) ).fillna(0).clip(lower=0)

# 1s 重采样
FREQ = '1s'
bar_data = raw[['bidPrice1','askPrice1','bidVolume1','askVolume1',
                'bidPrice2','askPrice2','bidVolume2','askVolume2',
                'bidPrice3','askPrice3','bidVolume3','askVolume3',
                'bidPrice4','askPrice4','bidVolume4','askVolume4',
                'bidPrice5','askPrice5','bidVolume5','askVolume5',
                'bidPrice6','askPrice6','bidVolume6','askVolume6',
                'bidPrice7','askPrice7','bidVolume7','askVolume7',
                'bidPrice8','askPrice8','bidVolume8','askVolume8',
                'bidPrice9','askPrice9','bidVolume9','askVolume9',
                'bidPrice10','askPrice10','bidVolume10','askVolume10',
                'value_dt','volume_dt']].resample(FREQ).first().ffill()

# 必要派生特征
bar_data['close_price'] = raw['lastPrice'].resample(FREQ).last().ffill()
bar_data['volume']      = raw['volume_dt'].resample(FREQ).sum()
bar_data['value_dt']    = raw['value_dt'].resample(FREQ).sum()
bar_data['volume_dt']   = raw['volume_dt'].resample(FREQ).sum()

vol_bar  = bar_data['volume'].replace(0, np.nan)
val_bar  = raw['value_dt'].resample(FREQ).sum()
bar_data['vwap'] = (val_bar / vol_bar).fillna(bar_data['close_price'])

# 5min 滚动 vwap
bar_data['volume_5min'] = bar_data['volume_dt'].rolling(300, min_periods=1).sum()
bar_data['vwap_5min']   = (bar_data['value_dt'].rolling(300, min_periods=1).sum() /
                            bar_data['volume_5min'].replace(0, np.nan)).fillna(bar_data['close_price'])

# 微结构特征
bar_data['ask_bid_spread'] = (bar_data['askPrice1'] - bar_data['bidPrice1']).fillna(0)
bar_data['trend']          = (bar_data['close_price'] - bar_data['close_price'].shift(20)).fillna(0)
bar_data['volatility']     = bar_data['close_price'].rolling(20, min_periods=1).std().fillna(0)
bar_data['ab_volume_misbalance'] = (
    bar_data[['askVolume1','askVolume2','askVolume3','askVolume4','askVolume5']].sum(axis=1) -
    bar_data[['bidVolume1','bidVolume2','bidVolume3','bidVolume4','bidVolume5']].sum(axis=1)
)
bar_data['transaction_net_volume'] = 0.0
bar_data['immediate_market_order_cost_bid'] = 0.0
bar_data['immediate_market_order_cost_ask'] = 0.0
bar_data['VOLR'] = 0.0; bar_data['PCTN'] = 0.0; bar_data['MidMove'] = 0.0
bar_data['BSP']  = 0.0; bar_data['weighted_price'] = 1.0; bar_data['order_imbalance'] = 0.0
bar_data['trend_strength'] = 0.0
bar_data['market_order_flow'] = 0.0; bar_data['limit_order_flow'] = 0.0
bar_data['max_last_price'] = bar_data['close_price']
bar_data['min_last_price'] = bar_data['close_price']
bar_data['ask1_deal_volume'] = 0.0; bar_data['bid1_deal_volume'] = 0.0
bar_data['high_price'] = bar_data['close_price']
bar_data['low_price']  = bar_data['close_price']
bar_data['open_price'] = bar_data['close_price']
bar_data['volume_tol'] = bar_data['volume_dt'][::-1].rolling(300, min_periods=1).sum()[::-1]

# 规范化基准
basis_price  = float(bar_data['close_price'].iloc[0]) or 1.0
basis_volume = float(bar_data['volume_dt'].sum()) or 1e6
bar_data['basis_price']  = basis_price
bar_data['basis_volume'] = basis_volume

# 时间特征
bar_data['time'] = bar_data.index
bar_data['time_diff'] = (bar_data.index - bar_data.index[0]).total_seconds() / 19800.0
bar_data = bar_data.reset_index(drop=True)

print(f"  bar_data 形状: {bar_data.shape}")
print(f"  basis_price={basis_price:.2f}, basis_volume={basis_volume:.0f}")

# 保存为 pkl（路径匹配 env 读取逻辑）
stock_dir = os.path.join(path_pkl_data, f"SH{code}")
os.makedirs(stock_dir, exist_ok=True)
pkl_path  = os.path.join(stock_dir, f"{trade_date}.pkl")
with open(pkl_path, 'wb') as f:
    pickle.dump(bar_data, f)
print(f"  已保存: {pkl_path}")

# 生成 orders pkl（env 的 data_exists() 同时检查两个文件）
orders_df = pd.read_csv(ORDER_CSV)
orders_df['OrderTime'] = pd.to_datetime(orders_df['OrderTime'], format='mixed')
order_dir  = os.path.join(path_orders, f"SH{code}")
os.makedirs(order_dir, exist_ok=True)
order_pkl  = os.path.join(order_dir, f"{trade_date}.pkl")
with open(order_pkl, 'wb') as f:
    pickle.dump(orders_df, f)
print(f"  已保存: {order_pkl}")
print()

# ──────────────────────────────────────────────────────────
# Step 2: Patch Data.__init__ — 跳过 DataPrepare 自动运行
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 2: Patch env_bs.Data (跳过 DataPrepare)")
print("=" * 55)

import env_bs

# 用只保存 config 的轻量 __init__ 替换
_orig_data_init = env_bs.Data.__init__
def _patched_data_init(self, config):
    self.config = config
    # skip DataPrepare entirely
env_bs.Data.__init__ = _patched_data_init

# Patch obtain_data 允许行数不等于 14222（样例数据可能更少）
_orig_obtain_data = env_bs.Data.obtain_data
def _patched_obtain_data(self, code, date, start_index=None, do_normalization=False):
    pkl = os.path.join(path_pkl_data, f"SH{code}", f"{date}.pkl")
    with open(pkl, 'rb') as f:
        self.data = pickle.load(f)
    # 不做行数断言，直接设 horizon
    if start_index is None:
        start_index = self._random_valid_start_index()
    self._set_horizon(start_index)
    self.basis_price  = float(self.data['basis_price'].iloc[0]) or 1.0
    self.basis_volume = float(self.data['basis_volume'].iloc[0]) or 1e6
env_bs.Data.obtain_data = _patched_obtain_data

print("  Patch 完成\n")

# ──────────────────────────────────────────────────────────
# Step 3: 构造 Config，只用样例股票
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 3: 构造 Config")
print("=" * 55)

from env_bs import SimplifiedEnvConfig, make_simplified_env

class SampleConfig(SimplifiedEnvConfig):
    code_list = [code]
    date_list = [trade_date]
    full_code_list = [code]
    full_date_list = [trade_date]
    simulation_lookback_horizon = 5          # 缩小回看窗口加快测试
    simulation_planning_horizon = 50         # 缩短 episode 长度
    simulation_volume_ratio = 0.001          # 极小交易量，适配样例数据量级

config = SampleConfig()
print(f"  code={code}, date={trade_date}")
print(f"  lookback={config.simulation_lookback_horizon}, "
      f"horizon={config.simulation_planning_horizon}\n")

# ──────────────────────────────────────────────────────────
# Step 4: env.reset()
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 4: env.reset()")
print("=" * 55)

env = make_simplified_env(config)
market_state, private_state = env.reset()

print(f"  market_state  shape = {market_state.shape}")
print(f"  private_state shape = {private_state.shape}")
print(f"  market_state_dim    = {env.market_state_dim}")
print(f"  private_state_dim   = {env.private_state_dim}")
print(f"  action_space        = {env.action_space}")
print(f"  total_quantity      = {env.total_quantity}")

assert market_state.shape[0] == env.market_state_dim, \
    f"维度不匹配: {market_state.shape[0]} vs {env.market_state_dim}"
assert private_state.shape[0] == env.private_state_dim
print("  ✅ reset 接口正常\n")

# ──────────────────────────────────────────────────────────
# Step 5: 跑完一个 episode（随机动作）
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 5: 随机策略跑一个完整 episode")
print("=" * 55)

market_state, private_state = env.reset()
done, step, total_reward = False, 0, 0.0

while not done:
    action = env.action_space.sample()
    ret    = env.step(action)
    assert len(ret) == 5, f"step 应返回 5 元组，实际: {len(ret)}"
    market_state, private_state, reward, done, info = ret
    assert np.all(np.isfinite(market_state)),  f"step {step}: market_state 含 nan/inf"
    assert np.all(np.isfinite(private_state)), f"step {step}: private_state 含 nan/inf"
    assert np.isfinite(reward),                f"step {step}: reward={reward} 非有限值"
    total_reward += float(reward)
    step += 1

print(f"  共 {step} 步，总奖励 = {total_reward:.4f}")
metrics = env.get_trading_metrics()
print(f"  price_performance_bp = {metrics.get('price_performance_bp', 'N/A'):.4f}")
print(f"  cancel_rate          = {metrics.get('cancel_rate', 'N/A'):.4f}")
print(f"  total_orders         = {metrics.get('total_orders', 'N/A')}")
print("  ✅ episode 正常\n")

# ──────────────────────────────────────────────────────────
# Step 6: PPOAgent 前向 + 一次梯度更新
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("Step 6: PPOAgent 前向传播 + 一次 update()")
print("=" * 55)

from ppo_bs import PPOAgent

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  设备: {device}")

agent = PPOAgent(
    market_state_dim  = env.market_state_dim,
    private_state_dim = env.private_state_dim,
    action_dim        = env.action_space.shape[0],
    device=device
)

# 跑一个 episode，收集 buffer
market_state, private_state = env.reset()
done = False
while not done:
    ms_t = torch.FloatTensor(market_state).unsqueeze(0).to(device)
    ps_t = torch.FloatTensor(private_state).unsqueeze(0).to(device)

    with torch.no_grad():
        action, log_prob, entropy = agent.select_action(market_state, private_state)
        value = agent.value_net(ms_t, ps_t).squeeze().item()

    ret = env.step(action)
    next_ms, next_ps, reward, done, info = ret

    # buffer 格式: (ms, ps, action, log_prob, reward, done, next_ms, next_ps, value)
    agent.buffer.append((
        market_state, private_state,
        action, float(log_prob),
        float(reward), done,
        next_ms, next_ps,
        value
    ))
    market_state, private_state = next_ms, next_ps

print(f"  buffer 大小: {len(agent.buffer)} 条")

# 执行一次 PPO update
policy_loss, value_loss, entropy_loss = agent.update()
print(f"  policy_loss={policy_loss:.4f}  value_loss={value_loss:.4f}  entropy={entropy_loss:.4f}")
print("  ✅ PPO 梯度更新正常\n")

# ──────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────
print("=" * 55)
print("✅ 全部通过！接口验证成功。")
print()
print("已确认：")
print("  • md.csv 可正确处理成 bar_data")
print("  • env reset / step / get_trading_metrics 接口正常")
print("  • PPO 前向传播和梯度更新正常")
print()
print("去服务器前，只需将 const_path_ob.py 路径确认正确，")
print("并确保 pkl 数据已预处理完成，即可直接运行 ppo_bs.py。")
print("=" * 55)
