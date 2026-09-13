import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
接口验证脚本 - 运行前先确认 const_path_ob.py 中的路径正确
运行方式: python "check_env.py"
"""
import sys
import os
import traceback
import numpy as np

# ─── 第1关：路径是否存在 ─────────────────────────────────────────────────────
print("=" * 55)
print("第1关: 检查数据路径")
print("=" * 55)

try:
    from const_path_ob import path_orders, path_snapshots, path_pkl_data
    paths = {
        "path_orders    (委托流水)": path_orders,
        "path_snapshots (LOB快照)":  path_snapshots,
        "path_pkl_data  (预处理pkl)": path_pkl_data,
    }
    all_exist = True
    for label, p in paths.items():
        exists = os.path.isdir(p)
        tag = "✅" if exists else "❌ 不存在"
        print(f"  {tag}  {label}")
        print(f"        {p}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⛔ 路径不存在，需要先修改 const_path_ob.py 指向正确位置，或把数据上传到对应路径。")
        print("   当前机器是 macOS，路径写的是 Linux 服务器路径，直接在本机跑会失败。")
        sys.exit(1)
    else:
        print("\n✅ 路径全部存在，继续检查。\n")

except ImportError as e:
    print(f"❌ 无法导入 const_path_ob: {e}")
    sys.exit(1)


# ─── 第2关：能否找到至少一条有效数据 ─────────────────────────────────────────
print("=" * 55)
print("第2关: 检查 pkl 数据文件")
print("=" * 55)

try:
    from const_simple import CODE_LIST, TRAIN_DATE_LIST

    found = None
    for code in CODE_LIST[:5]:          # 只查前5只股票
        for date in TRAIN_DATE_LIST[:3]: # 只查前3天
            pkl_path = os.path.join(path_pkl_data, f"SH{code}", f"{date}.pkl")
            if os.path.isfile(pkl_path):
                found = (code, date, pkl_path)
                break
        if found:
            break

    if found is None:
        print("❌ 在 path_pkl_data 下找不到任何 pkl 文件。")
        print("   可能需要先运行 DataPrepare 生成预处理数据。")
        sys.exit(1)
    else:
        code, date, pkl_path = found
        print(f"✅ 找到 pkl 文件: {pkl_path}\n")

except Exception as e:
    print(f"❌ 检查数据文件失败: {e}")
    sys.exit(1)


# ─── 第3关：env 能否 reset ────────────────────────────────────────────────────
print("=" * 55)
print("第3关: env.reset() 接口")
print("=" * 55)

try:
    from env_bs import SimplifiedEnvConfig, make_simplified_env

    config = SimplifiedEnvConfig()
    env = make_simplified_env(config)

    reset_ret = env.reset()

    # 检查返回值格式
    assert isinstance(reset_ret, tuple) and len(reset_ret) == 2, \
        f"reset() 应返回 (market_state, private_state)，实际返回: {type(reset_ret)}"

    market_state, private_state = reset_ret

    print(f"  market_state  shape = {np.array(market_state).shape}")
    print(f"  private_state shape = {np.array(private_state).shape}")
    print(f"  env.market_state_dim  = {env.market_state_dim}")
    print(f"  env.private_state_dim = {env.private_state_dim}")
    print(f"  action_space          = {env.action_space}")

    # 验证维度一致
    assert np.array(market_state).shape[0] == env.market_state_dim, \
        f"维度不匹配: 实际 {np.array(market_state).shape[0]} vs 配置 {env.market_state_dim}"
    assert np.array(private_state).shape[0] == env.private_state_dim, \
        f"维度不匹配: 实际 {np.array(private_state).shape[0]} vs 配置 {env.private_state_dim}"

    print("\n✅ reset() 接口正常\n")

except Exception as e:
    print(f"❌ reset() 失败:")
    traceback.print_exc()
    sys.exit(1)


# ─── 第4关：env 能否跑完一个 episode ─────────────────────────────────────────
print("=" * 55)
print("第4关: 完整跑一个 episode (随机动作)")
print("=" * 55)

try:
    market_state, private_state = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        # 随机动作：[limit_ratio, market_ratio] ∈ [0,1]^2
        action = env.action_space.sample()
        ret = env.step(action)

        assert len(ret) == 5, f"step() 应返回 5 元组，实际: {len(ret)}"
        market_state, private_state, reward, done, info = ret

        assert np.isfinite(reward), f"step {step}: reward={reward} 不是有限值"
        assert np.all(np.isfinite(market_state)), f"step {step}: market_state 含 nan/inf"

        total_reward += float(reward)
        step += 1

        if step > 400:  # 防止死循环
            print("  ⚠️  超过 400 步未结束，强制退出循环")
            break

    print(f"  共运行 {step} 步，累计奖励 = {total_reward:.4f}")
    print(f"\n✅ episode 正常结束\n")

except Exception as e:
    print(f"❌ step() 失败 (第 {step} 步):")
    traceback.print_exc()
    sys.exit(1)


# ─── 第5关：get_trading_metrics 能否正常返回 ─────────────────────────────────
print("=" * 55)
print("第5关: get_trading_metrics() 输出")
print("=" * 55)

try:
    metrics = env.get_trading_metrics()
    print(f"  price_performance_bp = {metrics.get('price_performance_bp', 'MISSING')}")
    print(f"  cancel_rate          = {metrics.get('cancel_rate', 'MISSING')}")
    print(f"  total_orders         = {metrics.get('total_orders', 'MISSING')}")
    print(f"  active_trades        = {metrics.get('active_trades', 'MISSING')}")
    print(f"  passive_trades       = {metrics.get('passive_trades', 'MISSING')}")
    print("\n✅ 指标输出正常\n")

except Exception as e:
    print(f"❌ get_trading_metrics() 失败:")
    traceback.print_exc()
    sys.exit(1)


# ─── 第6关：PPOTrainer 初始化能否成功 ────────────────────────────────────────
print("=" * 55)
print("第6关: PPOTrainer 初始化")
print("=" * 55)

try:
    import torch
    from ppo_bs import PPOTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  使用设备: {device}")

    trainer = PPOTrainer(config, result_dir="./results/check_run", device=device)
    print(f"  agent.policy:    {type(trainer.agent.policy).__name__}")
    print(f"  agent.value_net: {type(trainer.agent.value_net).__name__}")
    print("\n✅ PPOTrainer 初始化成功\n")

except Exception as e:
    print(f"❌ PPOTrainer 初始化失败:")
    traceback.print_exc()
    sys.exit(1)


# ─── 汇总 ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("✅ 全部 6 关通过，接口对齐，可以开始训练。")
print("   运行训练: python ppo_bs.py")
print("=" * 55)
