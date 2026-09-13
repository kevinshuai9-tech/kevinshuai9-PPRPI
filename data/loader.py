import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
训练时的数据加载层。

职责：加载预处理好的 bar_data pkl，管理时间步游标（current_index），
供 SimplifiedExecutionEnv 在每个 step 中查询当前市场数据。

依赖：data/prepare.py 已经跑过，pkl 文件存在于 path_pkl_data。
"""

import os
import pickle
import pandas as pd
import numpy as np

from const_path_ob import path_pkl_data, path_orders


class Data(object):
    """bar_data pkl 加载器 + 时间步游标。

    不触发任何数据下载或预处理，直接读取已有的 pkl 文件。
    """

    def __init__(self, config):
        self.config = config
        self.data   = None
        self.orders = None

    # ── 文件存在性检查 ────────────────────────────────────────────────────────

    def data_exists(self, code: str, date: str) -> bool:
        bar_ok   = os.path.isfile(os.path.join(path_pkl_data, f'SH{code}', f'{date}.pkl'))
        order_ok = os.path.isfile(os.path.join(path_orders,   f'SH{code}', f'{date}.pkl'))
        return bar_ok and order_ok

    # ── 数据加载 ──────────────────────────────────────────────────────────────

    def obtain_data(self, code: str, date: str, start_index=None, do_normalization=True):
        """加载指定股票/日期的 bar_data，设定 episode 起止索引。"""
        with open(os.path.join(path_pkl_data, f'SH{code}', f'{date}.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        try:
            with open(os.path.join(path_orders, f'SH{code}', f'{date}.pkl'), 'rb') as f:
                orders = pickle.load(f)
            orders = orders[
                (pd.to_datetime(f'{date} 09:30:00') <= orders['OrderTime']) &
                (orders['OrderTime'] < pd.to_datetime(f'{date} 14:57:00'))
            ]
            self.orders = orders[~orders['OrderTime'].between(f'{date} 11:30:01', f'{date} 12:59:59')]
        except FileNotFoundError:
            self.orders = None

        expected = self._expected_bar_count()
        assert self.data.shape[0] == expected, \
            f'bar_data 行数应为 {expected}，实际为 {self.data.shape[0]}'

        if start_index is None:
            start_index = self._random_valid_start_index()
        self._set_horizon(start_index)
        if start_index is not None:
            assert self._sanity_check(), \
                f'code={code} date={date} start_index={start_index} 数据异常'

        self.basis_price  = float(self.data.loc[self.start_index, 'close_price'])
        self.basis_volume = float(self.data['basis_volume'].values[0])

    def _expected_bar_count(self) -> int:
        freq_map = {'1min': 239, '10s': 1424, '1s': 14222}
        return freq_map.get(self.config.simulation_commission_freq, 14222)

    # ── 时间步管理 ────────────────────────────────────────────────────────────

    def _random_valid_start_index(self) -> int:
        cols  = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']
        valid = (self.data[cols] > 0).all(axis=1)
        look  = getattr(self.config, 'simulation_lookback_horizon', 1)
        fwd   = self.config.simulation_planning_horizon

        # 往前 look 步、往后 fwd 步都要有效
        tmp1 = valid.rolling(look).apply(lambda x: x.all())
        tmp2 = valid[::-1].rolling(fwd + 1).apply(lambda x: x.all())[::-1]
        candidates = valid.loc[(tmp1 > 0) & (tmp2 > 0)].index.tolist()
        assert len(candidates) > 0, '无有效起始点'
        return int(np.random.choice(candidates))

    def _set_horizon(self, start_index: int):
        self.start_index   = start_index
        self.current_index = start_index
        self.end_index     = start_index + self.config.simulation_planning_horizon

    def _sanity_check(self) -> bool:
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']
        return not (self.data.loc[self.start_index:self.end_index, cols] == 0).any(axis=None)

    # ── 环境查询接口 ──────────────────────────────────────────────────────────

    def obtain_level(self, name: str, level=''):
        """取当前时间步的某个字段值，例如 obtain_level('bidPrice', 1)。"""
        return self.data.loc[self.current_index, f'{name}{level}']

    def step(self):
        """时间步前进一格。"""
        self.current_index += 1
