import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
离线数据预处理脚本。

职责：把服务器 ClickHouse 的原始 LOB 快照 pkl 处理成 bar_data pkl，
供 SimplifiedExecutionEnv 在训练时直接加载。

只需在数据更新时手动运行一次：
    python data/prepare.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool

from const_path_ob import path_snapshots, path_pkl_data
from orderbook import download_snapshots

NUM_CORES = 5


class DataPrepare(object):
    """原始 LOB 快照 → 1s bar_data pkl 的离线预处理管道。"""

    def __init__(self, config):
        self.config = config
        self.download_raw_data()
        os.makedirs(path_pkl_data, exist_ok=True)
        file_paths = self.obtain_file_paths()
        pool = Pool(NUM_CORES)
        pool.map(self.process_file, file_paths)
        pool.close()
        pool.join()

    def download_raw_data(self, level=10):
        os.makedirs(path_snapshots, exist_ok=True)
        download_snapshots(self.config.full_code_list, self.config.full_date_list, level)

    def obtain_file_paths(self):
        file_paths = []
        tickers = os.listdir(path_snapshots)
        for ticker in tickers:
            if ticker[2:] in self.config.full_code_list:
                dates = os.listdir(os.path.join(path_snapshots, ticker))
                pkl_dir = os.path.join(path_pkl_data, ticker)
                os.makedirs(pkl_dir, exist_ok=True)
                file_paths.extend([
                    (os.path.join(path_snapshots, ticker, date),
                     os.path.join(pkl_dir, date.split('.')[0] + '.pkl'))
                    for date in dates
                ])
        return file_paths

    def process_file(self, paths, debug=False):
        snapshot_path, pkl_path = paths

        with open(snapshot_path, 'rb') as f:
            data = pickle.load(f)
        data.rename(columns=self.config.column_trans_dic, inplace=True)

        trade_date = data.iloc[0]['tradeDate']
        data.iloc[-1, data.columns.get_loc('time')] = pd.to_datetime(trade_date + ' 14:57:00.0')
        snapshot_shape0, snapshot_shape1 = data.shape

        # 过滤异常日
        if snapshot_shape0 == 1:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='EMPTY')
        if data['volume'].max() <= 0:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='NO_VOL')
        if data['lastPrice'][data['lastPrice'] > 0].mean() >= 1.09 * data['prevClosePrice'].values[0]:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='LIMIT_UP')
        if data['lastPrice'][data['lastPrice'] > 0].mean() <= 0.91 * data['prevClosePrice'].values[0]:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='LIMIT_DO')

        if debug:
            print('Processing: {} → {}'.format(snapshot_path, pkl_path))

        # 原始快照 → 100ms
        data.index = pd.DatetimeIndex(data['time'])
        data = data.resample('100ms', closed='right', label='right').last().ffill()

        total_vol = data['volume'].values[-1]
        data['volume_dt'] = (data['volume'] - data['volume'].shift(1)).fillna(0)
        data['value_dt']  = (data['value']  - data['value'].shift(1) ).fillna(0)

        spread = (data['askPrice1'] - data['bidPrice1']).replace(0, np.nan)
        ask1_deal_volume_tick = ((data['value_dt'] - data['volume_dt'] * data['bidPrice1']) / spread).clip(upper=data['volume_dt'], lower=0)
        bid1_deal_volume_tick = ((data['volume_dt'] * data['askPrice1'] - data['value_dt']) / spread).clip(upper=data['volume_dt'], lower=0)
        data['market_order_flow_tick'] = ask1_deal_volume_tick + bid1_deal_volume_tick
        data['bid_volume_change']      = data['bidVolume1'] - data['bidVolume1'].shift(1)
        data['ask_volume_change']      = data['askVolume1'] - data['askVolume1'].shift(1)
        data['limit_order_flow_tick']  = data['bid_volume_change'] - data['ask_volume_change']

        data = data[data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        data = data[~data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]

        freq = self.config.simulation_commission_freq

        ask1_deal_volume_tick = ((data['value_dt'] - data['volume_dt'] * data['bidPrice1']) /
                                 (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)
        bid1_deal_volume_tick = ((data['volume_dt'] * data['askPrice1'] - data['value_dt']) /
                                 (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)

        max_last_price = data['lastPrice'].resample(freq).max().reindex(data.index).ffill()
        min_last_price = data['lastPrice'].resample(freq).min().reindex(data.index).ffill()
        ask1_deal_volume = ((data['askPrice1'] == max_last_price) * ask1_deal_volume_tick).resample(freq).sum()
        bid1_deal_volume = ((data['bidPrice1'] == min_last_price) * bid1_deal_volume_tick).resample(freq).sum()
        max_last_price = data['askPrice1'].resample(freq).max()
        min_last_price = data['bidPrice1'].resample(freq).min()

        level_infos = [f'{side}Price{i}'  for i in range(1, 11) for side in ('bid', 'ask')] + \
                      [f'{side}Volume{i}' for i in range(1, 11) for side in ('bid', 'ask')] + \
                      ['value_dt', 'volume_dt']
        # 保持原始顺序
        level_infos = [
            'bidPrice1','askPrice1','bidVolume1','askVolume1',
            'bidPrice2','askPrice2','bidVolume2','askVolume2',
            'bidPrice3','askPrice3','bidVolume3','askVolume3',
            'bidPrice4','askPrice4','bidVolume4','askVolume4',
            'bidPrice5','askPrice5','bidVolume5','askVolume5',
            'bidPrice6','askPrice6','bidVolume6','askVolume6',
            'bidPrice7','askPrice7','bidVolume7','askVolume7',
            'bidPrice8','askPrice8','bidVolume8','askVolume8',
            'bidPrice9','askPrice9','bidVolume9','askVolume9',
            'bidPrice10','askPrice10','bidVolume10','askVolume10',
            'value_dt','volume_dt',
        ]
        bar_data = data[level_infos].resample(freq).first()
        bar_data.iloc[-1] = bar_data.iloc[-1].replace(0.0, np.nan)
        bar_data.ffill(inplace=True)

        bar_data['max_last_price']    = max_last_price
        bar_data['min_last_price']    = min_last_price
        bar_data['ask1_deal_volume']  = ask1_deal_volume
        bar_data['bid1_deal_volume']  = bid1_deal_volume
        bar_data['market_order_flow'] = data['market_order_flow_tick'].resample(freq).sum()
        bar_data['limit_order_flow']  = data['limit_order_flow_tick'].resample(freq).sum()

        bar_data['basis_price']  = data['openPrice'].values[0]
        bar_data['basis_volume'] = total_vol

        bar_data['high_price']  = data['lastPrice'].resample(freq, closed='right', label='right').max()
        bar_data['low_price']   = data['lastPrice'].resample(freq, closed='right', label='right').min()
        bar_data['open_price']  = data['lastPrice'].resample(freq, closed='right', label='right').first()
        bar_data['close_price'] = data['lastPrice'].resample(freq, closed='right', label='right').last()
        bar_data['volume']      = data['volume_dt'].resample(freq, closed='right', label='right').sum()
        horizon = self.config.simulation_planning_horizon
        bar_data['volume_tol']  = bar_data['volume'][::-1].rolling(horizon, min_periods=1).sum()[::-1]

        value_bar = data['value_dt'].resample(freq, closed='right', label='right').sum()
        bar_data['vwap'] = np.where(bar_data['volume'] > 0,
                                    value_bar / bar_data['volume'],
                                    bar_data['close_price'])

        bar_data['volume_5min'] = data['volume_dt'].rolling(horizon).sum()
        bar_data['vwap_5min']   = (data['value_dt'].rolling(horizon).sum() / bar_data['volume_5min']).fillna(bar_data['close_price'])

        bar_data['ask_bid_spread'] = bar_data['askPrice1'] - bar_data['bidPrice1']
        bar_data['trend'] = (data['lastPrice'] - data['lastPrice'].shift(20)).fillna(0) \
                             .resample(freq, closed='right', label='right').last()

        bar_data['time'] = bar_data.index
        bar_data = bar_data[bar_data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        bar_data = bar_data[~bar_data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]
        bar_data['time_diff'] = (bar_data['time'] - bar_data['time'].values[0]) / np.timedelta64(1, 's') / 19800
        bar_data = bar_data.reset_index(drop=True)

        with open(pkl_path, 'wb') as f:
            pickle.dump(bar_data, f, pickle.HIGHEST_PROTOCOL)

        return dict(snapshot_path=snapshot_path, pkl_path=pkl_path,
                    snapshot_shape0=snapshot_shape0, snapshot_shape1=snapshot_shape1,
                    res_shape0=bar_data.shape[0], res_shape1=bar_data.shape[1])


if __name__ == '__main__':
    from const_simple import SimplifiedEnvConfig
    config = SimplifiedEnvConfig()
    print(f'开始预处理 {len(config.full_code_list)} 只股票 × {len(config.full_date_list)} 天...')
    DataPrepare(config)
    print('完成。')
