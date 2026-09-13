import os
import pickle
import pandas as pd
import numpy as np  
from const_path_ob import *
from const_simple import *
from orderbook import download_snapshots
from multiprocessing import Pool
import itertools

prefixes = ['bid', 'ask']
# 需要几档的快照就可以生成几档的快照数据
levels = 3                                         
FEATURE_LOB = []
for prefix in prefixes:
    for level in range(1, levels + 1):
        FEATURE_LOB.append(f"{prefix}Price{level}")
    for level in range(1, levels + 1):
        FEATURE_LOB.append(f"{prefix}Volume{level}")
    for level in range(1, levels + 1):
        FEATURE_LOB.append(f"NumOrders{'B' if prefix == 'bid' else 'S'}{level}")

FEATURE_SET_LOB = FEATURE_LOB + ['close_price', 'volume', 'vwap', 'time_diff']

FEATURE_SET_FULL = FEATURE_SET_LOB + [
    'ask_bid_spread', 'ab_volume_misbalance', 'transaction_net_volume', 
    'volatility', 'trend', 'immediate_market_order_cost_bid', 
    'VOLR', 'PCTN', 'MidMove', 'weighted_price', 'order_imbalance', 
    'trend_strength'
]

NUM_CORES = 5

class DefaultConfig(object):
    # 订单簿数据字段列名重置
    column_trans_dic = {'TradVolume':'volume','LastPrice':'lastPrice','PreCloPrice':'prevClosePrice','Turnover':'value','OpenPrice':'openPrice','UpdateTime':'time'}
    for i in range(1,11):
        column_trans_dic['BidPrice{}'.format(i)] = 'bidPrice{}'.format(i)
        column_trans_dic['AskPrice{}'.format(i)] = 'askPrice{}'.format(i)
        column_trans_dic['BidVolume{}'.format(i)] = 'bidVolume{}'.format(i)
        column_trans_dic['AskVolume{}'.format(i)] = 'askVolume{}'.format(i)
    # 训练集合的code与date
    code_list = CODE_LIST
    date_list = TRAIN_DATE_LIST
    # 所有需要的 code 与 date
    full_code_list = CODE_LIST + VALIDATION_CODE_LIST
    full_date_list = TRAIN_DATE_LIST + VALIDATION_DATE_LIST
    # ##################################### 交易参数设置 ######################################
    # 计划交易周期为: 5 mins = 300 seconds
    # 模拟下单频率： 1min, 10s, 1s  (可以选择为交易延迟时间(便于后续改进)，也可以选择为忽略延迟的时间间隔(更贴近假设条件))
    simulation_commission_freq = '1s'
    simulation_planning_horizon = 300 # = 300s / simulation_commission_freq
    feature_horizon = 10 # = simulation_commission_freq / 100ms 变化率特征的回溯尺度
    # ############################### 拆单方式及交易量(动作空间) ###############################
    # 订单交易量 单位 = 总交易量 / 拆单数
    simulation_num_shares = 10
    # 离散动作空间步数 最多一次打出几个单位的交易量
    simulation_discrete_quantities = 3
    if simulation_discrete_quantities > simulation_num_shares:
        simulation_discrete_quantities = simulation_num_shares
    # 交易执行量与标的股票该时间段总交易量的比例 
    simulation_volume_ratio = 0.05
    # ############################### 奖励函数的惩罚系数 ###############################
    # 鼓励即时成交, 只有足够信心才鼓励持仓被动成交
    holding_cost_coeff = 0.0
    # 鼓励时间上均匀清算
    simulation_linear_reg_coeff = 0.0
    # 任务量报完后停止报单
    simulation_liq_coeff = 100.0
    # 鼓励委托尽可能成交
    simulation_cancel_coeff = 0.0
    # 市场变量特征名
    simulation_features = FEATURE_SET_FULL
    # 当前时刻向前特征数据的聚合数量
    simulation_lookback_horizon = 10
    # Whether return flattened or stacked features of the past x bars
    simulation_do_feature_flatten = True
    # 清算任务方向
    simulation_direction = 'sell' # or 'buy'
    qty_slices = 10

    # 市场冲击参数
    market_impact_coeff = 0.1  # 市场冲击系数
    
    # ############################### 交易验证参数 ###############################
    # 价格偏离容忍度
    price_deviation_tolerance = 0.02  # 2%的价格偏离容忍度

    # ------------------- 分层动作空间定义 -------------------
    # 添加被动成交默认数量比例配置
    passive_default_qty_ratio = 0.25  # 默认为剩余头寸的25%
    # 高层动作空间：0 -> 'active', 1 -> 'passive'
    simulation_high_level_actions = ['active', 'passive', 'no operation']
    # 低层动作空间 - 被动成交 (挂单) - 价格偏移 (单位: 基点)
    bps_passive = np.linspace(-10, 40, 51, dtype=np.float32)  # 价格偏移从-10到+40基点
    # 被动成交的离散动作组合 (价格偏移)
    simulation_discrete_actions_passive = list(bps_passive)

    # 低层动作空间 - 主动成交
    # 数量比例 (相对于剩余头寸的比例)
    active_qty_ratios = np.array([0.0, 0.33, 0.66, 1.0]) # 0%, 33%, 66%, 100%
    simulation_discrete_actions_active = list(active_qty_ratios)
    
    simulation_discrete_actions_0 = [(0, 0)]
    
    len_active = len(simulation_discrete_actions_active)
    len_passive = len(simulation_discrete_actions_passive)
    simulation_discrete_actions = list(simulation_discrete_actions_active + simulation_discrete_actions_passive + simulation_discrete_actions_0)
    ################################ END ###############################

class DataPrepare(object):
    """
    数据准备: 
        处理原始snapshot文件来模拟环境所需的文件
        即是将逐笔快照pkl数据转为bar级pkl数据
    """
    def __init__(self, config):
        self.config = config

        self.download_raw_data()
        if not os.path.isdir(path_pkl_data):
            os.makedirs(path_pkl_data)
        file_paths = self.obtain_file_paths()

        res = []
        for file_path in file_paths:
            res.append(self.process_file(file_path))
        pool = Pool(NUM_CORES)
        res = pool.map(self.process_file, file_paths)
        # pd.DataFrame(res).to_csv('data_generation_report.csv')

    def download_raw_data(self, level=10):
        if not os.path.isdir(path_snapshots):
            os.makedirs(path_snapshots)
        download_snapshots(self.config.full_code_list, self.config.full_date_list, level)

    def obtain_file_paths(self):

        file_paths = []
        tickers = os.listdir(path_snapshots)
        for ticker in tickers:
            if ticker[2:] in self.config.full_code_list:
                dates = os.listdir(os.path.join(path_snapshots, ticker))
                file_paths.extend([
                    (os.path.join(path_snapshots, ticker, date), 
                    os.path.join(path_pkl_data, ticker, date.split('.')[0] + '.pkl')) for date in dates])
                if not os.path.isdir(os.path.join(path_pkl_data, ticker)):
                    os.makedirs(os.path.join(path_pkl_data, ticker))
        return file_paths

    def process_file(self, paths, debug=True):

        snapshot_path, pkl_path = paths

        # Step 1: 读取数据,将第一列UpdateTime设为索引
        with open(snapshot_path, 'rb') as f:
            data = pickle.load(f)
        data.rename(columns=self.config.column_trans_dic, inplace=True)

        # 为要求交易在收盘竞价之前完成,将最后一行数据的时间戳设为14:57:00
        # 逐笔数据倒数第二行时间戳在14:57:00之前, 倒数第一行时间戳为15:00:00, 因此仅需直接修改最后一行时间戳
        trade_date = data.iloc[0]['tradeDate']
        data.iloc[-1, data.columns.get_loc('time')] = pd.to_datetime(trade_date + ' 14:57:00.0')
        snapshot_shape0, snapshot_shape1 = data.shape

        # 过滤异常数据 (如:股票今日不可交易)
        if snapshot_shape0 == 1:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='EMPTY')
        if data['volume'].max() <= 0:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='NO_VOL')
        if data['lastPrice'][data['lastPrice'] > 0].mean() >= 1.09 * data['prevClosePrice'].values[0]:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='LIMIT_UP')
        if data['lastPrice'][data['lastPrice'] > 0].mean() <= 0.91 * data['prevClosePrice'].values[0]:
            return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, status='LIMIT_DO')

        if debug:
            print('Current process: {} {} Shape: {}'.format(snapshot_path, pkl_path, data.shape))

        # Step 2: 原始快照数据格式化
        data.index = pd.DatetimeIndex(data['time'])

        # 统一原始逐笔订单簿的时间尺度为100ms间隔, 为后续计算变化率等特征提供快照数据
        data = data.resample('100ms', closed='right', label='right').last().ffill()

        total_vol = data['volume'].values[-1]              

        # 计算相关变化量
        # volume列表示截至当前时间戳的累计交易量
        data['volume_dt'] = (data['volume'] - data['volume'].shift(1)).fillna(0)
        # value列表示截至当前时间戳的累计交易金额
        data['value_dt'] = (data['value'] - data['value'].shift(1)).fillna(0)

        # 除去集合竞价期间快照数据
        data = data[data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        data = data[~data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]

        # Step 3: 生成需要的bar-level信息
        # 转换数据时间间隔
        #   1) 当前订单簿快照(5档卖/买价/量)
        #   2) 表明部分成交的最低/最高卖/买价
        ask1_deal_volume_tick = ((data['value_dt'] - data['volume_dt'] * data['bidPrice1']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)
        bid1_deal_volume_tick = ((data['volume_dt'] * data['askPrice1'] - data['value_dt']) \
            / (data['askPrice1'] - data['bidPrice1'])).clip(upper=data['volume_dt'], lower=0)

        max_last_price = data['lastPrice'].resample(self.config.simulation_commission_freq).max().reindex(data.index).ffill()
        min_last_price = data['lastPrice'].resample(self.config.simulation_commission_freq).min().reindex(data.index).ffill()

        ask1_deal_volume = ((data['askPrice1'] == max_last_price) * ask1_deal_volume_tick).resample(self.config.simulation_commission_freq).sum()
        bid1_deal_volume = ((data['bidPrice1'] == min_last_price) * bid1_deal_volume_tick).resample(self.config.simulation_commission_freq).sum()
        max_last_price = data['askPrice1'].resample(self.config.simulation_commission_freq).max()
        min_last_price = data['bidPrice1'].resample(self.config.simulation_commission_freq).min()

        # 当前10档卖/买价/量 (为模拟MO对市场的暂时影响建模)
        level_infos = FEATURE_LOB
        bar_data = data[level_infos].resample(self.config.simulation_commission_freq).first()

        # 修复bug:最后一行snapshot数据缺失
        bar_data.iloc[-1] = bar_data.iloc[-1].replace(0.0, np.nan)
        bar_data.ffill(inplace=True)

        # 当前直到下个bar数据之前的最优可执行价格/数量 (for modeling temporary market impact of LOs)
        bar_data['max_last_price'] = max_last_price
        bar_data['min_last_price'] = min_last_price
        bar_data['ask1_deal_volume'] = ask1_deal_volume
        bar_data['bid1_deal_volume'] = bid1_deal_volume

        # Step 4: 生成状态特征
        # 规范化所需常数
        bar_data['basis_price'] = data['openPrice'].values[0]
        bar_data['basis_volume'] = total_vol

        # Bar信息
        bar_data['high_price'] = data['lastPrice'].resample(self.config.simulation_commission_freq, closed='right', label='right').max()
        bar_data['low_price'] = data['lastPrice'].resample(self.config.simulation_commission_freq, closed='right', label='right').min()
        bar_data['high_low_price_diff'] = bar_data['high_price'] - bar_data['low_price']
        bar_data['open_price'] = data['lastPrice'].resample(self.config.simulation_commission_freq, closed='right', label='right').first()
        bar_data['close_price'] = data['lastPrice'].resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['volume'] = data['volume_dt'].resample(self.config.simulation_commission_freq, closed='right', label='right').sum()
        bar_data['volume_tol'] = bar_data['volume'][::-1].rolling(self.config.simulation_planning_horizon, min_periods=1).sum()[::-1] # 当前时间戳未来5min的总交易量
        
        bar_data['vwap'] = data['value_dt'].resample(self.config.simulation_commission_freq, closed='right', label='right').sum() / bar_data['volume']
        bar_data['vwap'] = bar_data['vwap'].fillna(bar_data['close_price'])

        bar_data['volume_5min'] = data['volume_dt'].rolling(self.config.simulation_planning_horizon).sum()
        bar_data['vwap_5min'] = data['value_dt'].rolling(self.config.simulation_planning_horizon).sum() / bar_data['volume_5min']
        bar_data['vwap_5min'] = bar_data['vwap_5min'].fillna(bar_data['close_price'])

        # 限价订单簿特征
        bar_data['ask_bid_spread'] = bar_data['askPrice1'] - bar_data['bidPrice1']
        bar_data['ab_volume_misbalance'] = \
            (bar_data['askVolume1'] + bar_data['askVolume2'] + bar_data['askVolume3'] + bar_data['askVolume4'] + bar_data['askVolume5']) \
            - (bar_data['bidVolume1'] + bar_data['bidVolume2'] + bar_data['bidVolume3'] + bar_data['bidVolume4'] + bar_data['bidVolume5']) 
        bar_data['transaction_net_volume'] = (ask1_deal_volume_tick - bid1_deal_volume_tick).resample(self.config.simulation_commission_freq, closed='right', label='right').sum()
        bar_data['volatility'] = data['lastPrice'].rolling(20, min_periods=1).std().fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['trend'] = (data['lastPrice'] - data['lastPrice'].shift(20)).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['immediate_market_order_cost_ask'] = self._calculate_immediate_market_order_cost(bar_data, 'ask')
        bar_data['immediate_market_order_cost_bid'] = self._calculate_immediate_market_order_cost(bar_data, 'bid')

        # 新限价订单簿特征
        horizon = self.config.feature_horizon
        bar_data['VOLR'] = self._VOLR(data).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['PCTN'] = self._PCTN(data, n=horizon).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['MidMove'] = self._MidMove(data, n=horizon).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['BSP'] = self._BSP(data).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['weighted_price'] = self._weighted_price(data).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['order_imbalance'] = self._order_imbalance(data).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()
        bar_data['trend_strength'] = self._trend_strength(data, n=horizon).fillna(0).resample(self.config.simulation_commission_freq, closed='right', label='right').last()

        # 订单簿时序特征
        # 动量&反转

        bar_data['time'] = bar_data.index
        bar_data = bar_data[bar_data['time'].between(trade_date + ' 09:30:00', trade_date + ' 14:57:00')]
        bar_data = bar_data[~bar_data['time'].between(trade_date + ' 11:30:01', trade_date + ' 12:59:59')]
        bar_data['time_diff'] = (bar_data['time'] - bar_data['time'].values[0]) / np.timedelta64(1, 's') / 19800
        bar_data = bar_data.reset_index(drop=True)

        # Step 5: 保存为pickle文件
        with open(pkl_path, 'wb') as f:
            pickle.dump(bar_data, f, pickle.HIGHEST_PROTOCOL)
        
        return dict(snapshot_path=snapshot_path, pkl_path=pkl_path, 
            snapshot_shape0=snapshot_shape0, snapshot_shape1=snapshot_shape1, 
            res_shape0=bar_data.shape[0], res_shape1=bar_data.shape[1])

    @staticmethod
    def _calculate_immediate_market_order_cost(bar_data, direction='ask'):
        # Assume the market order quantity is 1/500 of the basis volume
        remaining_quantity = (bar_data['basis_volume'] / 500).copy()  
        total_fee = pd.Series(0, index=bar_data.index)
        for i in range(1,11):
            total_fee = total_fee \
                + bar_data['{}Price{}'.format(direction, i)] \
                * np.minimum(bar_data['{}Volume{}'.format(direction, i)], remaining_quantity)
            remaining_quantity = (remaining_quantity - bar_data['{}Volume{}'.format(direction, i)]).clip(lower=0)

        if direction == 'ask':
            return total_fee / (bar_data['basis_volume'] / 20) - bar_data['askPrice1']
        elif direction == 'bid':
            return bar_data['bidPrice1'] - total_fee / (bar_data['basis_volume'] / 20)

    @staticmethod
    def _VOLR(df, beta1=0.551, beta2=0.778, beta3=0.699):
        """
        Volume Ratio: 
            反应投资行为的供求关系.
        Unit: Volume
        """
        volr = beta1 * (df['bidVolume1'] - df['askVolume1']) / (df['bidVolume1'] + df['askVolume1']) + \
            beta2 * (df['bidVolume2'] - df['askVolume2']) / (df['bidVolume2'] + df['askVolume2']) + \
            beta3 * (df['bidVolume3'] - df['askVolume3']) / (df['bidVolume3'] + df['askVolume3'])

        return volr

    @staticmethod
    def _PCTN(df, n):
        """
        Price Percentage Change: 
            表示价格随时间的变化率.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        pctn = (mid - mid.shift(n)) / mid

        return pctn

    @staticmethod
    def _MidMove(df, n):
        """
        Middle Price Move: 
            表示中间价的偏移程度.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        mean = mid.rolling(n).mean()
        mid_move = (mid - mean) / mean
        return mid_move

    @staticmethod
    def _BSP(df):
        """
        Buy-Sell Pressure: 
            the distribution of chips in the buying and selling direction.
        Unit: Volume
        """

        EPS = 1e-5
        mid = (df['askPrice1'] + df['bidPrice1']) / 2

        w_buy_list = []
        w_sell_list = []

        for level in range(1,11):
            w_buy_level = mid / (df['bidPrice{}'.format(level)] - mid - EPS)
            w_sell_level = mid / (df['askPrice{}'.format(level)] - mid + EPS)

            w_buy_list.append(w_buy_level)
            w_sell_list.append(w_sell_level)

        sum_buy = pd.concat(w_buy_list, axis=1).sum(axis=1)
        sum_sell = pd.concat(w_sell_list, axis=1).sum(axis=1)

        p_buy_list = []
        p_sell_list = []
        for level in range(5):
            l = level + 1
            p_buy = (df['bidVolume{}'.format(l)] * w_buy_list[level]) / sum_buy
            p_sell = (df['askVolume{}'.format(l)] * w_sell_list[level]) / sum_sell
            p_buy_list.append(p_buy)
            p_sell_list.append(p_sell)
    
        p_buy = pd.concat(p_buy_list, axis=1).sum(axis=1)
        p_sell = pd.concat(p_sell_list, axis=1).sum(axis=1)
        p = np.log((p_sell + EPS) / (p_buy + EPS))

        return p

    @staticmethod
    def _weighted_price(df):
        """
        Weighted price: The average price of ask and bid weighted 
            by corresponding volumn (divided by last price).
        Unit: One
        """

        price_list = []
        for level in range(1,11):

            price_level = (df['bidPrice{}'.format(level)] * df['bidVolume{}'.format(level)] + \
                           df['askPrice{}'.format(level)] * df['askVolume{}'.format(level)]) / \
                          (df['bidVolume{}'.format(level)] + df['askVolume{}'.format(level)])

            price_list.append(price_level)

        weighted_price = pd.concat(price_list, axis=1).mean(axis=1)
        weighted_price = weighted_price / (df['lastPrice'] + 1e-5)
        return weighted_price

    @staticmethod
    def _order_imbalance(df):
        """
        Order imbalance: 
            a situation resulting from an excess of buy or sell orders 
            for a specific security on a trading exchange, 
            making it impossible to match the orders of buyers and sellers.
        Unit: One
        """

        oi_list = []
        for level in range(1,11):

            oi_level = (df['bidVolume{}'.format(level)] - df['askVolume{}'.format(level)]) / \
                (df['bidVolume{}'.format(level)] + df['askVolume{}'.format(level)])

            oi_list.append(oi_level)

        oi = pd.concat(oi_list, axis=1).mean(axis=1)

        return oi

    @staticmethod
    def _trend_strength(df, n):
        """
        Trend strength: describes the strength of the short-term trend.
        Unit: One
        """

        mid = (df['askPrice1'] + df['bidPrice1']) / 2
        diff_mid = mid - mid.shift(1)
        sum1 = diff_mid.rolling(n).sum()
        sum2 = diff_mid.abs().rolling(n).sum()
        TS = sum1 / sum2

        return TS
    
    @staticmethod
    def _orderbook_slope(df):
        """
        Orderbook Slope: describes the sensitivity of price spread with volume difference.  
        Unit: log(Price)/log(Volume)
        """
        return (np.log(df['askPrice1']) - np.log(df['bidPrice1'])) / (np.log(df['askVolume1']) - np.log(df['bidVolume1']))

    @staticmethod
    def _pct_autocorr(df):
        pass

# 支持数据在模拟环境中进行迭代
class Data(object):
    price_5level_features = [
        'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5',
        'bidPrice6', 'bidPrice7', 'bidPrice8', 'bidPrice9', 'bidPrice10',
        'askPrice1', 'askPrice2', 'askPrice3', 'askPrice4', 'askPrice5', 
        'askPrice6', 'askPrice7', 'askPrice8', 'askPrice9', 'askPrice10', 
    ]
    volume_5level_features = [
        'bidVolume1', 'bidVolume2', 'bidVolume3', 'bidVolume4', 'bidVolume5',  
        'bidVolume6', 'bidVolume7', 'bidVolume8', 'bidVolume9', 'bidVolume10',
        'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5',        
        'askVolume6', 'askVolume7', 'askVolume8', 'askVolume9', 'askVolume10', 
    ]
    seq_5level_features = [
        'NumOrdersB1', 'NumOrdersB2', 'NumOrdersB3', 'NumOrdersB4', 'NumOrdersB5',  
        'NumOrdersB6', 'NumOrdersB7', 'NumOrdersB8', 'NumOrdersB9', 'NumOrdersB10',
        'NumOrdersS1', 'NumOrdersS2', 'NumOrdersS3', 'NumOrdersS4', 'NumOrdersS5',        
        'NumOrdersS6', 'NumOrdersS7', 'NumOrdersS8', 'NumOrdersS9', 'NumOrdersS10', 
    ]
    other_price_features = [
        'high_price', 'low_price', 'open_price', 'close_price', 'vwap',
    ]
    price_delta_features = [
        'ask_bid_spread', 'trend', 'immediate_market_order_cost_ask', 
        'immediate_market_order_cost_bid', 'volatility', 'high_low_price_diff',
    ]
    other_volume_features = [
        'volume', 'ab_volume_misbalance', 'transaction_net_volume', 
        'VOLR', 'BSP',
    ]
    backtest_lo_features = [
        'max_last_price', 'min_last_price', 'ask1_deal_volume', 'bid1_deal_volume',
    ]

    def __init__(self, config):

        self.config = config
        self.data = None
        self.backtest_data = None
        self.orders = None

    def _maintain_backtest_data(self):

        self.backtest_data = \
            self.data[self.price_5level_features + self.volume_5level_features + self.seq_5level_features + self.backtest_lo_features].copy()
        self.backtest_data['latest_price'] = \
            (self.backtest_data['askPrice1'] + self.backtest_data['bidPrice1']) / 2
        self.backtest_data['vwap'] = self.data['vwap']
        self.backtest_data['vwap_5min'] = self.data['vwap_5min']
        self.backtest_data['time'] = self.data['time']
        self.backtest_data['volume_tol'] = self.data['volume_tol']
        

    def _normalization(self):

        # Keep normalization units
        self.basis_price = self.backtest_data.loc[self.start_index, 'latest_price']
        self.basis_volume = self.data['basis_volume'].values[0]
        self.base_volume = self.backtest_data.loc[self.start_index, 'volume_tol']
        # Approximation: 平均价格变动率 2% * 50 = 1.0
        self.data[self.price_5level_features] = \
            (self.data[self.price_5level_features] - self.basis_price) / self.basis_price * 50
        self.data[self.other_price_features] = \
            (self.data[self.other_price_features] - self.basis_price) / self.basis_price * 50
        # 涨跌停限制：10% * 10 = 1.0
        self.data[self.price_delta_features] = \
            self.data[self.price_delta_features] / self.basis_price * 10

        # Such that the volumes are equally distributed in the range [-1, 1]
        self.data[self.volume_5level_features] = \
            self.data[self.volume_5level_features] / self.basis_volume * 100
        self.data[self.other_volume_features] = \
            self.data[self.other_volume_features] / self.basis_volume * 100

    def data_exists(self, code='300733', date='2024-12-02'):

        return os.path.isfile(os.path.join(path_pkl_data, 'SH' + code, date + '.pkl')) and os.path.isfile(os.path.join(path_orders, 'SH' + code, date + '.pkl'))

    def _get_pkl_shape(self):
        if self.config.simulation_commission_freq == '1min':
            return 239
        elif self.config.simulation_commission_freq == '10s':
            return 1424
        elif self.config.simulation_commission_freq == '1s':
            return 14222

    def obtain_data(self, code='300733', date='2024-12-02', start_index=None, do_normalization=True):

        with open(os.path.join(path_pkl_data, f'SH{code}', date + '.pkl'), 'rb') as f:
            self.data = pickle.load(f)

        try:
            with open(os.path.join(path_orders, f'SH{code}', date + '.pkl'), 'rb') as f:
                orders = pickle.load(f)
            orders = orders[(pd.to_datetime(date + ' 09:30:00') <= orders['OrderTime']) & (orders['OrderTime'] < pd.to_datetime(date + ' 14:57:00'))]
            self.orders = orders[~orders['OrderTime'].between(date + ' 11:30:01', date + ' 12:59:59')]
        except FileNotFoundError:
            self.orders = None  # 文件不存在

        assert self.data.shape[0] == self._get_pkl_shape(), \
            'The data should be of the shape ({}, 49), instead of {}'.format(self._get_pkl_shape(), self.data.shape)
        
        if start_index is None:
            # randomly choose a valid start_index
            start_index = self._random_valid_start_index()
            self._set_horizon(start_index)
        else:
            self._set_horizon(start_index)
            assert self._sanity_check(), "code={} date={} with start_index={} is invalid".format(code, date, start_index)
        self._maintain_backtest_data()
        if do_normalization:
            self._normalization()

    def _random_valid_start_index(self):
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']

        tmp = (self.data[cols] > 0).all(axis=1)
        tmp1 = tmp.rolling(self.config.simulation_lookback_horizon).apply(lambda x: x.all())
        tmp2 = tmp[::-1].rolling(self.config.simulation_planning_horizon + 1).apply(lambda x: x.all())[::-1]
        available_indx = tmp1.loc[(tmp1 > 0) & (tmp2  > 0)].index.tolist()
        assert len(available_indx) > 0, "The data is invalid"
        return np.random.choice(available_indx)

    def _set_horizon(self, start_index):
        '''initialize for a specific index'''
        self.start_index = start_index
        self.current_index = self.start_index
        self.end_index = self.start_index + self.config.simulation_planning_horizon

    def obtain_features(self, do_flatten=True):
        features = self.data.loc[self.current_index - self.config.simulation_lookback_horizon + 1: self.current_index, 
            self.config.simulation_features][::-1].fillna(0).values
        if do_flatten:
            return features.flatten()
        else:
            return features

    def obtain_feature(self, feature):
        return self.data.loc[self.current_index, feature]

    def obtain_future_features(self, features):
        return self.data.loc[self.current_index:self.end_index, features]

    def obtain_level(self, name, level=''):
        return self.backtest_data.loc[self.current_index, '{}{}'.format(name, level)]

    def step(self):
        self.current_index += 1

    def _sanity_check(self):
        """ When the price reaches daily limit, the price and volume"""
        cols = ['bidPrice1', 'bidVolume1', 'askPrice1', 'askVolume1']
        if (self.data.loc[self.start_index:self.end_index, cols] == 0).any(axis=None):
            return False
        else:
            return True


class BaseWrapper(object):
    def __init__(self, env):
        self.env = env 

    def reset(self, code=None, date=None, start_index=None):
        return self.env.reset(code, date, start_index)

    def step(self, action):
        return self.env.step(action)

    @property
    def quantity(self):
        return self.env.quantity

    @property
    def total_quantity(self):
        return self.env.total_quantity

    @property
    def cash(self):
        return self.env.cash

    @property
    def config(self):
        return self.env.config

    @property
    def data(self):
        return self.env.data

    @property
    def observation_dim(self):
        return self.env.observation_dim

    def get_metric(self, mtype='IS'):
        return self.env.get_metric(mtype)

    def get_future(self, features, padding=None):
        return self.env.get_future(features, padding=padding)


class DiscreteActionBaseWrapper(BaseWrapper):
    def __init__(self, env):
        super(DiscreteActionBaseWrapper, self).__init__(env)

    @property
    def action_sample_func(self):
        return lambda: np.random.randint(len(self.discrete_actions))

    @property
    def action_dim(self):
        return len(self.discrete_actions)
        

class DiscretePriceQuantityWrapper(DiscreteActionBaseWrapper):
    def __init__(self, env):
        super(DiscretePriceQuantityWrapper, self).__init__(env)
        self.discrete_actions = self.config.simulation_discrete_actions
        self.simulation_discrete_quantities = self.config.simulation_discrete_quantities
        self.base_quantity_ratio = self.config.simulation_volume_ratio \
            / self.config.simulation_num_shares 

    def step(self, action):
        price, quantity = self.discrete_actions[action]
        # 修正1：通过 self.env.data 访问数据
        if self.config.simulation_direction == 'buy':
            price = np.round((self.env.data.obtain_level('bidPrice', 1) + price / 100) * 100) / 100
        elif self.config.simulation_direction == 'sell':
            price = np.round((self.env.data.obtain_level('askPrice', 1) + price / 100) * 100) / 100
        
        # 修正2：通过 self.env.data 访问数据
        if self.env.data.base_volume != 0:
            quantity = int(self.env.data.base_volume * self.base_quantity_ratio * quantity)
        else:
            quantity = 100
        return self.env.step(dict(price=price, quantity=quantity))



class RealisticOrderMatching:
    """
    真实订单撮合引擎
    """
    def __init__(self, config):
        self.config = config
    
    def match_limit_order(self, order, orderbook_snapshot, order_flow=None):
        """
        更真实的限价订单撮合
        """
        executed_trades = []
        remaining_quantity = order['quantity']
        
        # 1. 按价格优先、时间优先原则撮合
        if order['side'] == 'sell':
            # 卖单与买方订单簿撮合
            for level in range(1, 11):
                if remaining_quantity <= 0:
                    break
                    
                bid_price = orderbook_snapshot[f'bidPrice{level}']
                bid_volume = orderbook_snapshot[f'bidVolume{level}']
                
                # 价格匹配条件
                if order['price'] <= bid_price and bid_volume > 0:
                    executed_qty = min(remaining_quantity, bid_volume)
                    executed_trades.append({
                        'price': bid_price,
                        'quantity': executed_qty,
                        'timestamp': pd.Timestamp.now(),  # 实际应该使用当前时间戳
                        'match_type': 'orderbook'
                    })
                    remaining_quantity -= executed_qty
        
        elif order['side'] == 'buy':
            # 买单与卖方订单簿撮合
            for level in range(1, 11):
                if remaining_quantity <= 0:
                    break
                    
                ask_price = orderbook_snapshot[f'askPrice{level}']
                ask_volume = orderbook_snapshot[f'askVolume{level}']
                
                # 价格匹配条件
                if order['price'] >= ask_price and ask_volume > 0:
                    executed_qty = min(remaining_quantity, ask_volume)
                    executed_trades.append({
                        'price': ask_price,
                        'quantity': executed_qty,
                        'timestamp': pd.Timestamp.now(),  # 实际应该使用当前时间戳
                        'match_type': 'orderbook'
                    })
                    remaining_quantity -= executed_qty
        
        # 2. 与订单流撮合（如果有）
        if order_flow is not None and not order_flow.empty:
            # 按时间顺序处理订单流
            for idx, flow_order in order_flow.iterrows():
                if remaining_quantity <= 0:
                    break
                    
                # 撮合条件检查
                if self._can_match(order, flow_order, remaining_quantity):
                    executed_qty = min(remaining_quantity, flow_order['Balance'])
                    executed_trades.append({
                        'price': flow_order['OrderPrice'],
                        'quantity': executed_qty,
                        'timestamp': flow_order['OrderTime'],
                        'match_type': 'orderflow'
                    })
                    remaining_quantity -= executed_qty
        
        return executed_trades, remaining_quantity
    
    def _can_match(self, order, flow_order, remaining_quantity):
        """
        判断订单是否可以撮合
        """
        if remaining_quantity <= 0 or flow_order['Balance'] <= 0:
            return False
            
        # 检查买卖方向是否匹配
        if order['side'] == 'sell' and flow_order['OrderBSFlag'] == 'B':
            # 我方卖单与买方订单流撮合
            return order['price'] <= flow_order['OrderPrice']
        elif order['side'] == 'buy' and flow_order['OrderBSFlag'] == 'S':
            # 我方买单与卖方订单流撮合
            return order['price'] >= flow_order['OrderPrice']
            
        return False
    
    def calculate_market_impact(self, order_quantity, orderbook_snapshot, side):
        """
        计算市场冲击价格（更真实的模型）
        """
        if order_quantity <= 0:
            return None
            
        # 累计订单簿深度来估算市场冲击
        accumulated_volume = 0
        total_value = 0.0
        remaining_qty = order_quantity
        
        if side == 'sell':
            # 卖单冲击买方订单簿
            for level in range(1, 11):
                if remaining_qty <= 0:
                    break
                    
                bid_volume = orderbook_snapshot[f'bidVolume{level}']
                bid_price = orderbook_snapshot[f'bidPrice{level}']
                
                if bid_volume > 0:
                    executed_qty = min(remaining_qty, bid_volume)
                    total_value += executed_qty * bid_price
                    accumulated_volume += executed_qty
                    remaining_qty -= executed_qty
                    
        elif side == 'buy':
            # 买单冲击卖方订单簿
            for level in range(1, 11):
                if remaining_qty <= 0:
                    break
                    
                ask_volume = orderbook_snapshot[f'askVolume{level}']
                ask_price = orderbook_snapshot[f'askPrice{level}']
                
                if ask_volume > 0:
                    executed_qty = min(remaining_qty, ask_volume)
                    total_value += executed_qty * ask_price
                    accumulated_volume += executed_qty
                    remaining_qty -= executed_qty
        
        # 计算加权平均成交价格
        if accumulated_volume > 0:
            avg_price = total_value / accumulated_volume
            return avg_price
        else:
            return None


class ExecutionEnv(object):
    """
    模拟交易执行环境
      Feature 1: There is no model misspecification error since the simulator is based on historical data.
      Featrue 2: We can model temporary market impact for MO and LO. 
          For MO, we assume that the order book is resilient between bars
          For LO, we assume that 1) full execution when the price passes through;   
              2) partial execution when the price reaches;   
              3) no execution otherwise
    """
    def __init__(self, config):
        self.config = config
        self.current_code = None 
        self.current_date = None 
        self.data = Data(config)
        self.cash = 0
        self.cost = 0
        self.total_quantity = 0
        self.quantity = 0
        self.pending_quantity = 0  # 已报但未成交的数量
       
        self.valid_code_date_list = self.get_valid_code_date_list()
        
         # 订单统计相关
        self.total_orders = 0           # 总订单数
        self.canceled_orders = 0        # 撤单数
        self.immediate_orders = 0       # 立即成交订单数
        self.holding_orders = []        # 当前挂单列表
        self.order_history = []         # 历史订单记录
        self.trade_history = []         # 交易历史记录

        # 初始化订单撮合引擎
        self.order_matching_engine = RealisticOrderMatching(config)

    def get_valid_code_date_list(self):
        code_date_list = []
        for code in self.config.code_list:
            for date in self.config.date_list:
                if self.data.data_exists(code, date):
                    code_date_list.append((code, date))
        return code_date_list

    def reset(self, code=None, date=None, start_index=None):
        """
        重置交易环境，随机选择或指定股票代码和日期，初始化状态信息。
        包含订单队列、头寸、现金等关键变量的初始化。
        """
        count = 0
        while True:
            if code is None and date is None:
                # 随机选择有效代码和日期
                ind = np.random.choice(len(self.valid_code_date_list))
                self.current_code, self.current_date = self.valid_code_date_list[ind]
            else:
                # 使用指定的代码和日期
                self.current_code, self.current_date = code, date

            try:
                # 加载数据
                self.data.obtain_data(self.current_code, self.current_date, start_index)
                # 确保数据加载成功后才退出循环
                break
            except AssertionError as e:
                # 数据验证失败，记录日志并重试
                count += 1
                print(f'Invalid data: code={self.current_code}, date={self.current_date}, error={str(e)}')
                if count > 100:
                    raise ValueError(f"code={code}, date={date} is invalid after 100 retries")
            except Exception as e:
                # 其他异常直接抛出
                print(f"Critical error loading data: {str(e)}")
                raise

        # 初始化交易状态
        self.cash = 0
        self.cost = 0
        self.pending_quantity = 0  # 重置挂单数量

        # 安全计算 total_quantity（避免除零）
        if self.data.base_volume != 0:
            self.total_quantity = int(self.config.simulation_volume_ratio * self.data.base_volume)
        elif self.data.basis_volume != 0:
            self.total_quantity = int(self.config.simulation_volume_ratio * self.data.basis_volume / 48)
        else:
            self.total_quantity = 100  # 默认值

        self.quantity = self.total_quantity
        self.latest_price = self.data.obtain_level('latest_price')

        # 订单统计信息初始化
        self.total_orders = 0
        self.canceled_orders = 0
        self.immediate_orders = 0
        self.holding_orders = []  # 使用列表存储当前挂单
        self.order_history = []   # 使用列表存储历史订单
        self.trade_history = []   # 使用列表存储交易历史

        # 数据索引初始化（跳过第一个时间步，若存在零成交量）
        if hasattr(self.data, 'start_index'):
            self.data.current_index = self.data.start_index
        else:
            self.data.current_index = 0

        # 生成初始市场状态和私有状态
        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state
    
    def _generate_private_state(self):
        # 时间进度
        elapsed_time = (self.data.current_index - self.data.start_index) / self.config.simulation_planning_horizon
        # 剩余头寸比例
        remaining_quantity = self.quantity / self.total_quantity if self.total_quantity > 0 else 0

        snapshot = {}
        for level in range(1, 11):
            snapshot[f'askPrice{level}'] = self.data.obtain_level('askPrice', level)
            snapshot[f'askVolume{level}'] = self.data.obtain_level('askVolume', level)
            snapshot[f'NumOrdersS{level}'] = self.data.obtain_level('NumOrdersS', level) 
            snapshot[f'bidPrice{level}'] = self.data.obtain_level('bidPrice', level)
            snapshot[f'bidVolume{level}'] = self.data.obtain_level('bidVolume', level)
            snapshot[f'NumOrdersB{level}'] = self.data.obtain_level('NumOrdersB', level) 

        # 历史成交订单价格分布（加权平均）
        filled_orders = [o for o in self.order_history if o['status'] == 'FILLED']
        if filled_orders:
            prices = [o['price'] for o in filled_orders]
            quantities = [o['quantity'] for o in filled_orders]
            weighted_avg_price = np.average(prices, weights=quantities)
        else:
            weighted_avg_price = 0

        # 当前订单簿价格档位匹配
        bid_prices = [snapshot[f'bidPrice{i}'] for i in range(1, 11)]
        ask_prices = [snapshot[f'askPrice{i}'] for i in range(1, 11)]

        # 订单队列信息 [所在档位, 档位订单数, 总订单数, 是否最优]
        my_queue_info = [0] * 4  # [档位, 档位订单数, 总订单数, 是否最优]

        for order in self.holding_orders:
            if order['side'] == 'buy':
                # 买单匹配买档
                matched_level = next((i for i, p in enumerate(bid_prices) if abs(p - order['price']) < 1e-6), None)
                if matched_level is not None:
                    my_queue_info[0] = matched_level + 1  # 档位（1~10）
                    my_queue_info[1] = snapshot[f'NumOrdersB{matched_level + 1}'] + 1  # 档位订单数
                    my_queue_info[2] = snapshot[f'bidVolume{matched_level + 1}'] + order['quantity'] # 档位总订单量
                    my_queue_info[3] = 1 if matched_level == 0 else 0  # 是否买一档
                else: 
                    # 价格不在已有的买卖队列里面
                    my_queue_info[0] = 11
                    my_queue_info[1] = 1  # 该档位订单数为1
                    my_queue_info[2] = order['quantity']
                    my_queue_info[3] = 0
                    for matched_level in range(1,11):
                        if order['price'] > snapshot[f'bidPrice{matched_level}']:
                            my_queue_info[0] = matched_level
                            my_queue_info[1] = 1  # 该档位订单数为1
                            my_queue_info[2] = order['quantity']
                            my_queue_info[3] = 1 if matched_level == 0 else 0  # 是否买一档


            elif order['side'] == 'sell':
                # 卖单匹配卖档
                matched_level = next((i for i, p in enumerate(ask_prices) if abs(p - order['price']) < 1e-6), None)
                if matched_level is not None:
                    my_queue_info[0] = matched_level + 1
                    my_queue_info[1] = snapshot[f'NumOrdersS{matched_level+1}'] + 1
                    my_queue_info[2] = snapshot[f'askVolume{matched_level+1}'] + order['quantity'] 
                    my_queue_info[3] = 1 if matched_level == 0 else 0
                else: 
                    # 价格不在已有的买卖队列里面
                    my_queue_info[0] = 11
                    my_queue_info[1] = 1  # 该档位订单数为1
                    my_queue_info[2] = order['quantity']
                    my_queue_info[3] = 0    
                    for matched_level in range(1,11):
                        if order['price'] < snapshot[f'askPrice{matched_level}']:
                            my_queue_info[0] = matched_level
                            my_queue_info[1] = 1  # 该档位订单数为1
                            my_queue_info[2] = order['quantity']
                            my_queue_info[3] = 1 if matched_level == 0 else 0  # 是否买一档

        # 订单统计信息
        order_num = len(self.holding_orders)
        order_quantity = sum(order['quantity'] for order in self.holding_orders)
        filled_orders_count = sum(1 for o in self.order_history if o['status'] == 'FILLED')
        partial_orders_count = sum(1 for o in self.order_history if o['status'] == 'PARTIAL_FILLED')
        canceled_orders_count = self.canceled_orders

        return np.array([
            # 时间与头寸信息
            elapsed_time,                    # 时间进度
            remaining_quantity,              # 剩余头寸比例

            # 订单统计信息
            order_num,                       # 当前挂单数
            order_quantity,                  # 挂单总量
            filled_orders_count,             # 已成交订单数
            partial_orders_count,            # 部分成交订单数
            canceled_orders_count,           # 撤单数

            # 成交价格统计
            weighted_avg_price,              # 历史成交加权平均价

            # 订单队列信息
            *my_queue_info                 # 买单队列信息 [档位, 档位订单数, 总订单数, 是否最优]
            ])

    def get_future(self, features, padding=None):
        # padding参数决定是否进行填充，若填充，则在后面重复最后一行的future features直到总行数为padding（猜测应该用于作为神经网络训练时的统一大小训练数据）
        future = self.data.obtain_future_features(features)
        if padding is None:
            return future
        else:
            padding_width = padding - future.shape[0]
            future = np.pad(future, ((0, padding_width), (0, 0)), 'edge')
            return future

    def step(self, high_level_action, low_level_action=None):
        """
        分层强化学习的 step 方法。
        Args:
            high_level_action (str or int): 'active' (0) 或 'passive' (1)。
            low_level_action (dict or int): 
                如果 high_level_action 是 'active'，则为 {'quantity': int} 或表示数量的比例/索引。
                如果 high_level_action 是 'passive'，则为原始的离散动作索引，用于传递给旧的 wrapper 处理，
                或者是 {'price': float, 'quantity': int}。
        Returns:
            tuple: (market_state, private_state, reward, done, info)
        """
        # 核心修改：将最后时刻判断提前到最前面
        done = (self.data.current_index + 1 >= self.data.end_index)
        info = dict(
            code=self.current_code,
            date=self.current_date,
            start_index=self.data.start_index,
            end_index=self.data.end_index,
            current_index=self.data.current_index,
            canceled_orders=self.canceled_orders,
            total_orders=self.total_orders,
            immediate_orders=self.immediate_orders,
            immediate_ratio=self.immediate_orders / self.total_orders if self.total_orders > 0 else 0,
            status='NO_OPERATION', # 默认状态，会被子步骤覆盖
            delay_deal='NO_DELAY_DEAL',
            execution_mode=high_level_action # 记录执行模式
        )

        reward = 0.0
        # 核心修改：最后时刻强制撤单和清算
        if done:
            # 1. 撤消所有未成交的挂单
            pending_orders = [o for o in self.holding_orders if o['status'] in ['PENDING', 'PARTIAL_FILLED']]
            self.canceled_orders += len(pending_orders)
            # 释放挂单量
            for order in pending_orders:
                self.pending_quantity -= order['quantity']
            # 清空挂单列表
            self.holding_orders = []

            # 2. 用对方10档价格进行强制市价清算
            liquidation_price = 0.0
            price_penalty = 0.0
            if self.quantity > 0:
                if self.config.simulation_direction == 'sell':
                    liquidation_price = self.data.obtain_level('bidPrice', 10)
                elif self.config.simulation_direction == 'buy':
                    liquidation_price = self.data.obtain_level('askPrice', 10)

                if liquidation_price <= 0:  # 保护机制
                    liquidation_price = self.data.obtain_level('lastPrice')

                # 获取清算惩罚
                price_penalty = self.config.simulation_not_filled_penalty_bp / 10000 * liquidation_price if hasattr(self.config, 'simulation_not_filled_penalty_bp') else 0
                # 应用惩罚项到价格
                liquidation_price_adj = liquidation_price
                if self.config.simulation_direction == 'sell':
                    liquidation_price_adj -= price_penalty
                elif self.config.simulation_direction == 'buy':
                    liquidation_price_adj += price_penalty

                executed_volume = self.quantity
                self.cash += executed_volume * liquidation_price_adj
                self.cost += executed_volume * liquidation_price_adj
                self.quantity = 0
                self.pending_quantity = 0  # 清零挂单量

                info['status'] = 'LIQUIDATED'
                info['liquidation_price'] = liquidation_price
                info['liquidation_price_adj'] = liquidation_price_adj
                info['liquidation_volume'] = executed_volume
                info['price_penalty'] = price_penalty

            # 计算清算带来的奖励/惩罚
            # 这里可以单独计算一个清算奖励，或者将其包含在最后一步的总奖励中
            # 简单起见，我们将其包含在最后的 _calculate_reward 调用中
            # 但需要传递清算信息
            info['pre_cash_for_reward'] = self.cash - executed_volume * liquidation_price_adj if self.quantity == 0 and executed_volume > 0 else self.cash

            # 更新数据状态到 done
            self.data.step()

            # 生成最终状态
            market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
            private_state = self._generate_private_state()

            # 计算包含清算效果的最终奖励
            # 注意：这里我们用清算前的 cash 作为 pre_cash，模拟整个过程的最终效果
            final_reward = self._calculate_reward(info['pre_cash_for_reward'], status=info['status'], punish=0, is_final_step=True)

            return market_state, private_state, final_reward, done, info

        else:
            # 正常交易时间步的处理流程
            pre_cash = self.cash
            punish = 0.0 # 在非最终步，此惩罚由 _step_active/_step_passive 内部处理或在通用奖励中处理

            if high_level_action == 'active' or high_level_action == 0:
                market_state, private_state, reward, done, info = self._step_active(low_level_action, pre_cash, info)
            elif high_level_action == 'passive' or high_level_action == 1:
                market_state, private_state, reward, done, info = self._step_passive(low_level_action, pre_cash, info)
            else:
                raise ValueError(f"Unknown high level action: {high_level_action}")

            # 更新数据状态 (这在 _step_active/_step_passive 中已经调用)
            # self.data.step() 

            return market_state, private_state, reward, done, info


    def _step_active(self, low_level_action, pre_cash, info):
        """
        处理主动成交逻辑。
        Args:
            low_level_action: 包含 'quantity' 键的字典，或一个表示数量的数值（绝对值或相对索引）。
            pre_cash (float): 执行此步前的现金。
            info (dict): 初始 info 字典。
        Returns:
            tuple: (market_state, private_state, reward, done, info)
        """
        done = (self.data.current_index + 1 >= self.data.end_index) # Should be False here
        # 1. 解析低层动作得到主动成交量
        if isinstance(low_level_action, dict) and 'quantity' in low_level_action:
            order_quantity = low_level_action['quantity']
        elif isinstance(low_level_action, (int, float)):
            # 主动成交动作只处理数量
            if 0 <= low_level_action < len(self.config.simulation_discrete_actions_active):
                qty_slice_value = self.config.simulation_discrete_actions_active[low_level_action]
                # 基于剩余头寸计算实际数量
                order_quantity = int(self.quantity * qty_slice_value)
            else:
                order_quantity = 0  # 无效索引视为无操作
        else:
            order_quantity = 0 # 默认无主动成交

        # 2. 计算可用量 (不能超过剩余头寸)
        available_quantity = self.quantity
        order_quantity = max(0, min(available_quantity, order_quantity))

        # 3. 确定成交价格 (使用更真实的市场冲击模型)
        execution_price = 0.0
        price_penalty = 0.0
        if order_quantity > 0:
            # 获取订单簿快照
            orderbook_snapshot = self._get_orderbook_snapshot()
            # 计算市场冲击价格
            execution_price = self.order_matching_engine.calculate_market_impact(
                order_quantity, orderbook_snapshot, self.config.simulation_direction
            )
            
            if execution_price is None:
                # 如果无法计算市场冲击价格，回退到原来的简单模型
                if self.config.simulation_direction == 'sell':
                    execution_price = self.data.obtain_level('bidPrice', 10)
                elif self.config.simulation_direction == 'buy':
                    execution_price = self.data.obtain_level('askPrice', 10)

        # 4. 成交并更新 cash, cost, quantity
        if order_quantity > 0 and execution_price > 0:
            # 添加方向判断因子
            direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
            self.cash += order_quantity * execution_price * direction_sign
            self.cost += order_quantity * execution_price * direction_sign
            self.quantity -= order_quantity
            
            # 记录交易详情
            self.trade_history.append({
                'price': execution_price,
                'quantity': order_quantity,
                'timestamp': self.data.backtest_data.loc[self.data.current_index, 'time'],
                'match_type': 'market_order',
                'side': self.config.simulation_direction
            })
            
            info['status'] = 'ACTIVE_FILLED'
            info['active_executed_volume'] = order_quantity
            info['active_execution_price'] = execution_price

        # 5. 撤下所有当前挂单
        pending_orders = [o for o in self.holding_orders if o['status'] in ['PENDING', 'PARTIAL_FILLED']]
        self.canceled_orders += len(pending_orders)
        for order in pending_orders:
            self.pending_quantity -= order['quantity']
        self.holding_orders = [] # 清空挂单列表

        # 6. 计算奖励 (基于即时成交的收益/成本)
        reward = self._calculate_reward(pre_cash, status=info['status'], punish=0) # 主动成交通常不触发挂单惩罚

        # 7. 更新数据时间步
        self.data.step()
        done = (self.data.current_index + 1 >= self.data.end_index) # 更新 done 状态

        # 8. 生成状态
        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state, reward, done, info

    def _step_passive(self, low_level_action, pre_cash, info):
        """
        处理被动成交逻辑。
        Args:
            low_level_action: 原始的离散动作索引，或包含 'price' 和 'quantity' 的字典。
            pre_cash (float): 执行此步前的现金。
            info (dict): 初始 info 字典。
        Returns:
            tuple: (market_state, private_state, reward, done, info)
        """
        done = (self.data.current_index + 1 >= self.data.end_index) # Should be False here
        # 1. 解析 low_level_action 得到价格偏移
        action_dict = {'price': 0, 'quantity': 0}
        if isinstance(low_level_action, dict) and 'price' in low_level_action and 'quantity' in low_level_action:
            action_dict = low_level_action
        elif isinstance(low_level_action, (int, float)):
            # 被动成交动作只处理价格偏移
            active_action_count = len(self.config.simulation_discrete_actions_active)
            if active_action_count <= low_level_action < active_action_count + len(self.config.simulation_discrete_actions_passive):
                # 处理被动成交动作
                passive_index = low_level_action - active_action_count
                bps_value = self.config.simulation_discrete_actions_passive[passive_index]
                
                # 计算价格 (基于当前对手价)
                ref_price = 0
                if self.config.simulation_direction == 'sell':
                    ref_price = self.data.obtain_level('askPrice', 1)
                    # 修正基点计算单位错误
                    action_dict['price'] = ref_price + (bps_value * ref_price / 10000)
                elif self.config.simulation_direction == 'buy':
                    ref_price = self.data.obtain_level('bidPrice', 1)
                    # 修正基点计算单位错误
                    action_dict['price'] = ref_price + (bps_value * ref_price / 10000)

                # 使用配置的数量比例
                action_dict['quantity'] = self.quantity * self.config.passive_default_qty_ratio
            elif low_level_action == len(self.config.simulation_discrete_actions) - 1:
                # (0, 0) 动作
                action_dict = {'price': 0, 'quantity': 0}
            else:
                # 主动动作索引在被动步骤中视为无效操作
                action_dict = {'price': 0, 'quantity': 0}
        else:
            action_dict = {'price': 0, 'quantity': 0}

        # 2. 计算实际可报单量：剩余总量 - 已挂单量（包括部分成交的订单）
        # 重新计算挂单总量，包括所有未完成订单
        pending_quantity_total = sum(order['quantity'] for order in self.holding_orders 
                                   if order['status'] in ['PENDING', 'PARTIAL_FILLED'])
        available_quantity = self.quantity - pending_quantity_total
        # 确保报单量不超过可用量
        order_quantity = max(min(available_quantity, action_dict['quantity']), 0)
        new_order = None

        current_opposite_price = 0 # 对手价，用于判断是否立即成交
        if self.config.simulation_direction == 'sell':
            current_opposite_price = self.data.obtain_level('bidPrice', 1)
        elif self.config.simulation_direction == 'buy':
            current_opposite_price = self.data.obtain_level('askPrice', 1)

        if order_quantity > 0:
            new_order = {
                'price': action_dict['price'],
                'quantity': order_quantity,
                'timestamp': self.data.current_index,
                'status': 'PENDING',
                'side': self.config.simulation_direction,
                'filled_quantity': 0,
                'entry_time': self.data.backtest_data.loc[self.data.current_index, 'time']
            }
            self.pending_quantity += order_quantity
            self.order_history.append(new_order.copy())
            self.total_orders += 1

            # 判断是否主动成交 (价格优于对手价)
            immediate_match = False
            if self.config.simulation_direction == 'sell' and action_dict['price'] <= current_opposite_price:
                immediate_match = True
                self.immediate_orders += 1
            elif self.config.simulation_direction == 'buy' and action_dict['price'] >= current_opposite_price:
                immediate_match = True
                self.immediate_orders += 1

            if not immediate_match:
                # 使用二分插入保持挂单列表有序，而非每次排序
                self._insert_order_sorted(new_order)
            # 如果 immediate_match，订单会在撮合中被处理，这里不加到 holding_orders

            info['status'] = 'ORDER_PLACED' if not immediate_match else 'IMMEDIATE_MATCH_PENDING'
        else:
            # 无可报单量
            info['status'] = 'NO_OPERATION'
            order_quantity = 0

        # 3. 执行撮合逻辑（订单簿 + 订单流）
        # 获取当前订单簿信息
        orderbook_snapshot = self._get_orderbook_snapshot()
        
        # 获取订单簿时间范围内的委托 (订单流数据)
        order_flow = self._get_order_flow()

        new_holding_orders = []
        hist_deal = False # 标记是否有订单流撮合

        # 处理已持有的订单成交 (使用真实的订单撮合引擎)
        for my_order in self.holding_orders:
            if my_order['status'] in ['FILLED', 'CANCELED']:
                new_holding_orders.append(my_order)
                continue
                
            # 使用真实的订单撮合引擎
            executed_trades, remaining_qty = self.order_matching_engine.match_limit_order(
                my_order, orderbook_snapshot, order_flow
            )
            
            # 更新订单状态和账户信息
            for trade in executed_trades:
                # 添加方向判断因子
                direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
                self.cash += trade['quantity'] * trade['price'] * direction_sign
                self.cost += trade['quantity'] * trade['price'] * direction_sign
                self.quantity -= trade['quantity']
                self.pending_quantity -= trade['quantity']
                hist_deal = True
                
                # 记录交易详情
                self.trade_history.append({
                    'order_id': len(self.order_history),
                    'price': trade['price'],
                    'quantity': trade['quantity'],
                    'timestamp': trade['timestamp'],
                    'match_type': trade['match_type'],
                    'side': self.config.simulation_direction
                })
            
            my_order['quantity'] = remaining_qty
            my_order['filled_quantity'] = my_order['quantity'] - remaining_qty
            
            if remaining_qty == 0:
                my_order['status'] = 'FILLED'
            elif my_order['filled_quantity'] > 0:
                my_order['status'] = 'PARTIAL_FILLED'

            # 未完全成交的订单保留
            if my_order['quantity'] > 0:
                new_holding_orders.append(my_order)

        # 处理新订单的订单簿撮合 (如果它没有被立即撤单)
        if new_order and new_order.get('status') == 'PENDING':
            executed_trades, remaining_qty = self.order_matching_engine.match_limit_order(
                new_order, orderbook_snapshot, order_flow
            )
            
            # 更新订单状态和账户信息
            for trade in executed_trades:
                # 添加方向判断因子
                direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
                self.cash += trade['quantity'] * trade['price'] * direction_sign
                self.cost += trade['quantity'] * trade['price'] * direction_sign
                self.quantity -= trade['quantity']
                self.pending_quantity -= trade['quantity']
                hist_deal = True
                
                # 记录交易详情
                self.trade_history.append({
                    'order_id': len(self.order_history),
                    'price': trade['price'],
                    'quantity': trade['quantity'],
                    'timestamp': trade['timestamp'],
                    'match_type': trade['match_type'],
                    'side': self.config.simulation_direction
                })
            
            new_order['quantity'] = remaining_qty
            new_order['filled_quantity'] = new_order['quantity'] - remaining_qty
            
            if remaining_qty == 0:
                new_order['status'] = 'FILLED'
            elif new_order['filled_quantity'] > 0:
                new_order['status'] = 'PARTIAL_FILLED'
                # 添加到挂单列表
                self._insert_order_sorted(new_order)
            else: # quantity > 0 and filled_quantity == 0
                # 完全未成交，已在前面添加到挂单列表
            
                self.holding_orders = new_holding_orders

        # 更新 info 状态 (如果之前是 ORDER_PLACED 或 IMMEDIATE_MATCH_PENDING)
        if info['status'] == 'ORDER_PLACED' or info['status'] == 'IMMEDIATE_MATCH_PENDING':
            # 检查新订单的最终状态
            if new_order:
                info['status'] = new_order['status']
            else:
                # 如果没有新订单，检查是否有撮合发生
                if hist_deal:
                    info['status'] = 'PASSIVE_DEAL'
        if hist_deal:
            info['delay_deal'] = 'DELAY_DEAL'
        else:
            info['delay_deal'] = 'NO_DELAY_DEAL'

        # 4. 计算奖励 (基于时间推进、撮合情况等)
        reward = self._calculate_reward(pre_cash, status=info['status'], punish=0) # 被动成交步的奖励

        # 5. 更新数据时间步
        self.data.step()
        done = (self.data.current_index + 1 >= self.data.end_index) # 更新 done 状态

        # 6. 生成状态
        market_state = self.data.obtain_features(do_flatten=self.config.simulation_do_feature_flatten)
        private_state = self._generate_private_state()

        return market_state, private_state, reward, done, info

    def _insert_order_sorted(self, order):
        """
        使用二分插入保持挂单列表有序
        """
        import bisect
    
        if self.config.simulation_direction == 'sell':
            # 卖单按价格升序排列（价格低优先）
            price_key = lambda x: x['price']
        else:  # buy
            # 买单按价格降序排列（价格高优先）
            price_key = lambda x: -x['price']
    
        # 创建一个带有价格键的临时列表用于比较
        keys = [price_key(order) for order in self.holding_orders]
        # 使用 bisect 查找插入位置
        pos = bisect.bisect_left(keys, price_key(order))
        # 在正确位置插入订单
        self.holding_orders.insert(pos, order)

    def _calculate_reward(self, pre_cash, status, punish=0, is_final_step=False):
        """
        计算奖励函数。
        Args:
            pre_cash (float): 执行动作前的现金。
            status (str): 执行后的状态。
            punish (float): 即时惩罚项 (例如，最后时刻还有挂单)。
            is_final_step (bool): 是否是最后一个时间步 (用于清算奖励)。
        Returns:
            float: 计算出的奖励。
        """
        # 基准交易量计算
        if self.data.base_volume != 0:
            expected_volume = self.data.base_volume * self.config.simulation_volume_ratio
        elif self.data.basis_volume != 0:
            expected_volume = self.data.basis_volume * self.config.simulation_volume_ratio / 48
        else:
            expected_volume = 1  # 默认值，避免除零
        expected_volume_per_step = expected_volume / self.config.simulation_planning_horizon

        # 基础奖励：基于收益（或成本节约）
        # cash 增加（卖单）或减少（买单，为负）表示好的执行
        cash_change = self.cash - pre_cash
        # 归一化基础奖励
        basic_reward = cash_change / (self.data.basis_price + 1e-8) / (expected_volume_per_step + 1e-8)
        # 如果是买单，收益为负是好事，需要调整符号
        if self.config.simulation_direction == 'buy':
             basic_reward = -basic_reward # 买单希望 cash_change (负值) 越大越好，即成本越低

        # --- 原有惩罚项 ---
        # 撤单惩罚项 (基于订单状态)
        cancel_punishment = 0.0
        if status in ['FILLED', 'NO_OPERATION', 'ACTIVE_FILLED', 'LIQUIDATED']:
            cancel_punishment = 0
        elif status == 'PARTIAL_FILLED':
            cancel_punishment = 0.05
        elif status == 'NOT_FILLED' or status == 'ORDER_PLACED': # 被动挂单未成交也算轻微惩罚
            cancel_punishment = 0.1
        # 如果是最后一步且还有挂单，则在 step/done 逻辑中已处理撤单和惩罚

        # 持仓惩罚项：当前挂单量 / 基准交易量
        holding_punishment = sum(order['quantity'] for order in self.holding_orders) / (expected_volume_per_step + 1e-8)

        # 清算惩罚项 (由 step/done 逻辑传入)
        order_after_liquidation_punishment = punish

        # 均匀执行惩罚项 (鼓励时间上均匀执行)
        _recommand_quantity = self.total_quantity * (self.data.end_index - self.data.current_index) / self.config.simulation_planning_horizon
        if self.data.base_volume != 0:
            expected_total = self.data.base_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon
        elif self.data.basis_volume != 0:
            expected_total = self.data.basis_volume * self.config.simulation_volume_ratio / self.config.simulation_planning_horizon / 48
        else:
            expected_total = 1
        # 使用绝对差值归一化
        linear_reg = abs(self.quantity - _recommand_quantity) / (expected_total + 1e-8)

        # --- 新增/调整的奖励/惩罚 ---
        # 主动成交奖励/惩罚？ (例如，鼓励在合适时机主动成交)
        # 被动成交奖励/惩罚？ (例如，惩罚长时间挂单但不成交)

        # 最终奖励组合
        reward = (
                basic_reward
                - self.config.simulation_linear_reg_coeff * linear_reg
                - self.config.simulation_liq_coeff * order_after_liquidation_punishment
                - self.config.holding_cost_coeff * holding_punishment
                - self.config.simulation_cancel_coeff * cancel_punishment
        )

        # 如果是最后一步，可以考虑加入最终清算的特殊奖励/惩罚
        if is_final_step:
            # 例如，根据最终的 IS (Implementation Shortfall) 给予奖励
            # 或者根据清算价格与 TWAP/VWAP 的差距
            pass # 当前逻辑已包含在 basic_reward 中，因为 cash 已更新

        return reward
    
    def _get_orderbook_snapshot(self):
        """获取当前订单簿快照"""
        snapshot = {}
        for level in range(1, 11):
            snapshot[f'bidPrice{level}'] = self.data.obtain_level('bidPrice', level)
            snapshot[f'bidVolume{level}'] = self.data.obtain_level('bidVolume', level)
            snapshot[f'NumOrdersB{level}'] = self.data.obtain_level('NumOrdersB', level)
            snapshot[f'askPrice{level}'] = self.data.obtain_level('askPrice', level)
            snapshot[f'askVolume{level}'] = self.data.obtain_level('askVolume', level)
            snapshot[f'NumOrdersS{level}'] = self.data.obtain_level('NumOrdersS', level)
        return snapshot
    
    def _get_order_flow(self):
        """获取当前时间窗口内的订单流"""
        if self.data.orders is None:
            return None
            
        current_time = self.data.backtest_data.loc[self.data.current_index, 'time']
        next_time = self.data.backtest_data.loc[self.data.current_index + 1, 'time']
        
        # 根据交易方向过滤订单流
        if self.config.simulation_direction == 'sell':
            order_filter = (self.data.orders['OrderBSFlag'] == 'B') & (self.data.orders['OrderType'] == 'A')
        elif self.config.simulation_direction == 'buy':
            order_filter = (self.data.orders['OrderBSFlag'] == 'S') & (self.data.orders['OrderType'] == 'A')
        else:
            order_filter = pd.Series([False] * len(self.data.orders), index=self.data.orders.index)
        
        orders = self.data.orders[
            (self.data.orders['OrderTime'] >= current_time) &
            (self.data.orders['OrderTime'] < next_time) &
            order_filter
        ].copy()
        
        return orders

    @property
    def observation_dim(self):
        return len(self.config.simulation_features) * self.config.simulation_lookback_horizon

    def get_metric(self, mtype='IS'):
        # IS: implementation shortfall
        if mtype == 'IS': 
            return self.data.basis_price * (self.total_quantity - self.quantity) - self.cash
        # TWAP: bp over mid price TWAP
        if mtype == 'TWAP':
            if self.total_quantity == self.quantity:
                return 0
            avg_price = abs(self.cash / (self.total_quantity - self.quantity))
            TWAP_mid = self.data.backtest_data.loc[self.data.start_index:self.data.end_index, 'latest_price'].mean()
            bp = (avg_price - TWAP_mid) / self.data.basis_price * 10000
            if self.config.simulation_direction == 'buy':
                return bp
            return -bp
        # VWAP: bp over VWAP
        if mtype == 'VWAP':
            if self.total_quantity == self.quantity:
                return 0
            avg_price = abs(self.cash / (self.total_quantity - self.quantity))
            vwap = self.data.backtest_data.loc[self.data.end_index, 'vwap_5min']
            bp = (avg_price - vwap) / vwap * 10000
            if self.config.simulation_direction == 'buy':
                return bp
            return -bp
        if mtype == 'cancel_rate':
            return (self.canceled_orders / self.total_orders) if self.total_orders != 0 else 0
        if mtype == 'ODR_NUM':
            return self.total_orders + self.canceled_orders
        if mtype == "immediate_ratio":
            return self.immediate_orders / self.total_orders if self.total_orders != 0 else 0

class BacktestValidator:
    """
    回测验证器，用于验证交易的真实性
    """
    def __init__(self, env):
        self.env = env
    
    def validate_trade_realism(self):
        """
        验证交易的真实性
        """
        issues = []
        
        # 通过 env.env 访问 ExecutionEnv 的属性
        trade_history = self.env.env.trade_history if hasattr(self.env, 'env') else self.env.trade_history
        
        # 1. 检查价格合理性
        for trade in trade_history:
            if trade['price'] <= 0:
                issues.append(f"Invalid price: {trade['price']}")
            
            # 检查价格是否在合理范围内
            latest_bid = self.env.env.data.obtain_level('bidPrice', 1) if hasattr(self.env, 'env') else self.env.data.obtain_level('bidPrice', 1)
            latest_ask = self.env.env.data.obtain_level('askPrice', 1) if hasattr(self.env, 'env') else self.env.data.obtain_level('askPrice', 1)
            
            # 检查价格偏离是否在容忍范围内
            env_obj = self.env.env if hasattr(self.env, 'env') else self.env
            if hasattr(env_obj.config, 'price_deviation_tolerance'):
                tolerance = env_obj.config.price_deviation_tolerance
                mid_price = (latest_bid + latest_ask) / 2
                
                if abs(trade['price'] - mid_price) / mid_price > tolerance:
                    issues.append(f"Price deviation exceeds tolerance: {trade['price']} vs mid: {mid_price}")
        
        # 获取 ExecutionEnv 对象
        env_obj = self.env.env if hasattr(self.env, 'env') else self.env
        
        # 2. 检查成交量合理性
        total_traded = sum(trade['quantity'] for trade in trade_history)
        if total_traded > env_obj.total_quantity * 1.1:  # 10%超量警告
            issues.append(f"Total traded volume exceeds expected: {total_traded} vs {env_obj.total_quantity}")
        
        return issues
    
    def generate_execution_report(self):
        """
        生成执行报告，用于验证回测真实性
        """
        # 获取 ExecutionEnv 对象
        env_obj = self.env.env if hasattr(self.env, 'env') else self.env
        
        if not env_obj.trade_history:
            return {}
            
        # 计算各种统计指标
        trade_prices = [trade['price'] for trade in env_obj.trade_history]
        trade_quantities = [trade['quantity'] for trade in env_obj.trade_history]
        
        report = {
            'total_quantity': env_obj.total_quantity,
            'executed_quantity': env_obj.total_quantity - env_obj.quantity,
            'execution_rate (%) ': 100 * (env_obj.total_quantity - env_obj.quantity) / env_obj.total_quantity if env_obj.total_quantity > 0 else 0,
            'average_execution_price': np.average(trade_prices, weights=trade_quantities) if trade_prices else 0,
            'price_volatility': np.std(trade_prices) if trade_prices else 0,
            'trade_count': len(env_obj.trade_history),
            'order_count': env_obj.total_orders,
            'cancel_rate': env_obj.canceled_orders / env_obj.total_orders if env_obj.total_orders > 0 else 0,
            'immediate_execution_rate': env_obj.immediate_orders / env_obj.total_orders if env_obj.total_orders > 0 else 0,
            'market_order_ratio': len([t for t in env_obj.trade_history if t['match_type'] == 'market_order']) / len(env_obj.trade_history) if env_obj.trade_history else 0
        }
        
        return report


class HierarchicalActionWrapper(BaseWrapper):
    """
    分层动作包装器。
    高层动作：'active' (0) 或 'passive' (1)
    低层动作：
        - 如果高层是 'active'：一个表示主动成交量的数值（绝对量或相对量）。
        - 如果高层是 'passive'：一个表示挂单价和挂单量的数值或索引。
    为了兼容现有代码和简化，低层动作可以复用旧的 DiscretePriceQuantityWrapper 的逻辑，
    或者定义新的离散/连续空间。
    这里我们创建一个组合了高层离散选择和复用旧低层离散动作的包装器。
    """
    def __init__(self, env):
        super(HierarchicalActionWrapper, self).__init__(env)
        # 高层动作空间：0 -> 'active', 1 -> 'passive'
        self.high_level_actions = ['active', 'passive']
        self.high_level_action_dim = len(self.high_level_actions)
        
        # 低层动作空间：复用配置中的离散动作
        self.low_level_discrete_actions = self.config.simulation_discrete_actions
        self.low_level_action_dim = len(self.low_level_discrete_actions)
        
        # 总的组合动作维度（用于某些需要 flat action index 的场景）
        # 例如，action_index = high_action_index * low_level_action_dim + low_action_index
        # 但这通常不是直接使用的，而是分别采样高层和低层动作
        self.combined_action_dim = self.high_level_action_dim * self.low_level_action_dim

    @property
    def action_sample_func(self):
        """返回一个函数，用于同时采样高层和低层动作"""
        # 返回一个 lambda 函数，返回 (high_action_index, low_action_index) 的元组
        return lambda: (
            np.random.randint(self.high_level_action_dim),
            np.random.randint(self.low_level_action_dim)
        )

    @property
    def action_dim(self):
        """返回动作空间的维度信息，这里返回一个元组表示分层结构"""
        return (self.high_level_action_dim, self.low_level_action_dim)

    def step(self, action):
        """
        执行分层动作。
        Args:
            action: 一个元组 (high_level_action_index, low_level_action_index) 
                    或 (high_level_action_str, low_level_action_obj)。
                    为了简化，我们假设输入是索引元组。
        Returns:
            tuple: (market_state, private_state, reward, done, info)
        """
        if isinstance(action, (tuple, list)) and len(action) == 2:
            high_action_input, low_action_input = action
        else:
            # 如果输入不是元组，尝试解析或报错
            # 例如，可以将单一整数解码为组合动作
            # action_index = action
            # high_action_input = action_index // self.low_level_action_dim
            # low_action_input = action_index % self.low_level_action_dim
            # 为了清晰，我们要求输入必须是元组
            raise ValueError("Action for HierarchicalActionWrapper must be a tuple (high_action, low_action)")

        # 解析高层动作
        if isinstance(high_action_input, int):
            if 0 <= high_action_input < self.high_level_action_dim:
                high_level_action = self.high_level_actions[high_action_input]
            else:
                raise ValueError(f"Invalid high level action index: {high_action_input}")
        elif isinstance(high_action_input, str) and high_action_input in self.high_level_actions:
            high_level_action = high_action_input
        else:
            raise ValueError(f"Invalid high level action: {high_action_input}")

        # 解析低层动作 (直接传递索引给 env.step，env 内部会根据 passive/active 进行处理)
        low_level_action = low_action_input

        # 调用环境的分层 step 方法
        return self.env.step(high_level_action, low_level_action)

# 更新 make_env 函数以使用新的包装器
def make_env(config):
    return HierarchicalActionWrapper(ExecutionEnv(config))

def run_env_test():
    config = DefaultConfig()
    env = make_env(config) # 现在使用 HierarchicalActionWrapper
    market_state, private_state = env.reset()
    print(f'Initial market_state shape: {market_state.shape}')
    print(f'Initial private_state: {private_state}')
    # print('snapshot = ')
    # print(env.data.backtest_data.loc[env.data.current_index])

    # 测试主动成交
    print("\n--- Testing Active Execution ---")
    # 假设动作 0 是 'active'，低层动作索引 5 (例如，中等数量)
    high_action_index = 0 # 'active'
    low_action_index = 2   # 使用一个有效的主动成交动作索引
    market_state, private_state, reward, done, info = env.step((high_action_index, low_action_index))
    print(f'After Active Step - market_state shape: {market_state.shape}')
    print(f'After Active Step - private_state: {private_state}')
    print(f'After Active Step - reward: {reward}')
    print(f'After Active Step - done: {done}')
    print(f'After Active Step - info: {info}')
    # print('snapshot = ')
    # print(env.data.backtest_data.loc[env.data.current_index])

    # 重置环境以测试被动成交
    market_state, private_state = env.reset()
    print("\n--- Testing Passive Execution ---")
    # 测试被动成交
    # 假设动作 1 是 'passive'，低层动作索引 10 (例如，稍高的价格，中等数量)
    high_action_index = 1 # 'passive'
    low_action_index = len(config.simulation_discrete_actions_active) + 20 # 使用一个有效的被动成交动作索引
    market_state, private_state, reward, done, info = env.step((high_action_index, low_action_index))
    print(f'After Passive Step - market_state shape: {market_state.shape}')
    print(f'After Passive Step - private_state: {private_state}')
    print(f'After Passive Step - reward: {reward}')
    print(f'After Passive Step - done: {done}')
    print(f'After Passive Step - info: {info}')
    
    # 验证回测真实性
    validator = BacktestValidator(env)
    issues = validator.validate_trade_realism()
    report = validator.generate_execution_report()
    
    print("\n--- Backtest Validation ---")
    if issues:
        print("Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No validation issues found.")
    
    print("\nExecution Report:")
    for key, value in report.items():
        print(f"  {key}: {value}")

def run_sl():
    config = DefaultConfig()
    env = make_env(config) # 使用新的分层包装器
    market_state, private_state = env.reset()
    total_reward = 0
    for i in range(30):
        # 简单策略：交替进行主动和被动成交，或者随机选择
        # 这里随机选择高层动作，低层动作也随机选择一个索引
        high_action_index = np.random.randint(2) # 0 or 1
        low_action_index = np.random.randint(len(config.simulation_discrete_actions)) # Use the low-level action space from config
        market_state, private_state, reward, done, info = env.step((high_action_index, low_action_index))
        total_reward += reward
        # print('market_state = {}'.format(market_state))
        print(f'Step {i}: private_state = {private_state}')
        print(f'Step {i}: reward = {reward}, total_reward = {total_reward}')
        # print('done = {}'.format(done))
        # print('info = {}'.format(info))
        print(f'Step {i}: cancel_rate = {env.get_metric("cancel_rate")}')
        if done:
            print(f"Episode finished after {i+1} steps.")
            break # 退出循环
    print(f"Final total reward: {total_reward}")
    print(f"Final IS: {env.get_metric('IS')}")
    print(f"Final VWAP: {env.get_metric('VWAP')}")
        
if __name__ == '__main__':
    # run_data_prepare()
    run_env_test()
    # run_sl()
