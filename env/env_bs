import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

import os
import numpy as np
import pandas as pd
from const_simple import *
try:
    from gym import spaces
except ImportError:
    from gymnasium import spaces
from loader import Data

class SimplifiedEnvConfig(object):
    # 订单簿数据字段列名重置
    column_trans_dic = {
        'TradVolume':'volume',
        'LastPrice':'lastPrice',
        'PreCloPrice':'prevClosePrice',
        'Turnover':'value',
        'OpenPrice':'openPrice',
        'UpdateTime':'time'
    }
    
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

    # 评估数据集配置
    eval_configurations = {
        'train_train': (CODE_LIST, TRAIN_DATE_LIST),                     # 训练股票和日期
        'train_test' : (CODE_LIST, VALIDATION_DATE_LIST),                 # 训练股票和测试日期
        'test_train' : (VALIDATION_CODE_LIST, TRAIN_DATE_LIST),           # 测试股票和训练日期
        'test_test'  : (VALIDATION_CODE_LIST, VALIDATION_DATE_LIST)        # 测试股票和测试日期
    }
    
    # 交易参数设置
    simulation_commission_freq = '1s'
    simulation_planning_horizon = 300  # 5分钟 = 300秒
    feature_horizon = 10
    simulation_lookback_horizon = 100  # 添加回看窗口大小
    
    # 交易量参数
    simulation_num_shares = 10
    simulation_discrete_quantities = 5
    if simulation_discrete_quantities > simulation_num_shares:
        simulation_discrete_quantities = simulation_num_shares
        
    simulation_volume_ratio = 0.005
    
    # 奖励函数系数
    holding_cost_coeff = 0.0
    simulation_linear_reg_coeff = 1.0
    simulation_liq_coeff = 1000.0
    simulation_cancel_coeff = 2.0
    
    # 清算任务方向
    simulation_direction = 'sell'
    
    # 市场冲击参数
    market_impact_coeff = 0.1
    orderbook_ratio = 1.2
    volatility_limit = 0.02
    
    # 价格偏离容忍度
    price_deviation_tolerance = 0.02

    # --- 新增配置以避免硬编码 ---
    # 订单簿深度 (用于市场状态和订单处理)
    order_book_depth = 10
    # 私有状态中跟踪的最大活跃订单数
    max_tracked_orders = 20
    # 动作空间阈值 (用于判断是否执行挂单/市价单)
    action_threshold = 0.2


class SimplifiedExecutionEnv(object):
    def __init__(self, config, eval_mode=None):
        # 初始化核心交易属性
        self.config = config
        self.queue_position = 1.0  # 初始队列位置
        self.remaining_quantity = 0

        # 设置订单簿深度 (从配置读取)
        self.order_book_depth = getattr(self.config, 'order_book_depth', 5)
        self.max_tracked_orders = getattr(self.config, 'max_tracked_orders', 5)
        self.action_threshold = getattr(self.config, 'action_threshold', 0.2)
        
        # 只初始化一次data对象
        self.data = Data(config)
        
        self.eval_mode = eval_mode
        self.valid_code_date_list = self.get_valid_code_date_list()
        
        # 创建订单放置策略
        self._order_placement_strategy = self._create_order_placement_strategy()
        
        # 初始化历史指标
        self.last_market_order_flow = 0
        self.last_spread = 0
        self.last_bid_volumes = None
        self.last_ask_volumes = None
        self.canceled_orders_step = 0
        self.passive_trades_step = 0   # 本步被动成交量，供 reward 计算用
        self._terminal_forced_qty = 0  # 末期强制清算量，供 reward 计算用

        # 市场状态和私有状态的维度
        self._market_state_dim = self.market_state_dim
        self._private_state_dim = self.private_state_dim
        self.state_dim = self._market_state_dim + self._private_state_dim
        
        # 动作空间
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), 
                                    high=np.array([1.0, 1.0]), 
                                    dtype=np.float32)
            
        # 初始化其他组件
        self.current_code = None 
        self.current_date = None 
        # 移除重复的 self.data = Data(config) 初始化
        self.cash = 0
        self.pnl = 0
        self.cost = 0
        self.pending_quantity = 0
        self.total_quantity = 0
        self.quantity = 0
        self.active_orders = []
        self.order_history = []
        self.trade_history = []
        self.execution_prices = []
        self.execution_quantities = []
        self.market_vwap_history = []
        self.normalization_price = 1.0
        self.normalization_volume = 1.0
        self.normalization_time = 1.0
        self.market_state_history = []
        self.private_state_history = []
        
        # 交易指标统计
        self.total_orders = 0
        self.canceled_orders = 0
        self.active_trades = 0
        self.passive_trades = 0
        self.price_performance_bp = 0
        
        # 新增：记录上一个动作用于奖励计算
        self.last_action = None
        self.last_action_was_no_action = False

    def _update_queue_positions(self):
        """
        改进的队列位置更新算法 - 基于市场微观结构
        """
        # 获取当前市场数据
        current_bid_volumes = [self.data.obtain_level('bidVolume', i) for i in range(1, 11)]
        current_ask_volumes = [self.data.obtain_level('askVolume', i) for i in range(1, 11)]
        
        # 获取市场订单流数据
        try:
            market_order_flow = self.data.obtain_level('market_order_flow') if 'market_order_flow' in self.data.data.columns else 0
            limit_order_flow = self.data.obtain_level('limit_order_flow') if 'limit_order_flow' in self.data.data.columns else 0
        except:
            market_order_flow = 0
            limit_order_flow = 0
        
        for order in self.active_orders:
            level = order['level']
            if level > len(current_bid_volumes) or level > len(current_ask_volumes):
                continue
                
            if self.config.simulation_direction == 'sell':
                # 卖单在ask队列中
                total_volume = current_ask_volumes[level-1]
                if total_volume > 0:
                    # 基于市场订单流的队列前进
                    if market_order_flow > 0:  # 有买入市价单，队列前进
                        progress_ratio = min(0.3, market_order_flow / (total_volume + 1e-6) * 0.1)
                        order['queue_position'] = max(0.01, order['queue_position'] - progress_ratio)
                    
                    # 基于限价订单流的队列变化
                    if limit_order_flow < 0:  # 有卖单撤单或买单挂单，队列可能前进
                        order['queue_position'] = max(0.01, order['queue_position'] - 0.02)
                    elif limit_order_flow > 0:  # 有卖单挂单，队列可能后退
                        order['queue_position'] = min(0.99, order['queue_position'] + 0.01)
                    
                    # 基于历史订单簿变化的队列调整
                    if hasattr(self, 'last_ask_volumes') and self.last_ask_volumes:
                        volume_change = self.last_ask_volumes[level-1] - current_ask_volumes[level-1]
                        if volume_change > 0:
                            progress = min(0.2, volume_change / total_volume)
                            order['queue_position'] = max(0.01, order['queue_position'] - progress)
            else:
                # 买单在bid队列中
                total_volume = current_bid_volumes[level-1]
                if total_volume > 0:
                    # 基于市场订单流的队列前进
                    if market_order_flow < 0:  # 有卖出市价单，队列前进
                        progress_ratio = min(0.3, abs(market_order_flow) / (total_volume + 1e-6) * 0.1)
                        order['queue_position'] = max(0.01, order['queue_position'] - progress_ratio)
                    
                    # 基于限价订单流的队列变化
                    if limit_order_flow > 0:  # 有买单撤单或卖单挂单，队列可能前进
                        order['queue_position'] = max(0.01, order['queue_position'] - 0.02)
                    elif limit_order_flow < 0:  # 有买单挂单，队列可能后退
                        order['queue_position'] = min(0.99, order['queue_position'] + 0.01)
                    
                    # 基于历史订单簿变化的队列调整
                    if hasattr(self, 'last_bid_volumes') and self.last_bid_volumes:
                        volume_change = self.last_bid_volumes[level-1] - current_bid_volumes[level-1]
                        if volume_change > 0:
                            progress = min(0.2, volume_change / total_volume)
                            order['queue_position'] = max(0.01, order['queue_position'] - progress)
            
            # 添加小幅随机波动模拟市场噪声
            random_noise = np.random.normal(0, 0.01)
            order['queue_position'] = max(0.01, min(0.99, order['queue_position'] + random_noise))
        
        # 保存当前订单簿状态
        self.last_bid_volumes = current_bid_volumes.copy()
        self.last_ask_volumes = current_ask_volumes.copy()

        
    def _process_orders_with_market_impact(self):
        """统一市场冲击模型
        参数说明:
        - impact_coeff: 从config读取冲击系数(默认0.001)
        - remaining_quantity: 当前剩余交易量
        - market_depth: 买一价量 + 卖一价量"""
        effective_quantity = min(self.remaining_quantity, getattr(self.config, 'max_order_size', self.remaining_quantity))
        market_depth = sum(self.data.obtain_level('bidVolume', i) for i in range(1, 3)) + \
                        sum(self.data.obtain_level('askVolume', i) for i in range(1, 3))
        mid_price = 0.5 * (self.data.obtain_level('bidPrice',1) + self.data.obtain_level('askPrice',1))
        impact = getattr(self.config, 'impact_coeff', 0.001) * (effective_quantity / max(market_depth, 1)) * mid_price
        # 存储临时冲击，不要直接改 bar 数据
        self._last_market_impact = impact
        
        # # 应用冲击
        # self.data.adjust_price(impact, self.config.simulation_direction)
            
    def _should_place_order(self, level):
        """
        判断是否应该在指定档位挂单
        - param level: 挂单位置的档位
        - return: 是否应该挂单
        """
        # 获取订单簿数据
        orderbook = {
            'bid_volume': [self.data.obtain_level('bidVolume', i) for i in range(1, 10)],
            'ask_volume': [self.data.obtain_level('askVolume', i) for i in range(1, 10)]
        }
        
        # 获取波动率
        volatility = self.data.obtain_level('trend') if 'trend' in self.data.data.columns else 0
        
        # 使用策略函数判断
        return self._order_placement_strategy(orderbook, volatility)

    def get_valid_code_date_list(self):
        code_date_list = []
        
        # 根据评估模式选择数据集
        if self.eval_mode and hasattr(self.config, 'eval_configurations') and self.eval_mode in self.config.eval_configurations:
            code_list, date_list = self.config.eval_configurations[self.eval_mode]
        else:
            # 默认使用训练集
            code_list = self.config.code_list
            date_list = self.config.date_list
            
        for code in code_list:
            for date in date_list:
                if self.data.data_exists(code, date):
                    code_date_list.append((code, date))
        return code_date_list

    def reset(self, code=None, date=None, start_index=None):
        """
        重置环境
        """
        count = 0
        while True:
            if code is None and date is None:
                ind = np.random.choice(len(self.valid_code_date_list))
                self.current_code, self.current_date = self.valid_code_date_list[ind]
            else:
                self.current_code, self.current_date = code, date

            try:
                self.data.obtain_data(self.current_code, self.current_date, start_index)
                break
            except AssertionError as e:
                count += 1
                print(f'Invalid data: code={self.current_code}, date={self.current_date}, error={str(e)}')
                if count > 100:
                    raise ValueError(f"code={code}, date={date} is invalid after 100 retries")
            except Exception as e:
                print(f"Critical error loading data: {str(e)}")
                raise

        # 初始化交易状态
        self.cash = 0
        self.cost = 0
        self.pending_quantity = 0

        # 计算总交易量
        if self.data.basis_volume != 0:
            self.total_quantity = int(self.config.simulation_volume_ratio * self.data.basis_volume)
        else:
            self.total_quantity = 100

        self.quantity = self.total_quantity
        self._terminal_forced_qty = 0

        # 重置交易指标统计
        self.total_orders = 0
        self.canceled_orders = 0
        self.active_trades = 0
        self.passive_trades = 0
        self.vwap_cost = 0  # 保持兼容性，实际使用price_performance_bp
        
        # 设置规范化参数
        self.normalization_price = self.data.basis_price if self.data.basis_price > 0 else 1.0
        self.normalization_volume = self.data.basis_volume if self.data.basis_volume > 0 else 1.0
        self.normalization_time = self.config.simulation_planning_horizon
        
        # 初始化历史状态缓存
        self.active_orders = []
        self.execution_prices = []
        self.execution_quantities = []
        self.trade_history = []
        self.order_history = []
        self.market_state_history = []
        self.private_state_history = []

        # 生成初始状态
        market_state = self._get_market_state()
        private_state = self._get_private_state()
        
        # 填充历史状态缓存
        lookback_horizon = getattr(self.config, 'simulation_lookback_horizon', 1)
        self.market_state_history = [market_state] * lookback_horizon
        self.private_state_history = [private_state] * lookback_horizon

        return self._get_stacked_states()
    
    def _get_stacked_states(self):
        """
        获取堆叠的历史状态
        """
        # 堆叠市场状态
        stacked_market_state = np.concatenate(self.market_state_history[-getattr(self.config, 'simulation_lookback_horizon', 1):])
        
        # 堆叠私有状态 (只使用当前私有状态)
        current_private_state = self.private_state_history[-1] if self.private_state_history else np.array([])
        
        return stacked_market_state, current_private_state
    
    def _create_order_placement_strategy(self):
        """根据配置创建动态挂单策略。"""
        orderbook_ratio_threshold = self.config.orderbook_ratio
        volatility_threshold = self.config.volatility_limit
        
        def strategy(orderbook, volatility):
            if orderbook['ask_volume'][0] == 0 or orderbook['bid_volume'][0] == 0:
                return False  # 避免零除错误
            # 计算买卖盘口深度比
            bid_ask_ratio = orderbook['bid_volume'][0] / orderbook['ask_volume'][0]
            # 仅在买卖盘口不平衡且波动率较低时挂单
            if bid_ask_ratio > orderbook_ratio_threshold and volatility < volatility_threshold:
                return True
            return False
        
        # 修复：返回函数对象，而不是调用它
        return strategy
        
    def _place_limit_order(self, level):
        """
        放置限价单
        :param level: 挂单位置的档位
        """
        if self.quantity <= 0:
            return  # 无库存，不挂单
        
        if self.config.simulation_direction == 'sell':
            price = self.data.obtain_level('askPrice', level)
        else:
            price = self.data.obtain_level('bidPrice', level)
        
        # 计算挂单量：最多挂剩余库存的30%，但至少1股（前提是还有库存）
        quantity = max(1, int(self.quantity * 0.3))
        
        # 再次确保 quantity 不超过剩余库存（防御性）
        quantity = min(quantity, self.quantity)
        
        if quantity > 0 and price > 0:
            order = {
                'price': price,
                'quantity': quantity,
                'side': self.config.simulation_direction,
                'timestamp': self.data.current_index,
                'level': level,
                'queue_position': self.queue_position,
                'status': 'active'
            }
            self.active_orders.append(order)
            self.quantity -= quantity          # BUG FIX: 挂单时扣减库存
            self.pending_quantity += quantity
            self.total_orders += 1

    def _get_market_state(self):
        """
        获取市场状态  
        Market states:
        • The best bid and ask prices p_b(t) and p_a(t)
        • The first K-1 entries of both volume vectors (v_{b,1}(t),...,v_{b,K-1}(t)) and (v_{a,1}(t),...,v_{a,K-1}(t))
        • The market order flow delta_m(t)
        • The limit order flow delta_l(t)
        • The mid-price drift delta_p(t) = p(t) - p(t - delta_t)
        """
        # 最佳买卖价格
        pb_t = self.data.obtain_level('bidPrice', 1)
        pa_t = self.data.obtain_level('askPrice', 1)
        k_minus_one = self.order_book_depth - 1

        # 前K-1档买卖量 (K=5，所以取前4档)
        vb = [self.data.obtain_level('bidVolume', i) for i in range(1, k_minus_one + 1 )]  # vb,1 vb,2 vb,3 vb,4
        va = [self.data.obtain_level('askVolume', i) for i in range(1, k_minus_one + 1 )]  # va,1 va,2 va,3 va,4
        # 市场订单流 ∆m(t) = 主动买入量 + 主动卖出量
        delta_m = self.data.obtain_level('market_order_flow') if 'market_order_flow' in self.data.data.columns else 0
        # 限价订单流 ∆l(t) = 订单簿量的净变化（买一量变化 - 卖一量变化）
        delta_l = self.data.obtain_level('limit_order_flow') if 'limit_order_flow' in self.data.data.columns else 0
        # 中间价变化 ∆p(t)
        delta_p = self.data.obtain_level('trend') if 'trend' in self.data.data.columns else 0
        # 规范化市场状态
        market_state = np.array([
            pb_t / self.normalization_price,
            pa_t / self.normalization_price
        ] + [v / self.normalization_volume for v in vb] +
        [v / self.normalization_volume for v in va] +
        [delta_m / self.normalization_volume,
        delta_l / self.normalization_volume,
        delta_p / self.normalization_price])
        return market_state
    
    def _get_private_state(self):
        """
        Private states:
        • The time t within the execution horizon
        • The inventory M(t)
        • The number of lots m(t) in {0,1,...,M(t)} sitting in the order book
        • For k in {1,2,...,m(t)}, the levels and queue positions l_k(t), q_k(t) in N times N of the active limit orders
        """
        # 时间进度 (规范化)
        time_progress = (self.data.current_index - self.data.start_index) / self.normalization_time
        # 剩余库存 (规范化)
        inventory = self.quantity / self.total_quantity if self.total_quantity > 0 else 0
        # 挂单数量
        pending_lots = self.pending_quantity
        # 活跃订单信息 (最多考虑前5个订单)
        order_info = []
        for i in range(self.max_tracked_orders):
            if i < len(self.active_orders):
                order = self.active_orders[i]
                # 订单层级和队列位置
                level = order.get('level', 1)  # 使用实际的挂单位置
                queue_pos = order.get('queue_position', 1)  # 使用实际的队列位置
                order_info.extend([level, queue_pos])
            else:
                order_info.extend([0, 0])  # 填充零值
        # 规范化私有状态
        private_state = np.array([time_progress, inventory, pending_lots] + order_info)
        return private_state
    
    def get_trading_metrics(self):
        """
        计算交易指标 - 使用真实累计VWAP作为市场基准
        """
        metrics = {}
        # 撤单率
        metrics['cancel_rate'] = self.canceled_orders / self.total_orders if self.total_orders > 0 else 0.0

        strategy_vwap = 0.0
        if self.execution_quantities and sum(self.execution_quantities) > 0:
            strategy_vwap = sum(p * q for p, q in zip(self.execution_prices, self.execution_quantities)) / sum(self.execution_quantities)
        else:
            metrics['price_performance_bp'] = 0.0
            metrics.update({
                'active_rate': 0.0, 'passive_rate': 0.0,
                'total_orders': self.total_orders,
                'canceled_orders': self.canceled_orders,
                'active_trades': self.active_trades,
                'passive_trades': self.passive_trades,
                'total_executed': self.active_trades + self.passive_trades
            })
            return metrics

        # === 关键修正：使用真实累计VWAP ===
        start_idx = self.data.start_index
        end_idx = min(self.data.end_index, len(self.data.data) - 1)

        total_value = self.data.data.loc[start_idx:end_idx, 'value_dt'].sum()
        total_volume = self.data.data.loc[start_idx:end_idx, 'volume_dt'].sum()

        if total_volume > 0:
            market_vwap = total_value / total_volume
            vwap_source = "真实累计VWAP"
        else:
            market_vwap = self.data.data.loc[start_idx:end_idx, 'close_price'].mean()
            vwap_source = "收盘价均值"

        # print(f"VWAP对比 - 策略: {strategy_vwap:.4f}, 市场({vwap_source}): {market_vwap:.4f}")

        # 计算价格表现（bp）
        if self.config.simulation_direction == 'sell':
            metrics['price_performance_bp'] = (strategy_vwap - market_vwap) / market_vwap * 10000
        else:
            metrics['price_performance_bp'] = (market_vwap - strategy_vwap) / market_vwap * 10000

        # 主动/被动成交率
        total_executed = self.active_trades + self.passive_trades
        metrics['active_rate'] = self.active_trades / total_executed if total_executed > 0 else 0.0
        metrics['passive_rate'] = self.passive_trades / total_executed if total_executed > 0 else 0.0

        # 原始统计
        metrics.update({
            'total_orders': self.total_orders,
            'canceled_orders': self.canceled_orders,
            'active_trades': self.active_trades,
            'passive_trades': self.passive_trades,
            'total_executed': total_executed
        })

        return metrics

    def _get_market_vwap(self, use_5min=False):
        """
        统一的市场VWAP获取方法
        Args:
            use_5min: 是否使用5分钟VWAP
        Returns:
            float: 市场VWAP值
        """
        try:
            if use_5min and 'vwap_5min' in self.data.data.columns:
                return self.data.obtain_level('vwap_5min')
            elif 'vwap' in self.data.data.columns:
                return self.data.obtain_level('vwap')
            else:
                return self.data.obtain_level('close_price')
        except Exception as e:
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"获取市场VWAP失败: {e}")
            return self.data.basis_price if hasattr(self.data, 'basis_price') else 1.0

    def _get_execution_period_vwap(self, use_5min=True):
        """
        获取执行期间的市场VWAP平均值（统一使用真实累计VWAP）
        """
        start_idx = self.data.start_index
        end_idx = min(self.data.end_index, len(self.data.data) - 1)
        
        total_value = self.data.data.loc[start_idx:end_idx, 'value_dt'].sum()
        total_volume = self.data.data.loc[start_idx:end_idx, 'volume_dt'].sum()
        
        if total_volume > 0:
            return total_value / total_volume, "真实累计VWAP"
        else:
            fallback = self.data.data.loc[start_idx:end_idx, 'close_price'].mean()
            return fallback, "收盘价均值"
    
    def _update_state_history(self):
        """
        统一的状态历史更新方法
        """
        market_state = self._get_market_state()
        private_state = self._get_private_state()
        
        # 更新市场状态历史
        self.market_state_history.append(market_state)
        lookback_horizon = getattr(self.config, 'simulation_lookback_horizon', 1)
        if len(self.market_state_history) > lookback_horizon:
            self.market_state_history.pop(0)
        
        # 更新私有状态历史（只保留当前状态）
        self.private_state_history = [private_state]

    def step(self, action):
        """
        执行一步操作 - 优化状态管理
        """
        self.last_action = action
        done = (self.data.current_index + 1 >= self.data.end_index)
        
        # 执行前的信息收集
        info = self._collect_step_info()
        
        if done:
            self._force_liquidation()
            reward = self._calculate_reward()
            self._update_state_history()  # 统一状态更新
            stacked_market_state, stacked_private_state = self._get_stacked_states()
            info.update(self._get_final_step_info())
            return stacked_market_state, stacked_private_state, reward, done, info
        
        # 执行动作
        self._execute_action(action)
        self.data.step()
        
        # 计算奖励和更新状态
        reward = self._calculate_reward()
        self._update_state_history()  # 统一状态更新
        stacked_market_state, stacked_private_state = self._get_stacked_states()
        
        # 更新信息
        info.update(self._get_step_info())
        
        return stacked_market_state, stacked_private_state, reward, done, info

    def _collect_step_info(self):
        """收集步骤信息"""
        try:
            market_vwap = self._get_market_vwap()
        except:
            market_vwap = 0.0
        
        return {
            'code': self.current_code,
            'date': self.current_date,
            'current_index': self.data.current_index,
            'quantity': self.quantity,
            'pending_quantity': self.pending_quantity,
            'market_vwap': market_vwap,
            'cash': self.cash,
            'total_quantity': self.total_quantity
        }

    def _get_step_info(self):
        """获取步骤执行信息"""
        return {
            'market_vwap': self._get_market_vwap(),
            'cash': self.cash
        }

    def _get_final_step_info(self):
        """获取最终步骤信息"""
        info = {
            'status': 'LIQUIDATED',
            'market_vwap': self._get_market_vwap(),
            'cash': self.cash
        }
        if hasattr(self, 'get_trading_metrics'):
            info['metrics'] = self.get_trading_metrics()
        return info
    
    def _execute_action(self, action):
        """
        执行动作 (正确处理连续动作空间)
        action: [0-1] 表示挂单位置, [0-1] 表示执行强度
        """
        # --- 新增：无动作检查 ---
        if self._should_do_no_action(action):
            # 记录无动作决策
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"步骤 {self.data.current_index}: 执行无动作")
            return
        
        if self.quantity <= 0:
            # 已完成交易，不执行任何操作
            return

        # 确保动作格式正确
        if not isinstance(action, (np.ndarray, list, tuple)) or len(action) != 2:
            raise ValueError("动作必须是包含两个元素的数组或列表 [挂单位置, 执行强度]")
        
        place_level = float(action[0])
        execution_intensity = float(action[1])
        
        # 1. 挂单决策
        if place_level > self.action_threshold:
            # [原有挂单逻辑保持不变...]
            level = int(1 + (place_level - self.action_threshold) / (1.0 - self.action_threshold) * (self.order_book_depth - 1))
            level = max(1, min(level, self.order_book_depth))
            
            # 获取订单簿数据
            orderbook = {
                'bid_volume': [self.data.obtain_level('bidVolume', i) for i in range(1, 11)],
                'ask_volume': [self.data.obtain_level('askVolume', i) for i in range(1, 11)]
            }
            volatility = self.data.obtain_level('trend') if 'trend' in self.data.data.columns else 0
            
            if self._order_placement_strategy(orderbook, volatility):
                self._place_limit_order(level)
        
        # 2. 执行决策
        if execution_intensity > self.action_threshold:
            # [原有执行逻辑保持不变...]
            execution_ratio = (execution_intensity - self.action_threshold) / (1.0 - self.action_threshold) * (1.0 - self.action_threshold)
            self._market_order_execution(execution_ratio)
        
        # 3. 处理订单和更新状态
        self.canceled_orders_step = 0
        self.passive_trades_step = 0
        self._process_orders()
        self._update_queue_positions()
        self._cancel_unfavorable_orders()

    def _should_do_no_action(self, action):
        """
        判断是否应该执行无动作
        条件：
        1. 动作值接近无动作区域
        2. 市场条件不适合交易
        3. 剩余数量为0
        """
        place_level = float(action[0])
        execution_intensity = float(action[1])
        
        # 条件1: 两个动作维度都低于阈值
        both_below_threshold = (place_level <= self.action_threshold and 
                            execution_intensity <= self.action_threshold)
        
        # 条件2: 剩余数量为0
        no_remaining_quantity = self.quantity <= 0
        
        # 条件3: 市场条件恶劣（可选）
        market_conditions_bad = self._check_market_conditions()
        
        return both_below_threshold or no_remaining_quantity or market_conditions_bad

    def _check_market_conditions(self):
        """
        检查市场条件是否适合交易
        返回True表示市场条件恶劣，应该执行无动作
        """
        try:
            # 检查价差
            bid_price = self.data.obtain_level('bidPrice', 1)
            ask_price = self.data.obtain_level('askPrice', 1)
            if bid_price <= 0 or ask_price <= 0:
                return True
                
            spread = (ask_price - bid_price) / ((ask_price + bid_price) / 2)
            if spread > 0.05:  # 价差超过5%，市场流动性差
                return True
            
            # 检查波动率（trend 是价格绝对变动，需除以基准价转为相对变动率）
            if 'trend' in self.data.data.columns:
                trend_abs = abs(self.data.obtain_level('trend'))
                volatility = trend_abs / (self.data.basis_price + 1e-6)
            else:
                volatility = 0
            if volatility > 0.02:  # 相对波动超过 2% 才视为恶劣
                return True
                
            # 检查订单簿深度
            bid_volume = sum(self.data.obtain_level('bidVolume', i) for i in range(1, 4))
            ask_volume = sum(self.data.obtain_level('askVolume', i) for i in range(1, 4))
            if bid_volume == 0 or ask_volume == 0:
                return True
                
        except Exception as e:
            # 如果获取市场数据失败，保守起见执行无动作
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"检查市场条件时出错: {e}")
            return True
            
        return False

    def _cancel_unfavorable_orders(self):
        """
        基于订单表现的智能撤单逻辑
        """
        new_active_orders = []
        current_mid_price = (self.data.obtain_level('bidPrice', 1) + self.data.obtain_level('askPrice', 1)) / 2 if self.data.obtain_level('bidPrice', 1) > 0 and self.data.obtain_level('askPrice', 1) > 0 else 0
        
        for order in self.active_orders:
            # 计算订单价格偏离度
            if current_mid_price > 0:
                price_deviation = abs(order['price'] - current_mid_price) / current_mid_price
            else:
                price_deviation = 0.0
            
            # 计算订单预期成交概率
            if self.config.simulation_direction == 'sell':
                execution_prob = max(0.0, 1 - price_deviation / self.config.price_deviation_tolerance)
            else:
                execution_prob = max(0.0, 1 - price_deviation / self.config.price_deviation_tolerance)
            
            # 计算订单挂单时间 (相对执行窗口)
            time_elapsed = (self.data.current_index - order['timestamp']) / self.config.simulation_planning_horizon
            
            # 决定是否撤单（放宽：原来 0.2+0.6*t 在窗口后段阈值高达0.8，
            # 且硬性上限只放开 1.5 倍容忍度，导致几乎所有非触及价的限价单
            # 还没来得及成交就被撤，被动成交率被压到接近0、撤单率被顶到9成左右）
            cancel_threshold = 0.1 + 0.3 * time_elapsed  # 随时间增加的撤单阈值，起点更低、爬升更慢
            if execution_prob < cancel_threshold or price_deviation > self.config.price_deviation_tolerance * 2.5:
                # 撤单 - 恢复数量到可用库存
                order['status'] = 'canceled'
                self.order_history.append(order)
                self.canceled_orders += 1
                self.canceled_orders_step += 1
                # 修复：撤单时恢复数量
                self.quantity += order['quantity']
                self.pending_quantity -= order['quantity']
            else:
                new_active_orders.append(order)
        
        self.active_orders = new_active_orders

    
    def _market_order_execution(self, ratio):
        """
        改进的市价单执行模型，考虑订单簿流动性
        """
        if self.quantity <= 0:
            return  # 无库存，不执行市价单

        execution_quantity = max(1, int(self.quantity * ratio))
        execution_quantity = min(execution_quantity, self.quantity)
        
        if execution_quantity > 0:
            # 根据方向获取订单簿数据
            if self.config.simulation_direction == 'sell':
                # 卖单：从买一价开始吃单
                prices = [self.data.obtain_level('bidPrice', i) for i in range(1, self.order_book_depth+1)]
                volumes = [self.data.obtain_level('bidVolume', i) for i in range(1, self.order_book_depth+1)]
            else:
                # 买单：从卖一价开始吃单
                prices = [self.data.obtain_level('askPrice', i) for i in range(1, self.order_book_depth+1)]
                volumes = [self.data.obtain_level('askVolume', i) for i in range(1, self.order_book_depth+1)]
            
            # 计算成交量加权平均执行价格
            total_value = 0
            remaining_quantity = execution_quantity
            
            for i in range(self.order_book_depth):
                if remaining_quantity <= 0:
                    break
                    
                available_volume = min(remaining_quantity, volumes[i])
                total_value += available_volume * prices[i]
                remaining_quantity -= available_volume
            
            if execution_quantity - remaining_quantity > 0:
                avg_price = total_value / (execution_quantity - remaining_quantity)
                executed_quantity = execution_quantity - remaining_quantity  # 实际执行数量
                
                # 更新状态
                direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
                self.cash += executed_quantity * avg_price * direction_sign
                self.cost += executed_quantity * avg_price * direction_sign
                self.quantity -= executed_quantity
                
                # 更新指标和记录交易
                self.active_trades += executed_quantity
                
                # 记录执行价格和数量
                self.execution_prices.append(avg_price)
                self.execution_quantities.append(executed_quantity)
                
                # 记录交易 - 修复：使用实际执行数量而不是计划数量
                self.trade_history.append({
                    'price': avg_price,
                    'quantity': executed_quantity,  # 使用实际执行数量
                    'timestamp': self.data.current_index,
                    'side': self.config.simulation_direction,
                    'type': 'market'
                })
    
    def _process_orders(self):
        """
        处理挂单成交(基于实际订单簿数据) - 修复库存重复减少问题
        """
        new_active_orders = []
        for order in self.active_orders:
            order_executed = False
            executed_quantity = 0
            
            # 获取当前时刻的订单簿数据
            bid_prices = [self.data.obtain_level('bidPrice', i) for i in range(1, self.order_book_depth + 1)]
            ask_prices = [self.data.obtain_level('askPrice', i) for i in range(1, self.order_book_depth + 1)]
            bid_volumes = [self.data.obtain_level('bidVolume', i) for i in range(1, self.order_book_depth + 1)]
            ask_volumes = [self.data.obtain_level('askVolume', i) for i in range(1, self.order_book_depth + 1)]
            
            # 根据订单方向检查是否可以成交
            if self.config.simulation_direction == 'sell':
                # 卖单：检查订单价格是否优于或等于买方报价
                for i in range(self.order_book_depth):
                    if bid_prices[i] > 0 and bid_volumes[i] > 0 and order['price'] <= bid_prices[i]:
                        exec_vol = min(order['quantity'] - executed_quantity, bid_volumes[i])
                        if exec_vol > 0:
                            executed_quantity += exec_vol
                            direction_sign = 1  # 卖单
                            self.cash += exec_vol * order['price'] * direction_sign
                            self.cost += exec_vol * order['price'] * direction_sign
                            # 修复：只减少挂单数量，不重复减少总库存
                            self.pending_quantity -= exec_vol
                            self.passive_trades += exec_vol
                            self.passive_trades_step += exec_vol

                            # 记录执行
                            self.execution_prices.append(order['price'])
                            self.execution_quantities.append(exec_vol)
                            self.trade_history.append({
                                'price': order['price'],
                                'quantity': exec_vol,
                                'timestamp': self.data.current_index,
                                'side': order['side'],
                                'type': 'limit'
                            })
                            
                            if executed_quantity >= order['quantity']:
                                order_executed = True
                                break
            else:
                # 买单：检查订单价格是否优于或等于卖方报价
                for i in range(self.order_book_depth):
                    if ask_prices[i] > 0 and ask_volumes[i] > 0 and order['price'] >= ask_prices[i]:
                        exec_vol = min(order['quantity'] - executed_quantity, ask_volumes[i])
                        if exec_vol > 0:
                            executed_quantity += exec_vol
                            direction_sign = -1  # 买单
                            self.cash += exec_vol * order['price'] * direction_sign
                            self.cost += exec_vol * order['price'] * direction_sign
                            # 修复：只减少挂单数量，不重复减少总库存
                            self.pending_quantity -= exec_vol
                            self.passive_trades += exec_vol
                            self.passive_trades_step += exec_vol

                            # 记录执行
                            self.execution_prices.append(order['price'])
                            self.execution_quantities.append(exec_vol)
                            self.trade_history.append({
                                'price': order['price'],
                                'quantity': exec_vol,
                                'timestamp': self.data.current_index,
                                'side': order['side'],
                                'type': 'limit'
                            })

                            if executed_quantity >= order['quantity']:
                                order_executed = True
                                break
            
            # 订单状态处理保持不变
            if order_executed:
                order['status'] = 'executed'
                self.order_history.append(order)
            elif executed_quantity > 0:
                order['quantity'] -= executed_quantity
                new_active_orders.append(order)
                executed_order = order.copy()
                executed_order['quantity'] = executed_quantity
                executed_order['status'] = 'partially_executed'
                self.order_history.append(executed_order)
            else:
                new_active_orders.append(order)
        
        self.active_orders = new_active_orders
        
    def _force_liquidation(self):
        """
        强制清算
        """
        # 撤销所有挂单 - 先恢复数量
        canceled_count = len(self.active_orders)
        # 记录清算前剩余量，供 _calculate_reward 末期惩罚使用
        self._terminal_forced_qty = self.quantity
        for order in self.active_orders:
            order['status'] = 'canceled'
            self.order_history.append(order)
            self.canceled_orders += 1  # 增加撤单计数
            # 修复：撤单时恢复数量
            self.quantity += order['quantity']
            self.pending_quantity -= order['quantity']
        
        self.active_orders = []
        
        # 剩余订单市价成交
        if self.quantity > 0:
            if self.config.simulation_direction == 'sell':
                price = self.data.obtain_level('bidPrice', self.order_book_depth) * (1 - 50 / 10000)
            else:
                price = self.data.obtain_level('askPrice', self.order_book_depth) * (1 + 50 / 10000)
            if price <= 0:
                price = self.data.obtain_level('close_price')
            direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
            self.cash += self.quantity * price * direction_sign
            self.cost += self.quantity * price * direction_sign
            
            # 记录执行价格和数量
            self.execution_prices.append(price)
            self.execution_quantities.append(self.quantity)
            
            # 更新主动成交量指标（强制清算视为主动成交）
            self.active_trades += self.quantity
            
            # 计算相对市场VWAP的价格表现
            try:
                market_vwap = self.data.obtain_level('vwap') if 'vwap' in self.data.data.columns else price
                if market_vwap > 0:
                    if self.config.simulation_direction == 'sell':
                        price_advantage_bp = (price - market_vwap) / market_vwap * 10000
                    else:
                        price_advantage_bp = (market_vwap - price) / market_vwap * 10000
                    # price_advantage_bp 已经是"正值=优于VWAP"的约定（与
                    # get_trading_metrics() 一致），这里必须用 += 累加，用 -=
                    # 会让 self.price_performance_bp 变成符号相反的另一套约定。
                    self.price_performance_bp += price_advantage_bp * self.quantity
            except Exception as e:
                print(f"计算强制清算价格表现时出错: {e}")
            
            # 记录交易
            self.trade_history.append({
                'price': price,
                'quantity': self.quantity,
                'timestamp': self.data.current_index,
                'side': self.config.simulation_direction,
                'type': 'market',
                'status': 'liquidation'
            })
            self.quantity = 0
        
    def _calculate_reward(self):
        """
        Reward 设计（与论文评估指标对齐）

        r_t = α·price_bp_t  −  β·cancel_pen_t  −  γ·progress_pen_t
        r_T += δ·forced_pen    （仅终止步）

        各项含义
        ─────────────────────────────────────────────────────────────
        price_bp_t   : 本步成交均价 vs 当前市场 VWAP（bp），与
                       episode 结束时的 price_performance_bp 对齐
        cancel_pen_t : 本步撤单数 × config.simulation_cancel_coeff
        progress_pen : 累计执行量落后均匀计划的比例 × 100（bp 量级）
        forced_pen   : 末期剩余仓位被强制清算（50bp 折价惩罚）
        """
        α, β, γ, δ = 1.0, 0.5, 0.3, 2.0

        # ── 1. 价格质量（本步，bp）────────────────────────────────────
        exec_qty, exec_val = self._get_current_step_execution()
        current_vwap = self._get_market_vwap()
        if current_vwap > 0 and exec_qty > 0:
            exec_price = exec_val / exec_qty
            if self.config.simulation_direction == 'sell':
                price_bp = (exec_price - current_vwap) / current_vwap * 10000
            else:
                price_bp = (current_vwap - exec_price) / current_vwap * 10000
        else:
            price_bp = 0.0

        # ── 2. 撤单惩罚 ──────────────────────────────────────────────
        cancel_pen = self.canceled_orders_step * self.config.simulation_cancel_coeff

        # ── 3. 进度惩罚（落后均匀计划则惩罚，超前不额外奖励）─────────
        horizon = max(1, self.config.simulation_planning_horizon)
        time_progress = (self.data.current_index - self.data.start_index) / horizon
        ideal_qty = self.total_quantity * time_progress
        actual_qty = self.total_quantity - self.quantity
        behind_frac = max(0.0, ideal_qty - actual_qty) / self.total_quantity
        progress_pen = behind_frac * 100  # bp 量级

        # ── 3b. 被动成交奖励（鼓励 agent 使用限价单节省成本）────────────
        passive_bonus = (self.passive_trades_step / max(1, self.total_quantity)) * 5.0

        r = α * price_bp - β * cancel_pen - γ * progress_pen + passive_bonus

        # ── 4. 末期强制清算惩罚 ──────────────────────────────────────
        if self._terminal_forced_qty > 0:
            forced_frac = self._terminal_forced_qty / self.total_quantity
            r -= δ * 50 * forced_frac   # 50bp × 剩余比例
            self._terminal_forced_qty = 0

        return r




    def _get_current_step_execution(self):
        """获取当前时间步的执行数据"""
        executed_quantity = 0
        executed_value = 0
        
        for trade in self.trade_history:
            if trade.get('timestamp', -1) == self.data.current_index - 1:
                executed_quantity += trade['quantity']
                direction_sign = 1 if self.config.simulation_direction == 'sell' else -1
                executed_value += trade['quantity'] * trade['price'] * direction_sign
        
        return executed_quantity, executed_value

    @property
    def market_state_dim(self):
        if not hasattr(self, '_market_state_dim'):
            # 动态计算: 2个价格 + 2*(K-1)个量 + 3个流数据
            k_minus_one = self.order_book_depth - 1
            single_market_state_dim = 2 + (2 * k_minus_one) + 3
            lookback_horizon = getattr(self.config, 'simulation_lookback_horizon', 1)
            self._market_state_dim = single_market_state_dim * lookback_horizon
        return self._market_state_dim

    @property
    def private_state_dim(self):
        if not hasattr(self, '_private_state_dim'):
            # 动态计算: 3个基础状态 + N个订单 * 2个维度
            self._private_state_dim = 3 + (self.max_tracked_orders * 2)
        return self._private_state_dim

def make_simplified_env(config, eval_mode=None):
    """
    创建简化环境
    """
    return SimplifiedExecutionEnv(config, eval_mode=eval_mode)  # 正确传递eval_mode参数

def test_simplified_env():
    """
    测试简化环境 - 完整运行300步
    """
    print("开始测试简化版交易执行环境...")
    
    # 创建配置
    config = SimplifiedEnvConfig()
    print("1. 配置创建成功")
    print(f"   回看窗口大小: {config.simulation_lookback_horizon}")
    print(f"   交易方向: {config.simulation_direction}")
    print(f"   总交易时长: {config.simulation_planning_horizon}秒")
    
    # 创建环境
    env = make_simplified_env(config)
    print("2. 环境创建成功")
    
    # 重置环境
    market_state, private_state = env.reset()
    print(f"3. 环境重置成功")
    print(f"   初始剩余数量: {env.quantity}")
    print(f"   总数量: {env.total_quantity}")
    print(f"   初始现金: {env.cash:.2f}")
    
    # 获取初始市场信息
    try:
        initial_vwap = env.data.obtain_level('vwap') if 'vwap' in env.data.data.columns else env.data.obtain_level('close_price')
        initial_vwap_5min = env.data.obtain_level('vwap_5min') if 'vwap_5min' in env.data.data.columns else initial_vwap
        print(f"   初始市场VWAP: {initial_vwap:.4f}")
        print(f"   初始5分钟VWAP: {initial_vwap_5min:.4f}")
    except Exception as e:
        print(f"   初始市场VWAP获取失败: {e}")
    
    # 运行完整300步
    print(f"4. 开始执行完整{config.simulation_planning_horizon}步测试:")
    total_reward = 0
    step_count = 0
    done = False
    
    # 记录关键指标
    execution_prices = []
    execution_quantities = []
    market_vwap_history = []
    
    while not done and step_count < config.simulation_planning_horizon + 10:  # 额外10步缓冲
        # 生成连续动作 [挂单位置, 执行强度]
        action = np.random.uniform(0, 1, size=(2,))
        
        # 执行步骤
        market_state, private_state, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # 记录执行数据
        if hasattr(env, 'execution_prices') and env.execution_prices:
            execution_prices.extend(env.execution_prices[len(execution_prices):])
            execution_quantities.extend(env.execution_quantities[len(execution_quantities):])
        
        # 记录市场VWAP
        try:
            current_vwap = env.data.obtain_level('vwap') if 'vwap' in env.data.data.columns else env.data.obtain_level('close_price')
            market_vwap_history.append(current_vwap)
        except:
            market_vwap_history.append(0)
        
        # 每50步输出一次进度
        if step_count % 50 == 0 or done:
            print(f"   步骤 {step_count}:")
            print(f"   - 剩余数量: {info['quantity']}")
            print(f"   - 挂单数量: {info['pending_quantity']}")
            print(f"   - 当前现金: {info['cash']:.2f}")
            
            # 输出市场VWAP
            if 'market_vwap' in info:
                print(f"   - 当前市场VWAP: {info['market_vwap']:.4f}")
            else:
                try:
                    current_vwap = env.data.obtain_level('vwap') if 'vwap' in env.data.data.columns else env.data.obtain_level('close_price')
                    print(f"   - 当前市场VWAP: {current_vwap:.4f}")
                except:
                    print(f"   - 当前市场VWAP: 无法获取")
            
            # 输出活跃订单信息
            if env.active_orders:
                print(f"   - 活跃订单数: {len(env.active_orders)}")
            else:
                print(f"   - 活跃订单数: 0")
            
            print(f"   - 累计奖励: {total_reward:.6f}")
            print(f"   - 是否完成: {done}")
            print()
    
    print(f"5. 测试完成，共执行 {step_count} 步")
    
     # 计算完整的VWAP对比
    print(f"6. VWAP对比分析:")
    # 计算策略VWAP
    if execution_prices and execution_quantities and sum(execution_quantities) > 0:
        strategy_vwap = sum(p * q for p, q in zip(execution_prices, execution_quantities)) / sum(execution_quantities)
        print(f"   - 策略执行均价: {strategy_vwap:.4f}")
    else:
        strategy_vwap = 0
        print(f"   - 策略执行均价: 无成交")

    # === 关键修正：使用真实累计VWAP ===
    try:
        start_idx = env.data.start_index
        end_idx = min(env.data.end_index, len(env.data.data) - 1)
        total_value = env.data.data.loc[start_idx:end_idx, 'value_dt'].sum()
        total_volume = env.data.data.loc[start_idx:end_idx, 'volume_dt'].sum()

        if total_volume > 0:
            market_vwap_avg = total_value / total_volume
            vwap_type = "真实累计VWAP"
        else:
            market_vwap_avg = env.data.data.loc[start_idx:end_idx, 'close_price'].mean()
            vwap_type = "收盘价均值"

        print(f"   - 市场VWAP({vwap_type}): {market_vwap_avg:.4f}")

    except Exception as e:
        print(f"   - 市场VWAP计算失败: {e}")
        market_vwap_avg = 0
        vwap_type = "无法计算"

    # 计算VWAP对比表现（正值=优于市场VWAP，与 get_trading_metrics() 约定一致，
    # 之前这里多写了一个取负号，导致这个诊断脚本自成一套相反的符号约定）
    if strategy_vwap > 0 and market_vwap_avg > 0:
        if config.simulation_direction == 'sell':
            vwap_comparison_bp = (strategy_vwap - market_vwap_avg) / market_vwap_avg * 10000
        else:
            vwap_comparison_bp = (market_vwap_avg - strategy_vwap) / market_vwap_avg * 10000
        print(f"   - VWAP对比({vwap_type}): {vwap_comparison_bp:.2f} bp")
        if vwap_comparison_bp > 0:
            print(f"   - 表现: 优于市场 ({vwap_comparison_bp:.2f} bp)")
        else:
            print(f"   - 表现: 差于市场 ({vwap_comparison_bp:.2f} bp)")
    else:
        print(f"   - VWAP对比: 无法计算")

    
    # 输出最终交易指标
    print(f"7. 最终交易指标:")
    metrics = env.get_trading_metrics()
    print(f"   - 总订单数: {metrics['total_orders']}")
    print(f"   - 撤单数: {metrics['canceled_orders']}")
    print(f"   - 撤单率: {metrics['cancel_rate']:.2%}")
    print(f"   - 主动成交: {metrics['active_trades']}")
    print(f"   - 被动成交: {metrics['passive_trades']}")
    print(f"   - 总成交: {metrics['total_executed']}")
    print(f"   - 主动成交率: {metrics['active_rate']:.2%}")
    print(f"   - 价格表现(bp): {metrics['price_performance_bp']:.2f}")
    
    # 输出最终财务信息
    print(f"8. 最终财务信息:")
    print(f"   - 最终现金: {env.cash:.2f}")
    print(f"   - 最终成本: {env.cost:.2f}")
    print(f"   - 最终剩余数量: {env.quantity}")
    print(f"   - 最终挂单数量: {env.pending_quantity}")
    print(f"   - 完成率: {(env.total_quantity - env.quantity) / env.total_quantity * 100:.2f}%")
    
    # 计算执行统计
    if execution_prices and execution_quantities:
        total_executed = sum(execution_quantities)
        print(f"   - 总执行数量: {total_executed}")
        print(f"   - 执行价格区间: {min(execution_prices):.4f} - {max(execution_prices):.4f}")
    
    print("完整测试完成!")

if __name__ == "__main__":
    test_simplified_env()
