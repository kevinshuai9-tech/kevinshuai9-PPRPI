import sys as _sys, os as _os
_r = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('config', 'data', 'env', 'agents', 'baselines', 'survival'):
    _p = _os.path.join(_r, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
del _sys, _os, _r, _d, _p

"""
使用order与trade数据生成沪市特定时间指定股票的逐笔委托数据 以及逐笔订单簿快照数据   
"""
import os
import pickle
import pandas as pd
import bisect
import sys
sys.path.append('.')
from clickhouse_driver import Client

from const_path_ob import *  # 记录了 path_orders path_snapshots path_pkl_data
from const_base import *     # 记录了 CODE_LIST VALIDATION_CODE_LIST TRAIN_DATE_LIST VALIDATION_DATE_LIST
CODES = CODE_LIST + VALIDATION_CODE_LIST
DATES = TRAIN_DATE_LIST + VALIDATION_DATE_LIST

class Data_loader(object):
    '''
    数据加载器,通过get_data(code:str, date: str 'yyyy-mm-dd')函数获取 
    逐笔成交数据、逐笔委托数据和MD三秒快照
    '''
    def _get_cols(self, client, db, tb):
        sql = 'DESCRIBE TABLE %s.`%s`'%(db, tb)
        res = client.execute(sql)
        cols = [r[0] for r in res]
        return cols

    def _get_one_stk(self, client, db, tb, stk, cols):
        str_cols = ",".join(cols)
        try:
            command = """SELECT %s\
            FROM  %s.`%s`\
            WHERE SecurityID='%s'"""%(str_cols, db, tb, stk)
            res = client.execute(command)
            df = pd.DataFrame(res)
            df.columns = cols
            return df
        except Exception as e:
            print(db, 'Missed', stk)
            return 1

    def get_data(self, code, date):
        """
        统一接口 获取指定股票在指定日期的逐笔成交、逐笔委托和MD快照数据
        根据股票代码自动调用对应的沪市或深市数据获取函数
        """
        # 根据股票代码前缀确定市场类型
        if code.startswith(('60', '688', '900')):
            # 沪市股票
            return self.get_data_sh(code, date)
        elif code.startswith(('000', '002', '003', '300', '200')):
            # 深市股票
            return self.get_data_sz(code, date)
        else:
            raise ValueError(f"不支持的股票代码: {code}")


    def get_data_sh(self, code, date):
        '''
        获取指定沪市股票、指定日期的逐笔成交数据、逐笔委托数据和MD三秒快照
        '''
        client = Client(host='100.103.0.4', port='9000', database='TLMDSH', user='YQLi',
                password='98JD#yq.13@')
        
        db = 'TLTradeSH'
        tb = date.replace('-','')
        cols = self._get_cols(client=client, db=db, tb=tb)

        stk = code
        df1 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df1['TradTime'] = date + ' ' + df1['TradTime']
        df1['TradTime'] = pd.to_datetime(df1['TradTime'], format='%Y-%m-%d %H:%M:%S.%f')
        df1.sort_values('BizIndex', inplace=True)

        db = 'TLOrderSH'
        cols = self._get_cols(client=client, db=db, tb=tb)
        df2 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df2['OrderTime'] = date + ' ' + df2['OrderTime']
        df2['OrderTime'] = pd.to_datetime(df2['OrderTime'], format='%Y-%m-%d %H:%M:%S.%f')
        df2.sort_values('BizIndex', inplace=True)

        db = 'TLMDSH'
        cols = self._get_cols(client=client, db=db, tb=tb)
        df3 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df3.sort_values('UpdateTime', inplace=True)
        return df1, df2, df3
    

    def get_data_sz(self, code, date):
        '''
        获取指定深市股票、指定日期的逐笔成交数据、逐笔委托数据和MD三秒快照
        '''
        client = Client(host='100.103.0.4', port='9000', database='TLMDSZ', user='YQLi',
                password='98JD#yq.13@')
        
        db = 'TLTradeSZ'
        tb = date.replace('-','')
        cols = self._get_cols(client=client, db=db, tb=tb)

        stk = code
        df1 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df1['TransactTime'] = date + ' ' + df1['TransactTime']
        df1['TransactTime'] = pd.to_datetime(df1['TransactTime'], format='%Y-%m-%d %H:%M:%S.%f')
        df1.sort_values('SeqNo', inplace=True)

        db = 'TLOrderSZ'
        cols = self._get_cols(client=client, db=db, tb=tb)
        df2 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df2['TransactTime'] = date + ' ' + df2['TransactTime']
        df2['TransactTime'] = pd.to_datetime(df2['TransactTime'], format='%Y-%m-%d %H:%M:%S.%f')
        df2.sort_values('SeqNo', inplace=True)

        db = 'TLMDSZ'
        cols = self._get_cols(client=client, db=db, tb=tb)
        df3 = self._get_one_stk(client=client, db=db, tb=tb, stk=stk, cols=cols)
        df3.sort_values('UpdateTime', inplace=True)
        return df1, df2, df3    

class Order():
    '''
    构建沪市某股票某日期的完整委托单

    参数：
    code: 股票代码  
    date: 日期  
    '''
    def __init__(self, code, date):
        self.data_loader = Data_loader()
        self.SecurityID = code
        self.date = date
        self.trade_merge_col = ['TradTime','TradPrice','TradVolume','TradeBuyNo','TradeSellNo','TradeBSFlag','BizIndex']
        self.order_merge_col = ['OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex']
    
    def _trade_to_order(self, Trades, Orders):
        """
        从交易数据还原对应原本委托信息  
        参数：  
        Orders: pd.DataFrame  要求Order为剔除撤单信息后的剩余委托信息, 且OrderNO确保唯一。 包括字段'OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex'  
        Trades: pd.DataFrame  包括字段'TradTime','TradPrice','TradVolume','TradeBuyNo','TradeSellNo','TradeBSFlag','BizIndex'  
        """
        # 构建整合数据字典
        Trade_merge_dic = {key: [] for key in self.order_merge_col}

        # 预生成订单号到BizIndex的映射字典（假设Order中的OrderNO唯一）
        assert Orders['OrderNO'].is_unique
        order_no_to_bizindex = pd.Series(
            Orders['BizIndex'].values, 
            index=Orders['OrderNO']
        ).to_dict()
        
        Trades = Trades.to_dict(orient='records')

        for row in Trades:
            if (row['TradTime'] < pd.to_datetime(self.date + ' 09:30:00')) or (row['TradTime'] > pd.to_datetime(self.date + ' 14:57:00')):
                # 仅考虑连续竞价期间的交易复原，集合竞价期间无需复原，在后续订单簿构建时才进行处理
                # 我们的Trade数据集合竞价成交时间分别是9：25：00和15：00：01
                continue

            # 只要是在该笔交易发生之前，逐笔委托中出现过的订单，就无需再复原
            # 检查买方订单是否存在于之前的委托中并生成复原信号
            buy_biz = order_no_to_bizindex.get(row['TradeBuyNo'], None)
            buy_in_order = (buy_biz is not None) and (buy_biz < row['BizIndex'])

            # 检查卖方订单是否存在于之前的委托中并生成复原信号
            sell_biz = order_no_to_bizindex.get(row['TradeSellNo'], None)
            sell_in_order = (sell_biz is not None) and (sell_biz < row['BizIndex'])

            # 判断需要补充的订单情况
            if buy_in_order ^ sell_in_order:  # XOR 异或操作
                order_no = row['TradeSellNo'] if buy_in_order else row['TradeBuyNo']
                bs_flag = 'S' if buy_in_order else 'B'
                Trade_merge_dic['OrderNO'].append(order_no)
                Trade_merge_dic['OrderBSFlag'].append(bs_flag)
                Trade_merge_dic['OrderTime'].append(row['TradTime'])
                Trade_merge_dic['OrderType'].append('A')
                Trade_merge_dic['OrderPrice'].append(row['TradPrice'])
                Trade_merge_dic['Balance'].append(row['TradVolume'])
                Trade_merge_dic['BizIndex'].append(row['BizIndex'])
                
            elif not (buy_in_order or sell_in_order):  # 该笔交易双方的订单均未在报单信息中显示
                # 确定订单顺序
                if row['TradeBSFlag'] == 'B': # 此时主动买，以卖一价成交;说明挂单时先有卖单，后有买单
                    shunxu = [(row['TradeSellNo'], 'S'), (row['TradeBuyNo'], 'B')]
                elif row['TradeBSFlag'] == 'S': # 此时主动卖，以买一价成交;说明挂单时先有买单，后有卖单
                    shunxu = [(row['TradeBuyNo'], 'B'), (row['TradeSellNo'], 'S')]
                else: 
                    # 此时连续竞价期间的委托不知道成交方向，需单独处理(一般不会出现该情况)
                    # print(f'BizIndex为{row['BizIndex']}的交易不知道成交方向，需单独处理,故先不添加到Order中')
                    break
                
                # 批量添加订单
                for order_no, bs_flag in shunxu:
                    Trade_merge_dic['OrderNO'].append(order_no)
                    Trade_merge_dic['OrderBSFlag'].append(bs_flag)
                    Trade_merge_dic['OrderTime'].append(row['TradTime'])
                    Trade_merge_dic['OrderType'].append('A')
                    Trade_merge_dic['OrderPrice'].append(row['TradPrice'])
                    Trade_merge_dic['Balance'].append(row['TradVolume'])
                    Trade_merge_dic['BizIndex'].append(row['BizIndex'])
        Trade_merge = pd.DataFrame(data=Trade_merge_dic)
        return Trade_merge
    
    def _recover_full_order(self, Orders, Trade_order, Orders_D):
        """
        拼接构建沪市委托单  
        参数：  
        Orders: pd.DataFrame  所有剩余委托中的报单。包括字段'OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex'  
        Trade_order: pd.DataFrame  由交易拆分出的(未经合并的)委托单。包括字段'OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex'  
        Orders_D: pd.DataFrame  所有(剩余)委托中的撤单。包括字段'OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex'  
        """
        # 拼接成交委托和剩余报单委托
        Orders = pd.concat([Orders, Trade_order], axis=0, ignore_index=True)
        Orders.sort_values(by='BizIndex' ,inplace=True)
        Orders.reset_index(inplace=True, drop=True)
        # 根据OrderNO整合委托
        Orders = Orders.groupby('OrderNO').agg(
            OrderTime=('OrderTime', 'first'),
            BizIndex=('BizIndex', 'first'),
            OrderType=('OrderType', 'first'),
            OrderBSFlag=('OrderBSFlag', 'first'),
            Balance=('Balance', 'sum'),
            OrderPrice_max=('OrderPrice', 'max'),
            OrderPrice_min=('OrderPrice', 'min')
        ).reset_index()

        # 根据OrderBSFlag选择对应的价格
        Orders['OrderPrice'] = Orders.apply(
            lambda row: row['OrderPrice_max'] if row['OrderBSFlag'] == 'B' else row['OrderPrice_min'],
            axis=1
        )

        # 删除临时列
        Orders = Orders.drop(['OrderPrice_max', 'OrderPrice_min'], axis=1)
        assert Orders['OrderNO'].is_unique
        # 拼接撤单委托
        Orders = pd.concat([Orders, Orders_D], axis=0, ignore_index=True)
        Orders.sort_values(by='BizIndex' ,inplace=True)
        Orders.reset_index(inplace=True, drop=True)
        return Orders
    
    def construct_order_SH(self):
        Trades, Orders, MD = self.data_loader.get_data_sh(self.SecurityID, self.date)
        precloseprc = MD['PreCloPrice'].values[0]
        # 数据选取
        Orders = Orders[self.order_merge_col]
        Trades = Trades[self.trade_merge_col]

        Orders = Orders.sort_values('BizIndex')
        Orders.reset_index(drop=True, inplace=True)
        Trades = Trades.sort_values('BizIndex')
        Trades.reset_index(drop=True, inplace=True)

        Orders['OrderTime'] = pd.to_datetime(Orders['OrderTime'])
        Trades['TradTime'] = pd.to_datetime(Trades['TradTime'])
        
        # 拆分报撤单信息
        Orders_D = Orders[Orders['OrderType'] == 'D']
        Orders = Orders[Orders['OrderType'] == 'A']
        assert Orders['OrderNO'].is_unique

        Trade_order = self._trade_to_order(Trades, Orders)
        Orders = self._recover_full_order(Orders, Trade_order, Orders_D)
        Orders.insert(0,'SecurityID', self.SecurityID)
        Orders['PreCloPrice'] = precloseprc

        save_dir = os.path.join(path_orders, f'SH{self.SecurityID}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, f'{self.date}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(Orders, f, pickle.HIGHEST_PROTOCOL)
        print(f'股票{self.SecurityID}日期{self.date}的委托信息已保存至{file_path}')

class OrderBook:
    def __init__(self, code, date, level=10):
        self.date = date
        self.SecurityID = code
        self.orders = Order(code, date)
        self.level = level

        # 实时状态信息记录
        self.time = None # current timestamp
        self.status = 'START' # current trading stage
        self.pre_status = None # previous trading stage
        self.current_price = None # current execution price

        # 实时订单簿信息记录
        self.order_map = {} # order_no -> order details
        self.bid_book = {} # price -> {order_no: order}
        self.ask_book = {} # price -> {order_no: order}
        self.sorted_bid_prices = [] # descending order
        self.sorted_ask_prices = [] # ascending order

        # 实时交易信息记录
        self.trades = [] # transaction record
        self.highprice = 0
        self.lowprice = 1e9
        self.tradvolume = 0
        self.turnover = 0

        # 逐笔订单簿信息记录
        self.snapshots = []
    
    def get_orders(self):
        file_path = os.path.join(path_orders, f'SH{self.SecurityID}', f'{self.date}.csv')
        if not os.path.exists(file_path):
            self.orders.construct_order_SH()
        
    def process_order(self, order):
        '''
        处理订单   
        参数：  
        order: 字典,要求键包括字段'SecurityID','OrderTime','OrderType','OrderPrice','Balance','OrderNO','OrderBSFlag','BizIndex'  
        '''
        self.time = order['OrderTime']
        can_cancel = self._set_status(order['OrderTime']) # 设置该订单所处时间状态以及判断此时是否能撤单
        if self.status == 'OCALL':
            self._process_call_auction(order, can_cancel)
        elif self.status == 'TRADE':
            if self.pre_status == 'OCALL':
                self._cal_call_auction_snapshot()
                self._get_call_auc_transaction()
                self.get_tick_snapshot(time = pd.to_datetime(f'{self.date} 09:30:00.0'))
            self._cont_auc_transaction(order, can_cancel)
            self.get_tick_snapshot()
        elif self.status == 'CCALL':
            self._process_call_auction(order, can_cancel)
        self.pre_status = self.status
        
    def _set_status(self, order_time):
        '''
        设置新委托所处的时间状态,返回值表示能否撤单   
        START:盘前  
        OCALL:开盘竞价  
        TRADE:连续竞价  
        CCALL:收盘竞价  
        ENDTR:盘后  
        '''
        order_time = pd.to_datetime(order_time)
        if order_time < pd.to_datetime(f'{self.date} 09:15:00', format='%Y-%m-%d %H:%M:%S'):
            self.status = 'START'
        elif pd.to_datetime(f'{self.date} 09:15:00', format='%Y-%m-%d %H:%M:%S') <= order_time < pd.to_datetime(f'{self.date} 09:25:00', format='%Y-%m-%d %H:%M:%S'):
           self.status = 'OCALL' # 允许撤单
           return True
        elif pd.to_datetime(f'{self.date} 09:25:00', format='%Y-%m-%d %H:%M:%S') <= order_time < pd.to_datetime(f'{self.date} 09:30:00', format='%Y-%m-%d %H:%M:%S'):
           self.status = 'OCALL' # 不允许撤单
        elif pd.to_datetime(f'{self.date} 09:30:00', format='%Y-%m-%d %H:%M:%S') <= order_time < pd.to_datetime(f'{self.date} 14:57:00', format='%Y-%m-%d %H:%M:%S'):
            self.status = 'TRADE'
            return True
        elif pd.to_datetime(f'{self.date} 14:57:00', format='%Y-%m-%d %H:%M:%S') <= order_time < pd.to_datetime(f'{self.date} 15:00:00', format='%Y-%m-%d %H:%M:%S'):
            self.status = 'CCALL'
        else:
            self.status = 'ENDTR'
        return False

    def _add_order(self, order):
        ''' 
        将无法成交的或者剩余的委托插入到订单簿中
        '''
        order_no = order['OrderNO']
        price = order['OrderPrice']
        balance = order['Balance']
        direction = order['OrderBSFlag']
        order_time = order['OrderTime']

        if order_no in self.order_map:
            return # 订单已存在，不处理

        order_details = {
        'order_no': order_no,
        'price': price,
        'balance': balance,
        'direction': direction,
        'time': order_time
        }
        self.order_map[order_no] = order_details

        if direction == 'B':
            book = self.bid_book
            sorted_prices = self.sorted_bid_prices
        else:
            book = self.ask_book
            sorted_prices = self.sorted_ask_prices

        if price not in book:
            book[price] = {}
            # 插入到sorted_prices的正确位置
            if direction == 'B':
                temp = [-p for p in sorted_prices]
                idx = bisect.bisect_left(temp, -price)
                sorted_prices.insert(idx, price)
            else:
                idx = bisect.bisect_left(sorted_prices, price)
                sorted_prices.insert(idx, price)
        book[price][order_no] = order_details # 注意！这里其实默认了在按照price插入订单时的{OrderNO -> order details}的字典是有顺序的，要求Python版本在3.7以上

    def _cancel_order(self, order, can_cancel):
        ''' 
        处理撤单
        '''
        if not can_cancel:
            return # 当前不在撤单时间内

        order_no = order['OrderNO']
        cancel_balance = order['Balance']

        if order_no not in self.order_map:
            return # 订单不存在或已被删除
        order_details = self.order_map[order_no]
        direction = order_details['direction']
        price = order_details['price']

        if direction == 'B':
            book = self.bid_book
            sorted_prices = self.sorted_bid_prices
        else:
            book = self.ask_book
            sorted_prices = self.sorted_ask_prices

        if price not in book:
            # 有这笔委托但订单簿中却没有这档价格
            del self.order_map[order_no]
            return
        
        price_orders = book[price]
        if order_no not in price_orders:
            # 有这笔委托的价格但订单簿中该价格下却没有这笔委托
            del self.order_map[order_no]
            return

        current_balance = order_details['balance']
        new_balance = current_balance - cancel_balance
        if new_balance <= 0:
            del price_orders[order_no] # 从订单簿中删去
            del self.order_map[order_no] # 从委托单中删去
            if not price_orders:
                del book[price] # 若该价格下没有其他委托，则删去这档价格
                # 从sorted_prices中移除
                try:
                    idx = sorted_prices.index(price)
                    sorted_prices.pop(idx)
                except ValueError:
                    pass
        else:
            order_details['balance'] = new_balance
            book[price][order_no] = order_details

    def _transaction(self, time, price, volume, buy_no, sell_no, bsflag):
        '''执行交易'''
        transaction = {
            'TradTime': time,
            'TradPrice': price,
            'TradVolume': volume,
            'TradeBuyNo': buy_no,
            'TradeSellNo': sell_no,
            'TradeBSFlag': bsflag
        }
        self.trades.append(transaction) # 补充交易数据
        self.current_price = price # 更新最新执行价格
        self.tradvolume += volume # 更新总成交量
        self.turnover += price * volume # 更新总成交额
        self.highprice = max(self.highprice, price) # 更新最高价
        self.lowprice = min(self.lowprice, price) # 更新最低价

    def _process_call_auction(self, new_order, can_cancel):
        '''
        整合集合竞价订单到订单簿中
        '''
        if new_order['OrderType'] == 'D':
            self._cancel_order(new_order, can_cancel)
        elif new_order['OrderType'] == 'A':
            self._add_order(new_order)

    def _cal_call_auction_snapshot(self):
        """
        模拟开盘集合竞价,得出开盘撮合价    
        """
        # 分离买卖单并聚合数量
        buy_orders = {}
        sell_orders = {}

        for price in self.sorted_bid_prices:
            buy_orders[price] = sum(order_detail['balance'] for order_detail in self.bid_book[price].values())

        for price in self.sorted_ask_prices:
            sell_orders[price] = sum(order_detail['balance'] for order_detail in self.ask_book[price].values())

        # 生成候选价格（所有出现过的价格）
        candidate_prices = sorted({p for p in buy_orders.keys()} | {p for p in sell_orders.keys()}, reverse=True)
        
        # 寻找最优成交价
        max_volume = 0
        candidates = []
        
        for price in candidate_prices:
            # 计算可成交量
            buy_volume = sum(qty for p, qty in buy_orders.items() if p >= price)
            sell_volume = sum(qty for p, qty in sell_orders.items() if p <= price)
            volume = min(buy_volume, sell_volume)
            imbalance = abs(buy_volume - sell_volume)
            
            if volume > max_volume:
                max_volume = volume
                candidates = [(price, imbalance)]
            elif volume == max_volume:
                candidates.append((price, imbalance))

        # 处理无成交情况
        if max_volume == 0 or not candidates:
            print('开盘无法撮合成交')

        # 筛选候选价格
        min_imbalance = min(imbalance for _, imbalance in candidates)
        final_prices = [price for price, imbalance in candidates if imbalance == min_imbalance]
        
        # 确定最终成交价
        final_prices.sort()
        if len(final_prices) % 2 == 1:
            match_price = final_prices[len(final_prices)//2]
        else:
            match_price = (final_prices[len(final_prices)//2-1] + final_prices[len(final_prices)//2])/2

        if self.pre_status == 'OCALL':
            # 更新开盘价格
            self.openprice = match_price
            print(f'股票{self.SecurityID}日期{self.date}今开盘价为：{self.openprice}')
        else:
            # 更新收盘价格
            self.closeprice = match_price
            print(f'股票{self.SecurityID}日期{self.date}今收盘价为：{self.closeprice}')
        
    def _get_call_auc_transaction(self):
        ''' 
        执行集合竞价撮合后的集中交易
        '''
        if self.pre_status == 'OCALL':
            transaction_time = pd.to_datetime(f'{self.date} 09:25:00.0')
            call_price = self.openprice
        else:
            transaction_time = pd.to_datetime(f'{self.date} 15:00:00.0')
            call_price = self.closeprice

        best_ask_price = self.sorted_ask_prices[0] if self.sorted_ask_prices else None
        best_bid_price = self.sorted_bid_prices[0] if self.sorted_bid_prices else None

        while (best_ask_price is not None and
            best_bid_price is not None and
            best_bid_price >= call_price and
            best_ask_price <= call_price):

            # 获取当前价格层的订单列表（按时间排序）
            ask_orders = list(self.ask_book[best_ask_price].items())
            bid_orders = list(self.bid_book[best_bid_price].items())
            i, j = 0, 0 # 分别追踪当前处理到的卖单和买单

            ask_remaining = 0
            bid_remaining = 0
            while i < len(ask_orders) and j < len(bid_orders):
                if ask_remaining == 0:
                    ask_order_no, ask_order_details = ask_orders[i]
                    ask_remaining = ask_order_details['balance']
                if bid_remaining == 0:
                    bid_order_no, bid_order_details = bid_orders[j]
                    bid_remaining = bid_order_details['balance']

                # 跳过已删除的订单
                if (ask_order_no not in self.order_map) or (bid_order_no not in self.order_map):
                    if ask_order_no not in self.order_map:
                        i += 1
                    if bid_order_no not in self.order_map:
                        j += 1
                    continue

                # 计算撮合量
                trade_qty = min(ask_remaining, bid_remaining)

                # 执行交易
                self._transaction(
                    time=transaction_time,
                    price=call_price,
                    volume=trade_qty,
                    buy_no=bid_order_no,  # 根据数据结构调整
                    sell_no=ask_order_no,
                    bsflag='N'
                )

                # 更新订单余额
                ask_remaining -= trade_qty
                bid_remaining -= trade_qty

                # 处理成交的订单
                if ask_remaining == 0:
                    del self.order_map[ask_order_no]
                    del self.ask_book[best_ask_price][ask_order_no]
                    i += 1
                else:
                    ask_order_details['balance'] = ask_remaining
                    self.order_map[ask_order_no] = ask_order_details
                    self.ask_book[best_ask_price][ask_order_no] = ask_order_details

                if bid_remaining == 0:
                    del self.order_map[bid_order_no]
                    del self.bid_book[best_bid_price][bid_order_no]
                    j += 1
                else:
                    bid_order_details['balance'] = bid_remaining
                    self.order_map[bid_order_no] = bid_order_details
                    self.bid_book[best_bid_price][bid_order_no] = bid_order_details

            # 清理空价格层
            if best_ask_price in self.ask_book and not self.ask_book[best_ask_price]:
                del self.ask_book[best_ask_price]
                if best_ask_price in self.sorted_ask_prices:
                    self.sorted_ask_prices.remove(best_ask_price)
            if best_bid_price in self.bid_book and not self.bid_book[best_bid_price]:
                del self.bid_book[best_bid_price]
                if best_bid_price in self.sorted_bid_prices:
                    self.sorted_bid_prices.remove(best_bid_price)

            # 更新当前最优价格
            best_ask_price = self.sorted_ask_prices[0] if self.sorted_ask_prices else None
            best_bid_price = self.sorted_bid_prices[0] if self.sorted_bid_prices else None

    def _cont_auc_transaction(self, new_order, can_cancel):
        '''连续竞价期间交易执行'''
        if new_order['OrderType'] == 'D':
            self._cancel_order(new_order, can_cancel)
        elif new_order['OrderType'] == 'A':
            remaining = new_order['Balance']
            if new_order['OrderBSFlag'] == 'B':
                # 处理买单撮合
                best_ask_price = self.sorted_ask_prices[0] if self.sorted_ask_prices else None
                while (best_ask_price is not None and
                    best_ask_price <= new_order['OrderPrice'] and
                    remaining > 0):

                    # 获取当前价格层的订单列表（按时间排序）
                    ask_orders = list(self.ask_book[best_ask_price].items())
                    ask_remaining = 0
                    i = 0 # 分别追踪当前处理到的卖单和买单
                    while (i < len(ask_orders) and remaining > 0):
                        if ask_remaining == 0:
                            ask_order_no, ask_order_details = ask_orders[i]
                            ask_remaining = ask_order_details['balance']

                        # 跳过已删除的订单
                        if ask_order_no not in self.order_map:
                            i += 1
                            continue

                        # 计算撮合量
                        trade_qty = min(ask_remaining, remaining)

                        # 执行交易
                        self._transaction(
                            time=new_order['OrderTime'],
                            price=best_ask_price,
                            volume=trade_qty,
                            buy_no=new_order['OrderNO'],
                            sell_no=ask_order_no,
                            bsflag='B'
                        )

                        # 更新订单余额
                        ask_remaining -= trade_qty
                        remaining -= trade_qty

                        # 处理成交的订单
                        if ask_remaining == 0:
                            del self.order_map[ask_order_no]
                            del self.ask_book[best_ask_price][ask_order_no]
                            i += 1
                        else:
                            ask_order_details['balance'] = ask_remaining
                            self.order_map[ask_order_no] = ask_order_details
                            self.ask_book[best_ask_price][ask_order_no] = ask_order_details

                    
                    # 清理空价格层
                    if best_ask_price in self.ask_book and not self.ask_book[best_ask_price]:
                        del self.ask_book[best_ask_price]
                        if best_ask_price in self.sorted_ask_prices:
                            self.sorted_ask_prices.remove(best_ask_price)


                    # 更新当前最优价格
                    best_ask_price = self.sorted_ask_prices[0] if self.sorted_ask_prices else None

            elif new_order['OrderBSFlag'] == 'S':
                # 处理买单撮合
                best_bid_price = self.sorted_bid_prices[0] if self.sorted_bid_prices else None
                while (best_bid_price is not None and
                    best_bid_price >= new_order['OrderPrice'] and
                    remaining > 0):

                    # 获取当前价格层的订单列表（按时间排序）
                    bid_orders = list(self.bid_book[best_bid_price].items())
                    bid_remaining = 0
                    i = 0 # 分别追踪当前处理到的卖单和买单
                    while i < len(bid_orders) and remaining > 0:
                        if bid_remaining == 0:
                            bid_order_no, bid_order_details = bid_orders[i]
                            bid_remaining = bid_order_details['balance']

                        # 跳过已删除的订单
                        if bid_order_no not in self.order_map:
                            i += 1
                            continue

                        # 计算撮合量
                        trade_qty = min(bid_remaining, remaining)

                        # 执行交易
                        self._transaction(
                            time=new_order['OrderTime'],
                            price=best_bid_price,
                            volume=trade_qty,
                            buy_no=bid_order_no,
                            sell_no=new_order['OrderNO'],
                            bsflag='S'
                        )

                        # 更新订单余额
                        bid_remaining -= trade_qty
                        remaining -= trade_qty

                        # 处理成交的订单
                        if bid_remaining == 0:
                            del self.order_map[bid_order_no]
                            del self.bid_book[best_bid_price][bid_order_no]
                            i += 1
                        else:
                            bid_order_details['balance'] = bid_remaining
                            self.order_map[bid_order_no] = bid_order_details
                            self.bid_book[best_bid_price][bid_order_no] = bid_order_details
                    
                    # 清理空价格层
                    if best_bid_price in self.bid_book and not self.bid_book[best_bid_price]:
                        del self.bid_book[best_bid_price]
                        if best_bid_price in self.sorted_bid_prices:
                            self.sorted_bid_prices.remove(best_bid_price)


                    # 更新当前最优价格
                    best_bid_price = self.sorted_bid_prices[0] if self.sorted_bid_prices else None
                
            # 处理剩余未成交部分
            if remaining > 0:
                new_order['Balance'] = remaining
                self._add_order(new_order)

    def get_tick_snapshot(self, time=None):
        snapshot = {}
        snapshot['UpdateTime'] = self.time if time is None else time
        snapshot['SecurityID'] = self.SecurityID
        snapshot['OpenPrice'] = self.openprice
        snapshot['HighPrice'] = self.highprice
        snapshot['LowPrice'] = self.lowprice
        snapshot['LastPrice'] = self.current_price
        snapshot['Turnover'] = self.turnover
        snapshot['TradVolume'] = self.tradvolume

        for i in range(self.level):
            if i < len(self.sorted_bid_prices):
                price = self.sorted_bid_prices[i]
                bid_volume = sum(order_detail['balance'] for order_detail in self.bid_book[price].values())
                bid_orders = len(self.bid_book[price])  # 获取订单数量
                snapshot[f'BidPrice{i+1}'] = price
                snapshot[f'BidVolume{i+1}'] = bid_volume
                snapshot[f'NumOrdersB{i+1}'] = bid_orders
            else:
                snapshot[f'BidPrice{i+1}'] = 0
                snapshot[f'BidVolume{i+1}'] = 0
                snapshot[f'NumOrdersB{i+1}'] = 0

            if i < len(self.sorted_ask_prices):
                price = self.sorted_ask_prices[i]
                ask_volume = sum(order_detail['balance'] for order_detail in self.ask_book[price].values())
                ask_orders = len(self.ask_book[price])  # 获取订单数量
                snapshot[f'AskPrice{i+1}'] = price
                snapshot[f'AskVolume{i+1}'] = ask_volume
                snapshot[f'NumOrdersS{i+1}'] = ask_orders
            else:
                snapshot[f'AskPrice{i+1}'] = 0
                snapshot[f'AskVolume{i+1}'] = 0
                snapshot[f'NumOrdersS{i+1}'] = 0

        self.snapshots.append(snapshot)
        return snapshot

    def update_orderbook(self):
        '''更新逐笔订单簿数据'''
        self.get_orders() # 获取数据
        with open(os.path.join(path_orders, 'SH' + self.SecurityID, self.date + '.pkl'), 'rb') as f:
            orders = pickle.load(f)
        precloseprc = orders['PreCloPrice'].values[0]
        # 处理时间格式
        orders['OrderTime'] = pd.to_datetime(orders['OrderTime'])
        # 将 DataFrame 按行转换为字典构成的列表
        orders = orders.to_dict(orient='records')
        for order in orders:
            self.process_order(order)
        self.status = 'ENDTR'
        self._cal_call_auction_snapshot()
        self._get_call_auc_transaction()
        self.get_tick_snapshot(time = pd.to_datetime(f'{self.date} 15:00:00.0'))

        snapshots = pd.DataFrame(self.snapshots)
        snapshots['tradeDate'] = self.date
        snapshots['PreCloPrice'] = precloseprc
    
        # 保存逐笔订单簿
        snapshot_path = os.path.join(path_snapshots, f'SH{self.SecurityID}', f'{self.date}.pkl')
        if not os.path.exists(os.path.dirname(snapshot_path)):
            os.makedirs(os.path.dirname(snapshot_path))
        with open(snapshot_path, 'wb') as f:
            pickle.dump(snapshots, f, pickle.HIGHEST_PROTOCOL)
        print(f'股票{self.SecurityID}日期{self.date}的逐笔订单簿已保存至{snapshot_path}')

    def run_orderbook(self):
        '''创建指定股票指定日期的逐笔订单簿'''
        # 检查是否已经存在该股票该日期的订单簿
        csv_path = os.path.join(path_snapshots, 'SH' + self.SecurityID, self.date + '.pkl')
        if os.path.exists(csv_path):
            print('{}文件已存在'.format(csv_path))
        else:
            self.update_orderbook()

def download_snapshots(codes, dates, level=10):
    for code in codes:
        for date in dates:
            ob = OrderBook(code=code, date=date, level=level)
            ob.run_orderbook()
# ------------------------------------------------------------------------------------------------------示例使用----------------------------------------------------------------------------------------- 

if __name__ == "__main__":
    # 测试股票下载
    loader = Data_loader()
    try:
        print("测试沪市股票 511880...")
        trade_sh, order_sh, md_sh = loader.get_data(code="511880", date='2024-12-02')
        print(f"逐笔成交数据: {len(trade_sh)}行, 列: {list(trade_sh.columns)}")
        print(f"逐笔委托数据: {len(order_sh)}行, 列: {list(order_sh.columns)}")
        print(f"MD快照数据: {len(md_sh)}行, 列: {list(md_sh.columns)}")
        print(f"MD快照数据: {md_sh.head}")
        print("沪市股票测试通过!\n")
    except Exception as e:
        print(f"沪市测试失败: {e}")
    try:
        print("测试深市股票 000001...")
        trade_sz, order_sz, md_sz = loader.get_data(code="000001", date='2024-12-02')
        print(f"逐笔成交数据: {len(trade_sz)}行, 列: {list(trade_sz.columns)}")
        print(f"逐笔委托数据: {len(order_sz)}行, 列: {list(order_sz.columns)}")
        print(f"MD快照数据: {len(md_sz)}行, 列: {list(md_sz.columns)}")
        print("深市股票测试通过!\n")
    except Exception as e:
        print(f"深市测试失败: {e}")
    # 测试快照的生成
    download_snapshots(codes=["600000"], dates=['2024-12-02'], level=20)
