#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略16 - 订单流分析
原理：
    分析订单簿和成交量的微观结构
    识别大单入场/离场信号

参数：
    - 合约：SHFE.rb2505
    - 周期：1分钟
    - 大单阈值：5手
    - 成交确认：3笔

适用行情：短线交易
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim, TargetPosTask
import pandas as pd
import numpy as np
from collections import deque

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"           # 螺纹钢
KLINE_DURATION = 60              # 1分钟K线
VOLUME = 1                       # 交易手数
DATA_LENGTH = 30                 # 历史K线数量
LARGE_ORDER_THRESHOLD = 5        # 大单阈值（手）
CONFIRMATION_COUNT = 3           # 成交确认次数
TIME_WINDOW = 5                  # 时间窗口（分钟）


def detect_order_flow(trades_df, large_threshold=5):
    """检测订单流"""
    if len(trades_df) == 0:
        return 0, 0
    
    # 大单买入
    large_buy = trades_df[trades_df["volume"] >= large_threshold]
    buy_volume = large_buy[large_buy["side"] == "buy"]["volume"].sum()
    
    # 大单卖出
    large_sell = trades_df[trades_df["volume"] >= large_threshold]
    sell_volume = large_sell[large_sell["side"] == "sell"]["volume"].sum()
    
    net_flow = buy_volume - sell_volume
    
    return net_flow, len(large_buy)


def analyze_order_imbalance(quote):
    """分析订单不平衡"""
    if quote is None:
        return 0
    
    bid_volume = 0
    ask_volume = 0
    
    # 累加买卖盘量
    for i in range(5):
        bid_volume += quote.get(f"bid_volume_{i}", 0)
        ask_volume += quote.get(f"ask_volume_{i}", 0)
    
    if bid_volume + ask_volume == 0:
        return 0
    
    # 不平衡度：正值为买方主导
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    return imbalance


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：订单流分析")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, DATA_LENGTH)
    target_pos = TargetPosTask(api, SYMBOL)
    
    print(f"分析参数:")
    print(f"  - 大单阈值: {LARGE_ORDER_THRESHOLD}手")
    print(f"  - 成交确认: {CONFIRMATION_COUNT}笔")
    print(f"  - 时间窗口: {TIME_WINDOW}分钟")
    
    # 订单流历史
    order_flow_history = deque(maxlen=TIME_WINDOW)
    large_order_count = 0
    position = 0
    
    while True:
        api.wait_update()
        
        # 获取实时行情
        quote = api.get_quote(SYMBOL)
        
        if api.is_changing(klines.iloc[-1], "datetime"):
            close = klines["close"].iloc[-1]
            volume = klines["volume"].iloc[-1]
            
            # 计算订单不平衡
            imbalance = analyze_order_imbalance(quote)
            
            # 记录订单流
            order_flow = {
                "price": close,
                "volume": volume,
                "imbalance": imbalance
            }
            order_flow_history.append(order_flow)
            
            # 分析最近订单流
            recent_imbalance = np.mean([o["imbalance"] for o in order_flow_history])
            recent_volume = np.mean([o["volume"] for o in order_flow_history])
            
            # 大单检测
            is_large_order = volume >= LARGE_ORDER_THRESHOLD * 10
            
            print(f"\n=== 订单流分析 ===")
            print(f"价格: {close:.2f}")
            print(f"成交量: {volume:.0f}")
            print(f"订单不平衡: {imbalance:+.3f}")
            print(f"平均不平衡: {recent_imbalance:+.3f}")
            print(f"平均成交量: {recent_volume:.0f}")
            print(f"大单信号: {'是 ⚡' if is_large_order else '否'}")
            
            # 订单流信号判断
            buy_signal = False
            sell_signal = False
            
            # 买单主导 + 大单入场
            if recent_imbalance > 0.3 and is_large_order:
                print("\n📈 信号: 大单买入")
                buy_signal = True
            
            # 卖单主导 + 大单入场
            elif recent_imbalance < -0.3 and is_large_order:
                print("\n📉 信号: 大单卖出")
                sell_signal = True
            
            # 订单流极端不平衡
            if abs(recent_imbalance) > 0.5:
                direction = "买方" if recent_imbalance > 0 else "卖方"
                print(f"⚠️ 极端不平衡: {direction}主导")
            
            # 执行交易
            if buy_signal and position == 0:
                print("\n✅ 执行买入")
                target_pos.set_target_volume(VOLUME)
                position = VOLUME
                large_order_count = 0
            elif sell_signal and position > 0:
                print("\n✅ 执行卖出")
                target_pos.set_target_volume(0)
                position = 0
                large_order_count = 0
            
            # 更新大单计数
            if is_large_order:
                large_order_count += 1
            
            print(f"当前仓位: {'多头' if position > 0 else '空仓'}")
            print("-" * 40)
    
    api.close()


if __name__ == "__main__":
    main()
