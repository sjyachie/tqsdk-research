#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略05 - 策略分析：RSI超买超卖分析工具
原理：
    统计历史数据中RSI超买超卖后的价格走势，
    分析策略有效性。

参数：
    - 合约：SHFE.rb2505
    - RSI周期：14
    - 超买阈值：70
    - 超卖阈值：30

作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
from tqsdk.ta import RSI
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"
KLINE_DURATION = 60 * 60        # 1小时K线
RSI_PERIOD = 14
OVERBOUGHT = 70
OVERSOLD = 30

# ============ 分析函数 ============
def analyze_rsi_signals(klines):
    """分析RSI信号"""
    
    rsi = RSI(klines, RSI_PERIOD)
    
    signals = []
    
    for i in range(50, len(rsi)):
        current_rsi = rsi.iloc[i]
        prev_rsi = rsi.iloc[i-1]
        
        # 超卖后反转
        if prev_rsi < OVERSOLD and current_rsi >= OVERSOLD:
            signals.append({
                'type': 'BUY',
                'rsi': current_rsi,
                'price': klines['close'].iloc[i]
            })
        # 超买后反转
        elif prev_rsi > OVERBOUGHT and current_rsi <= OVERBOUGHT:
            signals.append({
                'type': 'SELL',
                'rsi': current_rsi,
                'price': klines['close'].iloc[i]
            })
            
    return signals


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：RSI超买超卖分析")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, data_length=500)
    
    while len(klines) < 500:
        api.wait_update()
    
    signals = analyze_rsi_signals(klines)
    
    print(f"\n共发现 {len(signals)} 个RSI信号")
    
    buy_signals = [s for s in signals if s['type'] == 'BUY']
    sell_signals = [s for s in signals if s['type'] == 'SELL']
    
    print(f"买入信号: {len(buy_signals)}")
    print(f"卖出信号: {len(sell_signals)}")
    
    api.close()

if __name__ == "__main__":
    main()
