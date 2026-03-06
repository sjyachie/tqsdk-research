#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略06 - 策略分析：移动均线交叉分析工具
原理：
    分析MA金叉死叉的历史表现，
    统计不同参数组合的收益率。

参数：
    - 合约：SHFE.rb2505
    - 短周期：5, 10, 15
    - 长周期：20, 30, 60

作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
from tqsdk.ta import MA
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"
KLINE_DURATION = 60 * 60        # 1小时K线

# ============ 分析函数 ============
def analyze_ma_crossover(klines, short_period, long_period):
    """分析均线交叉"""
    
    ma_short = MA(klines, short_period)
    ma_long = MA(klines, long_period)
    
    trades = []
    position = 0
    
    for i in range(long_period, len(ma_short)):
        s = ma_short.iloc[i]
        l = ma_long.iloc[i]
        ps = ma_short.iloc[i-1]
        pl = ma_long.iloc[i-1]
        
        if position == 0:
            if ps < pl and s > l:  # 金叉
                position = 1
                trades.append(('BUY', klines['close'].iloc[i]))
        else:
            if ps > pl and s < l:  # 死叉
                position = 0
                trades.append(('SELL', klines['close'].iloc[i]))
                
    return trades


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：均线交叉分析")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, data_length=200)
    
    while len(klines) < 200:
        api.wait_update()
    
    # 测试不同参数
    short_periods = [5, 10, 15]
    long_periods = [20, 30, 60]
    
    for sp in short_periods:
        for lp in long_periods:
            if sp >= lp:
                continue
                
            trades = analyze_ma_crossover(klines, sp, lp)
            
            if len(trades) >= 2:
                total_return = 1.0
                for i in range(0, len(trades)-1, 2):
                    if i+1 < len(trades):
                        ret = (trades[i+1][1] - trades[i][1]) / trades[i][1]
                        total_return *= (1 + ret)
                        
                print(f"MA({sp}, {lp}): 交易次数 {len(trades)//2}, 累计收益 {(total_return-1)*100:.2f}%")
    
    api.close()

if __name__ == "__main__":
    main()
