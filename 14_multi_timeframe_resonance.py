#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略14 - 多周期共振分析
原理：
    多个时间周期信号共振确认
    提高信号可靠性，减少假信号

参数：
    - 合约：SHFE.rb2505
    - 周期：5分钟、15分钟、60分钟
    - 确认周期：3个周期同时发出信号

适用行情：趋势确认
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim, TargetPosTask
import pandas as pd

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"           # 螺纹钢
KLINE_DURATIONS = [5*60, 15*60, 60*60]  # 5分钟、15分钟、60分钟
MA_PERIOD = 20                   # 均线周期
VOLUME = 1                       # 交易手数
DATA_LENGTH = 50                 # 历史K线数量


def get_trend_signal(klines, ma_period=20):
    """获取趋势信号：1=多头，-1=空头，0=震荡"""
    close = klines["close"]
    ma = close.rolling(window=ma_period).mean()
    
    # 当前价格与均线关系
    current_price = close.iloc[-1]
    current_ma = ma.iloc[-1]
    
    # 均线斜率
    ma_slope = (ma.iloc[-1] - ma.iloc[-5]) / ma.iloc[-5]
    
    if current_price > current_ma and ma_slope > 0:
        return 1  # 多头
    elif current_price < current_ma and ma_slope < 0:
        return -1  # 空头
    else:
        return 0  # 震荡


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：多周期共振分析")
    
    # 获取多个周期的K线
    klines_list = []
    for duration in KLINE_DURATIONS:
        kline = api.get_kline_serial(SYMBOL, duration, DATA_LENGTH)
        klines_list.append(kline)
    
    target_pos = TargetPosTask(api, SYMBOL)
    
    position = 0
    
    while True:
        api.wait_update()
        
        # 检查每个周期是否更新
        signals = []
        period_names = ["5分钟", "15分钟", "60分钟"]
        
        for i, kline in enumerate(klines_list):
            if api.is_changing(kline.iloc[-1], "datetime"):
                signal = get_trend_signal(kline, MA_PERIOD)
                signals.append(signal)
                print(f"[{period_names[i]}] 信号: {'多头' if signal==1 else '空头' if signal==-1 else '震荡'}")
        
        # 如果所有周期都有信号
        if len(signals) == len(KLINE_DURATIONS):
            # 共振判断：多数一致
            bullish_count = sum(1 for s in signals if s == 1)
            bearish_count = sum(1 for s in signals if s == -1)
            
            print(f"\n--- 共振分析 ---")
            print(f"多头周期数: {bullish_count}")
            print(f"空头周期数: {bearish_count}")
            
            # 3个周期完全共振
            if bullish_count == 3 and position == 0:
                print(f"\n✅ 买入信号：3周期多头共振")
                target_pos.set_target_volume(VOLUME)
                position = VOLUME
            elif bearish_count == 3 and position > 0:
                print(f"\n✅ 卖出信号：3周期空头共振")
                target_pos.set_target_volume(0)
                position = 0
            
            print(f"---------------\n")
    
    api.close()


if __name__ == "__main__":
    main()
