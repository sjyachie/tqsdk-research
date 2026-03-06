#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略07 - 研究分析：布林带宽度分析策略
原理：
    布林带宽度（Bandwidth）反映市场波动性。
    带宽收窄后放大，往往预示着趋势行情即将启动。

参数：
    - 合约：SHFE.rb2505
    - 周期：1小时
    - 布林带周期：20
    - 带宽阈值：0.03

适用行情：盘整结束即将突破时
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
from tqsdk.ta import BOLL
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"          # 螺纹钢
KLINE_DURATION = 60 * 60        # 1小时K线
BOLL_PERIOD = 20                # 布林带周期
BOLL_STD = 2.0                  # 布林带倍数
BANDWIDTH_THRESHOLD = 0.03      # 带宽收窄阈值

# ============ 主策略 ============
def calculate_bandwidth(upper, lower, middle):
    """计算布林带宽度"""
    return (upper - lower) / middle

def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：布林带宽度分析策略")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, data_length=BOLL_PERIOD + 10)
    
    position = 0
    entry_price = 0
    
    while True:
        api.wait_update()
        
        if api.is_changing(klines):
            if len(klines) < BOLL_PERIOD + 5:
                continue
                
            boll = BOLL(klines, BOLL_PERIOD, BOLL_STD)
            upper = boll['upper'].iloc[-1]
            lower = boll['lower'].iloc[-1]
            middle = boll['mid'].iloc[-1]
            
            bandwidth = calculate_bandwidth(upper, lower, middle)
            current_price = klines['close'].iloc[-1]
            
            print(f"价格: {current_price}, 带宽: {bandwidth:.4f}")
            
            # 带宽收窄预警
            if bandwidth < BANDWIDTH_THRESHOLD:
                print(f"[预警] 带宽收窄，可能突破")
                
            if position == 0:
                # 突破上轨做多
                if current_price > upper:
                    position = 1
                    entry_price = current_price
                    print(f"[买入突破] 价格: {current_price}")
                # 突破下轨做空
                elif current_price < lower:
                    position = -1
                    entry_price = current_price
                    print(f"[卖出突破] 价格: {current_price}")
                    
            elif position == 1:
                if current_price < middle:
                    print(f"[平仓] 回归中轨")
                    position = 0
                    
            elif position == -1:
                if current_price > middle:
                    print(f"[平仓] 回归中轨")
                    position = 0
    
    api.close()

if __name__ == "__main__":
    main()
