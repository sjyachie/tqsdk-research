#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略08 - 研究分析：成交量加权价格分析策略
原理：
    VWAP（成交量加权平均价格）是重要的支撑阻力位。
    价格在VWAP上方偏多，下方偏空。

参数：
    - 合约：SHFE.rb2505
    - 周期：5分钟
    - 止损：0.5%

适用行情：日内交易
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"          # 螺纹钢
KLINE_DURATION = 5 * 60         # 5分钟K线
STOP_LOSS = 0.005               # 0.5%止损

# ============ 主策略 ============
def calculate_vwap(klines):
    """计算VWAP"""
    typical_price = (klines['high'] + klines['low'] + klines['close']) / 3
    volume = klines['volume']
    
    vwap = (typical_price * volume).sum() / volume.sum()
    return vwap

def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：VWAP分析策略")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, data_length=100)
    
    position = 0
    entry_price = 0
    
    while True:
        api.wait_update()
        
        if api.is_changing(klines):
            if len(klines) < 20:
                continue
                
            vwap = calculate_vwap(klines)
            current_price = klines['close'].iloc[-1]
            
            print(f"价格: {current_price}, VWAP: {vwap:.2f}")
            
            if position == 0:
                # 价格上穿VWAP做多
                if current_price > vwap:
                    position = 1
                    entry_price = current_price
                    print(f"[买入] 价格上穿VWAP, 价格: {current_price}")
                # 价格下穿VWAP做空
                elif current_price < vwap:
                    position = -1
                    entry_price = current_price
                    print(f"[卖出] 价格下穿VWAP, 价格: {current_price}")
                    
            elif position == 1:
                # 止损检查
                if current_price < entry_price * (1 - STOP_LOSS):
                    print(f"[止损] 价格: {current_price}")
                    position = 0
                # 价格下穿VWAP平仓
                elif current_price < vwap:
                    print(f"[平仓] 价格下穿VWAP, 价格: {current_price}")
                    position = 0
                    
            elif position == -1:
                # 止损检查
                if current_price > entry_price * (1 + STOP_LOSS):
                    print(f"[止损] 价格: {current_price}")
                    position = 0
                # 价格上穿VWAP平仓
                elif current_price > vwap:
                    print(f"[平仓] 价格上穿VWAP, 价格: {current_price}")
                    position = 0
    
    api.close()

if __name__ == "__main__":
    main()
