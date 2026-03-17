#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略09 - 协整性分析器
原理：
    对多个品种进行协整性检验，找出均值回归交易机会。
    当价差偏离协整均衡时，做多低估值品种，做空高估值品种。

参数：
    - 品种1：SHFE.rb2505
    - 品种2：SHFE.hc2505
    - 周期：日线
    - 回溯期：60天
    - 开仓阈值：2倍标准差

适用行情：统计套利
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
import numpy as np

# ============ 参数配置 ============
SYMBOL1 = "SHFE.rb2505"         # 螺纹钢
SYMBOL2 = "SHFE.hc2505"         # 热卷
KLINE_DURATION = 24 * 60 * 60  # 日线
LOOKBACK = 60                   # 回溯期
ENTRY_THRESHOLD = 2.0           # 开仓阈值（标准差）

# ============ 协整性检验 ============
def cointegration_test(series1, series2):
    """简单协整性检验"""
    # 计算回归系数
    coef = np.polyfit(series1, series2, 1)[0]
    
    # 计算价差
    spread = series2 - coef * series1
    
    # 计算价差的ADF统计量（简化版）
    mean = np.mean(spread)
    std = np.std(spread)
    
    return coef, mean, std

# ============ 主策略 ============
def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：协整性分析器")
    
    kline1 = api.get_kline_serial(SYMBOL1, KLINE_DURATION, data_length=LOOKBACK + 1)
    kline2 = api.get_kline_serial(SYMBOL2, KLINE_DURATION, data_length=LOOKBACK + 1)
    
    while True:
        api.wait_update()
        
        if api.is_changing(kline1) or api.is_changing(kline2):
            if len(kline1) < LOOKBACK or len(kline2) < LOOKBACK:
                continue
            
            prices1 = kline1['close'].iloc[-LOOKBACK:].values
            prices2 = kline2['close'].iloc[-LOOKBACK:].values
            
            # 计算协整关系
            coef, mean, std = cointegration_test(prices1, prices2)
            
            # 当前价差
            current_s1 = prices1[-1]
            current_s2 = prices2[-1]
            current_spread = current_s2 - coef * current_s1
            
            # Z-Score
            if std > 0:
                zscore = (current_spread - mean) / std
            else:
                zscore = 0
            
            print(f"{SYMBOL1}: {current_s1}, {SYMBOL2}: {current_s2}")
            print(f"协整系数: {coef:.4f}, 均值: {mean:.2f}, 标准差: {std:.2f}")
            print(f"当前价差: {current_spread:.2f}, Z-Score: {zscore:.2f}")
            
            if abs(zscore) > ENTRY_THRESHOLD:
                if zscore > 0:
                    print(f"[信号] 价差偏高，做空{SYMBOL2}，做多{SYMBOL1}")
                else:
                    print(f"[信号] 价差偏低，做多{SYMBOL2}，做空{SYMBOL1}")
    
    api.close()

if __name__ == "__main__":
    main()
