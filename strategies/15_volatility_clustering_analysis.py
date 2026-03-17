#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略15 - 波动率聚类分析
原理：
    分析价格波动的聚类特征
    识别市场状态转换的关键节点

参数：
    - 合约：SHFE.rb2505
    - 周期：15分钟
    - 聚类窗口：20
    - 高波动阈值：2倍ATR

适用行情：波动率分析
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim, TargetPosTask
import pandas as pd
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"           # 螺纹钢
KLINE_DURATION = 15 * 60         # 15分钟K线
VOLUME = 1                       # 交易手数
DATA_LENGTH = 100                # 历史K线数量
CLUSTER_WINDOW = 20              # 聚类窗口
ATR_PERIOD = 14                  # ATR周期
HIGH_VOLATILITY_MULTI = 2.0      # 高波动倍数


def calculate_atr(klines, period=14):
    """计算ATR"""
    high = klines["high"]
    low = klines["low"]
    close = klines["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_volatility_clusters(returns, window=20):
    """计算波动率聚类"""
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility


def detect_volatility_regime(volatility, atr, current_price):
    """检测波动率状态"""
    avg_vol = volatility.iloc[-window_mean:]
    current_vol = volatility.iloc[-1]
    
    # 高波动状态
    if current_vol > avg_vol * HIGH_VOLATILITY_MULTI:
        return "high"
    # 低波动状态
    elif current_vol < avg_vol / HIGH_VOLATILITY_MULTI:
        return "low"
    # 正常波动
    else:
        return "normal"


window_mean = 20


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：波动率聚类分析")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, DATA_LENGTH)
    target_pos = TargetPosTask(api, SYMBOL)
    
    print(f"分析参数:")
    print(f"  - 聚类窗口: {CLUSTER_WINDOW}")
    print(f"  - 高波动阈值: {HIGH_VOLATILITY_MULTI}x ATR")
    
    position = 0
    
    while True:
        api.wait_update()
        
        if api.is_changing(klines.iloc[-1], "datetime"):
            close = klines["close"]
            returns = close.pct_change().dropna()
            
            if len(returns) < CLUSTER_WINDOW + 10:
                continue
            
            # 计算波动率聚类
            volatility = calculate_volatility_clusters(returns, CLUSTER_WINDOW)
            atr = calculate_atr(klines, ATR_PERIOD)
            
            current_vol = volatility.iloc[-1]
            current_atr = atr.iloc[-1]
            price = close.iloc[-1]
            
            # 计算波动率统计
            vol_ma = volatility.rolling(window=window_mean).mean()
            current_vol_ma = vol_ma.iloc[-1]
            
            # 波动率聚类分析
            vol_percentile = (volatility < current_vol).sum() / len(volatility)
            
            # 检测状态转换
            regime = detect_volatility_regime(volatility, atr, price)
            
            print(f"\n=== 波动率聚类分析 ===")
            print(f"价格: {price:.2f}")
            print(f"ATR: {current_atr:.2f}")
            print(f"年化波动率: {current_vol*100:.2f}%")
            print(f"波动率均值: {current_vol_ma*100:.2f}%")
            print(f"波动率分位: {vol_percentile*100:.1f}%")
            print(f"当前状态: {regime}")
            
            # 状态解读
            if regime == "high":
                print("📊 状态: 高波动期 - 风险加剧，谨慎交易")
                print("   建议: 缩小仓位，等待波动率回归")
            elif regime == "low":
                print("📊 状态: 低波动期 - 蓄势待发")
                print("   建议: 关注突破信号，准备入场")
            else:
                print("📊 状态: 正常波动")
            
            # 交易信号（基于波动率聚类）
            # 波动率从极低开始上升 - 突破信号
            if vol_percentile < 0.1 and position == 0:
                print("\n🔔 信号: 波动率极低，可能爆发")
                print("   建议观望，等待确认")
            
            # 波动率达到极端高 - 可能是转折点
            elif vol_percentile > 0.9 and position == 0:
                print("\n🔔 信号: 波动率极端高")
                print("   建议: 关注均值回归机会")
            
            # 波动率回归均值
            elif abs(current_vol - current_vol_ma) / current_vol_ma < 0.1:
                print("📊 波动率回归均值")
            
            print("-" * 40)
    
    api.close()


if __name__ == "__main__":
    main()
