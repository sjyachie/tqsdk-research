#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略13 - 策略绩效归因分析
原理：
    分析策略收益来源
    归因到：方向收益、波动收益、时间价值

参数：
    - 基准周期：日线
    - 分析维度：持仓收益分解

适用行情：策略评估
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"           # 螺纹钢
KLINE_DURATION = 60 * 60         # 日线
DATA_LENGTH = 60                  # 历史数据数量
POSITION_SIZE = 1                # 持仓手数


def calculate_returns(price_series):
    """计算收益率分解"""
    returns = price_series.pct_change().dropna()
    
    # 方向收益：每日涨跌
    direction_return = returns.mean() * len(returns)
    
    # 波动收益：波动率贡献
    volatility_return = returns.std() * np.sqrt(len(returns))
    
    # 波动收益（高卖低买）
    high_low_return = (price_series.max() - price_series.min()) / price_series.mean()
    
    return {
        "direction": direction_return,
        "volatility": volatility_return,
        "high_low": high_low_return,
        "total": price_series.iloc[-1] / price_series.iloc[0] - 1
    }


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：策略绩效归因分析")
    
    # 获取历史数据
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, DATA_LENGTH)
    
    # 等待数据加载
    while len(klines) < DATA_LENGTH:
        api.wait_update()
    
    # 计算收益归因
    close_prices = klines["close"]
    open_prices = klines["open"]
    high_prices = klines["high"]
    low_prices = klines["low"]
    
    # 收益分解
    attribution = calculate_returns(close_prices)
    
    # 持仓期间分析
    position_returns = (close_prices.iloc[-1] - open_prices.iloc[0]) / open_prices.iloc[0]
    
    print(f"\n{'='*50}")
    print(f"策略绩效归因分析")
    print(f"{'='*50}")
    print(f"分析品种: {SYMBOL}")
    print(f"分析周期: {DATA_LENGTH} 天")
    print(f"\n--- 收益分解 ---")
    print(f"总收益: {attribution['total']*100:.2f}%")
    print(f"方向收益: {attribution['direction']*100:.2f}%")
    print(f"波动收益: {attribution['volatility']*100:.2f}%")
    print(f"高低收益: {attribution['high_low']*100:.2f}%")
    print(f"\n--- 统计指标 ---")
    print(f"平均日收益率: {close_prices.pct_change().mean()*100:.4f}%")
    print(f"收益率标准差: {close_prices.pct_change().std()*100:.4f}%")
    print(f"夏普比率(假设无风险利率2%): {(attribution['total'] - 0.02) / attribution['volatility']:.2f}")
    print(f"最大回撤: {(close_prices / close_prices.cummax() - 1).min()*100:.2f}%")
    print(f"{'='*50}\n")
    
    api.close()


if __name__ == "__main__":
    main()
