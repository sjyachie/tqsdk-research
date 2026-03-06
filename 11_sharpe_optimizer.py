#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略11 - 夏普比率优化器
原理：
    通过滚动计算策略的夏普比率，动态调整仓位

参数：
    - 合约：SHFE.rb2505
    - 周期：日线
    - 回看周期：20天
    - 夏普阈值：1.0

适用行情：所有行情
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim
import numpy as np

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2505"           # 螺纹钢
KLINE_DURATION = 24 * 60 * 60    # 日线
LOOKBACK = 20                    # 回看周期
SHARPE_THRESHOLD = 1.0          # 夏普比率阈值
VOLUME = 1                       # 每次交易手数
DATA_LENGTH = 100                # 历史K线数量


def calc_sharpe(returns, risk_free_rate=0.0):
    """计算夏普比率"""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：夏普比率优化器")
    
    klines = api.get_kline_serial(SYMBOL, KLINE_DURATION, DATA_LENGTH)
    
    position = 0
    
    while True:
        api.wait_update()
        
        if api.is_changing(klines.iloc[-1], "datetime"):
            close = klines["close"]
            
            if len(close) < LOOKBACK + 1:
                continue
            
            returns = close.pct_change().dropna()
            recent_returns = returns[-LOOKBACK:]
            
            sharpe = calc_sharpe(recent_returns)
            
            print(f"近期夏普比率: {sharpe:.2f}")
            
            if position == 0:
                if sharpe > SHARPE_THRESHOLD:
                    print(f"[开仓] 夏普比率高于阈值，开仓")
                    position = 1
            else:
                if sharpe < SHARPE_THRESHOLD * 0.5:
                    print(f"[平仓] 夏普比率低于阈值，平仓")
                    position = 0
    
    api.close()


if __name__ == "__main__":
    main()
