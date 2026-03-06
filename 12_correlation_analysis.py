#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略12 - 策略相关性分析
原理：
    分析多个策略/品种之间的相关性，实现组合优化

参数：
    - 品种1：SHFE.rb2505
    - 品种2：SHFE.hc2505
    - 周期：日线
    - 相关性窗口：30天

适用行情：所有行情
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth, TqSim
import numpy as np

# ============ 参数配置 ============
SYMBOL1 = "SHFE.rb2505"          # 螺纹钢
SYMBOL2 = "SHFE.hc2505"          # 热轧卷板
KLINE_DURATION = 24 * 60 * 60    # 日线
CORR_WINDOW = 30                 # 相关性窗口
VOLUME = 1                       # 每次交易手数
DATA_LENGTH = 100                # 历史K线数量


def calc_correlation(returns1, returns2):
    """计算相关性"""
    if len(returns1) < 2:
        return 0.0
    return np.corrcoef(returns1, returns2)[0, 1]


def main():
    api = TqApi(account=TqSim(), auth=TqAuth("账号", "密码"))
    print("启动：策略相关性分析")
    
    klines1 = api.get_kline_serial(SYMBOL1, KLINE_DURATION, DATA_LENGTH)
    klines2 = api.get_kline_serial(SYMBOL2, KLINE_DURATION, DATA_LENGTH)
    
    while True:
        api.wait_update()
        
        if api.is_changing(klines1.iloc[-1], "datetime"):
            close1 = klines1["close"]
            close2 = klines2["close"]
            
            if len(close1) < CORR_WINDOW + 1:
                continue
            
            returns1 = close1.pct_change().dropna()[-CORR_WINDOW:]
            returns2 = close2.pct_change().dropna()[-CORR_WINDOW:]
            
            corr = calc_correlation(returns1, returns2)
            
            print(f"{SYMBOL1} vs {SYMBOL2} 相关性: {corr:.3f}")
            
            # 高相关性时，可以做配对交易
            if corr > 0.8:
                print(f"[提示] 高相关性，可考虑配对交易")
    
    api.close()


if __name__ == "__main__":
    main()
