#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略03 - 策略相关性分析工具
原理：
    分析不同策略之间的相关性，帮助构建低相关组合：
    1. 读取多个策略的回测收益序列
    2. 计算策略间的皮尔逊相关系数
    3. 输出相关性矩阵

参数：
    - 策略文件列表
    - 计算周期

作者：sjyachie / tqsdk-research
"""

import pandas as pd
import numpy as np
from itertools import combinations

# ============ 参数配置 ============
STRATEGY_FILES = [
    "01_double_ma.py",
    "02_boll_breakout.py",
    "03_rsi_mean_reversion.py",
]
RETURNS_COLUMN = "returns"    # 收益列名
CORRELATION_THRESHOLD = 0.5   # 相关性阈值

def load_returns(filepath):
    """从回测结果文件加载收益序列"""
    # 实际实现需要解析回测输出文件
    # 这里用模拟数据
    dates = pd.date_range("2024-01-01", periods=252)
    returns = np.random.randn(252) * 0.02
    return pd.Series(returns, index=dates)

def calc_correlation_matrix(strategies_returns):
    """计算相关性矩阵"""
    df = pd.DataFrame(strategies_returns)
    return df.corr()

def find_low_correlation_pairs(corr_matrix):
    """找出低相关性策略对"""
    low_corr = []
    for (s1, s2), corr in np.ndindex(corr_matrix.shape):
        if s1 < s2 and corr < CORRELATION_THRESHOLD:
            low_corr.append((s1, s2, corr))
    return sorted(low_corr, key=lambda x: x[2])

def main():
    print("启动：策略相关性分析工具")
    
    # 加载各策略收益
    strategies_returns = {}
    for file in STRATEGY_FILES:
        name = file.replace(".py", "")
        strategies_returns[name] = load_returns(file)
        print(f"加载策略: {name}")
    
    # 计算相关性矩阵
    corr_matrix = calc_correlation_matrix(strategies_returns)
    
    print("\n=== 相关性矩阵 ===")
    print(corr_matrix)
    
    # 找出低相关性配对
    low_corr_pairs = find_low_correlation_pairs(corr_matrix)
    
    print(f"\n=== 低相关性策略对 (相关系数 < {CORRELATION_THRESHOLD}) ===")
    for s1, s2, corr in low_corr_pairs:
        print(f"  {s1} + {s2}: {corr:.3f}")
    
    print("\n分析完成")

if __name__ == "__main__":
    main()
