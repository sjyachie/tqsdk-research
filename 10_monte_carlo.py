#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略10 - 蒙特卡洛模拟器
原理：
    基于历史收益率分布，使用蒙特卡洛方法模拟未来可能的收益路径。
    帮助评估策略风险和收益分布。

参数：
    - 初始资金：100万
    - 模拟次数：10000
    - 交易天数：252
    - 历史收益率序列

适用行情：策略评估
作者：sjyachie / tqsdk-research
"""

from tqsdk import TqApi, TqAuth
import numpy as np
import random

# ============ 参数配置 ============
INITIAL_CAPITAL = 1000000       # 初始资金
NUM_SIMULATIONS = 10000         # 模拟次数
TRADING_DAYS = 252              # 一年交易日

# ============ 蒙特卡洛模拟 ============
def monte_carlo_simulation(returns, initial_capital, num_simulations, num_days):
    """
    蒙特卡洛模拟
    returns: 历史收益率序列
    initial_capital: 初始资金
    num_simulations: 模拟次数
    num_days: 模拟天数
    """
    final_values = []
    percentile_5 = []
    percentile_95 = []
    
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    for _ in range(num_simulations):
        # 生成随机收益路径
        daily_returns = np.random.normal(mu, sigma, num_days)
        
        # 计算最终权益
        final_value = initial_capital * np.prod(1 + daily_returns)
        final_values.append(final_value)
    
    final_values = np.array(final_values)
    
    return {
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'std': np.std(final_values),
        'min': np.min(final_values),
        'max': np.max(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_95': np.percentile(final_values, 95)
    }

# ============ 主策略 ============
def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("启动：蒙特卡洛模拟器")
    
    # 模拟历史日收益率（应从真实策略获取）
    # 假设：日均收益0.1%，波动2%
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 60)
    
    print(f"历史日均收益: {np.mean(returns)*100:.3f}%")
    print(f"历史日波动率: {np.std(returns)*100:.3f}%")
    print(f"开始模拟...")
    
    # 运行蒙特卡洛模拟
    results = monte_carlo_simulation(
        returns, 
        INITIAL_CAPITAL, 
        NUM_SIMULATIONS, 
        TRADING_DAYS
    )
    
    print(f"\n===== 蒙特卡洛模拟结果 ({NUM_SIMULATIONS}次) =====")
    print(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"模拟天数: {TRADING_DAYS}天")
    print(f"平均最终权益: {results['mean']:,.0f}")
    print(f"中位数最终权益: {results['median']:,.0f}")
    print(f"权益标准差: {results['std']:,.0f}")
    print(f"最小权益: {results['min']:,.0f}")
    print(f"最大权益: {results['max']:,.0f}")
    print(f"5%分位数: {results['percentile_5']:,.0f}")
    print(f"95%分位数: {results['percentile_95']:,.0f}")
    
    # 计算盈利概率
    prob_profit = len([v for v in [results['mean']] if v > INITIAL_CAPITAL) / 1 * 100
    print(f"\n预期年化收益: {(results['mean']/INITIAL_CAPITAL - 1)*100:.1f}%")
    print(f"最大回撤风险: {(1 - results['percentile_5']/INITIAL_CAPITAL)*100:.1f}%")
    
    api.close()

if __name__ == "__main__":
    main()
