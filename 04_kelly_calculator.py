#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略04 - 凯利公式仓位计算器
原理：
    凯利公式（Kelly Criterion）用于计算最优投资比例：
    f* = (bp - q) / b
    其中：
      f* = 应投入的资金比例
      b = 赔率（盈亏比）
      p = 胜率
      q = 1 - p

参数：
    - 历史胜率
    - 历史盈亏比
    - 风险偏好系数（0.5-1.0）

作者：sjyachie / tqsdk-research
"""

import numpy as np

# ============ 参数配置 ============
WIN_RATE = 0.55              # 历史胜率
AVG_WIN_LOSS_RATIO = 1.5     # 平均盈亏比
KELLY_FRACTION = 0.5         # 凯利分数（风险偏好，0.5为半凯利）

def kelly_formula(win_rate, win_loss_ratio, fraction=1.0):
    """
    计算凯利公式
    
    Args:
        win_rate: 胜率 (0-1)
        win_loss_ratio: 盈亏比
        fraction: 凯利分数（用于控制风险）
    
    Returns:
        最优仓位比例
    """
    q = 1 - win_rate
    b = win_loss_ratio
    
    # 凯利公式
    kelly = (b * win_rate - q) / b
    
    # 限制在合理范围
    kelly = max(0, min(kelly, 1))
    
    # 应用凯利分数
    return kelly * fraction

def simulate_kelly(win_rate, win_loss_ratio, num_trades=100, initial_capital=100000):
    """模拟凯利公式效果"""
    kelly = kelly_formula(win_rate, win_loss_ratio)
    
    capital = initial_capital
    position = 0
    
    for i in range(num_trades):
        # 按凯利比例开仓
        position = capital * kelly
        
        # 随机决定盈亏
        if np.random.rand() < win_rate:
            capital += position * win_loss_ratio
        else:
            capital -= position
    
    return capital

def main():
    print("启动：凯利公式仓位计算器")
    print(f"输入参数: 胜率={WIN_RATE}, 盈亏比={AVG_WIN_LOSS_RATIO}")
    
    # 计算最优仓位
    optimal_kelly = kelly_formula(WIN_RATE, AVG_WIN_LOSS_RATIO)
    half_kelly = kelly_formula(WIN_RATE, AVG_WIN_LOSS_RATIO, KELLY_FRACTION)
    
    print(f"\n=== 凯利公式计算结果 ===")
    print(f"全凯利仓位: {optimal_kelly:.1%}")
    print(f"半凯利仓位 ({KELLY_FRACTION}x): {half_kelly:.1%}")
    
    # 风险提示
    if optimal_kelly <= 0:
        print("\n警告: 胜率/盈亏比组合不建议使用凯利公式")
    elif optimal_kelly > 0.5:
        print("\n警告: 全凯利仓位较高，建议使用半凯利降低风险")
    
    # 模拟测试
    print("\n=== 模拟测试 (100笔交易) ===")
    final_capital = simulate_kelly(WIN_RATE, AVG_WIN_LOSS_RATIO, 100)
    print(f"初始资金: 100,000")
    print(f"最终资金: {final_capital:,.0f}")
    print(f"总收益: {(final_capital/100000-1)*100:.1f}%")
    
    print("\n注意: 凯利公式假设交易独立同分布，实际市场可能有差异")

if __name__ == "__main__":
    main()
