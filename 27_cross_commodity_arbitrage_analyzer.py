#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 27
策略名称: 跨品种套利分析器
生成日期: 2026-03-18
仓库地址: tqsdk-research
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【TqSdk 简介】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TqSdk（天勤量化 SDK）是由信易科技（北京）有限公司开发的专业期货量化交易框架，
完全免费开源（Apache 2.0 协议），基于 Python 语言设计，支持 Python 3.6+ 环境。

官网: https://www.shinnytech.com/tianqin/
文档: https://doc.shinnytech.com/tqsdk/latest/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【策略背景与原理】
跨品种套利分析器对多个相关品种进行截面分析，识别价格关系的均值回复机会。
支持产业链套利（螺纹钢/热卷/铁矿石/焦煤/焦炭）、跨交易所套利、
统计套利（Z-Score）等模式。自动计算价差/价比的历史分布，生成套利信号。

【策略参数】
- SYMBOL_PAIRS: 套利配对列表
- LOOKBACK_PERIOD: 历史回溯期
- ZSCORE_ENTRY: Z分数入场阈值
- ZSCORE_EXIT: Z分数出场阈值
- SPREAD_TYPE:价差类型（spread_ratio/spread_diff）
- MEAN_REVERSION_WINDOW: 均值回复计算窗口

【风险提示】
跨品种套利涉及两个品种的单边风险。价差可能长期维持异常，慎用高杠杆。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json


# ============ 参数配置 ============
# 产业链套利配对：钢铁产业链
SYMBOL_PAIRS = [
    ("rb2501.SHF", "hc2501.SHF", "螺纹-热卷套利"),
    ("rb2501.SHF", "i2501.DCE", "螺纹-铁矿套利"),
    ("j2501.DCE", "jm2501.DCE", "焦炭-焦煤套利"),
]
LOOKBACK_PERIOD = 60           # 回溯天数
ZSCORE_ENTRY = 2.0            # 入场Z分数阈值
ZSCORE_EXIT = 0.5             # 出场Z分数阈值
SPREAD_TYPE = "ratio"         # 比价(spread_ratio) or 价差(spread_diff)
MEAN_REVERSION_WINDOW = 20    # 均值回复窗口
KELLY_FRACTION = 0.25         # Kelly比例使用分数


class CrossCommodityArbitrageAnalyzer:
    """跨品种套利分析器"""

    def __init__(self, api, symbol_a, symbol_b, name="跨品种套利"):
        self.api = api
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.pair_name = name
        self.price_a_history = deque(maxlen=LOOKBACK_PERIOD + 10)
        self.price_b_history = deque(maxlen=LOOKBACK_PERIOD + 10)
        self.spread_history = deque(maxlen=LOOKBACK_PERIOD + 10)
        self.current_signal = "neutral"  # neutral / long_a / long_b
        self.position_open_time = None
        self.entry_zscore = 0.0
        self.trades = []
        self.current_pnl = 0.0
        self.quote_a = self.api.get_quote(symbol_a)
        self.quote_b = self.api.get_quote(symbol_b)

    def _compute_spread(self, price_a, price_b):
        """计算价差/价比"""
        if SPREAD_TYPE == "ratio":
            return price_a / price_b if price_b != 0 else 1.0
        else:
            return price_a - price_b

    def _compute_zscore(self, spread_series):
        """计算Z分数"""
        if len(spread_series) < MEAN_REVERSION_WINDOW:
            return 0.0
        window_data = list(spread_series)[-MEAN_REVERSION_WINDOW:]
        mean = np.mean(window_data)
        std = np.std(window_data)
        if std < 1e-10:
            return 0.0
        current = spread_series[-1]
        return (current - mean) / std

    def update_prices(self):
        """更新价格数据"""
        price_a = self.quote_a.get('last_price', 0)
        price_b = self.quote_b.get('last_price', 0)
        if price_a > 0:
            self.price_a_history.append(price_a)
        if price_b > 0:
            self.price_b_history.append(price_b)
        if price_a > 0 and price_b > 0:
            spread = self._compute_spread(price_a, price_b)
            self.spread_history.append(spread)

    def generate_signal(self):
        """生成套利信号"""
        if len(self.spread_history) < MEAN_REVERSION_WINDOW:
            return "neutral", 0.0

        zscore = self._compute_zscore(list(self.spread_history))
        spread_series = list(self.spread_history)

        if self.current_signal == "neutral":
            if zscore > ZSCORE_ENTRY:
                # 价差高于均值，做空价差（卖A买B）
                return "short_spread", zscore
            elif zscore < -ZSCORE_ENTRY:
                # 价差低于均值，做多价差（买A卖B）
                return "long_spread", zscore
        elif self.current_signal == "short_spread":
            if zscore < ZSCORE_EXIT:
                return "exit", zscore
        elif self.current_signal == "long_spread":
            if zscore > -ZSCORE_EXIT:
                return "exit", zscore

        return "hold", zscore

    def compute_position_size(self, account_balance):
        """计算仓位数量"""
        # Kelly公式简化版
        if len(self.spread_history) < MEAN_REVERSION_WINDOW:
            return 0
        # 使用波动率调整仓位
        spread_series = list(self.spread_history)
        spread_std = np.std(spread_series[-MEAN_REVERSION_WINDOW:])
        if spread_std < 1e-10:
            return 0
        kelly_size = account_balance * KELLY_FRACTION / (spread_std * 2)
        return max(1, int(kelly_size))

    def print_analysis(self, zscore):
        """打印分析结果"""
        spread_series = list(self.spread_history)
        if len(spread_series) < MEAN_REVERSION_WINDOW:
            return

        mean = np.mean(spread_series[-MEAN_REVERSION_WINDOW:])
        std = np.std(spread_series[-MEAN_REVERSION_WINDOW:])
        current = spread_series[-1]

        print(f"\n{'='*60}")
        print(f"【跨品种套利分析】{self.pair_name}")
        print(f"{'='*60}")
        print(f"  品种A: {self.symbol_a} | 品种B: {self.symbol_b}")
        print(f"  当前{'价比' if SPREAD_TYPE == 'ratio' else '价差'}: {current:.6f}")
        print(f"  均值({MEAN_REVERSION_WINDOW}日): {mean:.6f}")
        print(f"  标准差: {std:.6f}")
        print(f"  Z分数: {zscore:.3f}")
        print(f"  当前信号: {self.current_signal}")
        print(f"\n  套利逻辑:")
        print(f"    Z > {ZSCORE_ENTRY:.1f}: 价差偏高 → 做空A/做多B (short spread)")
        print(f"    Z < -{ZSCORE_ENTRY:.1f}: 价差偏低 → 做多A/做多做B (long spread)")
        print(f"    |Z| < {ZSCORE_EXIT:.1f}: 回归均值 → 平仓")

        if zscore > ZSCORE_ENTRY:
            print(f"\n  ⚠️ 建议: 做空{self.symbol_a}/做多{self.symbol_b}")
        elif zscore < -ZSCORE_ENTRY:
            print(f"\n  ⚠️ 建议: 做多{self.symbol_a}/做空{self.symbol_b}")
        else:
            print(f"\n  ✅ 建议: 持有/观望")

    def run(self, interval=60):
        """运行分析主循环"""
        print(f"跨品种套利分析器启动: {self.pair_name}")
        print(f"  {self.symbol_a} vs {self.symbol_b}")
        print(f"  Z分数阈值: 入场 {ZSCORE_ENTRY}, 出场 {ZSCORE_EXIT}")

        while True:
            self.api.wait_update()
            self.update_prices()
            signal, zscore = self.generate_signal()

            # 信号变化时打印报告
            if signal != self.current_signal and signal != "hold":
                print(f"\n🔔 信号变化: {self.current_signal} → {signal} (Z={zscore:.3f})")
                self.current_signal = signal
                self.entry_zscore = zscore
            elif self.current_signal == "neutral" or signal == "exit":
                self.print_analysis(zscore)
                if signal == "exit":
                    self.current_signal = "neutral"

            # 每隔一定时间打印一次分析
            if len(self.spread_history) % 10 == 0:
                self.print_analysis(zscore)


# ============ 主程序 ============
if __name__ == "__main__":
    import sys
    pair_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if pair_idx >= len(SYMBOL_PAIRS):
        print(f"无效配对索引，有效范围: 0-{len(SYMBOL_PAIRS)-1}")
        sys.exit(1)

    sym_a, sym_b, name = SYMBOL_PAIRS[pair_idx]
    api = TqApi(auth=TqAuth("auto", "auto"))
    analyzer = CrossCommodityArbitrageAnalyzer(api, sym_a, sym_b, name)
    try:
        analyzer.run(interval=60)
    except KeyboardInterrupt:
        print("\n分析器已停止")
    finally:
        api.close()
