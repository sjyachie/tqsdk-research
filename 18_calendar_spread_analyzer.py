#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 18
策略名称: 跨期价差分析器
生成日期: 2026-03-10
仓库地址: tqsdk-research
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【TqSdk 简介】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TqSdk（天勤量化 SDK）是由信易科技（北京）有限公司开发的专业期货量化交易框架，
完全免费开源（Apache 2.0 协议），基于 Python 语言设计，支持 Python 3.6+ 环境。
TqSdk 已服务于数万名国内期货量化投资者，是国内使用最广泛的期货量化框架之一。

TqSdk 核心能力包括：

1. **统一行情接口**：对接国内全部7大期货交易所（SHFE/DCE/CZCE/CFFEX/INE/GFEX）
   及主要期权品种，统一的 get_quote / get_kline_serial 接口，告别繁琐的协议适配；

2. **高性能数据推送**：天勤服务器行情推送延迟通常在5ms以内，Tick 级数据实时到达，
   K线自动合并，支持自定义周期（秒/分钟/小时/日/周/月）；

3. **同步式编程范式**：独特的 wait_update() + is_changing() 设计，策略代码像
   写普通Python一样自然流畅，无需掌握异步编程，大幅降低开发门槛；

4. **完整回测引擎**：内置 TqBacktest 回测模式，历史数据精确到Tick级别，
   支持滑点、手续费等真实市场参数，回测结果可信度高；

5. **实盘/模拟一键切换**：代码结构不变，仅替换 TqApi 初始化参数即可从
   模拟盘切换至实盘，极大降低策略上线风险；

6. **多账户并发**：支持同时连接多个期货账户，适合机构投资者和量化团队；

7. **活跃生态**：官方提供策略示例库、在线文档、量化社区论坛，更新维护活跃。

官网: https://www.shinnytech.com/tianqin/
文档: https://doc.shinnytech.com/tqsdk/latest/
GitHub: https://github.com/shinnytech/tqsdk-python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【策略背景与原理】
跨期价差（Calendar Spread）是指同一品种不同到期合约之间的价差分析。
当近月和远月合约价差偏离历史均值时，可能存在套利机会。

【策略参数】
- NEAR_CONTRACT: 近月合约
- FAR_CONTRACT: 远月合约
- LOOKBACK_PERIOD: 历史数据回看周期
- ENTRY_THRESHOLD: 入场阈值（标准差倍数）

【风险提示】
本策略仅用于研究分析，不构成投资建议。跨期套利存在流动性风险和基差风险。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 参数配置 ============
NEAR_CONTRACT = "SHFE.rb2405"    # 近月合约
FAR_CONTRACT = "SHFE.rb2410"    # 远月合约
LOOKBACK_PERIOD = 60            # 历史数据回看周期
ENTRY_THRESHOLD = 2.0           # 入场阈值（2倍标准差）
KLINE_DURATION = 60 * 60 * 24   # 日K线


class CalendarSpreadAnalyzer:
    """跨期价差分析器"""
    
    def __init__(self, api, near, far, lookback):
        self.api = api
        self.near = near
        self.far = far
        self.lookback = lookback
        
    def get_historical_data(self, symbol, days):
        """获取历史K线数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days + 30)
        
        klines = self.api.get_kline_serial(
            symbol,
            KLINE_DURATION,
            start_time=int(start_time.timestamp()),
            end_time=int(end_time.timestamp())
        )
        
        df = pd.DataFrame(klines)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        
        return df[['close']].rename(columns={'close': symbol})
    
    def calculate_spread(self, near_df, far_df):
        """计算价差"""
        spread = far_df.iloc[:, 0] - near_df.iloc[:, 0]
        return spread
    
    def calculate_spread_stats(self, spread):
        """计算价差统计特征"""
        mean = spread.mean()
        std = spread.std()
        current = spread.iloc[-1]
        z_score = (current - mean) / std
        
        return {
            'mean': mean,
            'std': std,
            'current': current,
            'z_score': z_score
        }
    
    def generate_signal(self, stats):
        """生成交易信号"""
        z = stats['z_score']
        
        if z > ENTRY_THRESHOLD:
            return -1, "做空价差（近月相对强势）", "价差高于均值，可能收敛"
        elif z < -ENTRY_THRESHOLD:
            return 1, "做多价差（远月相对强势）", "价差低于均值，可能扩大"
        elif abs(z) < 0.5:
            return 0, "观望", "价差接近均值"
        else:
            return 0, "观望", "价差未达到入场阈值"
    
    def analyze(self):
        """执行完整分析"""
        # 获取两个合约的历史数据
        near_df = self.get_historical_data(self.near, self.lookback + 30)
        far_df = self.get_historical_data(self.far, self.lookback + 30)
        
        # 对齐数据
        combined = pd.concat([near_df, far_df], axis=1)
        combined = combined.dropna()
        
        # 计算价差
        spread = self.calculate_spread(combined[self.near], combined[self.far])
        
        # 计算统计特征
        stats = self.calculate_spread_stats(spread)
        
        # 生成信号
        signal, signal_name, reason = self.generate_signal(stats)
        
        # 获取最新价格
        near_quote = self.api.get_quote(self.near)
        far_quote = self.api.get_quote(self.far)
        
        return {
            "near_contract": self.near,
            "far_contract": self.far,
            "near_price": near_quote.last_price,
            "far_price": far_quote.last_price,
            "current_spread": stats['current'],
            "mean_spread": stats['mean'],
            "std_spread": stats['std'],
            "z_score": stats['z_score'],
            "signal": signal,
            "signal_name": signal_name,
            "reason": reason
        }
    
    def generate_report(self, results):
        """生成分析报告"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          跨期价差分析报告 - {datetime.now().strftime('%Y-%m-%d')}                            ║
╠══════════════════════════════════════════════════════════════╣
║ 近月合约: {results['near_contract']:15s} 价格: {results['near_price']:10.2f}           ║
║ 远月合约: {results['far_contract']:15s} 价格: {results['far_price']:10.2f}           ║
╠══════════════════════════════════════════════════════════════╣
║ 价差统计:                                                      ║
║   • 当前价差: {results['current_spread']:10.2f}                                        ║
║   • 历史均值: {results['mean_spread']:10.2f}                                        ║
║   • 标准差:   {results['std_spread']:10.2f}                                        ║
║   • Z-Score: {results['z_score']:10.2f}                                        ║
╠══════════════════════════════════════════════════════════════╣
║ 交易信号: {results['signal_name']:20s}                        ║
║ 原因: {results['reason']:46s}   ║
╚══════════════════════════════════════════════════════════════╝"""
        
        return report


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("=" * 60)
    print("跨期价差分析器启动")
    print("=" * 60)
    
    analyzer = CalendarSpreadAnalyzer(api, NEAR_CONTRACT, FAR_CONTRACT, LOOKBACK_PERIOD)
    
    # 执行分析
    results = analyzer.analyze()
    
    # 生成报告
    report = analyzer.generate_report(results)
    print(report)
    
    api.close()


if __name__ == "__main__":
    main()
