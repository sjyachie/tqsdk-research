#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 17
策略名称: 时间序列动量分析器
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
时间序列动量（Time Series Momentum）是一种基于资产自身历史收益率预测未来走势的策略。
该策略假设过去收益为正的资产在未来一段时间内倾向于继续上涨，反之亦然。

【策略参数】
- SYMBOL: 交易合约
- LOOKBACK_PERIOD: 回看周期（天）
- HOLDING_PERIOD: 持有周期（天）
- VOLATILITY_ADJUSTED: 是否进行波动率调整

【风险提示】
本策略仅用于研究分析，不构成投资建议。历史业绩不代表未来表现。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 参数配置 ============
SYMBOL = "SHFE.rb2405"           # 交易合约
LOOKBACK_PERIOD = 60             # 回看60天
HOLDING_PERIOD = 20               # 持有20天
VOLATILITY_ADJUSTED = True       # 波动率调整
KLINE_DURATION = 60 * 60 * 24    # 日K线


class TimeSeriesMomentumAnalyzer:
    """时间序列动量分析器"""
    
    def __init__(self, api, symbol, lookback, holding):
        self.api = api
        self.symbol = symbol
        self.lookback = lookback
        self.holding = holding
        
    def get_historical_data(self, days):
        """获取历史K线数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days + 30)
        
        klines = self.api.get_kline_serial(
            self.symbol,
            KLINE_DURATION,
            start_time=int(start_time.timestamp()),
            end_time=int(end_time.timestamp())
        )
        
        df = pd.DataFrame(klines)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
        df.set_index('datetime', inplace=True)
        
        return df
    
    def calculate_returns(self, df):
        """计算收益率序列"""
        df['returns'] = df['close'].pct_change()
        return df
    
    def calculate_volatility(self, df, window=20):
        """计算滚动波动率"""
        df['volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
        return df
    
    def calculate_ts_momentum(self, df):
        """计算时间序列动量信号"""
        # 过去lookback天的累计收益
        df['cumulative_return'] = df['close'].pct_change(self.lookback)
        
        if VOLATILITY_ADJUSTED:
            # 波动率调整后的动量
            df['ts_momentum'] = df['cumulative_return'] / df['volatility']
        else:
            df['ts_momentum'] = df['cumulative_return']
        
        return df
    
    def generate_signal(self, df):
        """生成交易信号"""
        if len(df) < self.lookback + 1:
            return None, "数据不足"
        
        latest_momentum = df['ts_momentum'].iloc[-1]
        
        if latest_momentum > 0:
            return 1, "做多"
        elif latest_momentum < 0:
            return -1, "做空"
        else:
            return 0, "观望"
    
    def analyze(self):
        """执行完整分析"""
        # 获取数据
        days_needed = self.lookback + self.holding + 30
        df = self.get_historical_data(days_needed)
        
        # 计算指标
        df = self.calculate_returns(df)
        df = self.calculate_volatility(df)
        df = self.calculate_ts_momentum(df)
        
        # 生成信号
        signal, signal_name = self.generate_signal(df)
        
        # 获取最新数据
        latest = df.iloc[-1]
        
        return {
            "symbol": self.symbol,
            "current_price": latest['close'],
            "momentum": latest['ts_momentum'],
            "volatility": latest['volatility'],
            "cumulative_return": latest['cumulative_return'],
            "signal": signal,
            "signal_name": signal_name,
            "lookback": self.lookback,
            "holding": self.holding,
            "data": df.tail(10)
        }
    
    def generate_report(self, results):
        """生成分析报告"""
        df = results["data"]
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          时间序列动量分析报告 - {datetime.now().strftime('%Y-%m-%d')}                          ║
╠══════════════════════════════════════════════════════════════╣
║ 合约: {results['symbol']:20s} 当前价格: {results['current_price']:10.2f}     ║
║ 回看周期: {results['lookback']:3d}天  持有周期: {results['holding']:3d}天                          ║
╠══════════════════════════════════════════════════════════════╣
║ 指标数值:                                                      ║
║   • 累计收益: {results['cumulative_return']:8.2%}                                        ║
║   • 波动率:   {results['volatility']:8.2%}                                        ║
║   • 动量值:   {results['momentum']:8.4f}                                        ║
╠══════════════════════════════════════════════════════════════╣
║ 交易信号: {results['signal_name']:10s}  (信号值: {results['signal']:2d})                            ║
╚══════════════════════════════════════════════════════════════╝"""
        
        return report


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print("=" * 60)
    print("时间序列动量分析器启动")
    print("=" * 60)
    
    analyzer = TimeSeriesMomentumAnalyzer(api, SYMBOL, LOOKBACK_PERIOD, HOLDING_PERIOD)
    
    # 执行分析
    results = analyzer.analyze()
    
    # 生成报告
    report = analyzer.generate_report(results)
    print(report)
    
    api.close()


if __name__ == "__main__":
    main()
