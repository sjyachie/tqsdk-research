#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 19
策略名称: 策略相关性分析器
生成日期: 2026-03-11
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
策略相关性分析器用于分析多个策略或合约之间的相关性，帮助投资者构建
低相关性的策略组合，降低整体风险，提高风险调整后收益。

【策略参数】
- STRATEGIES: 要分析的策略/合约列表
- LOOKBACK_PERIOD: 回溯期（天）
- CORRELATION_THRESHOLD: 相关性阈值

【风险提示】
相关性分析基于历史数据，过去表现不代表未来。实际使用时需结合其他分析。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============ 参数配置 ============
STRATEGIES = [
    "SHFE.rb2405",   # 螺纹钢
    "SHFE.hc2405",   # 热卷
    "DCE.m2405",     # 豆粕
    "CZCE.rm2405",   # 菜粕
    "CFFEX IF2405"   # 股指期货
]
LOOKBACK_PERIOD = 60       # 回溯期（天）
CORRELATION_THRESHOLD = 0.7  # 高相关性阈值


class StrategyCorrelationAnalyzer:
    """策略相关性分析器"""
    
    def __init__(self, api):
        self.api = api
        self.price_data = {}
        
    def get_historical_prices(self, symbol, days):
        """获取历史价格数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        try:
            klines = self.api.get_kline_serial(
                symbol, 
                86400, 
                data_length=days
            )
            
            if len(klines) > 0:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df.set_index('timestamp', inplace=True)
                df['returns'] = df['close'].pct_change()
                
                return df[['close', 'returns']].dropna()
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            
        return pd.DataFrame()
    
    def collect_data(self):
        """收集所有策略/合约的数据"""
        print(f"正在收集 {len(STRATEGIES)} 个标的的历史数据...")
        
        for symbol in STRATEGIES:
            df = self.get_historical_prices(symbol, LOOKBACK_PERIOD)
            if not df.empty:
                self.price_data[symbol] = df
                print(f"  {symbol}: 获取 {len(df)} 条数据")
            else:
                print(f"  {symbol}: 数据获取失败")
        
        return len(self.price_data) > 0
    
    def calculate_correlation_matrix(self):
        """计算收益率相关性矩阵"""
        returns_dict = {}
        
        for symbol, df in self.price_data.items():
            returns_dict[symbol] = df['returns']
        
        returns_df = pd.DataFrame(returns_dict)
        correlation_matrix = returns_df.corr()
        
        return correlation_matrix
    
    def find_high_correlation_pairs(self):
        """找出高相关性配对"""
        corr_matrix = self.calculate_correlation_matrix()
        high_corr_pairs = []
        
        symbols = corr_matrix.columns
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > CORRELATION_THRESHOLD:
                    high_corr_pairs.append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': corr,
                        'level': 'HIGH' if abs(corr) > 0.8 else 'MEDIUM'
                    })
        
        return sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def calculate_rolling_correlation(self, symbol1, symbol2, window=20):
        """计算滚动相关性"""
        if symbol1 not in self.price_data or symbol2 not in self.price_data:
            return None
            
        df1 = self.price_data[symbol1]['returns']
        df2 = self.price_data[symbol2]['returns']
        
        # 对齐数据
        aligned = pd.DataFrame({'s1': df1, 's2': df2}).dropna()
        
        if len(aligned) < window:
            return None
        
        rolling_corr = aligned['s1'].rolling(window).corr(aligned['s2'])
        
        return {
            'current': rolling_corr.iloc[-1],
            'mean': rolling_corr.mean(),
            'std': rolling_corr.std(),
            'series': rolling_corr.dropna()
        }
    
    def calculate_portfolio_diversification(self):
        """计算组合分散度"""
        corr_matrix = self.calculate_correlation_matrix()
        
        # 计算平均相关性
        n = len(corr_matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = corr_matrix.where(mask).stack().mean()
        
        # 计算分散度指标 (1 - 平均相关性)
        diversification = 1 - abs(avg_corr) if not np.isnan(avg_corr) else 0
        
        # 分散度评级
        if diversification > 0.7:
            rating = "优秀"
        elif diversification > 0.5:
            rating = "良好"
        elif diversification > 0.3:
            rating = "一般"
        else:
            rating = "较差"
        
        return {
            'diversification': diversification,
            'rating': rating,
            'average_correlation': avg_corr
        }
    
    def recommend_portfolio_weights(self):
        """推荐组合权重（基于相关性）"""
        corr_matrix = self.calculate_correlation_matrix()
        
        # 简单等权重，但避免高相关配对
        n = len(STRATEGIES)
        weights = {s: 1.0/n for s in STRATEGIES}
        
        # 检测高相关配对，降低其中一个的权重
        high_corr = self.find_high_correlation_pairs()
        
        for pair in high_corr:
            if pair['correlation'] > 0:
                # 正相关：降低第二个的权重
                weights[pair['symbol2']] *= 0.5
            else:
                # 负相关：保持（这是好的）
                pass
        
        # 归一化
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def generate_analysis_report(self):
        """生成分析报告"""
        report = []
        report.append("=" * 70)
        report.append("策略相关性分析报告")
        report.append("=" * 70)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"回溯期: {LOOKBACK_PERIOD} 天")
        report.append(f"标的数量: {len(STRATEGIES)}")
        report.append("")
        
        # 相关性矩阵
        report.append("【相关性矩阵】")
        corr_matrix = self.calculate_correlation_matrix()
        report.append(corr_matrix.round(3).to_string())
        report.append("")
        
        # 高相关性配对
        report.append("【高相关性配对】")
        high_corr = self.find_high_correlation_pairs()
        if high_corr:
            for pair in high_corr:
                level_icon = "🔴" if pair['level'] == 'HIGH' else "🟡"
                report.append(f"  {level_icon} {pair['symbol1']} <-> {pair['symbol2']}: {pair['correlation']:.3f}")
        else:
            report.append("  未发现高相关性配对")
        report.append("")
        
        # 分散度分析
        report.append("【分散度分析】")
        div = self.calculate_portfolio_diversification()
        report.append(f"  分散度指标: {div['diversification']:.2%}")
        report.append(f"  评级: {div['rating']}")
        report.append(f"  平均相关性: {div['average_correlation']:.3f}")
        report.append("")
        
        # 推荐权重
        report.append("【推荐权重】")
        weights = self.recommend_portfolio_weights()
        for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {symbol}: {weight:.1%}")
        report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """主函数"""
    api = TqApi(auth=TqAuth("用户名", "密码"))
    
    analyzer = StrategyCorrelationAnalyzer(api)
    
    if analyzer.collect_data():
        report = analyzer.generate_analysis_report()
        print(report)
    else:
        print("数据收集失败")
    
    api.close()


def backtest_with_strategy():
    """与策略结合使用"""
    from tqsdk import TqBacktest
    
    api = TqBacktest(
        front_broker="经纪商代码",
        broker_id="经纪商ID",
        auth=TqAuth("用户名", "密码")
    )
    
    analyzer = StrategyCorrelationAnalyzer(api)
    
    # 在回测过程中定期分析
    while True:
        api.wait_update()
        if api.is_changing():
            # 可以添加定期相关性检查
            pass


if __name__ == "__main__":
    # main()  # 实盘模式
    # backtest_with_strategy()  # 回测模式
    print("策略相关性分析器模块")
    print("请根据需要选择运行模式")
