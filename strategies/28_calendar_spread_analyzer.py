#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 28
策略名称: 跨期套利分析器
生成日期: 2026-03-17
仓库地址: https://github.com/sjyachie/tqsdk-research
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【TqSdk 简介】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TqSdk（天勤量化 SDK）是由信易科技（北京）有限公司开发的专业期货量化交易框架，
完全免费开源（Apache 2.0 协议），基于 Python 语言设计，支持 Python 3.6+ 环境。

TqSdk 核心能力包括：
1. 统一行情接口 - 对接国内全部7大期货交易所
2. 高性能数据推送 - 延迟通常在5ms以内
3. 同步式编程范式 - 无需掌握异步编程
4. 完整回测引擎 - 支持Tick级回测
5. 实盘/模拟一键切换

官网: https://www.shinnytech.com/tianqin/
文档: https://doc.shinnytech.com/tqsdk/latest/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【策略背景与原理】
本策略是一个跨期套利分析器，用于分析期货不同到期月份合约之间的价差关系，
识别套利机会并计算理论价格。

核心功能：
1. 跨期价差分析 - 分析近月与远月合约的价差
2. 协整关系检验 - 检验价差的平稳性
3. 套利空间计算 - 计算理论价差与实际价差的偏离
4. 季节性分析 - 分析价差的季节性规律

支持的套利模式：
- 正向套利（Contango）：近月合约价格 < 远月合约价格
- 反向套利（Backwardation）：近月合约价格 > 远月合约价格

【策略参数】
- BASE_SYMBOLS: 基础品种列表
- MONTH_DIFF: 近远月合约月份差
- LOOKBACK_PERIOD: 回看周期（分钟）
- ENTRY_THRESHOLD: 入场阈值（标准差倍数）
- EXIT_THRESHOLD: 出场阈值（标准差倍数）

【风险提示】
- 本策略仅供研究学习使用
- 跨期套利存在基差风险
- 实际交易请做好风险控制
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============ 参数配置 ============
BASE_SYMBOLS = [
    "SHFE.rb",     # 螺纹钢
    "SHFE.hc",     # 热卷
    "SHFE.cu",     # 铜
    "SHFE.al",     # 铝
    "SHFE.zn",     # 锌
    "DCE.m",       # 豆粕
    "DCE.y",       # 豆油
    "DCE.j",       # 焦炭
    "DCE.jm",      # 焦煤
    "CZCE.SR",     # 白糖
    "CZCE.CF",     # 棉花
    "CZCE.MA",     # 甲醇
    "CZCE.TA",     # PTA
]
MONTH_DIFF = 2           # 近远月月份差
LOOKBACK_PERIOD = 120    # 回看周期（分钟）
ENTRY_THRESHOLD = 2.0    # 入场阈值（标准差）
EXIT_THRESHOLD = 0.5     # 出场阈值（标准差）
CURRENT_MONTH = 2405     # 当前月份


class CalendarSpreadAnalyzer:
    """跨期套利分析器"""
    
    def __init__(self, api):
        self.api = api
        self.base_symbols = BASE_SYMBOLS
        self.month_diff = MONTH_DIFF
        self.lookback = LOOKBACK_PERIOD
        self.current_month = CURRENT_MONTH
        
    def get_contract_symbol(self, base, month):
        """获取合约代码"""
        return f"{base}{month}"
    
    def get_near_far_contracts(self, base):
        """获取近月和远月合约"""
        near_month = self.current_month
        far_month = self.current_month + self.month_diff
        
        near_symbol = self.get_contract_symbol(base, near_month)
        far_symbol = self.get_contract_symbol(base, far_month)
        
        return near_symbol, far_symbol
    
    def get_historical_data(self, symbol, count):
        """获取历史K线数据"""
        try:
            klines = self.api.get_kline_serial(symbol, 60, count)
            if klines is not None and len(klines) > 0:
                return klines
        except Exception as e:
            pass
        return None
    
    def calculate_spread_statistics(self, near_closes, far_closes):
        """计算价差统计"""
        if len(near_closes) < 30 or len(far_closes) < 30:
            return None
        
        min_len = min(len(near_closes), len(far_closes))
        near = near_closes[:min_len]
        far = far_closes[:min_len]
        
        # 计算价差（近月 - 远月）
        spread = near - far
        
        # 统计指标
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        
        # 当前价差
        current_spread = spread[-1]
        
        # Z-score
        z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0
        
        # 价差百分比（相对于远月价格）
        spread_pct = (current_spread / far[-1] * 100) if far[-1] > 0 else 0
        
        return {
            'mean': mean_spread,
            'std': std_spread,
            'current': current_spread,
            'z_score': z_score,
            'spread_pct': spread_pct,
            'spread': spread
        }
    
    def check_spread_stationarity(self, spread):
        """检验价差平稳性（简化版ADF检验）"""
        if len(spread) < 30:
            return False, None
        
        # 计算价差的差分
        spread_diff = np.diff(spread)
        
        # 简单平稳性检验：差分的标准差应该小于原序列
        spread_std = np.std(spread)
        diff_std = np.std(spread_diff)
        
        # 如果差分标准差显著小于原序列，认为是平稳的
        ratio = diff_std / spread_std if spread_std > 0 else 1
        
        is_stationary = ratio < 0.8
        
        return is_stationary, ratio
    
    def calculate_theoretical_spread(self, near_symbol, far_symbol):
        """计算理论价差（基于持有成本）"""
        # 获取当前价格
        try:
            near_quote = self.api.get_quote(near_symbol)
            far_quote = self.api.get_quote(far_symbol)
            
            near_price = near_quote.get('last_price', 0)
            far_price = far_quote.get('last_price', 0)
            
            if near_price == 0 or far_price == 0:
                return None
            
            # 计算持仓成本（简化）
            # 假设年化利率5%，月份差为month_diff
            months = self.month_diff
            annual_rate = 0.05
            holding_cost_rate = annual_rate * months / 12
            
            # 理论近月价格 = 远月价格 / (1 + 持有成本)
            theoretical_near = far_price / (1 + holding_cost_rate)
            
            # 理论价差 = 实际近月 - 理论近月
            theoretical_spread = near_price - theoretical_near
            
            return {
                'near_price': near_price,
                'far_price': far_price,
                'theoretical_near': theoretical_near,
                'theoretical_spread': theoretical_spread,
                'actual_spread': near_price - far_price
            }
            
        except Exception as e:
            return None
    
    def analyze_calendar_spread(self, base):
        """分析单个品种的跨期套利机会"""
        near_symbol, far_symbol = self.get_near_far_contracts(base)
        
        # 获取历史数据
        near_data = self.get_historical_data(near_symbol, self.lookback)
        far_data = self.get_historical_data(far_symbol, self.lookback)
        
        if near_data is None or far_data is None:
            return None
        
        near_closes = list(near_data.get('close', []))
        far_closes = list(far_data.get('close', []))
        
        if len(near_closes) < 30 or len(far_closes) < 30:
            return None
        
        # 计算价差统计
        stats_result = self.calculate_spread_statistics(near_closes, far_closes)
        
        if stats_result is None:
            return None
        
        # 检验平稳性
        is_stationary, stationarity_ratio = self.check_spread_stationarity(stats_result['spread'])
        
        # 获取当前价格信息
        price_info = self.calculate_theoretical_spread(near_symbol, far_symbol)
        
        # 判断套利方向
        z_score = stats_result['z_score']
        
        if z_score > ENTRY_THRESHOLD:
            # 价差高于均值，做空价差（卖近月买远月）
            signal = "正向套利"
            action = "卖近月，买远月"
        elif z_score < -ENTRY_THRESHOLD:
            # 价差低于均值，做多价差（买近月卖远月）
            signal = "反向套利"
            action = "买近月，卖远月"
        else:
            signal = "观望"
            action = "等待"
        
        # 判断市场状态
        near_price = near_closes[-1]
        far_price = far_closes[-1]
        
        if near_price > far_price:
            market_type = "Backwardation（反向市场）"
        else:
            market_type = "Contango（正向市场）"
        
        return {
            'base': base,
            'near_symbol': near_symbol,
            'far_symbol': far_symbol,
            'statistics': stats_result,
            'stationary': is_stationary,
            'stationarity_ratio': stationarity_ratio,
            'signal': signal,
            'action': action,
            'market_type': market_type,
            'price_info': price_info
        }
    
    def run_analysis(self):
        """运行跨期套利分析"""
        print("=" * 60)
        print("跨期套利分析器")
        print("=" * 60)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"基础品种: {len(self.base_symbols)} 个")
        print(f"近远月差: {self.month_diff} 个月")
        print(f"回看周期: {self.lookback} 分钟")
        print(f"入场阈值: {ENTRY_THRESHOLD} 标准差")
        print(f"出场阈值: {EXIT_THRESHOLD} 标准差")
        
        results = {}
        opportunities = []
        
        for base in self.base_symbols:
            result = self.analyze_calendar_spread(base)
            if result:
                results[base] = result
                
                if result['signal'] != "观望":
                    opportunities.append(result)
        
        # 显示所有品种分析
        print(f"\n【跨期价差分析】")
        print("-" * 60)
        
        for base, result in results.items():
            stats = result['statistics']
            market = result['market_type']
            z = stats['z_score']
            
            print(f"\n{base} ({result['near_symbol']} vs {result['far_symbol']})")
            print(f"  市场状态: {market}")
            print(f"  当前价差: {stats['current']:.2f} ({stats['spread_pct']:.2f}%)")
            print(f"  历史均值: {stats['mean']:.2f}")
            print(f"  标准差: {stats['std']:.2f}")
            print(f"  Z-Score: {z:.2f}")
            print(f"  平稳性: {'✓ 平稳' if result['stationary'] else '⚠ 非平稳'}")
        
        # 显示套利机会
        print("\n" + "=" * 60)
        print("套利机会")
        print("=" * 60)
        
        if opportunities:
            for opp in opportunities:
                stats = opp['statistics']
                print(f"\n{opp['base']}:")
                print(f"  信号: {opp['signal']}")
                print(f"  操作: {opp['action']}")
                print(f"  Z-Score: {stats['z_score']:.2f}")
                print(f"  偏离程度: {abs(stats['z_score']) - ENTRY_THRESHOLD:.2f} 标准差")
        else:
            print("\n当前无明显套利机会（Z-Score在±2以内）")
        
        # 汇总
        print("\n" + "=" * 60)
        print("分析汇总")
        print("=" * 60)
        
        contango_count = sum(1 for r in results.values() if "Contango" in r['market_type'])
        backwardation_count = len(results) - contango_count
        
        print(f"正向市场(Contango): {contango_count} 个")
        print(f"反向市场(Backwardation): {backwardation_count} 个")
        print(f"平稳性品种对: {sum(1 for r in results.values() if r['stationary'])}/{len(results)}")
        
        return results, opportunities


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print(f"跨期套利分析器启动")
    
    analyzer = CalendarSpreadAnalyzer(api)
    results, opportunities = analyzer.run_analysis()
    
    api.close()


if __name__ == "__main__":
    main()
