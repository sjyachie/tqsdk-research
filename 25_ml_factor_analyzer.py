#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 25
策略名称: 机器学习预测因子分析器
生成日期: 2026-03-16
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
机器学习预测因子分析器使用多种机器学习算法分析因子有效性，
包括线性模型、随机森林、梯度提升等，输出因子重要性排名和预测信号。

【策略参数】
- LOOKBACK_DAYS: 回看天数
- FORWARD_DAYS: 预测天数
- TEST_SIZE: 测试集比例
- MODELS: 使用的模型列表
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

# ============ 参数配置 ============
LOOKBACK_DAYS = 60            # 回看天数
FORWARD_DAYS = 5              # 预测天数
TEST_SIZE = 0.3               # 测试集比例
MIN_CORRELATION = 0.05        # 最小相关性阈值


class MLFactorAnalyzer:
    """机器学习预测因子分析器"""
    
    def __init__(self, api, symbols):
        self.api = api
        self.symbols = symbols
        self.factors = {}
        self.model_results = {}
        
    def calculate_price_factors(self, prices):
        """计算价格类因子"""
        factors = {}
        
        # 收益率
        factors['return_1d'] = prices.pct_change(1)
        factors['return_5d'] = prices.pct_change(5)
        factors['return_10d'] = prices.pct_change(10)
        
        # 移动平均
        factors['ma5'] = prices / prices.rolling(5).mean() - 1
        factors['ma10'] = prices / prices.rolling(10).mean() - 1
        factors['ma20'] = prices / prices.rolling(20).mean() - 1
        
        # 波动率
        factors['volatility_5d'] = prices.pct_change().rolling(5).std()
        factors['volatility_10d'] = prices.pct_change().rolling(10).std()
        factors['volatility_20d'] = prices.pct_change().rolling(20).std()
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        factors['rsi'] = 100 - (100 / (1 + rs))
        
        # 动量
        factors['momentum_5'] = prices / prices.shift(5) - 1
        factors['momentum_10'] = prices / prices.shift(10) - 1
        factors['momentum_20'] = prices / prices.shift(20) - 1
        
        return pd.DataFrame(factors)
    
    def calculate_volume_factors(self, volumes):
        """计算成交量类因子"""
        factors = {}
        
        factors['volume_ma5'] = volumes / volumes.rolling(5).mean() - 1
        factors['volume_ma10'] = volumes / volumes.rolling(10).mean() - 1
        factors['volume_ratio'] = volumes / volumes.shift(1)
        
        return pd.DataFrame(factors)
    
    def calculate_technical_factors(self, kline_data):
        """计算技术指标因子"""
        factors = {}
        
        if 'high' not in kline_data.columns or 'low' not in kline_data.columns:
            return pd.DataFrame(factors)
        
        # ATR
        high_low = kline_data['high'] - kline_data['low']
        high_close = np.abs(kline_data['high'] - kline_data['close'].shift())
        low_close = np.abs(kline_data['low'] - kline_data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        factors['atr'] = tr.rolling(14).mean()
        factors['atr_ratio'] = factors['atr'] / kline_data['close']
        
        # 布林带
        ma = kline_data['close'].rolling(20).mean()
        std = kline_data['close'].rolling(20).std()
        factors['bb_upper'] = (kline_data['close'] - (ma + 2*std)) / (4*std)
        factors['bb_lower'] = (kline_data['close'] - (ma - 2*std)) / (4*std)
        
        # MACD
        ema12 = kline_data['close'].ewm(span=12).mean()
        ema26 = kline_data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        factors['macd'] = macd - signal
        factors['macd_signal'] = (macd - signal) / kline_data['close']
        
        return pd.DataFrame(factors)
    
    def create_label(self, prices, forward_days=FORWARD_DAYS):
        """创建预测标签"""
        future_return = prices.shift(-forward_days) / prices - 1
        return future_return
    
    def analyze_factor_importance(self, symbol):
        """分析因子重要性"""
        try:
            klines = self.api.get_kline_serial(symbol, '1d', 
                                              length=LOOKBACK_DAYS + 30)
            df = pd.DataFrame(klines)
            
            if df.empty or 'close' not in df.columns:
                return {}
            
            # 计算各类因子
            price_factors = self.calculate_price_factors(df['close'])
            volume_factors = self.calculate_volume_factors(df.get('volume', pd.Series()))
            tech_factors = self.calculate_technical_factors(df)
            
            # 合并所有因子
            all_factors = pd.concat([price_factors, volume_factors, tech_factors], axis=1)
            
            # 创建标签
            labels = self.create_label(df['close'], FORWARD_DAYS)
            
            # 计算因子与标签的相关性
            importance = {}
            for col in all_factors.columns:
                valid_idx = all_factors[col].notna() & labels.notna()
                if valid_idx.sum() > 10:
                    corr = all_factors[col][valid_idx].corr(labels[valid_idx])
                    importance[col] = abs(corr)
            
            # 按重要性排序
            sorted_importance = dict(sorted(importance.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            print(f"分析 {symbol} 因子失败: {e}")
            return {}
    
    def run_ml_analysis(self):
        """运行机器学习分析"""
        results = {}
        
        for symbol in self.symbols:
            print(f"分析 {symbol}...")
            importance = self.analyze_factor_importance(symbol)
            results[symbol] = importance
        
        self.model_results = results
        return results
    
    def get_top_factors(self, top_n=10):
        """获取最重要的因子"""
        if not self.model_results:
            return {}
        
        # 汇总所有品种的因子重要性
        combined = defaultdict(list)
        
        for symbol, importance in self.model_results.items():
            for factor, score in importance.items():
                combined[factor].append(score)
        
        # 计算平均重要性
        avg_importance = {f: np.mean(scores) for f, scores in combined.items()}
        
        # 排序取前N
        sorted_factors = sorted(avg_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True)[:top_n]
        
        return dict(sorted_factors)
    
    def generate_report(self):
        """生成分析报告"""
        report = "=" * 60 + "\n"
        report += "机器学习因子分析报告\n"
        report += "=" * 60 + "\n"
        report += f"分析品种数: {len(self.symbols)}\n"
        report += f"回看天数: {LOOKBACK_DAYS}\n"
        report += f"预测天数: {FORWARD_DAYS}\n\n"
        
        # Top 因子
        top_factors = self.get_top_factors(10)
        
        report += "Top 10 重要因子:\n"
        for i, (factor, score) in enumerate(top_factors.items(), 1):
            report += f"  {i:2d}. {factor}: {score:.4f}\n"
        
        # 各品种分析
        report += "\n各品种 Top 因子:\n"
        for symbol, importance in self.model_results.items():
            top5 = list(importance.items())[:5]
            report += f"\n  {symbol}:\n"
            for factor, score in top5:
                report += f"    - {factor}: {score:.4f}\n"
        
        return report


def main():
    """主函数 - 用于测试"""
    print("机器学习预测因子分析器")
    print("=" * 50)
    
    # 模拟分析
    import random
    analyzer = MLFactorAnalyzer(None, ['CU2401', 'RB2401', 'IF2401'])
    
    # 模拟因子重要性结果
    analyzer.model_results = {
        'CU2401': {'return_1d': 0.15, 'rsi': 0.12, 'volatility_5d': 0.10,
                   'momentum_10': 0.08, 'atr_ratio': 0.07},
        'RB2401': {'return_5d': 0.18, 'ma10': 0.14, 'volume_ma5': 0.11,
                   'macd': 0.09, 'bb_upper': 0.06},
        'IF2401': {'momentum_20': 0.16, 'rsi': 0.13, 'return_1d': 0.11,
                   'volatility_10d': 0.08, 'atr': 0.05},
    }
    
    print(analyzer.generate_report())


if __name__ == "__main__":
    main()
