#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
截面动量分析工具 (Cross-Sectional Momentum Analyzer)
分析不同品种间的截面动量关系，生成动量信号

Author: TqSdk Research
Update: 2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import warnings


class CrossSectionalMomentum:
    """截面动量分析器"""
    
    def __init__(self, ranking_period: int = 20,
                 formation_period: int = 60,
                 rebalance_period: int = 5):
        """
        初始化
        
        Args:
            ranking_period: 动量排名期（天）
            formation_period: 形成期（天）
            rebalance_period: 调仓周期（天）
        """
        self.ranking_period = ranking_period
        self.formation_period = formation_period
        self.rebalance_period = rebalance_period
        
        self.price_data = {}  # 各品种价格数据
        self.returns = {}  # 各品种收益率
        self.signals = {}  # 交易信号
        
    def add_price_series(self, symbol: str, prices: pd.Series):
        """添加价格序列"""
        self.price_data[symbol] = prices
        self.returns[symbol] = prices.pct_change().dropna()
        
    def calculate_momentum(self, symbol: str, period: Optional[int] = None) -> float:
        """
        计算动量因子
        
        Args:
            symbol: 品种代码
            period: 动量计算期
            
        Returns:
            动量因子值
        """
        if symbol not in self.returns:
            return 0.0
            
        period = period or self.ranking_period
        returns_series = self.returns[symbol]
        
        if len(returns_series) < period:
            return 0.0
            
        # 使用几何平均计算动量
        recent_returns = returns_series.iloc[-period:]
        momentum = (1 + recent_returns).prod() - 1
        
        return momentum
        
    def rank_cross_sectional(self, date: Optional[datetime] = None) -> pd.Series:
        """
        截面排名
        
        Args:
            date: 日期
            
        Returns:
            各品种动量排名
        """
        momenta = {}
        
        for symbol in self.price_data.keys():
            momentum = self.calculate_momentum(symbol)
            momenta[symbol] = momentum
            
        # 转换为Series并排名
        momentum_series = pd.Series(momenta)
        ranking = momentum_series.rank(ascending=True)
        
        return ranking
        
    def generate_signals(self, top_n: int = 3, bottom_n: int = 3) -> Dict:
        """
        生成交易信号
        
        Args:
            top_n: 做多顶部动量品种数
            bottom_n: 做空底部动量品种数
            
        Returns:
            交易信号字典
        """
        ranking = self.rank_cross_sectional()
        
        # 排序
        sorted_symbols = ranking.sort_values(ascending=False)
        
        # 做多动量最强的，做空动量最弱的
        long_symbols = sorted_symbols.head(top_n).index.tolist()
        short_symbols = sorted_symbols.tail(bottom_n).index.tolist()
        
        # 计算动量差（多空因子）
        if len(long_symbols) > 0 and len(short_symbols) > 0:
            long_momentum = ranking[long_symbols].mean()
            short_momentum = ranking[short_symbols].mean()
            momentum_spread = long_momentum - short_momentum
        else:
            momentum_spread = 0
            
        return {
            'long': long_symbols,
            'short': short_symbols,
            'momentum_spread': momentum_spread,
            'ranking': ranking.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
    def calculate_momentum_decay(self, symbol: str) -> Dict:
        """
        计算动量衰减因子（检测动量反转风险）
        
        Args:
            symbol: 品种代码
            
        Returns:
            衰减分析结果
        """
        if symbol not in self.returns:
            return {}
            
        returns_series = self.returns[symbol]
        
        # 计算不同周期的动量
        periods = [5, 10, 20, 40, 60]
        momentum_dict = {}
        
        for period in periods:
            if len(returns_series) >= period:
                momentum_dict[f'momentum_{period}'] = (
                    (1 + returns_series.iloc[-period:]).prod() - 1
                )
                
        # 检测动量是否在最近开始衰减
        if 'momentum_20' in momentum_dict and 'momentum_5' in momentum_dict:
            decay_ratio = momentum_dict['momentum_5'] / (momentum_dict['momentum_20'] + 1e-10)
        else:
            decay_ratio = 1.0
            
        return {
            'momentum_values': momentum_dict,
            'decay_ratio': decay_ratio,
            'reversal_risk': 'high' if decay_ratio < 0.3 else 'medium' if decay_ratio < 0.6 else 'low'
        }
        
    def calculate_cross_sectional_spread(self, symbol1: str, symbol2: str,
                                         lookback: int = 20) -> pd.Series:
        """
        计算两个品种的价差（用于配对交易）
        
        Args:
            symbol1: 品种1
            symbol2: 品种2
            lookback: 回看周期
            
        Returns:
            价差序列
        """
        if symbol1 not in self.price_data or symbol2 not in self.price_data:
            return pd.Series()
            
        prices1 = self.price_data[symbol1]
        prices2 = self.price_data[symbol2]
        
        # 对齐价格
        common_idx = prices1.index.intersection(prices2.index)
        p1 = prices1.loc[common_idx]
        p2 = prices2.loc[common_idx]
        
        # 计算价差（价格比率）
        spread = (p1 / p2).pct_change().dropna()
        
        return spread
        
    def momentum_clustering(self) -> Dict:
        """
        动量聚类分析
        
        Returns:
            聚类结果
        """
        # 计算所有品种的动量
        momenta = {}
        for symbol in self.price_data.keys():
            momenta[symbol] = self.calculate_momentum(symbol)
            
        momentum_series = pd.Series(momenta)
        
        # 使用分位数分组
        clusters = {
            'strong_momentum': momentum_series[momentum_series > momentum_series.quantile(0.75)].index.tolist(),
            'weak_momentum': momentum_series[momentum_series < momentum_series.quantile(0.25)].index.tolist(),
            'neutral': momentum_series[
                (momentum_series >= momentum_series.quantile(0.25)) & 
                (momentum_series <= momentum_series.quantile(0.75))
            ].index.tolist()
        }
        
        return clusters
        
    def calculate_momentum_half_life(self, symbol: str) -> float:
        """
        估计动量半衰期
        
        Args:
            symbol: 品种代码
            
        Returns:
            半衰期（天）
        """
        if symbol not in self.returns:
            return 0.0
            
        returns = self.returns[symbol]
        
        # 计算滚动自相关
        autocorrs = []
        for lag in range(1, min(20, len(returns)//2)):
            autocorr = returns.autocorr(lag=lag)
            autocorrs.append(autocorr)
            
        if not autocorrs or max(autocorrs) < 0:
            return 0.0
            
        # 半衰期估计
        avg_autocorr = np.mean(autocorrs)
        half_life = -1 / np.log2(avg_autocorr) if avg_autocorr > 0 else 0
        
        return half_life
        
    def generate_momentum_report(self) -> Dict:
        """生成完整动量分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'ranking_period': self.ranking_period,
                'formation_period': self.formation_period,
                'rebalance_period': self.rebalance_period
            },
            'momentum_ranking': self.rank_cross_sectional().to_dict(),
            'signals': self.generate_signals(),
            'clusters': self.momentum_clustering(),
            'decay_analysis': {},
            'half_life': {}
        }
        
        # 衰减分析
        for symbol in self.price_data.keys():
            report['decay_analysis'][symbol] = self.calculate_momentum_decay(symbol)
            report['half_life'][symbol] = self.calculate_momentum_half_life(symbol)
            
        return report


class MomentumRotation:
    """动量轮动策略"""
    
    def __init__(self, symbols: List[str], lookback: int = 20,
                 n_select: int = 2):
        """
        初始化轮动策略
        
        Args:
            symbols: 品种列表
            lookback: 回看周期
            n_select: 选择动量最强的品种数量
        """
        self.symbols = symbols
        self.lookback = lookback
        self.n_select = n_select
        
        self.price_data = {}
        self.historical_positions = []
        
    def add_price(self, symbol: str, price: float, timestamp: datetime):
        """添加价格数据"""
        if symbol not in self.price_data:
            self.price_data[symbol] = []
            
        self.price_data[symbol].append({'price': price, 'timestamp': timestamp})
        
    def should_rotate(self) -> bool:
        """判断是否需要轮动"""
        # 简化：每次调用都返回True进行重新选择
        return True
        
    def select_top_momentum(self) -> List[str]:
        """选择动量最强的品种"""
        momenta = {}
        
        for symbol in self.symbols:
            if symbol not in self.price_data or len(self.price_data[symbol]) < self.lookback:
                continue
                
            prices = [p['price'] for p in self.price_data[symbol][-self.lookback:]]
            returns = np.diff(prices) / prices[:-1]
            
            # 动量 = 累计收益率
            momentum = (1 + returns).prod() - 1
            momenta[symbol] = momentum
            
        if not momenta:
            return []
            
        # 排序并返回前N个
        sorted_symbols = sorted(momenta.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_symbols[:self.n_select]]


def demo():
    """演示用法"""
    # 模拟数据
    np.random.seed(42)
    symbols = ['SHFE.rb', 'SHFE.hc', 'DCE.i', 'DCE.j', 'CZCE.ma', 
               'CZCE.sf', 'SHFE.fu', 'SHFE.cu']
    
    n_days = 252
    prices_dict = {}
    
    for symbol in symbols:
        # 生成具有不同动量特征的价格
        base_price = np.random.uniform(1000, 10000)
        trend = np.random.uniform(-0.001, 0.003)  # 随机趋势
        prices = base_price * np.exp(
            np.cumsum(np.random.normal(trend, 0.02, n_days))
        )
        prices_dict[symbol] = pd.Series(prices)
    
    # 创建分析器
    analyzer = CrossSectionalMomentum(
        ranking_period=20,
        formation_period=60,
        rebalance_period=5
    )
    
    # 添加数据
    for symbol, prices in prices_dict.items():
        analyzer.add_price_series(symbol, prices)
    
    # 生成报告
    report = analyzer.generate_momentum_report()
    
    print("=" * 60)
    print("截面动量分析报告")
    print("=" * 60)
    print(f"分析时间: {report['timestamp']}")
    print()
    
    # 动量排名
    print("动量排名:")
    ranking = report['momentum_ranking']
    sorted_ranking = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    for i, (symbol, rank) in enumerate(sorted_ranking[:5], 1):
        print(f"  {i}. {symbol}: {ranking[symbol]:.4f}")
    print()
    
    # 交易信号
    signals = report['signals']
    print("交易信号:")
    print(f"  做多: {', '.join(signals['long'])}")
    print(f"  做空: {', '.join(signals['short'])}")
    print(f"  动量价差: {signals['momentum_spread']:.4f}")
    print()
    
    # 聚类
    print("动量聚类:")
    clusters = report['clusters']
    print(f"  强动量: {', '.join(clusters['strong_momentum'])}")
    print(f"  弱动量: {', '.join(clusters['weak_momentum'])}")
    print(f"  中性: {', '.join(clusters['neutral'])}")
    print()
    
    # 半衰期
    print("动量半衰期 (天):")
    half_life = report['half_life']
    for symbol, hl in list(half_life.items())[:5]:
        print(f"  {symbol}: {hl:.1f}")
        
    print("=" * 60)
    
    return report


if __name__ == '__main__':
    demo()
