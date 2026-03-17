#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子收益率分析工具 (Factor Returns Analyzer)
分析不同Alpha因子的收益率和有效性，支持因子衰减检测

Author: TqSdk Research
Update: 2026-03-13
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats, optimize
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FactorReturnsAnalyzer:
    """因子收益率分析器"""
    
    def __init__(self, 
                 formation_period: int = 60,
                 ranking_period: int = 20,
                 top_pct: float = 0.2,
                 bottom_pct: float = 0.2):
        """
        初始化分析器
        
        Args:
            formation_period: 因子形成期（天）
            ranking_period: 因子排名期（天）
            top_pct: 多头头部比例
            bottom_pct: 空头尾部比例
        """
        self.formation_period = formation_period
        self.ranking_period = ranking_period
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        
        self.factor_data = defaultdict(dict)  # 因子数据
        self.returns_data = defaultdict(dict)  # 收益率数据
        self.factor_results = {}
        
    def add_factor_data(self, factor_name: str, 
                       symbol: str, 
                       factor_value: float,
                       date: str):
        """添加因子数据"""
        if factor_name not in self.factor_data:
            self.factor_data[factor_name] = {}
            
        if date not in self.factor_data[factor_name]:
            self.factor_data[factor_name][date] = {}
            
        self.factor_data[factor_name][date][symbol] = factor_value
        
    def add_return_data(self, symbol: str, 
                       period_return: float,
                       date: str):
        """添加收益率数据"""
        if date not in self.returns_data:
            self.returns_data[date] = {}
            
        self.returns_data[date][symbol] = period_return
        
    def calculate_factor_returns(self, factor_name: str) -> Dict:
        """
        计算因子收益率
        
        Args:
            factor_name: 因子名称
            
        Returns:
            因子收益率分析结果
        """
        if factor_name not in self.factor_data:
            return {}
            
        dates = sorted(self.factor_data[factor_name].keys())
        if len(dates) < self.formation_period + self.ranking_period:
            return {'status': 'INSUFFICIENT_DATA'}
            
        long_returns = []
        short_returns = []
        spread_returns = []
        
        for i in range(self.ranking_period, len(dates) - 1):
            formation_start = i - self.formation_period
            formation_end = i
            ranking_end = i + self.ranking_period
            
            # 获取形成期因子值
            formation_factors = {}
            for d in dates[formation_start:formation_end]:
                formation_factors.update(self.factor_data[factor_name].get(d, {}))
                
            if not formation_factors:
                continue
                
            # 排序
            sorted_symbols = sorted(formation_factors.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            n = len(sorted_symbols)
            n_long = max(1, int(n * self.top_pct))
            n_short = max(1, int(n * self.bottom_pct))
            
            long_symbols = [s[0] for s in sorted_symbols[:n_long]]
            short_symbols = [s[0] for s in sorted_symbols[-n_short:]]
            
            # 获取持有期收益
            holding_date = dates[min(ranking_end, len(dates) - 1)]
            holding_returns = self.returns_data.get(holding_date, {})
            
            if not holding_returns:
                continue
                
            # 计算多空收益
            long_ret = np.mean([holding_returns.get(s, 0) for s in long_symbols])
            short_ret = np.mean([holding_returns.get(s, 0) for s in short_symbols])
            spread_ret = long_ret - short_ret
            
            long_returns.append(long_ret)
            short_returns.append(short_ret)
            spread_returns.append(spread_ret)
            
        if not spread_returns:
            return {'status': 'NO_DATA'}
            
        long_returns = np.array(long_returns)
        short_returns = np.array(short_returns)
        spread_returns = np.array(spread_returns)
        
        # 计算统计指标
        result = {
            'status': 'SUCCESS',
            'factor_name': factor_name,
            'periods': len(spread_returns),
            
            # 多头统计
            'long_mean': np.mean(long_returns),
            'long_std': np.std(long_returns),
            'long_sharpe': np.mean(long_returns) / max(np.std(long_returns), 1e-10),
            'long_winrate': np.mean(long_returns > 0),
            
            # 空头统计
            'short_mean': np.mean(short_returns),
            'short_std': np.std(short_returns),
            'short_sharpe': np.mean(short_returns) / max(np.std(short_returns), 1e-10),
            'short_winrate': np.mean(short_returns > 0),
            
            # 价差统计
            'spread_mean': np.mean(spread_returns),
            'spread_std': np.std(spread_returns),
            'spread_sharpe': np.mean(spread_returns) / max(np.std(spread_returns), 1e-10),
            'spread_winrate': np.mean(spread_returns > 0),
            
            # t检验
            't_statistic': stats.ttest_1samp(spread_returns, 0)[0],
            'p_value': stats.ttest_1samp(spread_returns, 0)[1]
        }
        
        self.factor_results[factor_name] = result
        return result
        
    def calculate_factor_decay(self, factor_name: str,
                              max_lag: int = 20) -> Dict:
        """
        计算因子衰减特性
        
        Args:
            factor_name: 因子名称
            max_lag: 最大滞后期
            
        Returns:
            因子衰减分析
        """
        if factor_name not in self.factor_data:
            return {}
            
        dates = sorted(self.factor_data[factor_name].keys())
        
        # 计算不同滞后期的IC
        ic_by_lag = []
        
        for lag in range(1, max_lag + 1):
            ics = []
            
            for i in range(lag, len(dates) - 1):
                # 因子值（滞后）
                factor_values = self.factor_data[factor_name].get(dates[i-lag], {})
                
                # 收益率（当前）
                returns = self.returns_data.get(dates[i], {})
                
                if not factor_values or not returns:
                    continue
                    
                # 计算IC
                symbols = set(factor_values.keys()) & set(returns.keys())
                if len(symbols) < 5:
                    continue
                    
                factor_list = [factor_values[s] for s in symbols]
                return_list = [returns[s] for s in symbols]
                
                ic = np.corrcoef(factor_list, return_list)[0, 1]
                ics.append(ic)
                
            if ics:
                ic_by_lag.append({
                    'lag': lag,
                    'ic_mean': np.mean(ics),
                    'ic_std': np.std(ics),
                    'ic_ir': np.mean(ics) / max(np.std(ics), 1e-10),
                    'sample_count': len(ics)
                })
                
        return {
            'factor_name': factor_name,
            'decay_data': ic_by_lag,
            'optimal_lag': max(ic_by_lag, key=lambda x: x['ic_mean'])['lag'] if ic_by_lag else 1,
            'half_life': self._estimate_half_life(ic_by_lag)
        }
        
    def _estimate_half_life(self, ic_by_lag: List[Dict]) -> float:
        """估计因子半衰期"""
        if len(ic_by_lag) < 3:
            return -1
            
        ic_values = [d['ic_mean'] for d in ic_by_lag]
        initial_ic = ic_values[0]
        
        if initial_ic <= 0:
            return -1
            
        for i, ic in enumerate(ic_values[1:], 1):
            if ic <= initial_ic * 0.5:
                return i
                
        return len(ic_by_lag)
        
    def calculate_factor_correlation(self, 
                                    factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算因子间相关性
        
        Args:
            factors: 因子列表（默认全部）
            
        Returns:
            因子相关性矩阵
        """
        if factors is None:
            factors = list(self.factor_data.keys())
            
        if len(factors) < 2:
            return pd.DataFrame()
            
        # 提取每个因子的收益率序列
        factor_returns_series = {}
        
        for factor in factors:
            result = self.calculate_factor_returns(factor)
            if result.get('status') == 'SUCCESS':
                factor_returns_series[factor] = result.get('spread_mean', 0)
                
        if len(factor_returns_series) < 2:
            return pd.DataFrame()
            
        # 计算滚动相关性
        dates = sorted(self.returns_data.keys())
        rolling_corr = defaultdict(list)
        
        window = 20
        for i in range(window, len(dates)):
            window_dates = dates[i-window:i]
            
            for f1 in factor_returns_series.keys():
                for f2 in factor_returns_series.keys():
                    if f1 >= f2:
                        continue
                        
                    # 获取两个因子的信号
                    f1_signals = []
                    f2_signals = []
                    
                    for d in window_dates:
                        f1_data = self.factor_data.get(f1, {}).get(d, {})
                        f2_data = self.factor_data.get(f2, {}).get(d, {})
                        
                        if f1_data and f2_data:
                            common = set(f1_data.keys()) & set(f2_data.keys())
                            if common:
                                f1_signals.append(np.mean([f1_data[s] for s in common]))
                                f2_signals.append(np.mean([f2_data[s] for s in common]))
                                
                    if len(f1_signals) > 5:
                        corr = np.corrcoef(f1_signals, f2_signals)[0, 1]
                        rolling_corr[f"{f1}_vs_{f2}"].append(corr)
                        
        # 返回平均相关性
        corr_matrix = pd.DataFrame(index=factors, columns=factors)
        
        for f1 in factors:
            for f2 in factors:
                if f1 == f2:
                    corr_matrix.loc[f1, f2] = 1.0
                else:
                    key = f1 if f1 < f2 else f2
                    key2 = f2 if f1 < f2 else f1
                    full_key = f"{key}_vs_{key2}"
                    
                    if full_key in rolling_corr:
                        corr_matrix.loc[f1, f2] = np.mean(rolling_corr[full_key])
                    else:
                        corr_matrix.loc[f1, f2] = 0.0
                        
        return corr_matrix.astype(float)
        
    def generate_factor_report(self) -> str:
        """生成因子分析报告"""
        report = """
========================================
        因子收益率分析报告
========================================
"""
        
        for factor_name in self.factor_data.keys():
            result = self.calculate_factor_returns(factor_name)
            decay = self.calculate_factor_decay(factor_name)
            
            if result.get('status') != 'SUCCESS':
                continue
                
            report += f"""
【{result['factor_name']}】
- 统计周期数: {result['periods']}
- 多头平均收益: {result['long_mean']:.4f} ({result['long_mean']*100:.2f}%)
- 空头平均收益: {result['short_mean']:.4f} ({result['short_mean']*100:.2f}%)
- 价差收益: {result['spread_mean']:.4f} ({result['spread_mean']*100:.2f}%)
- 价差夏普: {result['spread_sharpe']:.2f}
- 胜率: {result['spread_winrate']:.2%}
- t统计量: {result['t_statistic']:.2f}
- p值: {result['p_value']:.4f}
- 半衰期: {decay.get('half_life', 'N/A')}天
"""
            
        report += "========================================\n"
        return report


class CrossSectionalRegression:
    """截面回归分析"""
    
    def __init__(self):
        self.factor_exposures = defaultdict(dict)
        self.factor_returns = defaultdict(list)
        
    def add_exposure(self, date: str, symbol: str, 
                    factor_name: str, exposure: float):
        """添加因子暴露"""
        if date not in self.factor_exposures:
            self.factor_exposures[date] = {}
            
        if symbol not in self.factor_exposures[date]:
            self.factor_exposures[date][symbol] = {}
            
        self.factor_exposures[date][symbol][factor_name] = exposure
        
    def run_regression(self, date: str) -> Dict:
        """
        运行截面回归
        
        Args:
            date: 日期
            
        Returns:
            回归结果
        """
        if date not in self.factor_exposures:
            return {}
            
        symbols = list(self.factor_exposures[date].keys())
        
        # 构建因子矩阵
        factors = set()
        for symbol in symbols:
            factors.update(self.factor_exposures[date][symbol].keys())
            
        factors = sorted(list(factors))
        
        X = np.zeros((len(symbols), len(factors)))
        for i, symbol in enumerate(symbols):
            for j, factor in enumerate(factors):
                X[i, j] = self.factor_exposures[date][symbol].get(factor, 0)
                
        # 添加常数项
        X = np.column_stack([np.ones(len(symbols)), X])
        
        # 简化的收益率（使用1作为占位符）
        y = np.zeros(len(symbols))
        
        # 回归
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            return {
                'date': date,
                'factors': factors,
                'betas': beta[1:].tolist(),  # 排除常数项
                'r_squared': 0.0  # 需要收益率数据计算
            }
        except:
            return {}


def main():
    """主函数 - 演示用法"""
    # 创建分析器
    analyzer = FactorReturnsAnalyzer(
        formation_period=60,
        ranking_period=20
    )
    
    # 模拟添加因子数据
    np.random.seed(42)
    symbols = ['SHFE.rb', 'DCE.m', 'CZCE.CF', 'DCE.y', 'CZCE.SR']
    
    for date_idx in range(100):
        date = f"2024-01-{date_idx+1:02d}"
        
        for symbol in symbols:
            # 模拟因子值
            momentum = np.random.randn()
            volatility = np.random.randn()
            volume = np.random.randn()
            
            analyzer.add_factor_data('momentum', symbol, momentum, date)
            analyzer.add_factor_data('volatility', symbol, volatility, date)
            analyzer.add_factor_data('volume', symbol, volume, date)
            
            # 模拟收益率
            ret = momentum * 0.3 + volatility * 0.2 + np.random.randn() * 0.5
            analyzer.add_return_data(symbol, ret, date)
            
    # 计算因子收益
    for factor in ['momentum', 'volatility', 'volume']:
        result = analyzer.calculate_factor_returns(factor)
        if result.get('status') == 'SUCCESS':
            print(f"{factor}: 价差收益={result['spread_mean']:.4f}, 夏普={result['spread_sharpe']:.2f}")
            
    # 计算因子衰减
    decay = analyzer.calculate_factor_decay('momentum')
    print(f"动量因子半衰期: {decay.get('half_life', 'N/A')}天")
    
    # 生成报告
    report = analyzer.generate_factor_report()
    print(report)


if __name__ == "__main__":
    main()
