#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多因子分析工具 (Multi-Factor Analysis Tool)
支持Alpha因子、风险因子的构建与分析，因子有效性检验

Author: TqSdk Research
Update: 2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from scipy import stats
import warnings


class Factor:
    """因子基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.values = None
        self.returns = None
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class MomentumFactor(Factor):
    """动量因子"""
    
    def __init__(self, period: int = 20):
        super().__init__(f"momentum_{period}")
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].pct_change(self.period)


class VolatilityFactor(Factor):
    """波动率因子"""
    
    def __init__(self, period: int = 20):
        super().__init__(f"volatility_{period}")
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['close'].pct_change().rolling(self.period).std()


class VolumeFactor(Factor):
    """成交量因子"""
    
    def __init__(self, period: int = 20):
        super().__init__(f"volume_{period}")
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['volume'].rolling(self.period).mean()


class TurnoverFactor(Factor):
    """换手率因子"""
    
    def __init__(self, period: int = 20):
        super().__init__(f"turnover_{period}")
        self.period = period
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        return data['volume'].pct_change().abs().rolling(self.period).mean()


class MultiFactorAnalyzer:
    """多因子分析器"""
    
    def __init__(self):
        self.factors = {}
        self.factor_data = pd.DataFrame()
        self.returns = None
        
    def add_factor(self, factor: Factor, data: pd.DataFrame):
        """添加因子"""
        self.factors[factor.name] = factor
        self.factor_data[factor.name] = factor.calculate(data)
        
    def set_returns(self, returns: pd.Series):
        """设置收益序列"""
        self.returns = returns
        
    def ic_analysis(self, factor_name: str, n_groups: int = 10) -> Dict:
        """
        IC（信息系数）分析
        
        Args:
            factor_name: 因子名称
            n_groups: 分组数量
            
        Returns:
            IC分析结果
        """
        if factor_name not in self.factor_data.columns:
            raise ValueError(f"因子 {factor_name} 不存在")
            
        factor_values = self.factor_data[factor_name].dropna()
        aligned_returns = self.returns.reindex(factor_values.index).dropna()
        
        # 对齐
        common_idx = factor_values.index.intersection(aligned_returns.index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = aligned_returns.loc[common_idx]
        
        # IC
        ic = factor_aligned.corr(returns_aligned)
        
        # 滚动IC
        rolling_ic = pd.Series([
            factor_aligned.iloc[:i].corr(returns_aligned.iloc[:i])
            for i in range(20, len(factor_aligned))
        ], index=factor_aligned.index[20:])
        
        return {
            'ic': ic,
            'ic_mean': rolling_ic.mean(),
            'ic_std': rolling_ic.std(),
            'ic_ir': rolling_ic.mean() / rolling_ic.std() if rolling_ic.std() > 0 else 0,
            'rank_ic': factor_aligned.rank().corr(returns_aligned.rank()),
            'p_value': stats.pearsonr(factor_aligned, returns_aligned)[1]
        }
        
    def group_analysis(self, factor_name: str, n_groups: int = 5) -> Dict:
        """
        分组回测分析
        
        Args:
            factor_name: 因子名称
            n_groups: 分组数量
            
        Returns:
            分组收益
        """
        factor_values = self.factor_data[factor_name].dropna()
        aligned_returns = self.returns.reindex(factor_values.index).dropna()
        
        common_idx = factor_values.index.intersection(aligned_returns.index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = aligned_returns.loc[common_idx]
        
        # 分组
        factor_aligned_copy = factor_aligned.copy()
        factor_aligned_copy['group'] = pd.qcut(factor_aligned_copy, n_groups, labels=False, duplicates='drop')
        
        group_returns = {}
        for group in range(n_groups):
            group_mask = factor_aligned_copy['group'] == group
            group_returns[f'Group_{group+1}'] = returns_aligned[group_mask].mean()
            
        # 多空组合
        long_short = group_returns[f'Group_{n_groups}'] - group_returns['Group_1']
        
        return {
            'group_returns': group_returns,
            'long_short_return': long_short,
            'group_count': factor_aligned_copy['group'].value_counts().sort_index().to_dict()
        }
        
    def factor_correlation(self) -> pd.DataFrame:
        """计算因子间相关性"""
        return self.factor_data.corr()
        
    def orthogonalize_factor(self, factor_name: str, 
                             control_factors: List[str]) -> pd.Series:
        """
        因子正交化（对其他因子回归取残差）
        
        Args:
            factor_name: 待正交化因子
            control_factors: 控制变量
            
        Returns:
            正交化后的因子值
        """
        y = self.factor_data[factor_name]
        X = self.factor_data[control_factors]
        
        # 添加常数项
        X = sm.add_constant(X)
        
        # OLS回归
        model = sm.OLS(y, X).fit()
        
        # 残差即为正交化后的因子
        return model.resid
        
    def calculate_factor_returns(self) -> pd.DataFrame:
        """计算因子收益率（基于Fama-MacBeth回归）"""
        factor_names = list(self.factors.keys())
        
        # 标准化因子
        factor_std = (self.factor_data - self.factor_data.mean()) / self.factor_data.std()
        
        # 回归
        results = []
        for idx in factor_std.index:
            y = self.returns.loc[idx]
            X = factor_std.loc[idx]
            
            # 简单线性回归
            valid = y.notna() & X.notna()
            if valid.sum() < 10:
                continue
                
            slope = X[valid].corr(y[valid]) * y[valid].std() / X[valid].std() if X[valid].std() > 0 else 0
            results.append({'date': idx, 'factor_return': slope})
            
        return pd.DataFrame(results).set_index('date')
        
    def generate_factor_report(self) -> Dict:
        """生成完整因子分析报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'factors': list(self.factors.keys()),
            'factor_correlation': self.factor_correlation().to_dict(),
            'analysis': {}
        }
        
        for factor_name in self.factors.keys():
            try:
                report['analysis'][factor_name] = {
                    'ic': self.ic_analysis(factor_name),
                    'groups': self.group_analysis(factor_name)
                }
            except Exception as e:
                report['analysis'][factor_name] = {'error': str(e)}
                
        return report


class FactorOptimizer:
    """因子优化器"""
    
    def __init__(self, target_return: float = 0.0):
        self.target_return = target_return
        self.factor_returns = None
        self.factor_cov = None
        
    def optimize_weights(self, factor_returns: pd.Series,
                         factor_cov: pd.DataFrame,
                         long_only: bool = True) -> pd.Series:
        """
        优化因子权重
        
        Args:
            factor_returns: 因子收益率
            factor_cov: 因子协方差矩阵
            long_only: 是否只做多
            
        Returns:
            最优权重
        """
        try:
            import cvxpy as cp
            
            n = len(factor_returns)
            weights = cp.Variable(n)
            
            # 目标：最大化因子暴露带来的收益
            portfolio_return = factor_returns.values @ weights
            portfolio_risk = cp.quad_form(weights, factor_cov.values)
            
            # 优化目标：最大化夏普比率
            objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
            
            constraints = []
            
            if long_only:
                constraints.append(weights >= 0)
                
            # 权重和为1
            constraints.append(cp.sum(weights) == 1)
            
            # 求解
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                return pd.Series(weights.value, index=factor_returns.index)
            else:
                # 返回等权重
                return pd.Series(1/n, index=factor_returns.index)
                
        except ImportError:
            # 如果没有cvxpy，返回等权重
            return pd.Series(1/len(factor_returns), index=factor_returns.index)


def demo():
    """演示用法"""
    # 生成模拟数据
    np.random.seed(42)
    n_days = 252
    symbols = ['SHFE.rb', 'SHFE.hc', 'DCE.i', 'DCE.j', 'CZCE.ma']
    
    data_dict = {}
    for symbol in symbols:
        prices = 5000 + np.cumsum(np.random.normal(10, 100, n_days))
        volumes = np.random.lognormal(15, 0.5, n_days)
        
        df = pd.DataFrame({
            'symbol': symbol,
            'datetime': pd.date_range('2024-01-01', periods=n_days, freq='D'),
            'open': prices + np.random.normal(0, 20, n_days),
            'high': prices + np.abs(np.random.normal(50, 30, n_days)),
            'low': prices - np.abs(np.random.normal(50, 30, n_days)),
            'close': prices,
            'volume': volumes
        })
        data_dict[symbol] = df
    
    # 创建分析器
    analyzer = MultiFactorAnalyzer()
    
    # 添加因子（使用第一个品种演示）
    data = data_dict['SHFE.rb']
    analyzer.add_factor(MomentumFactor(20), data)
    analyzer.add_factor(VolatilityFactor(20), data)
    analyzer.add_factor(VolumeFactor(20), data)
    
    # 设置收益率
    returns = data['close'].pct_change().dropna()
    analyzer.set_returns(returns)
    
    # IC分析
    print("=" * 60)
    print("多因子分析报告")
    print("=" * 60)
    print()
    
    for factor_name in analyzer.factors.keys():
        ic_result = analyzer.ic_analysis(factor_name)
        print(f"因子: {factor_name}")
        print(f"  IC: {ic_result['ic']:.4f}")
        print(f"  IC均值: {ic_result['ic_mean']:.4f}")
        print(f"  IC_IR: {ic_result['ic_ir']:.4f}")
        print(f"  Rank IC: {ic_result['rank_ic']:.4f}")
        print(f"  P值: {ic_result['p_value']:.4f}")
        print()
        
    # 分组分析
    print("分组收益分析:")
    group_result = analyzer.group_analysis('momentum_20', n_groups=5)
    for group, ret in group_result['group_returns'].items():
        print(f"  {group}: {ret*100:.4f}%")
    print(f"  多空组合: {group_result['long_short_return']*100:.4f}%")
    print()
    
    # 因子相关性
    print("因子相关性矩阵:")
    print(analyzer.factor_correlation())
    print()
    
    print("=" * 60)
    
    return analyzer.generate_factor_report()


if __name__ == '__main__':
    demo()
