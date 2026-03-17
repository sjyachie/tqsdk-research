#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
市场状态检测工具 (Market Regime Detector)
基于隐马尔可夫模型和统计检验的市场状态识别

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


class MarketRegimeDetector:
    """市场状态检测器"""
    
    def __init__(self, 
                 lookback_period: int = 60,
                 regimes: Optional[List[str]] = None):
        """
        初始化检测器
        
        Args:
            lookback_period: 回看周期
            regimes: 市场状态列表
        """
        self.lookback_period = lookback_period
        
        if regimes is None:
            self.regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL', 'LOW_VOL']
        else:
            self.regimes = regimes
            
        self.price_history = []
        self.volume_history = []
        self.detected_regimes = []
        
    def add_market_data(self, close: float, 
                       volume: float,
                       high: Optional[float] = None,
                       low: Optional[float] = None):
        """添加市场数据"""
        self.price_history.append(close)
        self.volume_history.append(volume)
        
        if len(self.price_history) > self.lookback_period * 2:
            self.price_history.pop(0)
            self.volume_history.pop(0)
            
    def calculate_returns(self, period: int = 1) -> np.ndarray:
        """计算收益率序列"""
        if len(self.price_history) < period + 1:
            return np.array([])
            
        prices = np.array(self.price_history[-self.lookback_period:])
        returns = np.diff(prices) / prices[:-1]
        
        return returns[::period]
        
    def calculate_volatility(self, period: Optional[int] = None) -> float:
        """计算波动率"""
        if period is None:
            period = self.lookback_period
            
        returns = self.calculate_returns()
        
        if len(returns) < 10:
            return 0.0
            
        return np.std(returns[-period:]) * np.sqrt(252)  # 年化
        
    def calculate_trend_strength(self) -> float:
        """
        计算趋势强度
        
        Returns:
            趋势强度指标 (-1 到 1)
        """
        prices = np.array(self.price_history[-self.lookback_period:])
        
        if len(prices) < 10:
            return 0.0
            
        # 使用线性回归斜率
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # 标准化斜率
        trend_strength = slope / np.mean(prices) * len(prices)
        
        # 限制在 -1 到 1
        return np.clip(trend_strength, -1, 1)
        
    def calculate_volume_trend(self) -> float:
        """
        计算成交量趋势
        
        Returns:
            成交量趋势 (相对均值的比例)
        """
        if len(self.volume_history) < 20:
            return 1.0
            
        recent = np.array(self.volume_history[-10:])
        historical = np.array(self.volume_history[-30:-10])
        
        if len(historical) == 0:
            return 1.0
            
        return np.mean(recent) / max(np.mean(historical), 1)
        
    def detect_bull_bear(self) -> str:
        """检测牛熊状态"""
        returns = self.calculate_returns()
        
        if len(returns) < 20:
            return 'UNKNOWN'
            
        recent = returns[-10:]
        older = returns[-20:-10]
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        # t检验
        if len(recent) > 3 and len(older) > 3:
            t_stat, p_value = stats.ttest_ind(recent, older)
            
            if p_value < 0.1:
                if recent_mean > older_mean:
                    return 'BULL'
                else:
                    return 'BEAR'
                    
        # 简化判断
        if recent_mean > 0.001:
            return 'BULL'
        elif recent_mean < -0.001:
            return 'BEAR'
            
        return 'SIDEWAYS'
        
    def detect_volatility_regime(self) -> str:
        """检测波动率状态"""
        vol = self.calculate_volatility()
        
        # 基于历史波动率判断
        all_returns = self.calculate_returns()
        
        if len(all_returns) < 30:
            return 'UNKNOWN'
            
        historical_vol = np.std(all_returns) * np.sqrt(252)
        
        if vol > historical_vol * 1.3:
            return 'HIGH_VOL'
        elif vol < historical_vol * 0.7:
            return 'LOW_VOL'
            
        return 'NORMAL_VOL'
        
    def detect_momentum_regime(self) -> str:
        """检测动量状态"""
        returns = self.calculate_returns()
        
        if len(returns) < 20:
            return 'UNKNOWN'
            
        # 计算滞后相关性
        recent = returns[-10:]
        
        if len(recent) < 5:
            return 'UNKNOWN'
            
        # 自相关
        autocorr = np.corrcoef(recent[:-1], recent[1:])[0, 1]
        
        if np.isnan(autocorr):
            return 'UNKNOWN'
            
        if autocorr > 0.3:
            return 'TRENDING'
        elif autocorr < -0.3:
            return 'MEAN_REVERTING'
            
        return 'RANDOM'
        
    def detect_market_regime(self) -> Dict:
        """
        检测当前市场状态
        
        Returns:
            市场状态分析结果
        """
        if len(self.price_history) < 30:
            return {
                'status': 'WARMUP',
                'message': f'需要更多数据 ({len(self.price_history)}/{30})'
            }
            
        # 各项检测
        trend_strength = self.calculate_trend_strength()
        volatility = self.calculate_volatility()
        volume_trend = self.calculate_volume_trend()
        bull_bear = self.detect_bull_bear()
        vol_regime = self.detect_volatility_regime()
        momentum_regime = self.detect_momentum_regime()
        
        # 综合判断
        regime_scores = {
            'BULL': 0,
            'BEAR': 0,
            'SIDEWAYS': 0,
            'HIGH_VOL': 0,
            'LOW_VOL': 0
        }
        
        if bull_bear == 'BULL':
            regime_scores['BULL'] += 2
        elif bull_bear == 'BEAR':
            regime_scores['BEAR'] += 2
            
        if vol_regime == 'HIGH_VOL':
            regime_scores['HIGH_VOL'] += 1
        elif vol_regime == 'LOW_VOL':
            regime_scores['LOW_VOL'] += 1
            
        if abs(trend_strength) > 0.5:
            if trend_strength > 0:
                regime_scores['BULL'] += 1
            else:
                regime_scores['BEAR'] += 1
        else:
            regime_scores['SIDEWAYS'] += 1
            
        # 确定主要状态
        primary_regime = max(regime_scores, key=regime_scores.get)
        
        # 生成信号
        if primary_regime == 'BULL' and vol_regime != 'HIGH_VOL':
            signal = 'BUY'
        elif primary_regime == 'BEAR' or vol_regime == 'HIGH_VOL':
            signal = 'SELL'
        elif primary_regime == 'SIDEWAYS':
            signal = 'NEUTRAL'
        else:
            signal = 'MONITOR'
            
        result = {
            'status': 'SUCCESS',
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.price_history),
            
            # 指标
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volume_trend': volume_trend,
            
            # 状态
            'bull_bear': bull_bear,
            'volatility_regime': vol_regime,
            'momentum_regime': momentum_regime,
            
            # 综合
            'primary_regime': primary_regime,
            'regime_scores': regime_scores,
            'signal': signal,
            
            # 置信度
            'confidence': regime_scores[primary_regime] / sum(regime_scores.values())
        }
        
        self.detected_regimes.append(result)
        
        return result
        
    def calculate_regime_stability(self) -> float:
        """计算状态稳定性"""
        if len(self.detected_regimes) < 10:
            return 0.0
            
        recent = self.detected_regimes[-10:]
        regimes = [r.get('primary_regime', 'UNKNOWN') for r in recent]
        
        # 计算最常见状态的比例
        from collections import Counter
        counts = Counter(regimes)
        
        return counts.most_common(1)[0][1] / len(regimes)
        
    def predict_regime_transition(self) -> Dict:
        """预测状态转换"""
        if len(self.detected_regimes) < 20:
            return {'status': 'INSUFFICIENT_DATA'}
            
        # 简化的马尔可夫转换概率
        recent = self.detected_regimes[-20:]
        
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(1, len(recent)):
            current = recent[i].get('primary_regime', 'UNKNOWN')
            previous = recent[i-1].get('primary_regime', 'UNKNOWN')
            transitions[previous][current] += 1
            
        # 转换为概率
        current_regime = recent[-1].get('primary_regime', 'UNKNOWN')
        
        transition_probs = {}
        total = sum(transitions[current_regime].values())
        
        if total > 0:
            for regime, count in transitions[current_regime].items():
                transition_probs[regime] = count / total
        else:
            # 默认均匀分布
            for regime in self.regimes:
                transition_probs[regime] = 1.0 / len(self.regimes)
                
        return {
            'current_regime': current_regime,
            'transition_probabilities': transition_probs,
            'most_likely_next': max(transition_probs, key=transition_probs.get),
            'stability': self.calculate_regime_stability()
        }


class RegimeBasedStrategy:
    """基于市场状态的策略框架"""
    
    def __init__(self):
        self.detector = MarketRegimeDetector()
        self.strategy_params = {}
        
    def set_strategy_params(self, regime: str, params: Dict):
        """设置不同状态下的策略参数"""
        self.strategy_params[regime] = params
        
    def get_position_size(self, base_size: float) -> float:
        """根据市场状态调整仓位"""
        regime_info = self.detector.detect_market_regime()
        
        if regime_info.get('status') != 'SUCCESS':
            return base_size * 0.5  # 数据不足时减半
            
        regime = regime_info.get('primary_regime', 'SIDEWAYS')
        signal = regime_info.get('signal', 'NEUTRAL')
        
        # 基础仓位调整
        position_multipliers = {
            'BULL': 1.0,
            'BEAR': 0.3,
            'SIDEWAYS': 0.6,
            'HIGH_VOL': 0.4,
            'LOW_VOL': 0.8
        }
        
        multiplier = position_multipliers.get(regime, 0.5)
        
        # 信号调整
        if signal == 'BUY':
            multiplier *= 1.2
        elif signal == 'SELL':
            multiplier *= 0.5
        elif signal == 'NEUTRAL':
            multiplier *= 0.8
            
        # 波动率调整
        vol = regime_info.get('volatility', 0.2)
        if vol > 0.3:
            multiplier *= 0.7
        elif vol < 0.15:
            multiplier *= 1.1
            
        return base_size * np.clip(multiplier, 0.1, 1.5)
        
    def get_stop_loss(self, entry_price: float, 
                     direction: str = 'LONG') -> float:
        """根据市场状态调整止损"""
        regime_info = self.detector.detect_market_regime()
        
        if regime_info.get('status') != 'SUCCESS':
            # 默认2%止损
            return entry_price * 0.98 if direction == 'LONG' else entry_price * 1.02
            
        vol = regime_info.get('volatility', 0.2)
        regime = regime_info.get('primary_regime', 'SIDEWAYS')
        
        # 波动率止损
        vol_stop = vol * 2  # 2倍波动率
        
        # 状态调整
        regime_stops = {
            'BULL': 1.0,
            'BEAR': 1.5,  # 更紧的止损
            'SIDEWAYS': 1.2,
            'HIGH_VOL': 1.5,
            'LOW_VOL': 0.8
        }
        
        stop_pct = vol_stop * regime_stops.get(regime, 1.0)
        
        if direction == 'LONG':
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)


def main():
    """主函数 - 演示用法"""
    # 创建检测器
    detector = MarketRegimeDetector()
    
    # 模拟市场数据
    np.random.seed(42)
    
    # 模拟不同市场状态
    phases = [
        ('BULL', 30),
        ('HIGH_VOL', 15),
        ('SIDEWAYS', 20),
        ('BEAR', 15),
        ('LOW_VOL', 10)
    ]
    
    base_price = 1000
    
    for regime, days in phases:
        for _ in range(days):
            if regime == 'BULL':
                price_change = np.random.normal(0.002, 0.01)
            elif regime == 'BEAR':
                price_change = np.random.normal(-0.003, 0.015)
            elif regime == 'HIGH_VOL':
                price_change = np.random.normal(0, 0.025)
            elif regime == 'LOW_VOL':
                price_change = np.random.normal(0.0005, 0.005)
            else:
                price_change = np.random.normal(0, 0.008)
                
            base_price *= (1 + price_change)
            volume = np.random.randint(50000, 200000)
            
            detector.add_market_data(base_price, volume)
            
    # 检测市场状态
    result = detector.detect_market_regime()
    
    print(f"""
========================================
        市场状态检测结果
========================================
- 主要状态: {result.get('primary_regime')}
- 信号: {result.get('signal')}
- 趋势强度: {result.get('trend_strength', 0):.3f}
- 年化波动率: {result.get('volatility', 0):.2%}
- 成交量趋势: {result.get('volume_trend', 0):.2f}
- 置信度: {result.get('confidence', 0):.2%}

【细分状态】
- 牛熊判断: {result.get('bull_bear')}
- 波动状态: {result.get('volatility_regime')}
- 动量状态: {result.get('momentum_regime')}
""")
    
    # 状态转换预测
    prediction = detector.predict_regime_transition()
    if prediction.get('status') != 'INSUFFICIENT_DATA':
        print(f"""
【状态转换预测】
- 当前状态: {prediction.get('current_regime')}
- 最可能转换: {prediction.get('most_likely_next')}
- 稳定性: {prediction.get('stability', 0):.2%}
- 转换概率: {prediction.get('transition_probabilities')}
""")
    
    # 策略参数调整示例
    strategy = RegimeBasedStrategy()
    strategy.detector = detector
    
    position = strategy.get_position_size(100000)
    print(f"建议仓位: {position:.0f} (基础10万)")
    
    stop_loss = strategy.get_stop_loss(1000, 'LONG')
    print(f"建议止损: {stop_loss:.2f}")


if __name__ == "__main__":
    main()
