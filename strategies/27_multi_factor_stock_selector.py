#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 27
策略名称: 多因子选股模型
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
本策略是一个多因子选股模型，通过构建和回测多个因子来识别具有超额收益
潜力的期货品种。因子类型包括动量、波动率、成交量、期限结构等。

主要因子：
1. 动量因子 - 过去收益率的加权平均
2. 波动率因子 - 过去波动率的倒数
3. 成交量因子 - 成交量异常变化
4. 期限结构因子 - 远近月价差
5. 持仓变化因子 - 持仓量变化

【策略参数】
- SYMBOLS: 候选品种列表
- LOOKBACK_PERIOD: 因子计算回看周期
- FACTOR_WEIGHTS: 各因子权重
- TOP_N: 选取排名前N的品种
- REBALANCE_FREQ: 调仓频率（小时）

【风险提示】
- 本策略仅供研究学习使用
- 过往业绩不代表未来表现
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============ 参数配置 ============
SYMBOLS = [
    "SHFE.rb2405", "SHFE.hc2405", "SHFE.cu2405", "SHFE.al2405", "SHFE.zn2405",
    "DCE.m2405", "DCE.y2405", "DCE.j2405", "DCE.jm2405", "DCE.p2405",
    "CZCE.SR2405", "CZCE.CF2405", "CZCE.MA2405", "CZCE.TA2405", "CZCE.ZM2405",
    "CFFEX.IF2405", "CFFEX.IC2405", "CFFEX.IH2405",
]
LOOKBACK_PERIOD = 60        # 回看周期（分钟）
TOP_N = 5                   # 选取前N名
REBALANCE_FREQ = 24        # 调仓频率（小时）

# 因子权重
MOMENTUM_WEIGHT = 0.25      # 动量因子权重
VOLATILITY_WEIGHT = 0.20    # 波动率因子权重
VOLUME_WEIGHT = 0.15        # 成交量因子权重
TERM_STRUCTURE_WEIGHT = 0.20  # 期限结构权重
POSITION_WEIGHT = 0.20      # 持仓变化权重


class MultiFactorModel:
    """多因子选股模型"""
    
    def __init__(self, api):
        self.api = api
        self.symbols = SYMBOLS
        self.lookback = LOOKBACK_PERIOD
        self.factor_weights = {
            'momentum': MOMENTUM_WEIGHT,
            'volatility': VOLATILITY_WEIGHT,
            'volume': VOLUME_WEIGHT,
            'term_structure': TERM_STRUCTURE_WEIGHT,
            'position': POSITION_WEIGHT
        }
        
    def get_historical_data(self, symbol, count):
        """获取历史K线数据"""
        try:
            klines = self.api.get_kline_serial(symbol, 60, count)
            if klines is not None and len(klines) > 0:
                return klines
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
        return None
    
    def calculate_momentum_factor(self, closes):
        """计算动量因子"""
        if len(closes) < 20:
            return 0
        
        # 计算不同周期的动量
        returns_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 6 else 0
        returns_10d = (closes[-1] - closes[-11]) / closes[-11] if len(closes) > 11 else 0
        returns_20d = (closes[-1] - closes[-21]) / closes[-21] if len(closes) > 21 else 0
        
        # 加权动量
        momentum = 0.5 * returns_5d + 0.3 * returns_10d + 0.2 * returns_20d
        
        return momentum
    
    def calculate_volatility_factor(self, returns):
        """计算波动率因子（低波动率更好）"""
        if len(returns) < 20:
            return 0
        
        volatility = np.std(returns)
        
        # 波动率倒数（低波动率高分数）
        if volatility > 0:
            inv_vol = 1.0 / volatility
            # 归一化
            return min(1.0, inv_vol * 10)
        
        return 0
    
    def calculate_volume_factor(self, volumes):
        """计算成交量因子（成交量放大更好）"""
        if len(volumes) < 20:
            return 0
        
        # 计算近期成交量变化
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            # 归一化
            return min(1.0, volume_ratio / 2)
        
        return 0
    
    def calculate_term_structure_factor(self, symbol):
        """计算期限结构因子"""
        # 获取近月和远月数据
        # 这里简化处理，实际需要获取不同到期日的合约
        try:
            # 尝试获取当前合约的K线
            klines = self.api.get_kline_serial(symbol, 60*60*24, 5)  # 日K
            if klines is None or len(klines) < 5:
                return 0
            
            closes = list(klines.get('close', []))
            if len(closes) < 5:
                return 0
            
            # 简单期限结构：近期价格变化趋势
            # 上涨的期限结构（contango）通常表示需求旺盛
            trend = (closes[-1] - closes[-5]) / closes[-5]
            
            return trend
            
        except Exception as e:
            print(f"期限结构计算失败: {e}")
            return 0
    
    def calculate_position_factor(self, klines):
        """计算持仓变化因子"""
        try:
            if 'close' not in klines or 'volume' not in klines:
                return 0
            
            closes = np.array(klines.get('close', []))
            volumes = np.array(klines.get('volume', []))
            
            if len(closes) < 20 or len(volumes) < 20:
                return 0
            
            # 价格变化与成交量变化的关系
            price_change = (closes[-1] - closes[-20]) / closes[-20]
            
            # 成交量变化
            volume_change = (volumes[-1] - volumes[-20]) / volumes[-20] if volumes[-20] > 0 else 0
            
            # 持仓变化因子：价量齐涨为正
            position_factor = price_change * np.sign(volume_change)
            
            return position_factor
            
        except Exception as e:
            return 0
    
    def calculate_composite_score(self, symbol):
        """计算综合因子得分"""
        klines = self.get_historical_data(symbol, self.lookback)
        
        if klines is None or len(klines) < 20:
            return None, {}
        
        closes = np.array(klines.get('close', []))
        volumes = np.array(klines.get('volume', []))
        
        if len(closes) < 20:
            return None, {}
        
        # 计算收益率序列
        returns = np.diff(closes) / closes[:-1]
        
        # 计算各因子
        momentum = self.calculate_momentum_factor(closes)
        volatility = self.calculate_volatility_factor(returns)
        volume = self.calculate_volume_factor(volumes)
        term_structure = self.calculate_term_structure_factor(symbol)
        position = self.calculate_position_factor(klines)
        
        # 标准化各因子（简单处理）
        # 动量因子：越大越好
        momentum_norm = max(0, min(1, (momentum + 0.1) / 0.2))
        
        # 波动率因子：越低越好（已经是倒数形式）
        volatility_norm = volatility
        
        # 成交量因子：越大越好
        volume_norm = volume
        
        # 期限结构因子：标准化
        term_norm = max(0, min(1, (term_structure + 0.1) / 0.2))
        
        # 持仓因子：标准化
        position_norm = max(0, min(1, (position + 0.1) / 0.2))
        
        # 加权计算综合得分
        composite_score = (
            momentum_norm * self.factor_weights['momentum'] +
            volatility_norm * self.factor_weights['volatility'] +
            volume_norm * self.factor_weights['volume'] +
            term_norm * self.factor_weights['term_structure'] +
            position_norm * self.factor_weights['position']
        )
        
        factors = {
            'momentum': round(momentum_norm * 100, 1),
            'volatility': round(volatility_norm * 100, 1),
            'volume': round(volume_norm * 100, 1),
            'term_structure': round(term_norm * 100, 1),
            'position': round(position_norm * 100, 1)
        }
        
        return composite_score, factors
    
    def run_analysis(self):
        """运行多因子分析"""
        print("=" * 60)
        print("多因子选股模型")
        print("=" * 60)
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"候选品种: {len(self.symbols)} 个")
        print(f"回看周期: {self.lookback} 分钟")
        print(f"选取数量: TOP {TOP_N}")
        print("-" * 60)
        
        results = {}
        
        for symbol in self.symbols:
            score, factors = self.calculate_composite_score(symbol)
            if score is not None:
                results[symbol] = {
                    'score': score,
                    'factors': factors
                }
        
        # 排序
        sorted_results = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
        
        print(f"\n【因子权重】")
        print(f"  动量因子: {MOMENTUM_WEIGHT*100:.0f}%")
        print(f"  波动率因子: {VOLATILITY_WEIGHT*100:.0f}%")
        print(f"  成交量因子: {VOLUME_WEIGHT*100:.0f}%")
        print(f"  期限结构因子: {TERM_STRUCTURE_WEIGHT*100:.0f}%")
        print(f"  持仓变化因子: {POSITION_WEIGHT*100:.0f}%")
        
        print(f"\n【排名结果】")
        print("-" * 60)
        
        for i, (symbol, data) in enumerate(sorted_results[:TOP_N], 1):
            print(f"\n第{i}名: {symbol}")
            print(f"  综合得分: {data['score']*100:.1f}/100")
            print(f"  动量因子: {data['factors']['momentum']}")
            print(f"  波动率因子: {data['factors']['volatility']}")
            print(f"  成交量因子: {data['factors']['volume']}")
            print(f"  期限结构: {data['factors']['term_structure']}")
            print(f"  持仓变化: {data['factors']['position']}")
        
        # 建议
        print("\n" + "=" * 60)
        print("选股建议")
        print("=" * 60)
        top_symbols = [s for s, _ in sorted_results[:TOP_N]]
        print(f"推荐做多品种: {', '.join(top_symbols)}")
        print(f"调仓频率: 每 {REBALANCE_FREQ} 小时")
        
        return sorted_results[:TOP_N]


def main():
    api = TqApi(auth=TqAuth("账号", "密码"))
    
    print(f"多因子选股模型启动")
    
    model = MultiFactorModel(api)
    results = model.run_analysis()
    
    api.close()


if __name__ == "__main__":
    main()
