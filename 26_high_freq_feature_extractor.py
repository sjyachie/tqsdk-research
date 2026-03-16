#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 26
策略名称: 高频数据特征提取器
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
高频数据特征提取器从Tick级数据中提取微观市场特征，
包括订单流、成交量分布、价格冲击、市场微结构等指标。

【策略参数】
- WINDOW_SIZE: 特征计算窗口
- VOLUME_PROFILE_BINS: 成交量分布分箱数
- ORDER_FLOW_WINDOW: 订单流窗口
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json

# ============ 参数配置 ============
WINDOW_SIZE = 100              # 特征计算窗口
VOLUME_PROFILE_BINS = 20       # 成交量分布分箱数
ORDER_FLOW_WINDOW = 50         # 订单流窗口
TICK_BUFFER_SIZE = 500         # Tick缓冲大小


class HighFrequencyFeatureExtractor:
    """高频数据特征提取器"""
    
    def __init__(self, api, symbol):
        self.api = api
        self.symbol = symbol
        self.tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)
        self.features = {}
        
    def process_tick(self, tick):
        """处理单个Tick数据"""
        tick_data = {
            'time': tick.get('datetime', ''),
            'last_price': tick.get('last_price', 0),
            'volume': tick.get('volume', 0),
            'bid_price1': tick.get('bid_price1', 0),
            'ask_price1': tick.get('ask_price1', 0),
            'bid_volume1': tick.get('bid_volume1', 0),
            'ask_volume1': tick.get('ask_volume1', 0),
            'bid_price2': tick.get('bid_price2', 0),
            'ask_price2': tick.get('ask_price2', 0),
            'bid_price3': tick.get('bid_price3', 0),
            'ask_price3': tick.get('ask_price3', 0),
            'bid_volume2': tick.get('bid_volume2', 0),
            'ask_volume2': tick.get('ask_volume2', 0),
            'bid_volume3': tick.get('bid_volume3', 0),
            'ask_volume3': tick.get('ask_volume3', 0),
        }
        
        self.tick_buffer.append(tick_data)
        return tick_data
    
    def calculate_spread(self):
        """计算买卖价差"""
        if len(self.tick_buffer) < 2:
            return {}
        
        ticks = list(self.tick_buffer)
        spreads = []
        
        for t in ticks:
            if t['ask_price1'] > 0 and t['bid_price1'] > 0:
                spread = (t['ask_price1'] - t['bid_price1']) / t['last_price']
                spreads.append(spread)
        
        if not spreads:
            return {}
        
        return {
            'spread_mean': np.mean(spreads),
            'spread_std': np.std(spreads),
            'spread_current': spreads[-1] if spreads else 0,
        }
    
    def calculate_order_imbalance(self):
        """计算订单流不平衡"""
        if len(self.tick_buffer) < ORDER_FLOW_WINDOW:
            return {}
        
        ticks = list(self.tick_buffer)[-ORDER_FLOW_WINDOW:]
        
        bid_volumes = [t['bid_volume1'] + t.get('bid_volume2', 0) + t.get('bid_volume3', 0) 
                      for t in ticks]
        ask_volumes = [t['ask_volume1'] + t.get('ask_volume2', 0) + t.get('ask_volume3', 0) 
                      for t in ticks]
        
        total_bid = sum(bid_volumes)
        total_ask = sum(ask_volumes)
        
        if total_bid + total_ask == 0:
            return {}
        
        # 订单不平衡指标
        imbalance = (total_bid - total_ask) / (total_bid + total_ask)
        
        # 累计订单流
        order_flow = []
        cumsum = 0
        for bv, av in zip(bid_volumes, ask_volumes):
            cumsum += bv - av
            order_flow.append(cumsum)
        
        return {
            'order_imbalance': imbalance,
            'total_bid_volume': total_bid,
            'total_ask_volume': total_ask,
            'order_flow_trend': np.polyfit(range(len(order_flow)), order_flow, 1)[0] if len(order_flow) > 2 else 0,
        }
    
    def calculate_volume_profile(self):
        """计算成交量分布"""
        if len(self.tick_buffer) < WINDOW_SIZE:
            return {}
        
        ticks = list(self.tick_buffer)[-WINDOW_SIZE:]
        
        prices = [t['last_price'] for t in ticks if t['last_price'] > 0]
        volumes = [t.get('volume', 1) for t in ticks]
        
        if not prices or len(prices) != len(volumes):
            return {}
        
        # 创建价格分箱
        price_min, price_max = min(prices), max(prices)
        
        if price_max == price_min:
            return {}
        
        bins = np.linspace(price_min, price_max, VOLUME_PROFILE_BINS + 1)
        
        # 统计每个价格区间的成交量
        vol_profile = np.zeros(VOLUME_PROFILE_BINS)
        
        for p, v in zip(prices, volumes):
            bin_idx = np.digitize(p, bins) - 1
            if 0 <= bin_idx < VOLUME_PROFILE_BINS:
                vol_profile[bin_idx] += v
        
        # 归一化
        total_vol = vol_profile.sum()
        if total_vol > 0:
            vol_profile = vol_profile / total_vol
        
        # 寻找高成交量区域
        high_vol_indices = np.where(vol_profile > vol_profile.mean() * 1.5)[0]
        
        return {
            'vol_profile': vol_profile.tolist(),
            'high_vol_price_level': bins[high_vol_indices[0]] if len(high_vol_indices) > 0 else 0,
            'vol_concentration': vol_profile.max(),
            'vol_balance': 1 - abs(np.sum(vol_profile[:len(vol_profile)//2]) - 
                                  np.sum(vol_profile[len(vol_profile)//2:])),
        }
    
    def calculate_price_impact(self):
        """计算价格冲击"""
        if len(self.tick_buffer) < WINDOW_SIZE:
            return {}
        
        ticks = list(self.tick_buffer)[-WINDOW_SIZE:]
        
        prices = [t['last_price'] for t in ticks]
        volumes = [t.get('volume', 1) for t in ticks]
        
        if not prices:
            return {}
        
        # 成交量加权的平均价格变动
        returns = np.diff(prices) / prices[:-1]
        
        # 计算VWAP
        vwap = np.cumsum(np.array(prices[1:]) * np.array(volumes[1:])) / np.cumsum(volumes[1:])
        
        # 价格冲击: 单位成交量导致的价格变动
        price_impact = np.abs(returns) / np.array(volumes[1:]) * 1e6  # 标准化
        
        return {
            'price_impact_mean': np.mean(price_impact) if len(price_impact) > 0 else 0,
            'price_impact_std': np.std(price_impact) if len(price_impact) > 0 else 0,
            'price_trend': np.polyfit(range(len(prices)), prices, 1)[0] if len(prices) > 2 else 0,
            'volatility_per_volume': np.std(returns) / (np.mean(volumes) + 1) * 1e6,
        }
    
    def calculate_micro_features(self):
        """计算微观特征"""
        features = {}
        
        # 买卖价差
        features.update(self.calculate_spread())
        
        # 订单流
        features.update(self.calculate_order_imbalance())
        
        # 成交量分布
        features.update(self.calculate_volume_profile())
        
        # 价格冲击
        features.update(self.calculate_price_impact())
        
        self.features = features
        return features
    
    def get_current_features(self):
        """获取当前特征"""
        return self.features
    
    def generate_features_report(self):
        """生成特征报告"""
        if not self.features:
            return "暂无特征数据"
        
        report = "=" * 60 + "\n"
        report += f"高频特征报告 - {self.symbol}\n"
        report += "=" * 60 + "\n"
        report += f"Tick数量: {len(self.tick_buffer)}\n\n"
        
        report += "买卖价差特征:\n"
        for k, v in self.features.items():
            if 'spread' in k:
                report += f"  {k}: {v:.6f}\n"
        
        report += "\n订单流特征:\n"
        for k, v in self.features.items():
            if 'order' in k or 'bid' in k or 'ask' in k:
                if isinstance(v, float):
                    report += f"  {k}: {v:.4f}\n"
                else:
                    report += f"  {k}: {v}\n"
        
        report += "\n价格冲击特征:\n"
        for k, v in self.features.items():
            if 'price' in k or 'impact' in k or 'trend' in k or 'volatility' in k:
                if isinstance(v, float):
                    report += f"  {k}: {v:.6f}\n"
        
        return report


def main():
    """主函数 - 用于测试"""
    print("高频数据特征提取器")
    print("=" * 50)
    
    # 模拟测试
    extractor = HighFrequencyFeatureExtractor(None, 'IF2401')
    
    # 模拟Tick数据
    import random
    base_price = 3800
    
    for i in range(150):
        tick = {
            'datetime': f'2024-01-15 10:00:{i%60:02d}',
            'last_price': base_price + random.uniform(-10, 10),
            'volume': random.randint(1, 100),
            'bid_price1': base_price - 2 + random.uniform(-1, 1),
            'ask_price1': base_price + 2 + random.uniform(-1, 1),
            'bid_volume1': random.randint(10, 100),
            'ask_volume1': random.randint(10, 100),
        }
        extractor.process_tick(tick)
    
    # 计算特征
    features = extractor.calculate_micro_features()
    
    print("\n" + extractor.generate_features_report())


if __name__ == "__main__":
    main()
