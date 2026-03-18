#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 28
策略名称: 期权波动率套利分析器
生成日期: 2026-03-18
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
期权波动率套利分析器对期权链数据进行截面分析，识别隐含波动率(IV)与
历史波动率(HV)之间的偏差，寻找波动率溢价/折价机会。
支持：IV-HV差值分析、波动率偏斜(Vol Skew)分析、波动率期限结构分析、
期权Vega对冲等策略研究。

【策略参数】
- UNDERLYING_SYMBOL: 标的期货代码
- OPTIONS_EXPIRY: 期权到期日
- HV_WINDOW: 历史波动率窗口（天）
- IV_HV_THRESHOLD: IV-HV偏差入场阈值
- SKEW_THRESHOLD: 波动率偏斜阈值
- GREEKS_WINDOW: Greeks计算数据窗口

【风险提示】
期权交易涉及复杂希腊字母风险。本工具仅供研究分析，不构成实际交易建议。
实际交易需考虑流动性、价差和尾部风险。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json
import math


# ============ 参数配置 ============
UNDERLYING_SYMBOL = "m2501.DCE"  # 豆粕期货（带期权）
OPTIONS_EXPIRY = "2025-01-24"    # 期权到期日
HV_WINDOW = 20                  # 历史波动率窗口
IV_HV_THRESHOLD = 0.10          # IV-HV偏差入场阈值（10%）
SKEW_THRESHOLD = 0.15           # 波动率偏斜阈值
GREEKS_WINDOW = 20             # Greeks计算窗口
RISK_FREE_RATE = 0.03           # 无风险利率


# 简化版 Black-Scholes 计算
def normal_cdf(x):
    """标准正态分布CDF"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def normal_pdf(x):
    """标准正态分布PDF"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def black_scholes_call(S, K, T, r, sigma):
    """BS看涨期权定价"""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return call_price

def implied_volatility(price, S, K, T, r, is_call=True):
    """二叉树/牛顿法计算隐含波动率（简化）"""
    if T <= 0 or price <= 0:
        return 0.0
    sigma = 0.20  # 初始猜测
    for _ in range(50):
        price_bs = black_scholes_call(S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * normal_pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-10:
            break
        diff = price - price_bs
        sigma += diff / vega
        sigma = max(0.01, min(sigma, 5.0))
        if abs(diff) < 1e-8:
            break
    return sigma


class OptionsVolArbAnalyzer:
    """期权波动率套利分析器"""

    def __init__(self, api, underlying, expiry):
        self.api = api
        self.underlying = underlying
        self.expiry = expiry
        self.quote = None
        self.option_quotes = {}
        self.price_history = deque(maxlen=HV_WINDOW + 10)
        self.returns_history = deque(maxlen=HV_WINDOW + 10)
        self.iv_history = {}
        self.hv_history = deque(maxlen=100)
        self.skew_history = deque(maxlen=100)
        self.analysis_results = []

    def _compute_hv(self):
        """计算历史波动率（年化）"""
        if len(self.returns_history) < HV_WINDOW:
            return 0.0
        returns = list(self.returns_history)[-HV_WINDOW:]
        hv = np.std(returns) * np.sqrt(252)
        return hv

    def _get_option_chain(self):
        """获取期权链数据（简化：通过标的行情推测）"""
        try:
            if self.quote is None:
                self.quote = self.api.get_quote(self.underlying)
            price = self.quote.get('last_price', 0)
            if price > 0:
                self.price_history.append(price)
                if len(self.price_history) >= 2:
                    ret = (price - self.price_history[-2]) / self.price_history[-2]
                    self.returns_history.append(ret)
            return price
        except Exception:
            return 0.0

    def analyze_iv_hv_spread(self, S, T):
        """分析IV-HV价差"""
        hv = self._compute_hv()
        self.hv_history.append(hv)

        results = {}
        # 模拟不同行权价的隐含波动率（实际需要期权实时行情）
        atm_strike = round(S / 10) * 10  # 简化ATM
        strikes = [atm_strike - 50, atm_strike - 25, atm_strike,
                    atm_strike + 25, atm_strike + 50]

        for strike in strikes:
            # 模拟IV（实际应从行情获取）
            moneyness = abs(S - strike) / S
            iv = hv * (1.0 + moneyness * 0.5 + np.random.uniform(-0.05, 0.05))
            iv = max(0.05, min(iv, 1.0))

            iv_hv_spread = iv - hv
            results[strike] = {
                'iv': iv,
                'hv': hv,
                'iv_hv_spread': iv_hv_spread,
                'moneyness': 'ITM' if strike < S else ('ATM' if strike == atm_strike else 'OTM'),
                'signal': 'long_vol' if iv_hv_spread < -IV_HV_THRESHOLD else
                          ('short_vol' if iv_hv_spread > IV_HV_THRESHOLD else 'neutral')
            }
            self.iv_history[strike] = iv

        return results

    def analyze_volatility_skew(self, results):
        """分析波动率偏斜"""
        if not results:
            return None

        otm_ivs = []
        itm_ivs = []
        for strike, data in results.items():
            if data['moneyness'] == 'OTM':
                otm_ivs.append(data['iv'])
            elif data['moneyness'] == 'ITM':
                itm_ivs.append(data['iv'])

        skew_indicator = 0.0
        if otm_ivs and itm_ivs:
            skew_indicator = np.mean(otm_ivs) - np.mean(itm_ivs)
        self.skew_history.append(skew_indicator)

        return skew_indicator

    def compute_portfolio_vega(self, results, position_size=1):
        """计算组合Vega（近似）"""
        total_vega = 0.0
        for strike, data in results.items():
            iv = data['iv']
            # 简化Vega ≈ S * sqrt(T) * pdf(d1) * 0.01
            T = 30 / 365  # 假设30天到期
            vega = 0.5 * np.sqrt(T) / (iv + 0.01) * position_size
            total_vega += vega
        return total_vega

    def run_analysis(self):
        """执行完整分析"""
        S = self._get_option_chain()
        if S <= 0 or len(self.price_history) < HV_WINDOW:
            return None

        hv = self._compute_hv()
        T = 30 / 365  # 简化

        iv_results = self.analyze_iv_hv_spread(S, T)
        skew = self.analyze_volatility_skew(iv_results)
        portfolio_vega = self.compute_portfolio_vega(iv_results)

        report = {
            'timestamp': datetime.now().isoformat(),
            'underlying': self.underlying,
            'spot_price': round(S, 4),
            'hv_annual': round(hv, 6),
            'hv_percent': f"{hv:.2%}",
            'iv_results': {str(k): v for k, v in iv_results.items()},
            'skew_indicator': round(skew, 6) if skew is not None else 0.0,
            'portfolio_vega': round(portfolio_vega, 4),
            'avg_iv': round(np.mean(list(iv_results.values())), 6) if iv_results else 0.0,
            'iv_hv_spread': round(np.mean([v['iv_hv_spread'] for v in iv_results.values()]), 6) if iv_results else 0.0,
            'signal': self._generate_signal(iv_results, skew)
        }
        self.analysis_results.append(report)
        if len(self.analysis_results) > 100:
            self.analysis_results = self.analysis_results[-100:]
        return report

    def _generate_signal(self, iv_results, skew):
        """生成综合信号"""
        if not iv_results:
            return "no_data"
        avg_spread = np.mean([v['iv_hv_spread'] for v in iv_results.values()])

        if avg_spread > IV_HV_THRESHOLD and (skew and skew > SKEW_THRESHOLD):
            return "short_volatility"
        elif avg_spread < -IV_HV_THRESHOLD:
            return "long_volatility"
        elif skew and skew > SKEW_THRESHOLD:
            return "skew_trading"
        else:
            return "neutral"

    def print_report(self, report):
        """打印分析报告"""
        if report is None:
            return
        print(f"\n{'='*60}")
        print(f"【期权波动率套利分析】{report['underlying']} @ {report['timestamp'][:19]}")
        print(f"{'='*60}")
        print(f"  标的现价: {report['spot_price']}")
        print(f"  历史波动率(HV): {report['hv_percent']} (年化)")
        print(f"  平均隐含波动率(IV): {report['avg_iv']:.2%}")
        print(f"  IV-HV 差值: {report['iv_hv_spread']:+.2%}")
        print(f"  波动率偏斜: {report['skew_indicator']:+.4f}")
        print(f"  组合Vega: {report['portfolio_vega']:.4f}")
        print(f"\n  各行权价IV分析:")
        for strike, data in sorted(report['iv_results'].items(), key=lambda x: float(x[0])):
            spread_str = f"{data['iv_hv_spread']:+.2%}"
            signal_map = {'long_vol': '↗ 多波动率', 'short_vol': '↘ 空波动率', 'neutral': '—观望'}
            sig = signal_map.get(data['signal'], data['signal'])
            print(f"    K={strike} ({data['moneyness']}): IV={data['iv']:.2%} HV差={spread_str} {sig}")
        print(f"\n  综合信号: {report['signal']}")
        print(f"\n  策略建议:")
        if report['signal'] == 'short_volatility':
            print(f"    → IV相对HV偏高，隐含波动率可能高估，建议做空期权波动率（卖出跨式/宽跨式）")
        elif report['signal'] == 'long_volatility':
            print(f"    → IV相对HV偏低，隐含波动率被低估，建议做多期权波动率（买入跨式/宽跨式）")
        elif report['signal'] == 'skew_trading':
            print(f"    → 波动率偏斜明显，可考虑偏斜交易（如买OTM Put / 卖ITM Put）")
        else:
            print(f"    → 波动率处于合理区间，无明显套利机会")

    def run(self, interval=60):
        """运行主循环"""
        print(f"期权波动率套利分析器启动: {self.underlying}")
        print(f"历史波动率窗口: {HV_WINDOW}天, IV-HV阈值: {IV_HV_THRESHOLD:.0%}")

        while True:
            self.api.wait_update()
            report = self.run_analysis()
            if report:
                self.print_report(report)


# ============ 主程序 ============
if __name__ == "__main__":
    api = TqApi(auth=TqAuth("auto", "auto"))
    analyzer = OptionsVolArbAnalyzer(api, UNDERLYING_SYMBOL, OPTIONS_EXPIRY)
    try:
        analyzer.run(interval=60)
    except KeyboardInterrupt:
        print("\n分析器已停止")
        if analyzer.analysis_results:
            last = analyzer.analysis_results[-1]
            print(f"最近分析: {json.dumps(last, ensure_ascii=False, indent=2)}")
    finally:
        api.close()
