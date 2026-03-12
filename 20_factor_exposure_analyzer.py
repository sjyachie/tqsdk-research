#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
策略编号: 20
策略名称: 因子暴露度分析器
生成日期: 2026-03-11
仓库地址: tqsdk-research
================================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【TqSdk 简介】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TqSdk（天勤量化 SDK）是由信易科技（北京）有限公司开发的专业期货量化交易框架，
完全免费开源（0 协议），Apache 2.基于 Python 语言设计，支持 Python 3.6+ 环境。
TqSdk 已服务于数万名国内期货量化投资者，是国内使用最广泛的期货量化框架之一。

TqSdk 核心能力包括：

1. **统一行情接口**：对接国内全部7大期货交易所（SHFE/DCE/CZCE/CFFEX/INE/GFEX）
   及主要期权品种，统一的 get_quote / get_kline_serial 接口，告别繁琐的协议适配；

2. **高性能数据推送**：天勤服务器行情推送延迟通常 in 5ms以内，Tick 级数据实时到达，
   K线自动合并，支持自定义周期（秒/分钟/小时/日/周/月）；

3. **同步式编程范式**：独特的 wait_update() + is_changing() 设计，策略代码像
   写普通Python一样自然流畅，无需掌握异步编程，大幅降低开发门槛；

4. **完整回测引擎**：内置 TqBacktest 回测模式，历史数据精确到Tick级别，
   支持滑点、手续费等真实市场参数，回测结果可信度高；

5. **实盘/模拟一键切换**：代码结构不变，仅替换 TqApi 初始化参数即可从
   模拟盘切换至实盘，极大降低策略上线风险；

6. **多账户并发**：支持同时连接多个期货账户，适合机构投资者和量化团队；

7. **活跃生态**：官方提供策略示例库、在线文档、量化社区论坛，更关维护活跃。

官网: https://www.shinnytech.com/tianqin/
文档: https://doc.shinnytech.com/tqsdk/latest/
GitHub: https://github.com/shinnytech/tqsdk-python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【策略背景与原理】
因子暴露度分析器用于分析投资组合对各类风险因子（动量、价值、波动率、规模等）
的暴露程度，帮助投资者理解收益来源并进行风险管理。

【策略参数】
- FACTORS: 要分析的因子列表
- LOOKBACK_PERIOD: 回溯期
- EXPOSURE_THRESHOLD: 暴露度阈值

【风险提示】
因子暴露度分析基于历史回归，可能存在幸存者偏差。实际使用需谨慎解读。
================================================================================
"""

from tqsdk import TqApi, TqAuth
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ============ 参数配置 ============
SYMBOLS = [
    "SHFE.rb2405",   # 螺纹钢 - 工业品
    "SHFE.hc2405",   # 热卷 - 工业品
    "DCE.m2405",     # 豆粕 - 农产品
    "CZCE.rm2405",   # 菜粕 - 农产品
    "DCE.j2405",     # 焦炭 - 工业品
    "DCE.jm2405",    # 焦煤 - 工业品
    "CZCE.cs2405",   # 玉米 - 农产品
    "CFFEX IF2405"   # 股指期货 - 金融
]
LOOKBACK_PERIOD = 60        # 回溯期（天）
EXPOSURE_THRESHOLD = 0.3    # 高暴露度阈值


class FactorExposureAnalyzer:
    """因子暴露度分析器"""
    
    def __init__(self, api, symbols=None):
        self.api = api
        self.symbols = symbols or SYMBOLS
        self.factor_returns = {}
        self.asset_returns = {}
        
        # 定义商品因子
        self.factor_definitions = {
            '动量': self._momentum_factor,
            '波动率': self._volatility_factor,
            '反转': self._reversal_factor,
            '规模': self._size_factor,
            '期限结构': self._roll_yield_factor
        }
    
    def get_historical_data(self, symbol, days):
        """获取历史数据"""
        try:
            klines = self.api.get_kline_serial(symbol, 86400, data_length=days+10)
            
            if len(klines) > days:
                df = pd.DataFrame(klines)
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df.set_index('timestamp', inplace=True)
                df = df.tail(days)
                return df
        except:
            pass
        return None
    
    def _momentum_factor(self, prices):
        """动量因子：过去20天收益率"""
        return prices.pct_change(20).dropna()
    
    def _volatility_factor(self, prices):
        """波动率因子：过去20天收益率标准差"""
        returns = prices.pct_change().dropna()
        return returns.rolling(20).std().dropna()
    
    def _reversal_factor(self, prices):
        """反转因子：过去5天收益率"""
        return prices.pct_change(5).dropna()
    
    def _size_factor(self, prices):
        """规模因子：使用价格作为代理（低价=小市值）"""
        # 简化的规模因子
        return pd.Series(1/prices, index=prices.index).dropna()
    
    def _roll_yield_factor(self, prices):
        """期限结构因子：近月-远月价差（简化版）"""
        # 简化：使用价格变化率
        return prices.pct_change(10).dropna()
    
    def calculate_factor_returns(self):
        """计算各因子收益率"""
        print("正在计算因子收益率...")
        
        # 获取市场基准（使用IF作为代理）
        market_data = self.get_historical_data("CFFEX IF2405", LOOKBACK_PERIOD)
        
        if market_data is not None:
            market_returns = market_data['close'].pct_change().dropna()
            self.factor_returns['市场'] = market_returns
        
        # 计算各因子
        for factor_name, factor_func in self.factor_definitions.items():
            # 使用rb作为代表计算因子
            data = self.get_historical_data("SHFE.rb2405", LOOKBACK_PERIOD)
            if data is not None:
                try:
                    factor_return = factor_func(data['close'])
                    if not factor_return.empty:
                        self.factor_returns[factor_name] = factor_return
                        print(f"  {factor_name}: OK")
                except:
                    print(f"  {factor_name}: FAIL")
    
    def calculate_asset_returns(self):
        """计算各资产收益率"""
        print("\n正在计算资产收益率...")
        
        for symbol in self.symbols:
            data = self.get_historical_data(symbol, LOOKBACK_PERIOD)
            if data is not None:
                returns = data['close'].pct_change().dropna()
                self.asset_returns[symbol] = returns
                print(f"  {symbol}: {len(returns)} 条")
    
    def calculate_exposures(self):
        """计算因子暴露度"""
        exposures = {}
        
        # 对齐日期
        common_dates = None
        for ret in self.asset_returns.values():
            if common_dates is None:
                common_dates = set(ret.index)
            else:
                common_dates = common_dates & set(ret.index)
        
        if not common_dates or len(self.factor_returns) == 0:
            return exposures
        
        for symbol, returns in self.asset_returns.items():
            aligned_returns = returns[returns.index.isin(common_dates)]
            
            # 准备因子矩阵
            factor_data = []
            factor_names = []
            
            for factor_name, factor_ret in self.factor_returns.items():
                aligned_factor = factor_ret[factor_ret.index.isin(common_dates)]
                # 对齐
                aligned_factor = aligned_factor.reindex(aligned_returns.index)
                if len(aligned_factor) == len(aligned_returns):
                    factor_data.append(aligned_factor.values)
                    factor_names.append(factor_name)
            
            if len(factor_data) == 0:
                continue
            
            X = np.column_stack(factor_data)
            y = aligned_returns.values
            
            # 去除NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]
            
            if len(X) < 30:
                continue
            
            # 线性回归
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                exposure_dict = {}
                for i, name in enumerate(factor_names):
                    exposure_dict[name] = model.coef_[i]
                
                exposures[symbol] = {
                    'exposures': exposure_dict,
                    'r_squared': model.score(X, y)
                }
            except:
                continue
        
        return exposures
    
    def analyze_sector_exposure(self):
        """分析板块暴露度"""
        sectors = {
            '工业品': ['SHFE.rb2405', 'SHFE.hc2405', 'DCE.j2405', 'DCE.jm2405'],
            '农产品': ['DCE.m2405', 'CZCE.rm2405', 'CZCE.cs2405'],
            '金融': ['CFFEX IF2405']
        }
        
        sector_exposure = {}
        
        for sector, symbols in sectors.items():
            sector_assets = {s: r for s, r in self.asset_returns.items() if s in symbols}
            
            if sector_assets:
                # 计算板块平均收益
                returns_df = pd.DataFrame(sector_assets)
                avg_returns = returns_df.mean(axis=1).dropna()
                
                sector_exposure[sector] = {
                    'avg_return': avg_returns.mean() * 252,  # 年化
                    'volatility': avg_returns.std() * np.sqrt(252),  # 年化
                    'assets': len(sector_assets)
                }
        
        return sector_exposure
    
    def generate_report(self):
        """生成分析报告"""
        # 计算暴露度
        exposures = self.calculate_exposures()
        
        report = []
        report.append("=" * 70)
        report.append("因子暴露度分析报告")
        report.append("=" * 70)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"回溯期: {LOOKBACK_PERIOD} 天")
        report.append(f"资产数量: {len(self.asset_returns)}")
        report.append("")
        
        # 因子暴露度
        report.append("【因子暴露度】")
        if exposures:
            for symbol, data in exposures.items():
                report.append(f"\n  {symbol}:")
                report.append(f"    R²: {data['r_squared']:.3f}")
                for factor, exp in data['exposures'].items():
                    level = "高" if abs(exp) > EXPOSURE_THRESHOLD else "低"
                    report.append(f"    {factor}: {exp:+.3f} ({level})")
        else:
            report.append("  无法计算暴露度")
        report.append("")
        
        # 板块暴露度
        report.append("【板块暴露度】")
        sector_exp = self.analyze_sector_exposure()
        for sector, data in sector_exp.items():
            report.append(f"  {sector}:")
            report.append(f"    年化收益: {data['avg_return']:.1%}")
            report.append(f"    年化波动: {data['volatility']:.1%}")
            report.append(f"    资产数: {data['assets']}")
        report.append("")
        
        # 高暴露告警
        report.append("【高暴露告警】")
        high_exposure_found = False
        for symbol, data in exposures.items():
            for factor, exp in data['exposures'].items():
                if abs(exp) > EXPOSURE_THRESHOLD * 2:
                    direction = "正" if exp > 0 else "负"
                    report.append(f"  🔴 {symbol} {factor}: {exp:+.3f} ({direction}暴露)")
                    high_exposure_found = True
        
        if not high_exposure_found:
            report.append("  无高暴露告警")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def get_portfolio_exposure(self, positions):
        """获取组合的因子暴露度"""
        exposures = self.calculate_exposures()
        
        if not exposures:
            return None
        
        # 加权平均暴露度
        portfolio_exp = {}
        total_weight = 0
        
        for symbol, position in positions.items():
            if symbol in exposures and position != 0:
                weight = abs(position)
                for factor, exp in exposures[symbol]['exposures'].items():
                    if factor not in portfolio_exp:
                        portfolio_exp[factor] = 0
                    portfolio_exp[factor] += exp * weight
                total_weight += weight
        
        if total_weight > 0:
            portfolio_exp = {k: v/total_weight for k, v in portfolio_exp.items()}
        
        return portfolio_exp


def main():
    """主函数"""
    api = TqApi(auth=TqAuth("用户名", "密码"))
    
    analyzer = FactorExposureAnalyzer(api)
    
    # 计算因子和资产收益
    analyzer.calculate_factor_returns()
    analyzer.calculate_asset_returns()
    
    # 生成报告
    report = analyzer.generate_report()
    print(report)
    
    api.close()


if __name__ == "__main__":
    # main()  # 实盘模式
    print("因子暴露度分析器模块")
    print("请根据需要选择运行模式")
