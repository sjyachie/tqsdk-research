# tqsdk-research

> 基于 **TqSdk** 的量化研究工具，持续更新中。

## 项目简介

本仓库专注于**量化研究工具**，涵盖回测框架、技术指标库、数据分析等领域。  
所有策略使用 [天勤量化 TqSdk](https://github.com/shinnytech/tqsdk-python) 实现。

## 工具列表

| # | 工具名称 | 类型 | 文件 |
|---|---------|------|------|
| 01 | 回测分析工具 | 回测工具 | [01_backtest_analyzer.py](01_backtest_analyzer.py) |
| 02 | K线数据对齐工具 | 数据工具 | [02_kline_aligner.py](02_kline_aligner.py) |
| 03 | 策略相关性分析工具 | 分析工具 | [03_correlation_analyzer.py](03_correlation_analyzer.py) |
| 04 | 凯利公式仓位计算器 | 仓位工具 | [04_kelly_calculator.py](04_kelly_calculator.py) |
| 05 | RSI分析工具 | 指标工具 | [05_rsi_analyzer.py](05_rsi_analyzer.py) |
| 06 | 均线交叉分析工具 | 指标工具 | [06_ma_crossover_analyzer.py](06_ma_crossover_analyzer.py) |
| 07 | 布林带宽度分析 | 指标工具 | [07_boll_bandwidth.py](07_boll_bandwidth.py) |
| 08 | VWAP分析工具 | 指标工具 | [08_vwap_analysis.py](08_vwap_analysis.py) |
| 09 | 协整分析工具 | 统计分析 | [09_cointegration.py](09_cointegration.py) |
| 10 | 蒙特卡洛模拟工具 | 统计分析 | [10_monte_carlo.py](10_monte_carlo.py) |
| 11 | 夏普比率优化器 | 绩效分析 | [11_sharpe_optimizer.py](11_sharpe_optimizer.py) |
| 12 | 相关性分析工具 | 统计分析 | [12_correlation_analysis.py](12_correlation_analysis.py) |
| 13 | 绩效归因分析 | 绩效分析 | [13_performance_attribution.py](13_performance_attribution.py) |
| 14 | 多时间周期共振分析 | 指标工具 | [14_multi_timeframe_resonance.py](14_multi_timeframe_resonance.py) |
| 15 | 波动率聚类分析 | 统计分析 | [15_volatility_clustering_analysis.py](15_volatility_clustering_analysis.py) |
| 16 | 订单流分析工具 | 分析工具 | [16_order_flow_analysis.py](16_order_flow_analysis.py) |
| 17 | 时间序列动量分析 | 分析工具 | [17_ts_momentum_analyzer.py](17_ts_momentum_analyzer.py) |
| 18 | 跨期价差分析 | 分析工具 | [18_calendar_spread_analyzer.py](18_calendar_spread_analyzer.py) |
| 19 | 策略相关性分析器 | 分析工具 | [19_strategy_correlation_analyzer.py](19_strategy_correlation_analyzer.py) |
| 20 | 因子暴露分析器 | 分析工具 | [20_factor_exposure_analyzer.py](20_factor_exposure_analyzer.py) |
| 21 | 多因子分析工具 | 多因子 | [21_multi_factor_analysis.py](21_multi_factor_analysis.py) |
| 22 | 截面动量策略 | 多因子 | [22_cross_sectional_momentum.py](22_cross_sectional_momentum.py) |
| 23 | 因子收益率分析 | 多因子 | [23_factor_returns_analyzer.py](23_factor_returns_analyzer.py) |
| 24 | 市场状态检测 | 分析工具 | [24_market_regime_detector.py](24_market_regime_detector.py) |
| 25 | 机器学习预测因子分析器 | 机器学习 | [25_ml_factor_analyzer.py](25_ml_factor_analyzer.py) |
| 26 | 高频数据特征提取器 | 分析工具 | [26_high_freq_feature_extractor.py](26_high_freq_feature_extractor.py) |
| 27 | 跨品种套利分析器 | 分析工具 | [27_cross_commodity_arbitrage_analyzer.py](27_cross_commodity_arbitrage_analyzer.py) |
| 28 | 期权波动率套利分析器 | 多因子 | [28_options_volatility_arbitrage.py](28_options_volatility_arbitrage.py) |

## 工具分类

### 📊 回测工具
策略回测和绩效分析工具。

### 📈 指标工具
各类技术指标的计算和分析。

### 🔬 统计分析
相关性、协整、蒙特卡洛模拟、波动率聚类等。

### 📉 绩效分析
夏普比率、绩效归因等。

### ⚖️ 多因子
截面动量、因子暴露、多因子分析等。

## 环境要求

```bash
pip install tqsdk numpy pandas scipy statsmodels
```

## 风险提示

- 本仓库仅供研究学习使用
- 过往业绩不代表未来表现

---

**持续更新中，欢迎 Star ⭐ 关注**

*更新时间：2026-03-18*
