# tqsdk-research

> 基于 **TqSdk** 的量化研究工具，持续更新中。

## 项目简介

本仓库专注于**量化研究工具**，涵盖回测框架、技术指标库、数据分析等领域。  
所有策略使用 [天勤量化 TqSdk](https://github.com/shinnytech/tqsdk-python) 实现。

## 策略列表

| # | 策略名称 | 类型 | 文件 |
|---|---------|------|------|
| 01 | 回测分析工具 | 回测工具 | [01_backtest_analyzer.py](strategies/01_backtest_analyzer.py) |
| 02 | K线数据对齐工具 | 数据工具 | [02_kline_aligner.py](strategies/02_kline_aligner.py) |
| 03 | 策略相关性分析工具 | 分析工具 | [03_correlation_analyzer.py](strategies/03_correlation_analyzer.py) |
| 04 | 凯利公式仓位计算器 | 仓位工具 | [04_kelly_calculator.py](strategies/04_kelly_calculator.py) |

## 更新日志

- 2026-03-03: 新增策略03（相关性分析）、策略04（凯利公式）
