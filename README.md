# tqsdk-research

> 基于 **TqSdk** 的量化研究工具集合，持续更新中。

## 项目简介

本仓库专注于**量化研究工具**，所有工具使用 [天勤量化 TqSdk](https://github.com/shinnytech/tqsdk-python) 实现，覆盖回测分析、多品种数据处理、技术指标等方向。

**TqSdk 官方文档**：https://doc.shinnytech.com/tqsdk/latest/

---

## 工具列表

| 编号 | 文件名 | 工具名称 | 功能描述 | 上传日期 |
|------|--------|----------|----------|----------|
| 01 | [01_backtest_analyzer.py](strategies/01_backtest_analyzer.py) | BacktestAnalyzer — TqSdk回测绩效评估器 | 输入回测账户对象，自动计算年化收益率、夏普比率、最大回撤、胜率、盈亏比、日收益分布，支持输出DataFrame | 2026-03-02 |
| 02 | [02_kline_aligner.py](strategies/02_kline_aligner.py) | KlineAligner — 多品种K线对齐工具 | 同时订阅多个品种K线并自动对齐时间戳，处理缺失值，返回对齐DataFrame，用于跨品种相关性分析 | 2026-03-02 |

---

## 工具详情

### 01 · BacktestAnalyzer — TqSdk 回测绩效评估器

**文件**：`strategies/01_backtest_analyzer.py`

输入 TqSdk 回测账户对象（账户净值曲线 DataFrame + 交易记录 DataFrame），自动计算并输出：

- 📈 **年化收益率**：基于期末/期初净值和回测天数，复利公式计算
- ⚡ **夏普比率**：年化超额收益率 / 日收益标准差 × √252
- 📉 **最大回撤**：净值曲线峰值到谷底的最大跌幅
- 🎯 **胜率**：盈利平仓笔数 / 总平仓笔数
- ⚖️ **盈亏比**：平均盈利额 / 平均亏损额（绝对值）
- 📊 **日收益分布**：均值、标准差、偏度、峰度等统计量

**快速使用**：
```python
from strategies.01_backtest_analyzer import BacktestAnalyzer

analyzer = BacktestAnalyzer(account_df=account_df, trade_df=trade_df)
analyzer.print_report()
df = analyzer.to_dataframe()
```

---

### 02 · KlineAligner — 多品种 K 线对齐工具

**文件**：`strategies/02_kline_aligner.py`

同时订阅多个品种 K 线并自动对齐时间戳，解决因夜盘差异、停牌等导致的时间戳不一致问题：

- 🔗 **并发订阅**：通过 TqSdk API 同时获取多品种 K 线数据
- ⏱️ **时间戳对齐**：Outer Join 合并，以时间戳并集为索引
- 🔧 **缺失值处理**：支持 `ffill`（前向填充）/ `zero` / `drop` / `keep` 四种策略
- 📐 **相关性分析**：内置 `correlation_matrix()` 方法，支持 Pearson/Spearman/Kendall

**快速使用**：
```python
from strategies.02_kline_aligner import KlineAligner

aligner = KlineAligner(
    symbols=["SHFE.rb2501", "DCE.i2501", "SHFE.cu2501"],
    duration_seconds=86400,  # 日线
    fields=["close", "volume"],
    fill_method="ffill",
)
aligner.load_from_api(api)
aligned_df = aligner.align()
corr = aligner.correlation_matrix()
print(corr)
```

---

## 环境要求

```bash
pip install tqsdk pandas numpy
```

## 目录结构

```
tqsdk-research/
├── README.md
└── strategies/
    ├── 01_backtest_analyzer.py   # 回测绩效评估器
    └── 02_kline_aligner.py       # 多品种K线对齐工具
```

## 许可证

MIT License
