"""
================================================================================
TqSdk 多品种 K 线对齐工具 (KlineAligner)
================================================================================

【TqSdk 简介】
TqSdk（天勤量化 SDK）是由信易科技（Shinny Tech）开发的专业级期货量化交易框架，
官方文档：https://doc.shinnytech.com/tqsdk/latest/

TqSdk 是国内主流的期货量化交易 Python SDK，主要特点包括：
  - 实时行情：支持全品种期货合约的 Tick 与 K 线数据，延迟极低；
  - 历史数据：通过 get_kline_serial() 可获取任意周期的历史 K 线，
    时间精度从 1 秒到日线均支持；
  - 多品种订阅：可同时订阅多个品种的行情，适合跨品种套利和相关性分析；
  - 异步驱动：基于异步 IO 设计，wait_update() 机制高效处理实时数据推送；
  - 数据格式：K 线数据以 pandas DataFrame 形式返回，每行对应一根 K 线，
    列包括：datetime（纳秒时间戳）、open、high、low、close、volume、open_oi 等；
  - 完整交易支持：从行情到下单、风控、持仓管理一体化；
  - 跨品种分析支持：可同时拉取多品种数据，但各品种的时间戳可能因交易时段
    差异（如有无夜盘）而产生错位，需要进行对齐处理后才能用于相关性分析。

安装方式：
    pip install tqsdk

官方文档：https://doc.shinnytech.com/tqsdk/latest/
GitHub：https://github.com/shinnytech/tqsdk-python

================================================================================
【工具说明】KlineAligner — 多品种 K 线对齐工具
================================================================================

跨品种量化分析的核心挑战：K 线时间对齐问题
---------------------------------------------

在使用 TqSdk 进行多品种分析时，不同品种的 K 线往往存在时间戳不一致的情况：

1. 夜盘时段差异：部分品种（如金融期货 IF、IH）无夜盘，但商品期货（如螺纹钢
   rb、铜 cu）有夜盘，导致同一日期的 K 线数量不同；

2. 节假日差异：虽然所有期货品种遵循相同的交易日历，但偶尔因品种上市时间
   不同，早期历史数据会有缺失；

3. 停牌/临时停盘：个别品种可能因特殊原因停牌，导致该时刻无 K 线数据；

4. 数据延迟：实盘行情中，不同品种的数据推送时间可能有毫秒级差异；

本工具的解决方案：
------------------

KlineAligner 通过以下步骤实现多品种 K 线的精确对齐：

  1. 并发订阅：通过 TqSdk API 同时订阅多个品种的 K 线序列；

  2. 时间戳统一：将 TqSdk 返回的纳秒时间戳（int64）统一转换为 datetime 格式；

  3. 外连接对齐：以所有品种时间戳的并集为索引，使用 outer join 合并所有品种
     的 K 线数据，确保任何品种有行情的时刻都被保留；

  4. 缺失值处理：对缺失的 K 线数据提供多种填充策略：
     - 'ffill'：用前一个有效 K 线数据填充（适合价格类数据）；
     - 'zero'：用 0 填充（适合成交量类数据）；
     - 'drop'：直接删除含缺失值的行（适合要求所有品种同时有数据的场景）；
     - 'keep'：保留 NaN（适合需要自定义处理的场景）；

  5. 结果返回：返回对齐后的 pandas DataFrame，列名格式为"品种_字段"，
     如"SHFE.rb2501_close"，方便后续统计分析；

典型应用场景：
--------------
  - 跨品种相关性分析（计算 Pearson/Spearman 相关系数矩阵）；
  - 统计套利信号生成（价差、比价、协整检验）；
  - 多因子模型中的品种特征提取；
  - 跨市场联动分析（商品期货与股指期货的相关性）；

输出示例（日线对齐后的 DataFrame）：
--------------------------------------
    datetime            SHFE.rb2501_close  DCE.i2501_close  SHFE.cu2501_close
    2024-01-02 09:00    3800.0             820.0             68000.0
    2024-01-02 10:00    3810.0             825.0             68050.0
    ...

================================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from datetime import datetime


# ==============================================================================
# K 线对齐核心工具类
# ==============================================================================

class KlineAligner:
    """
    多品种 K 线对齐工具

    可以将多个 TqSdk K 线 DataFrame（由 api.get_kline_serial() 返回）
    对齐到统一时间轴，处理缺失值，并返回适合跨品种分析的宽表 DataFrame。

    参数说明：
    -----------
    symbols : list of str
        需要对齐的品种合约代码列表，格式为"交易所.合约代码"，
        如 ["SHFE.rb2501", "DCE.i2501", "SHFE.cu2501"]。

    duration_seconds : int
        K 线周期（秒），常用值：
          - 60: 1 分钟线
          - 300: 5 分钟线
          - 3600: 60 分钟线
          - 86400: 日线

    fields : list of str
        需要提取的 K 线字段，默认为 ["close"]。
        TqSdk K 线支持的字段：open、high、low、close、volume、open_oi 等。

    fill_method : str
        缺失值填充方式，可选：
          - 'ffill'：前向填充（默认）
          - 'zero'：填充 0
          - 'drop'：删除含缺失值的行
          - 'keep'：保留 NaN 不处理

    data_length : int
        每个品种拉取的 K 线条数，默认 2000 条。
    """

    def __init__(
        self,
        symbols: List[str],
        duration_seconds: int = 86400,
        fields: Optional[List[str]] = None,
        fill_method: str = "ffill",
        data_length: int = 2000,
    ):
        if not symbols:
            raise ValueError("symbols 不能为空，请至少提供 1 个品种代码。")

        self.symbols = symbols
        self.duration_seconds = duration_seconds
        self.fields = fields if fields else ["close"]
        self.fill_method = fill_method
        self.data_length = data_length

        # 存储每个品种的原始 K 线数据（由外部调用 load_from_api 或 load_from_dict 填充）
        self._raw_klines: Dict[str, pd.DataFrame] = {}

        # 对齐后的宽表 DataFrame
        self._aligned_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 数据加载方法
    # ------------------------------------------------------------------

    def load_from_api(self, api) -> None:
        """
        通过 TqSdk API 实时订阅并加载各品种 K 线数据。

        参数：
            api: 已初始化的 TqApi 实例

        注意：
            调用此方法前需确保 api 已完成初始化（即已调用过 api.wait_update()
            至少一次，确保行情数据已推送到本地）。
        """
        print(f"[KlineAligner] 开始订阅 {len(self.symbols)} 个品种的 K 线数据...")

        kline_serials = {}
        for symbol in self.symbols:
            print(f"  - 订阅 {symbol} ({self._duration_label()})...")
            kline_serials[symbol] = api.get_kline_serial(
                symbol=symbol,
                duration_seconds=self.duration_seconds,
                data_length=self.data_length,
            )

        # 等待所有品种数据就绪
        print("[KlineAligner] 等待行情数据推送完成...")
        api.wait_update()

        # 将 TqSdk 返回的 DataFrame 存入内部字典
        for symbol, klines in kline_serials.items():
            df = klines.copy()
            # TqSdk 的 datetime 字段是纳秒时间戳（int64），转换为 datetime
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ns")
            df = df.set_index("datetime")
            self._raw_klines[symbol] = df
            print(f"  - {symbol}: 共加载 {len(df)} 条 K 线")

        print("[KlineAligner] 数据加载完成。")

    def load_from_dict(self, klines_dict: Dict[str, pd.DataFrame]) -> None:
        """
        从外部传入已有的 K 线 DataFrame 字典（用于测试或离线分析）。

        参数：
            klines_dict: dict，key 为品种代码，value 为该品种的 K 线 DataFrame。
                DataFrame 必须包含 'datetime'（或已设为 index）和 fields 中指定的列。

        示例：
            aligner.load_from_dict({
                "SHFE.rb2501": rb_klines_df,
                "DCE.i2501": iron_klines_df,
            })
        """
        for symbol, df in klines_dict.items():
            df = df.copy()
            # 如果 datetime 不是 index，则设置为 index
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"品种 {symbol} 的 DataFrame 必须包含 datetime 列或使用 DatetimeIndex。"
                )
            self._raw_klines[symbol] = df
        print(f"[KlineAligner] 从字典加载了 {len(klines_dict)} 个品种的 K 线数据。")

    # ------------------------------------------------------------------
    # 对齐逻辑
    # ------------------------------------------------------------------

    def align(self) -> pd.DataFrame:
        """
        执行 K 线对齐操作，返回对齐后的宽表 DataFrame。

        对齐步骤：
          1. 提取各品种指定字段的数据；
          2. 以时间戳为 key，outer join 合并所有品种数据；
          3. 按选定策略处理缺失值；
          4. 按时间升序排序；

        返回：
            pd.DataFrame: 列名格式为"{symbol}_{field}"的宽表，索引为 datetime。
        """
        if not self._raw_klines:
            raise RuntimeError(
                "尚未加载数据，请先调用 load_from_api() 或 load_from_dict()。"
            )

        print(f"[KlineAligner] 开始对齐 {len(self._raw_klines)} 个品种的 K 线...")

        # 第一步：提取各品种的指定字段，重命名列为"品种_字段"
        dfs_to_merge = []
        for symbol, df in self._raw_klines.items():
            # 检查字段是否存在
            available_fields = [f for f in self.fields if f in df.columns]
            missing_fields = [f for f in self.fields if f not in df.columns]
            if missing_fields:
                print(f"  [警告] {symbol} 缺少字段: {missing_fields}，将跳过这些字段。")

            if not available_fields:
                print(f"  [警告] {symbol} 没有任何可用字段，跳过该品种。")
                continue

            sub_df = df[available_fields].copy()
            # 重命名列：close -> SHFE.rb2501_close
            sub_df.columns = [f"{symbol}_{f}" for f in available_fields]
            dfs_to_merge.append(sub_df)

        if not dfs_to_merge:
            raise ValueError("没有可用的 K 线数据可供对齐，请检查 fields 参数和数据。")

        # 第二步：使用 outer join 合并所有品种（以时间戳并集为索引）
        aligned = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            aligned = aligned.join(df, how="outer")

        # 第三步：按时间升序排序
        aligned = aligned.sort_index()

        # 第四步：处理缺失值
        aligned = self._handle_missing(aligned)

        print(f"[KlineAligner] 对齐完成：共 {len(aligned)} 行，{len(aligned.columns)} 列。")
        self._aligned_df = aligned
        return aligned

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据 fill_method 参数处理缺失值。

        参数：
            df: 待处理的 DataFrame

        返回：
            pd.DataFrame: 处理后的 DataFrame
        """
        method = self.fill_method.lower()

        if method == "ffill":
            # 前向填充：适合价格数据（无行情时延续上一个价格）
            return df.ffill()

        elif method == "zero":
            # 填充 0：适合成交量（无成交时为 0）
            return df.fillna(0)

        elif method == "drop":
            # 删除含缺失值的行：只保留所有品种都有数据的时刻
            before = len(df)
            df = df.dropna()
            after = len(df)
            print(f"  [fill=drop] 删除了 {before - after} 行含缺失值的数据，保留 {after} 行。")
            return df

        elif method == "keep":
            # 保留 NaN，由调用方自行处理
            return df

        else:
            raise ValueError(
                f"不支持的 fill_method: '{method}'，"
                f"请使用 'ffill'、'zero'、'drop' 或 'keep'。"
            )

    # ------------------------------------------------------------------
    # 统计分析辅助方法
    # ------------------------------------------------------------------

    def correlation_matrix(
        self,
        field: str = "close",
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        计算各品种指定字段的相关系数矩阵。

        先检查是否已执行 align()，如未执行则自动调用。

        参数：
            field: 用于计算相关性的字段，默认 'close'
            method: 相关系数计算方法，支持 'pearson'、'spearman'、'kendall'

        返回：
            pd.DataFrame: N×N 相关系数矩阵，N 为品种数
        """
        if self._aligned_df is None:
            self.align()

        # 提取指定字段的列
        close_cols = [f"{s}_{field}" for s in self.symbols if f"{s}_{field}" in self._aligned_df.columns]

        if len(close_cols) < 2:
            raise ValueError(f"至少需要 2 个品种的 '{field}' 数据才能计算相关性。")

        price_df = self._aligned_df[close_cols].dropna()
        # 计算对数收益率（更适合相关性分析）
        ret_df = price_df.pct_change().dropna()

        # 简化列名用于显示
        ret_df.columns = [c.replace(f"_{field}", "").split(".")[-1] for c in close_cols]

        corr = ret_df.corr(method=method)
        return corr

    def get_aligned(self) -> Optional[pd.DataFrame]:
        """
        返回最近一次对齐的结果 DataFrame。

        如果尚未调用 align()，返回 None。
        """
        return self._aligned_df

    def missing_stats(self) -> pd.DataFrame:
        """
        统计各品种在对齐后的缺失值情况。

        返回：
            pd.DataFrame: 每列的缺失行数和缺失率
        """
        if self._aligned_df is None:
            raise RuntimeError("请先调用 align() 方法。")

        total = len(self._aligned_df)
        stats = []
        for col in self._aligned_df.columns:
            missing = self._aligned_df[col].isna().sum()
            stats.append({
                "列名": col,
                "总行数": total,
                "缺失行数": missing,
                "缺失率": f"{missing / total * 100:.2f}%"
            })
        return pd.DataFrame(stats)

    def _duration_label(self) -> str:
        """返回 K 线周期的可读标签"""
        mapping = {
            60: "1分钟线",
            300: "5分钟线",
            900: "15分钟线",
            1800: "30分钟线",
            3600: "60分钟线",
            86400: "日线",
        }
        return mapping.get(self.duration_seconds, f"{self.duration_seconds}秒线")


# ==============================================================================
# 独立辅助函数：从 TqSdk API 快速对齐多品种 K 线
# ==============================================================================

def align_klines_from_api(
    api,
    symbols: List[str],
    duration_seconds: int = 86400,
    fields: Optional[List[str]] = None,
    fill_method: str = "ffill",
    data_length: int = 2000,
) -> pd.DataFrame:
    """
    便捷函数：直接传入 TqApi 实例，快速获取多品种对齐后的 K 线 DataFrame。

    参数：
        api: TqApi 实例
        symbols: 品种代码列表
        duration_seconds: K 线周期（秒）
        fields: 需要的字段列表，默认 ['close']
        fill_method: 缺失值填充方式
        data_length: 每个品种拉取的 K 线条数

    返回：
        pd.DataFrame: 对齐后的多品种宽表

    示例：
        from tqsdk import TqApi, TqAuth
        api = TqApi(auth=TqAuth("用户名", "密码"))
        df = align_klines_from_api(
            api,
            symbols=["SHFE.rb2501", "DCE.i2501"],
            duration_seconds=86400,
            fields=["close", "volume"]
        )
        print(df.tail())
        api.close()
    """
    aligner = KlineAligner(
        symbols=symbols,
        duration_seconds=duration_seconds,
        fields=fields,
        fill_method=fill_method,
        data_length=data_length,
    )
    aligner.load_from_api(api)
    return aligner.align()


# ==============================================================================
# 使用示例
# ==============================================================================

def run_demo():
    """
    多品种 K 线对齐使用示例（使用模拟数据，无需 TqSdk 账户）

    本示例演示：
      1. 生成模拟的多品种 K 线数据
      2. 使用 KlineAligner 对齐时间戳
      3. 处理缺失值
      4. 计算品种间相关性矩阵
    """
    print("=" * 65)
    print("    KlineAligner — 多品种 K 线对齐工具 演示")
    print("=" * 65)

    # ---------------------------------------------------------------
    # 第一步：生成模拟数据（模拟不同品种时间戳不一致的情况）
    # ---------------------------------------------------------------
    np.random.seed(2024)

    # 模拟 rb2501（螺纹钢）日线：全部工作日（含夜盘日期）
    rb_dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
    rb_close = 3800 + np.cumsum(np.random.normal(0, 30, len(rb_dates)))

    rb_df = pd.DataFrame({
        "datetime": rb_dates,
        "open": rb_close - np.random.uniform(5, 15, len(rb_dates)),
        "high": rb_close + np.random.uniform(10, 30, len(rb_dates)),
        "low": rb_close - np.random.uniform(10, 30, len(rb_dates)),
        "close": rb_close,
        "volume": np.random.randint(50000, 200000, len(rb_dates)),
    }).set_index("datetime")

    # 模拟 i2501（铁矿石）日线：缺少若干日期（模拟停牌/无数据情况）
    # 随机删除 10% 的交易日，模拟缺失
    iron_dates = rb_dates[np.random.choice([True, False], len(rb_dates), p=[0.9, 0.1])]
    iron_close = 820 + np.cumsum(np.random.normal(0, 8, len(iron_dates)))

    iron_df = pd.DataFrame({
        "datetime": iron_dates,
        "open": iron_close - np.random.uniform(1, 5, len(iron_dates)),
        "high": iron_close + np.random.uniform(3, 10, len(iron_dates)),
        "low": iron_close - np.random.uniform(3, 10, len(iron_dates)),
        "close": iron_close,
        "volume": np.random.randint(20000, 100000, len(iron_dates)),
    }).set_index("datetime")

    # 模拟 cu2501（铜）日线：只有工作日且无夜盘
    cu_dates = rb_dates[::2]   # 隔天一根（极端情况模拟）
    cu_close = 68000 + np.cumsum(np.random.normal(0, 300, len(cu_dates)))

    cu_df = pd.DataFrame({
        "datetime": cu_dates,
        "close": cu_close,
        "volume": np.random.randint(10000, 50000, len(cu_dates)),
    }).set_index("datetime")

    # ---------------------------------------------------------------
    # 第二步：初始化 KlineAligner 并加载模拟数据
    # ---------------------------------------------------------------
    symbols = ["SHFE.rb2501", "DCE.i2501", "SHFE.cu2501"]

    aligner = KlineAligner(
        symbols=symbols,
        duration_seconds=86400,
        fields=["close", "volume"],
        fill_method="ffill",   # 价格用前向填充，成交量实际应用中可用 'zero'
        data_length=2000,
    )

    aligner.load_from_dict({
        "SHFE.rb2501": rb_df,
        "DCE.i2501": iron_df,
        "SHFE.cu2501": cu_df,
    })

    # ---------------------------------------------------------------
    # 第三步：执行对齐
    # ---------------------------------------------------------------
    aligned_df = aligner.align()

    print(f"\n【对齐结果（前 8 行）】")
    print(aligned_df.head(8).to_string())

    print(f"\n【对齐结果（后 5 行）】")
    print(aligned_df.tail(5).to_string())

    # ---------------------------------------------------------------
    # 第四步：查看缺失值统计
    # ---------------------------------------------------------------
    print(f"\n【对齐前（ffill 填充前）的缺失情况统计】")
    # 用 keep 模式重新对齐查看缺失情况
    aligner_check = KlineAligner(
        symbols=symbols,
        duration_seconds=86400,
        fields=["close"],
        fill_method="keep",
    )
    aligner_check.load_from_dict({
        "SHFE.rb2501": rb_df,
        "DCE.i2501": iron_df,
        "SHFE.cu2501": cu_df,
    })
    aligner_check.align()
    print(aligner_check.missing_stats().to_string(index=False))

    # ---------------------------------------------------------------
    # 第五步：计算相关性矩阵
    # ---------------------------------------------------------------
    print(f"\n【品种间收盘价日收益率相关系数矩阵（Pearson）】")
    corr_matrix = aligner.correlation_matrix(field="close", method="pearson")
    print(corr_matrix.round(4).to_string())

    print(f"\n【品种间收盘价日收益率相关系数矩阵（Spearman）】")
    corr_spearman = aligner.correlation_matrix(field="close", method="spearman")
    print(corr_spearman.round(4).to_string())

    # ---------------------------------------------------------------
    # 第六步：演示与 TqSdk 实盘/回测结合的用法
    # ---------------------------------------------------------------
    print("\n" + "=" * 65)
    print("【TqSdk 实盘使用示例代码（需真实账户）】")
    print("=" * 65)
    print("""
from tqsdk import TqApi, TqAuth
from tools.02_kline_aligner import align_klines_from_api

# 初始化 TqSdk API
api = TqApi(auth=TqAuth("天勤用户名", "天勤密码"))

# 一行代码获取多品种对齐日线数据
aligned = align_klines_from_api(
    api=api,
    symbols=["SHFE.rb2501", "DCE.i2501", "SHFE.cu2501", "SHFE.hc2501"],
    duration_seconds=86400,        # 日线
    fields=["close", "volume"],
    fill_method="ffill",
    data_length=500,               # 最近 500 根日线
)

# 计算相关性矩阵
aligner = KlineAligner(
    symbols=["SHFE.rb2501", "DCE.i2501", "SHFE.cu2501", "SHFE.hc2501"],
    fields=["close"],
    fill_method="ffill",
)
aligner.load_from_dict({"SHFE.rb2501": ..., ...})
corr = aligner.correlation_matrix()
print(corr)

api.close()
""")

    print("=" * 65)
    print("[演示完成] KlineAligner 多品种 K 线对齐工具运行成功！")
    print("=" * 65)

    return aligned_df, aligner


if __name__ == "__main__":
    aligned_df, aligner = run_demo()

    # 提取收盘价列，方便直接用于后续分析
    close_cols = [c for c in aligned_df.columns if c.endswith("_close")]
    print(f"\n可直接用于分析的收盘价列：{close_cols}")
    print(aligned_df[close_cols].describe().round(2))
