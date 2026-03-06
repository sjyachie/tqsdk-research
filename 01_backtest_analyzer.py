"""
================================================================================
TqSdk 回测绩效评估器 (BacktestAnalyzer)
================================================================================

【TqSdk 简介】
TqSdk（天勤量化 SDK）是由信易科技（Shinny Tech）开发的专业级期货量化交易框架，
官方文档：https://doc.shinnytech.com/tqsdk/latest/

TqSdk 提供了完整的量化交易解决方案，涵盖：
  - 实时行情订阅：支持 Tick 级别和 K 线数据，覆盖国内全品种期货合约；
  - 自动化交易：支持限价单、市价单、条件单等多种委托方式，并提供持仓管理；
  - 历史回测：内置 `backtest` 模式，可以对任意时间段的历史数据进行策略回测；
  - 实盘模拟：支持在不消耗真实资金的情况下进行策略验证；
  - 多账户管理：可同时管理多个账户，适合机构量化团队；
  - 数据分析支持：提供 pandas DataFrame 格式的数据输出，便于策略研究和绩效评估。

TqSdk 兼容 Python 3.6+，安装方式：
    pip install tqsdk

注意事项：
  - 回测时需使用 TqBacktest 对象作为 auth 参数；
  - 账户对象 TqAccount 在回测模式下会模拟真实交易环境；
  - 资金和持仓信息通过 `api.get_account()` 和 `api.get_position()` 获取。

官方网站：https://www.shinnytech.com/
官方文档：https://doc.shinnytech.com/tqsdk/latest/
GitHub：https://github.com/shinnytech/tqsdk-python

================================================================================
【工具说明】BacktestAnalyzer — TqSdk 回测绩效评估器
================================================================================

本工具用于对 TqSdk 回测结果进行全面的绩效评估，主要功能包括：

1. 年化收益率计算
   - 基于回测起止日期和最终净值变化，计算策略的年化复利收益率；
   - 公式：年化收益率 = (期末资产 / 期初资产)^(365 / 回测天数) - 1

2. 夏普比率（Sharpe Ratio）
   - 衡量策略的风险调整后收益，公式：Sharpe = (日均收益 - 无风险利率) / 日收益标准差 * sqrt(252)
   - 无风险利率默认使用年化 3%（可自定义）；
   - 夏普比率越高，说明单位风险带来的超额收益越高；

3. 最大回撤（Max Drawdown）
   - 统计回测期间账户净值从峰值到谷底的最大跌幅；
   - 是衡量策略风险控制能力的重要指标；

4. 胜率（Win Rate）
   - 统计所有已平仓交易中盈利笔数的占比；
   - 胜率 = 盈利交易数 / 总交易数；

5. 盈亏比（Profit/Loss Ratio）
   - 平均单笔盈利额 / 平均单笔亏损额的绝对值；
   - 与胜率结合使用，评估策略的期望收益；

6. 日收益分布
   - 统计每日收益的均值、标准差、偏度、峰度等分布特征；
   - 支持输出为 pandas DataFrame，便于进一步分析和可视化；

使用方式：
    analyzer = BacktestAnalyzer(account_df=account_df_data, trade_df=trade_df_data)
    summary = analyzer.summary()
    print(summary)
    df = analyzer.to_dataframe()

================================================================================
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any


class BacktestAnalyzer:
    """
    TqSdk 回测绩效评估器

    输入回测过程中记录的账户资金曲线 DataFrame 和交易记录 DataFrame，
    自动计算并输出各项绩效指标，并支持导出为 DataFrame 格式。

    参数说明：
    -----------
    account_df : pd.DataFrame
        账户每日资产快照 DataFrame，必须包含以下列：
          - date (str/datetime): 日期
          - balance (float): 账户权益（即当日结算后账户总资产）
        该 DataFrame 通常由回测过程中每日记录 api.get_account().balance 得到。

    trade_df : pd.DataFrame
        交易记录 DataFrame，必须包含以下列：
          - trade_date_time (str/datetime): 成交时间
          - direction (str): 方向，'BUY' 或 'SELL'
          - offset (str): 开平，'OPEN' 或 'CLOSE'
          - volume (int): 成交手数
          - price (float): 成交价格
          - profit (float): 本笔平仓盈亏（仅平仓记录有效）
        如果没有交易记录，可传入空 DataFrame，胜率/盈亏比将返回 NaN。

    risk_free_rate : float
        年化无风险利率，默认 0.03（即 3%），用于计算夏普比率。
    """

    def __init__(
        self,
        account_df: pd.DataFrame,
        trade_df: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.03
    ):
        # 深拷贝，避免修改原始数据
        self.account_df = account_df.copy()
        self.trade_df = trade_df.copy() if trade_df is not None else pd.DataFrame()
        self.risk_free_rate = risk_free_rate

        # 确保 date 列为 datetime 类型
        if "date" in self.account_df.columns:
            self.account_df["date"] = pd.to_datetime(self.account_df["date"])
            self.account_df = self.account_df.sort_values("date").reset_index(drop=True)

        # 计算每日收益率
        if "balance" in self.account_df.columns and len(self.account_df) > 1:
            self.account_df["daily_return"] = self.account_df["balance"].pct_change()
        else:
            self.account_df["daily_return"] = np.nan

    # ------------------------------------------------------------------
    # 核心指标计算方法
    # ------------------------------------------------------------------

    def annual_return(self) -> float:
        """
        计算年化收益率

        公式：(期末资产 / 期初资产)^(365 / 回测总天数) - 1
        回测总天数不足 1 天时返回 NaN。

        返回：
            float: 年化收益率，如 0.25 表示 25%
        """
        if len(self.account_df) < 2:
            return float("nan")

        start_balance = self.account_df["balance"].iloc[0]
        end_balance = self.account_df["balance"].iloc[-1]
        start_date = self.account_df["date"].iloc[0]
        end_date = self.account_df["date"].iloc[-1]
        days = (end_date - start_date).days

        if days <= 0 or start_balance <= 0:
            return float("nan")

        # 年化复利公式
        return (end_balance / start_balance) ** (365.0 / days) - 1.0

    def sharpe_ratio(self) -> float:
        """
        计算年化夏普比率

        公式：(日均超额收益率) / (日收益标准差) * sqrt(252)
        日均超额收益率 = 日均收益率 - 日无风险利率（年化无风险利率 / 252）

        返回：
            float: 夏普比率
        """
        daily_returns = self.account_df["daily_return"].dropna()
        if len(daily_returns) < 2:
            return float("nan")

        # 日均收益率
        mean_return = daily_returns.mean()
        # 日无风险利率（将年化利率转换为日利率）
        daily_risk_free = self.risk_free_rate / 252.0
        # 日收益标准差
        std_return = daily_returns.std(ddof=1)

        if std_return == 0:
            return float("nan")

        # 年化夏普比率
        sharpe = (mean_return - daily_risk_free) / std_return * math.sqrt(252)
        return sharpe

    def max_drawdown(self) -> float:
        """
        计算最大回撤

        从账户净值曲线中找出最大的从峰值到谷底的下跌幅度。
        公式：max_dd = max((峰值 - 谷底) / 峰值)

        返回：
            float: 最大回撤（正值，如 0.15 表示 15%）
        """
        if "balance" not in self.account_df.columns or len(self.account_df) < 2:
            return float("nan")

        balance_series = self.account_df["balance"].values
        peak = balance_series[0]
        max_dd = 0.0

        for b in balance_series:
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def win_rate(self) -> float:
        """
        计算胜率

        基于交易记录中的平仓盈亏，统计盈利笔数占总平仓笔数的比例。
        只计算 offset == 'CLOSE'（或 'CLOSETODAY'）的记录。

        返回：
            float: 胜率（0~1），如 0.6 表示 60%
        """
        if self.trade_df.empty or "profit" not in self.trade_df.columns:
            return float("nan")

        # 过滤平仓记录
        close_trades = self.trade_df[
            self.trade_df.get("offset", pd.Series(dtype=str)).isin(
                ["CLOSE", "CLOSETODAY", "平仓"]
            )
        ] if "offset" in self.trade_df.columns else self.trade_df

        if len(close_trades) == 0:
            return float("nan")

        win_count = (close_trades["profit"] > 0).sum()
        return win_count / len(close_trades)

    def profit_loss_ratio(self) -> float:
        """
        计算盈亏比

        公式：平均盈利额 / 平均亏损额（绝对值）
        只统计有效平仓记录。

        返回：
            float: 盈亏比，如 2.0 表示平均每笔盈利是平均亏损的 2 倍
        """
        if self.trade_df.empty or "profit" not in self.trade_df.columns:
            return float("nan")

        close_trades = self.trade_df[
            self.trade_df.get("offset", pd.Series(dtype=str)).isin(
                ["CLOSE", "CLOSETODAY", "平仓"]
            )
        ] if "offset" in self.trade_df.columns else self.trade_df

        wins = close_trades[close_trades["profit"] > 0]["profit"]
        losses = close_trades[close_trades["profit"] < 0]["profit"]

        if len(wins) == 0 or len(losses) == 0:
            return float("nan")

        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        return avg_win / avg_loss if avg_loss > 0 else float("nan")

    def daily_return_stats(self) -> Dict[str, float]:
        """
        统计日收益分布特征

        返回日收益率的均值、标准差、偏度、峰度、最大值、最小值等统计量。

        返回：
            dict: 包含日收益分布各统计指标的字典
        """
        daily_returns = self.account_df["daily_return"].dropna()

        if len(daily_returns) < 2:
            return {}

        return {
            "日均收益率": round(daily_returns.mean(), 6),
            "日收益标准差": round(daily_returns.std(ddof=1), 6),
            "日收益偏度": round(float(daily_returns.skew()), 4),
            "日收益峰度": round(float(daily_returns.kurt()), 4),
            "日最大收益": round(daily_returns.max(), 6),
            "日最大亏损": round(daily_returns.min(), 6),
            "正收益天数": int((daily_returns > 0).sum()),
            "负收益天数": int((daily_returns < 0).sum()),
            "零收益天数": int((daily_returns == 0).sum()),
        }

    # ------------------------------------------------------------------
    # 综合汇总与导出
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """
        输出所有绩效指标的汇总字典

        返回：
            dict: 包含所有绩效指标的字典，可直接打印或转为 DataFrame
        """
        ar = self.annual_return()
        sr = self.sharpe_ratio()
        md = self.max_drawdown()
        wr = self.win_rate()
        pl = self.profit_loss_ratio()
        dr = self.daily_return_stats()

        result = {
            "年化收益率": f"{ar * 100:.2f}%" if not math.isnan(ar) else "N/A",
            "夏普比率": f"{sr:.4f}" if not math.isnan(sr) else "N/A",
            "最大回撤": f"{md * 100:.2f}%" if not math.isnan(md) else "N/A",
            "胜率": f"{wr * 100:.2f}%" if not math.isnan(wr) else "N/A",
            "盈亏比": f"{pl:.4f}" if not math.isnan(pl) else "N/A",
        }
        result.update(dr)

        # 附加账户信息
        if len(self.account_df) >= 2:
            result["期初资产"] = self.account_df["balance"].iloc[0]
            result["期末资产"] = self.account_df["balance"].iloc[-1]
            result["回测天数"] = (
                self.account_df["date"].iloc[-1] - self.account_df["date"].iloc[0]
            ).days

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        将绩效汇总指标导出为 pandas DataFrame

        返回：
            pd.DataFrame: 以"指标名称"和"指标数值"为两列的 DataFrame
        """
        summary = self.summary()
        df = pd.DataFrame(
            list(summary.items()),
            columns=["指标名称", "指标数值"]
        )
        return df

    def account_curve(self) -> pd.DataFrame:
        """
        返回带日收益率的账户净值曲线 DataFrame

        返回：
            pd.DataFrame: 包含 date、balance、daily_return 列的 DataFrame
        """
        cols = [c for c in ["date", "balance", "daily_return"] if c in self.account_df.columns]
        return self.account_df[cols].copy()

    def print_report(self) -> None:
        """
        在控制台打印格式化的绩效报告
        """
        print("=" * 60)
        print("          TqSdk 回测绩效报告 (BacktestAnalyzer)")
        print("=" * 60)
        for key, val in self.summary().items():
            print(f"  {key:<20s}: {val}")
        print("=" * 60)


# ==============================================================================
# 使用示例（结合 TqSdk 回测模式）
# ==============================================================================

def run_backtest_example():
    """
    TqSdk 回测示例：在回测模式下运行简单策略并进行绩效评估。

    注意：运行此示例需要 TqSdk 账户权限（天勤量化账户）。
    安装：pip install tqsdk

    本示例演示：
      1. 使用 TqBacktest 进行历史回测
      2. 逐日记录账户资产和交易记录
      3. 使用 BacktestAnalyzer 评估绩效
    """
    try:
        from tqsdk import TqApi, TqAuth, TqBacktest, TqAccount
        from tqsdk.backtest import TqBacktest
        import datetime

        # -------------------------------------------------------
        # 第一步：初始化 TqSdk API（回测模式）
        # -------------------------------------------------------
        # 请替换为您的天勤账户用户名和密码
        api = TqApi(
            account=TqAccount("期货公司名称", "资金账号", "密码"),
            auth=TqAuth("天勤用户名", "天勤密码"),
            backtest=TqBacktest(
                start_dt=datetime.date(2024, 1, 1),
                end_dt=datetime.date(2024, 6, 30),
            )
        )

        # -------------------------------------------------------
        # 第二步：订阅 K 线行情
        # -------------------------------------------------------
        symbol = "SHFE.rb2501"     # 螺纹钢主力合约
        klines = api.get_kline_serial(symbol, 60 * 60 * 24)  # 日线

        # -------------------------------------------------------
        # 第三步：运行回测主循环，记录账户数据
        # -------------------------------------------------------
        account_records = []   # 每日账户资产快照
        trade_records = []     # 交易记录

        position = api.get_position(symbol)
        account = api.get_account()

        prev_close = None
        in_position = False

        while True:
            api.wait_update()

            # 每当日线 K 线更新时进行决策
            if api.is_changing(klines.iloc[-1], "datetime"):
                current_close = klines.iloc[-2]["close"]  # 前一根完整日线收盘价
                current_date = str(klines.iloc[-2]["datetime"])[:10]

                # 记录当日账户净值
                account_records.append({
                    "date": current_date,
                    "balance": account.balance
                })

                # 简单均线策略示例（仅用于演示，非投资建议）
                if prev_close is not None:
                    # 上涨则做多（简化逻辑，实际策略应更完善）
                    if current_close > prev_close and not in_position:
                        api.insert_order(
                            symbol=symbol,
                            direction="BUY",
                            offset="OPEN",
                            volume=1,
                            limit_price=current_close
                        )
                        in_position = True
                    elif current_close < prev_close and in_position:
                        api.insert_order(
                            symbol=symbol,
                            direction="SELL",
                            offset="CLOSE",
                            volume=1,
                            limit_price=current_close
                        )
                        in_position = False

                prev_close = current_close

    except BacktestFinished:
        # 回测结束，进行绩效分析
        pass
    except Exception as e:
        print(f"TqSdk 回测异常（可能未安装或无账户）：{e}")
        print("使用模拟数据演示 BacktestAnalyzer 功能...\n")

        # -------------------------------------------------------
        # 模拟数据演示（无需 TqSdk 账户也可运行）
        # -------------------------------------------------------
        import numpy as np
        np.random.seed(42)

        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")  # 工作日
        n = len(dates)

        # 模拟账户净值曲线（初始 100 万，随机游走）
        daily_rets = np.random.normal(0.001, 0.015, n)   # 日均收益 0.1%，波动 1.5%
        balance = [1_000_000.0]
        for r in daily_rets[1:]:
            balance.append(balance[-1] * (1 + r))

        account_df = pd.DataFrame({
            "date": dates,
            "balance": balance
        })

        # 模拟交易记录
        profits = np.random.normal(500, 2000, 80)   # 80 笔交易，盈亏随机
        trade_df = pd.DataFrame({
            "offset": ["CLOSE"] * 80,
            "profit": profits
        })

        # -------------------------------------------------------
        # 初始化 BacktestAnalyzer 并输出报告
        # -------------------------------------------------------
        analyzer = BacktestAnalyzer(
            account_df=account_df,
            trade_df=trade_df,
            risk_free_rate=0.03
        )

        # 打印格式化报告
        analyzer.print_report()

        # 输出 DataFrame 格式
        print("\n【DataFrame 格式输出】")
        result_df = analyzer.to_dataframe()
        print(result_df.to_string(index=False))

        # 输出账户净值曲线（前5行）
        print("\n【账户净值曲线（前5行）】")
        print(analyzer.account_curve().head())

        return analyzer


if __name__ == "__main__":
    analyzer = run_backtest_example()
    if analyzer:
        # 也可以单独获取某个指标
        print(f"\n年化收益率: {analyzer.annual_return() * 100:.2f}%")
        print(f"夏普比率: {analyzer.sharpe_ratio():.4f}")
        print(f"最大回撤: {analyzer.max_drawdown() * 100:.2f}%")
