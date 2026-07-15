"""
指标计算引擎
从事件日志中聚合计算 AB 实验的各项指标。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from experiment.config_schema import MetricConfig


@dataclass
class MetricResult:
    """单个指标的计算结果"""
    metric_name: str
    control_value: float
    treatment_value: float
    control_sample_size: int
    treatment_sample_size: int
    relative_change: float      # 相对变化 (treatment - control) / control
    absolute_change: float      # 绝对变化
    control_variance: float
    treatment_variance: float


class MetricCalculator:
    """
    指标计算引擎。

    输入格式（event_log DataFrame）：
        user_id   | group     | event_name          | value | timestamp
        u001      | control   | booking_success      | 1     | 2026-07-14 10:00:00
        u002      | treatment | booking_success      | 1     | 2026-07-14 10:01:00
        u001      | control   | booking_flow_start   | 1     | 2026-07-14 09:59:00

    重要设计注意事项（面试会问）：
        - 比率指标（如 CTR）的"分析单元"是用户，而不是事件。
          若以事件为单元，会违反独立性假设（同一用户的多次事件相关）。
          正确做法：先在用户层聚合，再跨用户做统计检验。
        - 这就是为什么需要 delta method 来估计比率指标的方差。
    """

    def __init__(self, event_log: pd.DataFrame):
        """
        Args:
            event_log: 事件日志 DataFrame，必须包含 [user_id, group, event_name, value, timestamp]
        """
        required_cols = {"user_id", "group", "event_name", "value", "timestamp"}
        missing = required_cols - set(event_log.columns)
        if missing:
            raise ValueError(f"event_log 缺少列：{missing}")
        self.event_log = event_log.copy()
        self.event_log["timestamp"] = pd.to_datetime(self.event_log["timestamp"])

        # holdout 组数据不参与指标计算，提前告警避免静默丢失
        holdout_users = self.event_log[self.event_log["group"] == "holdout"]["user_id"].nunique()
        if holdout_users > 0:
            print(
                f"[MetricCalculator] 警告：event_log 中包含 {holdout_users} 个 holdout 组用户，"
                f"这些用户不参与 control/treatment 指标计算。"
                f"如需分析 holdout 长期效果，请单独传入 holdout 用户数据。"
            )

    def calculate(self, metric: MetricConfig, pre_data: Optional[Dict[str, pd.Series]] = None) -> "MetricResult":
        """
        计算单个指标，根据 aggregation 类型选择计算方法。

        Args:
            metric:   指标配置
            pre_data: 可选，CUPED 协变量数据，格式为
                      {"control": pd.Series(per_user_pre_value),
                       "treatment": pd.Series(per_user_pre_value)}
                      传入后自动对 post 指标做 CUPED 调整，降低方差、提升功效。
        """
        if metric.aggregation == "ratio":
            return self._calculate_ratio_metric(metric, pre_data=pre_data)
        elif metric.aggregation == "mean":
            return self._calculate_mean_metric(metric, pre_data=pre_data)
        elif metric.aggregation == "sum":
            return self._calculate_sum_metric(metric)
        else:
            raise ValueError(f"不支持的 aggregation 类型: {metric.aggregation}")

    def calculate_all(self, metrics: List[MetricConfig], pre_data: Optional[Dict] = None) -> Dict[str, "MetricResult"]:
        """批量计算多个指标。"""
        return {m.name: self.calculate(m, pre_data=pre_data) for m in metrics}

    def time_series_analysis(
        self, metric: MetricConfig, date_col: str = "timestamp"
    ) -> Dict[str, Dict]:
        """
        按天计算指标走势，用于检测 Novelty Effect。

        Novelty Effect：新功能上线初期，用户因好奇心产生异常行为（点击过多），
        前几天效果虚高，后续衰减到真实水平。如果在 novelty 窗口内下结论，
        会高估实验收益。

        判断方法：比较前 3 天 vs 后 4 天的相对变化，若后段相对变化 < 前段的 70%，
                  输出 novelty effect 警告。

        Returns:
            {
                "daily": [{date, control_value, treatment_value, relative_change}, ...],
                "novelty_warning": bool,
                "early_lift": float,   # 前 3 天平均相对变化
                "late_lift": float,    # 后 4 天平均相对变化
            }
        """
        df = self.event_log.copy()
        df["_date"] = pd.to_datetime(df[date_col]).dt.date
        dates = sorted(df["_date"].unique())
        daily_rows = []

        for d in dates:
            day_df = df[df["_date"] == d]
            day_calc = MetricCalculator.__new__(MetricCalculator)
            day_calc.event_log = day_df
            try:
                result = day_calc.calculate(metric)
                if result.control_sample_size > 0 and result.treatment_sample_size > 0:
                    daily_rows.append({
                        "date": str(d),
                        "control_value": result.control_value,
                        "treatment_value": result.treatment_value,
                        "relative_change": result.relative_change,
                    })
            except Exception:
                pass  # 某天数据不足，跳过

        # Novelty effect 检测：前 3 天 vs 后续天的相对变化衰减
        novelty_warning = False
        early_lift, late_lift = 0.0, 0.0
        if len(daily_rows) >= 5:
            early_changes = [r["relative_change"] for r in daily_rows[:3]]
            late_changes = [r["relative_change"] for r in daily_rows[3:]]
            early_lift = float(np.mean(early_changes))
            late_lift = float(np.mean(late_changes))
            # 如果后段提升 < 前段提升的 70%，且前段提升为正，触发警告
            if early_lift > 0 and late_lift < early_lift * 0.7:
                novelty_warning = True

        return {
            "daily": daily_rows,
            "novelty_warning": novelty_warning,
            "early_lift": early_lift,
            "late_lift": late_lift,
        }

    def subgroup_analysis(
        self, metric: MetricConfig, subgroup_col: str
    ) -> Dict[str, "MetricResult"]:
        """
        分层分析：按 subgroup_col 字段分别计算指标，返回各子组的 MetricResult。

        用途（面试高频）：
            新老用户对预约完成率的基线差异很大（新用户 ~5%，老用户 ~20%+）。
            若两组的新老用户比例不一致，全量指标会受 Simpson's Paradox 影响。
            分层分析可以确认实验在各子群体中的效果是否一致。

        Args:
            metric:       指标配置
            subgroup_col: 分层字段名，如 "user_type"（需在 event_log 中存在）

        Returns:
            {"new": MetricResult, "returning": MetricResult, ...}
        """
        if subgroup_col not in self.event_log.columns:
            raise ValueError(
                f"event_log 中不存在列 '{subgroup_col}'，"
                f"可用列：{list(self.event_log.columns)}"
            )
        subgroups = self.event_log[subgroup_col].dropna().unique()
        results = {}
        for sg_val in subgroups:
            sg_log = self.event_log[self.event_log[subgroup_col] == sg_val]
            sub_calc = MetricCalculator.__new__(MetricCalculator)
            sub_calc.event_log = sg_log
            results[str(sg_val)] = sub_calc.calculate(metric)
        return results

    # ──────────────────────────────────────────────────────────

    def _calculate_ratio_metric(self, metric: MetricConfig, pre_data: Optional[Dict] = None) -> "MetricResult":
        """
        计算比率类指标（如 CTR = 点击量 / 曝光量）。

        关键：以用户为分析单元（user-level aggregation），
              即先算每个用户的分子总数和分母总数，再跨用户聚合。
        若传入 pre_data，对每用户的比率值（numerator/denominator）做 CUPED 调整。
        """
        numerator_df = self.event_log[
            self.event_log["event_name"] == metric.numerator_event
        ].groupby(["user_id", "group"])["value"].sum().reset_index(name="numerator")

        denominator_df = self.event_log[
            self.event_log["event_name"] == metric.denominator_event
        ].groupby(["user_id", "group"])["value"].sum().reset_index(name="denominator")

        user_df = denominator_df.merge(numerator_df, on=["user_id", "group"], how="left")
        user_df["numerator"] = user_df["numerator"].fillna(0)

        results = {}
        for group in ["control", "treatment"]:
            grp = user_df[user_df["group"] == group]
            num_sum = grp["numerator"].sum()
            den_sum = grp["denominator"].sum()
            rate = num_sum / den_sum if den_sum > 0 else 0.0

            # 用户层比率（用于 CUPED 调整）
            user_rate = (grp["numerator"] / grp["denominator"].replace(0, np.nan)).fillna(0)

            if pre_data and group in pre_data:
                from stats.cuped import cuped_adjust
                pre_series = pd.Series(pre_data[group].values, index=user_rate.index)
                pre_series = pre_series.reindex(user_rate.index).fillna(pre_series.mean())
                user_rate_cuped = cuped_adjust(pre_series, user_rate)
                variance = user_rate_cuped.var(ddof=1) / len(user_rate_cuped) if len(user_rate_cuped) > 1 else 0
            else:
                # Delta method 方差估计（见 stats/delta_method.py 中的原理）
                variance = self._delta_method_variance(grp["numerator"], grp["denominator"])

            results[group] = {
                "value": rate,
                "n": len(grp),
                "variance": variance,
            }

        ctrl, trt = results["control"], results["treatment"]
        rel_change = (trt["value"] - ctrl["value"]) / ctrl["value"] if ctrl["value"] != 0 else 0

        return MetricResult(
            metric_name=metric.name,
            control_value=round(ctrl["value"], 6),
            treatment_value=round(trt["value"], 6),
            control_sample_size=ctrl["n"],
            treatment_sample_size=trt["n"],
            relative_change=round(rel_change, 6),
            absolute_change=round(trt["value"] - ctrl["value"], 6),
            control_variance=ctrl["variance"],
            treatment_variance=trt["variance"],
        )

    def _calculate_mean_metric(self, metric: MetricConfig, pre_data: Optional[Dict] = None) -> "MetricResult":
        """计算均值类指标（如平均停留时长）。若传入 pre_data 则做 CUPED 调整。"""
        value_df = self.event_log[self.event_log["event_name"].notna()]
        if metric.value_field and metric.value_field in self.event_log.columns:
            value_col = metric.value_field
        else:
            value_col = "value"

        results = {}
        for group in ["control", "treatment"]:
            grp_vals = self.event_log[self.event_log["group"] == group].groupby("user_id")[value_col].mean()
            if pre_data and group in pre_data:
                from stats.cuped import cuped_adjust
                pre_series = pd.Series(pre_data[group].values, index=grp_vals.index[:len(pre_data[group])])
                pre_series = pre_series.reindex(grp_vals.index).fillna(pre_series.mean())
                grp_vals_cuped = cuped_adjust(pre_series, grp_vals.reset_index(drop=True))
                results[group] = {
                    "value": float(grp_vals.mean()),
                    "n": len(grp_vals),
                    "variance": grp_vals_cuped.var(ddof=1) / len(grp_vals_cuped) if len(grp_vals_cuped) > 1 else 0,
                }
            else:
                results[group] = {
                    "value": grp_vals.mean(),
                    "n": len(grp_vals),
                    "variance": grp_vals.var() / len(grp_vals) if len(grp_vals) > 1 else 0,
                }

        ctrl, trt = results["control"], results["treatment"]
        rel_change = (trt["value"] - ctrl["value"]) / ctrl["value"] if ctrl["value"] != 0 else 0

        return MetricResult(
            metric_name=metric.name,
            control_value=round(float(ctrl["value"]), 4),
            treatment_value=round(float(trt["value"]), 4),
            control_sample_size=ctrl["n"],
            treatment_sample_size=trt["n"],
            relative_change=round(float(rel_change), 6),
            absolute_change=round(float(trt["value"] - ctrl["value"]), 4),
            control_variance=float(ctrl["variance"]),
            treatment_variance=float(trt["variance"]),
        )

    def _calculate_sum_metric(self, metric: MetricConfig) -> "MetricResult":
        """计算汇总类指标。"""
        value_col = metric.value_field or "value"
        results = {}
        for group in ["control", "treatment"]:
            grp = self.event_log[self.event_log["group"] == group]
            user_sums = grp.groupby("user_id")[value_col].sum()
            results[group] = {
                "value": user_sums.mean(),  # 人均 sum
                "n": len(user_sums),
                "variance": user_sums.var() / len(user_sums) if len(user_sums) > 1 else 0,
            }

        ctrl, trt = results["control"], results["treatment"]
        rel_change = (trt["value"] - ctrl["value"]) / ctrl["value"] if ctrl["value"] != 0 else 0

        return MetricResult(
            metric_name=metric.name,
            control_value=round(float(ctrl["value"]), 4),
            treatment_value=round(float(trt["value"]), 4),
            control_sample_size=ctrl["n"],
            treatment_sample_size=trt["n"],
            relative_change=round(float(rel_change), 6),
            absolute_change=round(float(trt["value"] - ctrl["value"]), 4),
            control_variance=float(ctrl["variance"]),
            treatment_variance=float(trt["variance"]),
        )

    @staticmethod
    def _delta_method_variance(numerator: pd.Series, denominator: pd.Series) -> float:
        """
        用 Delta Method 估计比率指标的用户层方差。
        详细推导见 stats/delta_method.py。
        """
        n = len(numerator)
        if n == 0:
            return 0.0
        mu_num = numerator.mean()
        mu_den = denominator.mean()
        if mu_den == 0:
            return 0.0

        var_num = numerator.var(ddof=1) if n > 1 else 0
        var_den = denominator.var(ddof=1) if n > 1 else 0
        cov_nd = numerator.cov(denominator) if n > 1 else 0

        # Delta method: Var(X/Y) ≈ (μY²·VarX + μX²·VarY - 2μXμY·Cov(X,Y)) / (n·μY⁴)
        variance = (
            mu_den ** 2 * var_num
            + mu_num ** 2 * var_den
            - 2 * mu_num * mu_den * cov_nd
        ) / (n * mu_den ** 4)

        return max(variance, 0.0)
