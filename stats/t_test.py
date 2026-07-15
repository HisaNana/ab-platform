"""
假设检验模块
支持：双样本 t 检验、比率检验（Z 检验）、多重比较校正。
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from dataclasses import dataclass
from metrics.metric_calculator import MetricResult
from experiment.config_schema import MetricConfig, MetricType


@dataclass
class TestResult:
    """单个指标的检验结果"""
    metric_name: str
    metric_type: str
    # 核心统计量
    control_value: float
    treatment_value: float
    relative_change: float
    absolute_change: float
    # 检验结果
    test_statistic: float
    p_value: float
    p_value_corrected: Optional[float]    # 多重比较校正后的 p_value
    confidence_interval: tuple            # 相对变化的 95% CI
    # 结论
    is_significant: bool
    direction_correct: bool               # 变化方向是否符合预期（越大/越小越好）
    conclusion: str
    sample_size_control: int
    sample_size_treatment: int
    # 序贯检验相关（默认 None，仅在 use_sequential_testing=True 时填充）
    sequential_boundary: Optional[dict] = None   # alpha_spending_test()/mSPRT 返回的完整结果
    sequential_method_used: Optional[str] = None  # "alpha_spending" | "msprt"


class HypothesisTester:
    """
    假设检验引擎。

    支持的检验方式：
        - 均值类指标：Welch's t-test（不假设方差相等，更鲁棒）
        - 比率类指标：基于 Delta Method 方差的 Z 检验

    多重比较校正：
        - 为什么需要？
          同时检验 10 个指标，每个 alpha=0.05，有约 40% 概率出现至少一个假阳性。
        - Bonferroni：最保守，将 alpha / 检验次数，适合少量重要指标
        - Benjamini-Hochberg（BH）：控制 FDR，探索性指标常用，比 Bonferroni 宽松

    最佳实践（面试高频）：
        - 只对北极星指标（PRIMARY）应用最严格的 Bonferroni
        - 护栏指标只要有任何显著下降就失败，不需要多重校正
        - 探索性指标仅供参考，不校正，结论中明确说明

    序贯检验（可选，解决 peeking problem）：
        默认关闭（use_sequential_testing=False），行为与固定样本量检验完全一致。
        开启后，test() 需额外传入 information_fraction（Alpha Spending）或直接依赖
        n/方差（mSPRT），内部调用 stats/sequential.py 覆盖 is_significant 判定，
        并将中间结果记录到 TestResult.sequential_boundary。
        注意：序贯模式下不建议再叠加 test_multiple 的多重比较校正
        （二者都是为控制假阳性率设计的机制，同时使用会过度保守，本类不做强制拦截，
        由调用方自行决定是否两者取一）。
    """

    def __init__(
        self,
        alpha: float = 0.05,
        correction_method: str = "bonferroni",  # "bonferroni" | "bh" | "none"
        use_sequential_testing: bool = False,
        sequential_method: str = "alpha_spending",           # "alpha_spending" | "msprt"
        sequential_spending_function: str = "obrien_fleming",  # "obrien_fleming" | "pocock"
    ):
        self.alpha = alpha
        self.correction_method = correction_method
        self.use_sequential_testing = use_sequential_testing
        self.sequential_method = sequential_method
        self.sequential_spending_function = sequential_spending_function

    def test(
        self,
        metric_result: MetricResult,
        metric_config: MetricConfig,
        is_one_sided: bool = False,
        information_fraction: Optional[float] = None,
    ) -> TestResult:
        """
        对单个指标执行假设检验。

        Args:
            metric_result: 指标计算结果
            metric_config: 指标配置（含方向、类型）
            is_one_sided:  是否使用单尾检验（通常用双尾，更保守）
            information_fraction: 当前信息量占比（已积累样本/计划总样本），(0, 1]。
                仅在 self.use_sequential_testing=True 且 sequential_method="alpha_spending"
                时必须提供；mSPRT 模式不需要此参数（可任意时刻调用）。
        """
        n_ctrl = metric_result.control_sample_size
        n_trt = metric_result.treatment_sample_size

        # 计算标准误差（基于 Delta Method 或直接方差）
        se_ctrl = np.sqrt(metric_result.control_variance)
        se_trt = np.sqrt(metric_result.treatment_variance)
        se_diff = np.sqrt(se_ctrl ** 2 + se_trt ** 2)

        diff = metric_result.treatment_value - metric_result.control_value
        test_stat = diff / se_diff if se_diff > 0 else 0.0

        # p-value（双尾）
        if metric_result.control_sample_size > 30:
            # 大样本：使用 Z 检验
            p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        else:
            # 小样本：使用 t 检验
            df = n_ctrl + n_trt - 2
            p_value = 2 * (1 - stats.t.cdf(abs(test_stat), df=df))

        if is_one_sided:
            p_value /= 2

        # 95% 置信区间
        z_critical = stats.norm.ppf(0.975)
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        # 转换为相对变化 CI
        base = metric_result.control_value
        rel_ci = (
            round(ci_lower / base, 4) if base != 0 else 0,
            round(ci_upper / base, 4) if base != 0 else 0,
        )

        is_significant = p_value < self.alpha
        # 方向判断：+1 表示越大越好，-1 表示越小越好
        direction_correct = (
            (metric_config.direction == 1 and diff > 0) or
            (metric_config.direction == -1 and diff < 0)
        )

        # 序贯检验路径（可选）：用序贯边界覆盖 is_significant 判定
        sequential_boundary = None
        sequential_method_used = None
        if self.use_sequential_testing:
            from stats.sequential import alpha_spending_test, mixture_sequential_probability_ratio

            sequential_method_used = self.sequential_method
            if self.sequential_method == "alpha_spending":
                if information_fraction is None:
                    raise ValueError(
                        "use_sequential_testing=True 且 sequential_method='alpha_spending' 时，"
                        "必须传入 information_fraction（已积累样本 / 计划总样本）。"
                    )
                sequential_boundary = alpha_spending_test(
                    z_statistic=test_stat,
                    information_fraction=information_fraction,
                    alpha=self.alpha,
                    spending_function=self.sequential_spending_function,
                )
            elif self.sequential_method == "msprt":
                pooled_variance = metric_result.control_variance + metric_result.treatment_variance
                sequential_boundary = mixture_sequential_probability_ratio(
                    n=n_ctrl + n_trt,
                    sample_mean_diff=diff,
                    sample_variance=pooled_variance,
                    alpha=self.alpha,
                )
            else:
                raise ValueError(
                    f"不支持的 sequential_method: {self.sequential_method}，"
                    "请使用 'alpha_spending' 或 'msprt'"
                )
            is_significant = sequential_boundary["is_significant"]

        conclusion = self._make_conclusion(
            metric_config, is_significant, direction_correct,
            metric_result.relative_change, p_value
        )

        return TestResult(
            metric_name=metric_result.metric_name,
            metric_type=metric_config.metric_type.value,
            control_value=metric_result.control_value,
            treatment_value=metric_result.treatment_value,
            relative_change=metric_result.relative_change,
            absolute_change=metric_result.absolute_change,
            test_statistic=round(test_stat, 4),
            p_value=round(p_value, 6),
            p_value_corrected=None,  # 由 test_multiple 填充
            confidence_interval=rel_ci,
            is_significant=is_significant,
            direction_correct=direction_correct,
            conclusion=conclusion,
            sample_size_control=n_ctrl,
            sample_size_treatment=n_trt,
            sequential_boundary=sequential_boundary,
            sequential_method_used=sequential_method_used,
        )

    def test_multiple(
        self,
        metric_results: Dict[str, MetricResult],
        metric_configs: Dict[str, MetricConfig],
    ) -> Dict[str, TestResult]:
        """
        批量检验多个指标，并根据校正方法调整 p_value。

        多重比较策略（针对通用健康产品场景）：
            - PRIMARY 指标：使用 Bonferroni 校正（只有 1 个时无需校正）
            - GUARDRAIL 指标：单独检验，不参与多重校正；任一显著下降即实验失败
            - EXPLORATORY 指标：不校正，在结论中注明仅供参考
        """
        raw_results = {
            name: self.test(metric_results[name], metric_configs[name])
            for name in metric_results
        }

        if self.correction_method == "none":
            return raw_results

        # 分组：只对 PRIMARY + EXPLORATORY 进行多重校正，GUARDRAIL 单独处理
        primary_names = [
            n for n, c in metric_configs.items() if c.metric_type == MetricType.PRIMARY
        ]
        exploratory_names = [
            n for n, c in metric_configs.items() if c.metric_type == MetricType.EXPLORATORY
        ]
        correct_names = primary_names + exploratory_names

        if not correct_names:
            return raw_results

        p_values = [raw_results[n].p_value for n in correct_names]

        if self.correction_method == "bonferroni":
            corrected = [min(p * len(correct_names), 1.0) for p in p_values]
        elif self.correction_method == "bh":
            corrected = self._bh_correction(p_values)
        else:
            corrected = p_values

        for name, p_corr in zip(correct_names, corrected):
            raw_results[name].p_value_corrected = round(p_corr, 6)
            raw_results[name].is_significant = p_corr < self.alpha

        return raw_results

    def make_experiment_conclusion(self, test_results: Dict[str, TestResult]) -> str:
        """
        综合所有指标结果，输出实验最终结论。

        决策框架：
            1. 任一护栏指标显著下降 → 实验失败，无论北极星如何
            2. 北极星指标显著正向 → 实验成功候选
            3. 北极星不显著 → 中性，继续观察或增加样本
            4. 北极星显著负向 → 实验失败
        """
        lines = ["## 实验综合结论\n"]

        # 护栏检查
        guardrails = [r for r in test_results.values() if r.metric_type == "guardrail"]
        failed_guardrails = [
            r for r in guardrails
            if r.is_significant and not r.direction_correct
        ]
        if failed_guardrails:
            lines.append("### ❌ 护栏指标告警（实验失败）")
            for r in failed_guardrails:
                lines.append(
                    f"- **{r.metric_name}**: 显著{('下降' if r.relative_change < 0 else '上升')} "
                    f"{r.relative_change:.2%}，p={r.p_value:.4f}"
                )
            lines.append("\n> 存在护栏指标显著劣化，**不建议推全量**，需排查原因。\n")
        else:
            lines.append("### ✅ 护栏指标：全部通过\n")

        # 北极星指标
        primaries = [r for r in test_results.values() if r.metric_type == "primary"]
        if primaries:
            r = primaries[0]
            p_display = r.p_value_corrected or r.p_value
            if r.is_significant and r.direction_correct:
                lines.append(
                    f"### 🎉 北极星指标显著正向：{r.metric_name}\n"
                    f"- 相对变化：{r.relative_change:+.2%}，p={p_display:.4f}\n"
                    f"- 95% CI：[{r.confidence_interval[0]:+.2%}, {r.confidence_interval[1]:+.2%}]\n"
                )
                if not failed_guardrails:
                    lines.append("**推荐：可推全量。**")
            elif r.is_significant and not r.direction_correct:
                lines.append(
                    f"### ❌ 北极星指标显著负向：{r.metric_name}\n"
                    f"- 相对变化：{r.relative_change:+.2%}，p={p_display:.4f}\n"
                    "**推荐：回滚实验组策略。**"
                )
            else:
                lines.append(
                    f"### 🔍 北极星指标不显著：{r.metric_name}\n"
                    f"- 相对变化：{r.relative_change:+.2%}，p={p_display:.4f}（未达到 α={self.alpha}）\n"
                    f"- 95% CI：[{r.confidence_interval[0]:+.2%}, {r.confidence_interval[1]:+.2%}]\n"
                    "**推荐：样本量不足或效果不明显，可延长实验或评估是否值得继续。**"
                )

        # 探索性指标摘要
        exploratory = [r for r in test_results.values() if r.metric_type == "exploratory"]
        if exploratory:
            lines.append("\n### 探索性指标（仅供参考，未校正 p-value）")
            for r in exploratory:
                sig_mark = "⭐" if r.is_significant else "—"
                direction = "↑" if r.relative_change > 0 else "↓"
                lines.append(
                    f"- {sig_mark} {r.metric_name}: {direction} {r.relative_change:+.2%}，p={r.p_value:.4f}"
                )

        return "\n".join(lines)

    @staticmethod
    def _bh_correction(p_values: List[float]) -> List[float]:
        """Benjamini-Hochberg FDR 校正。"""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_pvals = np.array(p_values)[sorted_indices]
        corrected = np.minimum(sorted_pvals * n / (np.arange(1, n + 1)), 1.0)
        # 保证单调性
        for i in range(n - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])
        result = np.empty(n)
        result[sorted_indices] = corrected
        return result.tolist()

    @staticmethod
    def _make_conclusion(metric_config, is_significant, direction_correct, rel_change, p_value) -> str:
        metric_type = metric_config.metric_type.value
        if not is_significant:
            return f"不显著（p={p_value:.4f}），变化 {rel_change:+.2%}，不足以下结论。"
        if direction_correct:
            return f"显著{'正向' if rel_change > 0 else '负向'}变化 {rel_change:+.2%}（p={p_value:.4f}）✅"
        else:
            dir_word = "下降" if metric_config.direction == 1 else "上升"
            return f"显著{dir_word} {rel_change:+.2%}（p={p_value:.4f}）⚠️ 方向不符预期。"
