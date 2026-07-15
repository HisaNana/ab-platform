"""
序贯检验单测（stats/sequential.py + HypothesisTester 序贯模式集成）
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from stats.sequential import (
    obrien_fleming_boundary,
    pocock_boundary,
    alpha_spending_test,
    mixture_sequential_probability_ratio,
)
from stats.t_test import HypothesisTester
from metrics.metric_calculator import MetricResult
from experiment.config_schema import MetricConfig, MetricType


class TestObrienFlemingBoundary:
    """O'Brien-Fleming 边界性质验证。"""

    def test_boundary_decreases_as_information_increases(self):
        """信息量占比越大，边界应越小（早期严格，后期宽松）。"""
        b_early = obrien_fleming_boundary(information_fraction=0.2, alpha=0.05)
        b_mid = obrien_fleming_boundary(information_fraction=0.5, alpha=0.05)
        b_late = obrien_fleming_boundary(information_fraction=1.0, alpha=0.05)
        assert b_early > b_mid > b_late, (
            f"边界应随信息量单调递减: t=0.2 -> {b_early:.4f}, "
            f"t=0.5 -> {b_mid:.4f}, t=1.0 -> {b_late:.4f}"
        )

    def test_boundary_converges_to_standard_z_at_t1(self):
        """信息量占比=1 时，边界应收敛到标准 z_{alpha/2}。"""
        boundary = obrien_fleming_boundary(information_fraction=1.0, alpha=0.05)
        z_alpha_half = scipy_stats.norm.ppf(1 - 0.05 / 2)
        assert abs(boundary - z_alpha_half) < 1e-9, (
            f"t=1 时边界 {boundary:.6f} 应等于标准 z_alpha/2 {z_alpha_half:.6f}"
        )

    def test_invalid_information_fraction_raises(self):
        """信息量占比超出 (0, 1] 应报错。"""
        with pytest.raises(ValueError):
            obrien_fleming_boundary(information_fraction=0.0)
        with pytest.raises(ValueError):
            obrien_fleming_boundary(information_fraction=1.5)


class TestPocockBoundary:
    """Pocock 边界性质验证。"""

    def test_boundary_constant_across_information_fraction(self):
        """Pocock 边界应恒定，不随信息量占比变化（与 O'Brien-Fleming 的核心区别）。"""
        b1 = pocock_boundary(information_fraction=0.2, alpha=0.05, total_analyses=5)
        b2 = pocock_boundary(information_fraction=0.6, alpha=0.05, total_analyses=5)
        b3 = pocock_boundary(information_fraction=1.0, alpha=0.05, total_analyses=5)
        assert b1 == b2 == b3, f"Pocock 边界应恒定，实际: {b1}, {b2}, {b3}"

    def test_boundary_higher_than_standard_z(self):
        """Pocock 边界应高于标准单次检验的 z_alpha/2（因为要覆盖多次查看的假阳性膨胀）。"""
        boundary = pocock_boundary(information_fraction=1.0, alpha=0.05, total_analyses=5)
        z_alpha_half = scipy_stats.norm.ppf(1 - 0.05 / 2)
        assert boundary > z_alpha_half

    def test_boundary_increases_with_more_planned_analyses(self):
        """计划查看次数越多，Pocock 边界应越高（每次都更难显著，控制总体假阳性率）。"""
        b_3 = pocock_boundary(information_fraction=1.0, alpha=0.05, total_analyses=3)
        b_10 = pocock_boundary(information_fraction=1.0, alpha=0.05, total_analyses=10)
        assert b_10 > b_3


class TestAlphaSpendingTest:
    """alpha_spending_test() 集成行为验证。"""

    def test_early_look_does_not_falsely_reject_under_null(self):
        """
        AA 场景（真实无差异）下，早期查看（低信息量占比）即使 z 值中等偏大，
        也不应轻易被 O'Brien-Fleming 边界判定为显著 —— 这正是序贯检验防止 peeking 的核心价值。
        """
        # 固定样本量检验下 z=2.0 已经显著（|z| > 1.96），但信息量仅 20% 时不应显著
        result = alpha_spending_test(
            z_statistic=2.0, information_fraction=0.2,
            alpha=0.05, spending_function="obrien_fleming",
        )
        assert not result["is_significant"], (
            f"早期查看（t=0.2）z=2.0 不应越过 O'Brien-Fleming 边界，"
            f"实际边界={result['boundary_z']}"
        )

    def test_full_information_matches_standard_test(self):
        """信息量占比=1 时，判定结果应与标准 Z 检验一致。"""
        z = 2.5
        result = alpha_spending_test(
            z_statistic=z, information_fraction=1.0,
            alpha=0.05, spending_function="obrien_fleming",
        )
        standard_significant = abs(z) > scipy_stats.norm.ppf(1 - 0.05 / 2)
        assert result["is_significant"] == standard_significant

    def test_pocock_spending_function(self):
        """pocock 分支可正常调用并返回结构化结果。"""
        result = alpha_spending_test(
            z_statistic=2.6, information_fraction=0.5,
            alpha=0.05, spending_function="pocock", total_analyses=5,
        )
        assert "boundary_z" in result
        assert result["spending_function"] == "pocock"

    def test_invalid_spending_function_raises(self):
        with pytest.raises(ValueError):
            alpha_spending_test(
                z_statistic=2.0, information_fraction=0.5,
                spending_function="not_a_real_method",
            )


class TestMSPRT:
    """mSPRT 似然比检验行为验证。"""

    def test_likelihood_ratio_increases_with_effect_size(self):
        """真实效应量越大（sample_mean_diff 越大），似然比应越大。"""
        r_small = mixture_sequential_probability_ratio(
            n=5000, sample_mean_diff=0.001, sample_variance=0.02,
        )
        r_large = mixture_sequential_probability_ratio(
            n=5000, sample_mean_diff=0.02, sample_variance=0.02,
        )
        assert r_large["likelihood_ratio"] > r_small["likelihood_ratio"]

    def test_likelihood_ratio_eventually_exceeds_threshold(self):
        """效应量足够大且样本量足够多时，似然比应越过 1/alpha 阈值，判定显著。"""
        result = mixture_sequential_probability_ratio(
            n=20000, sample_mean_diff=0.03, sample_variance=0.02, alpha=0.05,
        )
        assert result["is_significant"], (
            f"大效应量+大样本下应显著，实际 likelihood_ratio={result['likelihood_ratio']}, "
            f"threshold={result['threshold']}"
        )

    def test_false_positive_rate_lower_than_naive_repeated_testing(self):
        """
        模拟 H0 为真（无实际效应）时反复查看的场景：
        mSPRT 的假阳性率应明显低于"每次都用固定样本量 Z 检验判定"的 naive 重复检验方式。
        """
        np.random.seed(42)
        n_simulations = 100
        look_points = [1000, 2000, 3000, 4000, 5000]
        true_variance = 0.02

        naive_false_positives = 0
        msprt_false_positives = 0

        for _ in range(n_simulations):
            # H0 为真：全程无真实均值差，模拟每个 look_point 下的样本均值差（服从正态分布）
            naive_rejected = False
            msprt_rejected = False
            for n in look_points:
                se = np.sqrt(true_variance / n)
                observed_diff = np.random.normal(0, se)

                # naive：每次查看都做一次固定样本量 Z 检验，只要显著就"停"（peeking）
                z_stat = observed_diff / se
                if abs(z_stat) > scipy_stats.norm.ppf(1 - 0.05 / 2):
                    naive_rejected = True

                # mSPRT：anytime-valid，同样每次查看
                result = mixture_sequential_probability_ratio(
                    n=n, sample_mean_diff=observed_diff, sample_variance=true_variance,
                    alpha=0.05, tau_squared=0.001,
                )
                if result["is_significant"]:
                    msprt_rejected = True

            if naive_rejected:
                naive_false_positives += 1
            if msprt_rejected:
                msprt_false_positives += 1

        naive_fpr = naive_false_positives / n_simulations
        msprt_fpr = msprt_false_positives / n_simulations
        assert msprt_fpr <= naive_fpr, (
            f"mSPRT 假阳性率 {msprt_fpr:.2%} 应不高于 naive 重复检验假阳性率 {naive_fpr:.2%}"
        )

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            mixture_sequential_probability_ratio(n=0, sample_mean_diff=0.01, sample_variance=0.02)

    def test_invalid_variance_raises(self):
        with pytest.raises(ValueError):
            mixture_sequential_probability_ratio(n=1000, sample_mean_diff=0.01, sample_variance=-1)


class TestHypothesisTesterSequentialIntegration:
    """HypothesisTester 序贯模式集成 & 向后兼容回归测试。"""

    def _make_metric_result_and_config(self, control_value=0.15, treatment_value=0.18,
                                        n_ctrl=5000, n_trt=5000, variance=0.0002):
        metric_result = MetricResult(
            metric_name="booking_completion_rate",
            control_value=control_value,
            treatment_value=treatment_value,
            relative_change=(treatment_value - control_value) / control_value,
            absolute_change=treatment_value - control_value,
            control_variance=variance,
            treatment_variance=variance,
            control_sample_size=n_ctrl,
            treatment_sample_size=n_trt,
        )
        metric_config = MetricConfig(
            name="booking_completion_rate",
            metric_type=MetricType.PRIMARY,
            description="预约完成率",
            aggregation="ratio",
            direction=1,
        )
        return metric_result, metric_config

    def test_default_mode_unaffected_by_new_fields(self):
        """默认模式下（use_sequential_testing=False），sequential_boundary/method_used 应为 None。"""
        tester = HypothesisTester(alpha=0.05)
        metric_result, metric_config = self._make_metric_result_and_config()
        result = tester.test(metric_result, metric_config)
        assert result.sequential_boundary is None
        assert result.sequential_method_used is None

    def test_alpha_spending_mode_fills_sequential_fields(self):
        """开启 alpha_spending 序贯模式后，应正确填充 sequential_boundary 字段。"""
        tester = HypothesisTester(
            alpha=0.05, use_sequential_testing=True,
            sequential_method="alpha_spending", sequential_spending_function="obrien_fleming",
        )
        metric_result, metric_config = self._make_metric_result_and_config()
        result = tester.test(metric_result, metric_config, information_fraction=0.4)
        assert result.sequential_method_used == "alpha_spending"
        assert result.sequential_boundary is not None
        assert "boundary_z" in result.sequential_boundary

    def test_alpha_spending_mode_requires_information_fraction(self):
        """alpha_spending 模式下缺少 information_fraction 应报错。"""
        tester = HypothesisTester(alpha=0.05, use_sequential_testing=True, sequential_method="alpha_spending")
        metric_result, metric_config = self._make_metric_result_and_config()
        with pytest.raises(ValueError):
            tester.test(metric_result, metric_config)

    def test_msprt_mode_fills_sequential_fields(self):
        """开启 mSPRT 序贯模式后，应正确填充 sequential_boundary 字段，且不需要 information_fraction。"""
        tester = HypothesisTester(alpha=0.05, use_sequential_testing=True, sequential_method="msprt")
        metric_result, metric_config = self._make_metric_result_and_config()
        result = tester.test(metric_result, metric_config)
        assert result.sequential_method_used == "msprt"
        assert result.sequential_boundary is not None
        assert "likelihood_ratio" in result.sequential_boundary
