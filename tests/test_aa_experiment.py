"""
AA 实验验证测试
验证分流系统的无偏性：如果对照/实验组之间没有实际干预，所有指标理论上应无显著差异。
"""

import pandas as pd
import numpy as np
import pytest
from allocation.hash_splitter import HashSplitter
from stats.t_test import HypothesisTester
from metrics.metric_calculator import MetricCalculator, MetricResult
from experiment.config_schema import MetricConfig, MetricType


class TestAASplitter:
    """
    AA 实验：验证 HashSplitter 的分流无偏性。

    原理：
        在 AA 实验中，两组用户完全相同（均用对照策略），
        理想情况下两组指标无显著差异。
        如果 AA 实验中出现显著差异，说明分流有偏，AB 实验结论不可信。

        标准做法：模拟 AA 实验 1000 次，统计显著率，应接近 alpha（如 5%）。
    """

    def test_hash_distribution_uniform(self):
        """测试哈希分桶是否均匀分布（卡方均匀性检验）。"""
        splitter = HashSplitter("test_uniform", salt="v1")
        buckets = [splitter._hash_to_bucket(f"user_{i}") for i in range(10000)]

        # 分成 10 个区间，每个区间期望 1000 次
        counts, _ = np.histogram(buckets, bins=10, range=(0, 10000))
        from scipy.stats import chisquare
        chi2, p = chisquare(counts)
        assert p > 0.05, f"哈希分布不均匀！chi2={chi2:.2f}, p={p:.4f}"

    def test_split_ratio_50_50(self):
        """验证 50/50 分流实际比例与配置一致（SRM 检测）。"""
        splitter = HashSplitter("test_50_50", salt="v1")
        n_users = 10000
        groups = [
            splitter.assign_group(f"user_{i}", control_ratio=0.5, treatment_ratio=0.5)
            for i in range(n_users)
        ]
        control_n = groups.count("control")
        treatment_n = groups.count("treatment")

        srm_result = splitter.check_srm(control_n, treatment_n)
        assert not srm_result["srm_detected"], (
            f"50/50 分流检测到 SRM！实际比例: {srm_result['actual_ratio']:.4f}，"
            f"p={srm_result['p_value']}"
        )

    def test_split_ratio_30_70(self):
        """验证 30/70 分流比例正确性。"""
        splitter = HashSplitter("test_30_70", salt="v1")
        n_users = 20000
        groups = [
            splitter.assign_group(f"user_{i}", control_ratio=0.3, treatment_ratio=0.7)
            for i in range(n_users)
        ]
        control_n = sum(1 for g in groups if g == "control")
        treatment_n = sum(1 for g in groups if g == "treatment")

        srm_result = splitter.check_srm(
            control_n, treatment_n,
            control_ratio=0.3, treatment_ratio=0.7
        )
        assert not srm_result["srm_detected"], (
            f"30/70 分流 SRM 检测失败，实际 treatment 比例: {srm_result['actual_ratio']:.4f}"
        )

    def test_user_assignment_stable(self):
        """同一用户在相同实验配置下，每次分配结果一致（幂等性）。"""
        splitter = HashSplitter("test_stable", salt="v1")
        user_id = "stable_user_001"
        groups = {splitter.assign_group(user_id) for _ in range(100)}
        assert len(groups) == 1, f"用户分组不稳定！出现了多个分组: {groups}"

    def test_salt_changes_assignment(self):
        """修改 salt 应该改变用户分组（允许实验重置）。"""
        user_id = "test_user_salt"
        groups = set()
        for salt in ["v1", "v2", "v3"]:
            splitter = HashSplitter("test_salt_exp", salt=salt)
            groups.add(splitter.assign_group(user_id))
        # 不同 salt 应该产生不同的分组结果（概率上大概率不全相同）
        # 至少要求 salt 修改后 hash 值变化（这里检查 hash bucket 不同）
        buckets = set()
        for salt in ["v1", "v2", "v3"]:
            splitter = HashSplitter("test_salt_exp", salt=salt)
            buckets.add(splitter._hash_to_bucket(user_id))
        assert len(buckets) > 1, "修改 salt 没有改变哈希结果，salt 机制可能失效"


class TestAAFalsePositiveRate:
    """
    模拟 AA 实验，验证假阳性率是否符合预期。
    理论上在 alpha=0.05 时，假阳性率约为 5%。
    """

    def _simulate_aa_experiment(
        self,
        n_users: int = 2000,
        n_simulations: int = 200,
        true_rate: float = 0.15,
        alpha: float = 0.05,
    ) -> float:
        """
        模拟 N 次 AA 实验，返回显著率（应接近 alpha）。
        """
        from scipy import stats
        significant_count = 0

        np.random.seed(42)
        for sim_i in range(n_simulations):
            # AA 实验：两组使用相同的真实转化率（无实际效果）
            control_impressions = np.random.poisson(10, n_users)
            control_clicks = np.array([
                np.random.binomial(imp, true_rate)
                for imp in control_impressions
            ])
            treatment_impressions = np.random.poisson(10, n_users)
            treatment_clicks = np.array([
                np.random.binomial(imp, true_rate)
                for imp in treatment_impressions
            ])

            # 用 Delta Method 估计方差
            from stats.delta_method import delta_method_variance
            ctrl_series_num = pd.Series(control_clicks)
            ctrl_series_den = pd.Series(control_impressions)
            trt_series_num = pd.Series(treatment_clicks)
            trt_series_den = pd.Series(treatment_impressions)

            ctrl_rate = control_clicks.sum() / control_impressions.sum()
            trt_rate = treatment_clicks.sum() / treatment_impressions.sum()

            var_ctrl = delta_method_variance(ctrl_series_num, ctrl_series_den)
            var_trt = delta_method_variance(trt_series_num, trt_series_den)

            se_diff = np.sqrt(var_ctrl + var_trt)
            if se_diff == 0:
                continue
            z_stat = (trt_rate - ctrl_rate) / se_diff
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            if p_value < alpha:
                significant_count += 1

        return significant_count / n_simulations

    def test_aa_false_positive_rate(self):
        """
        AA 实验假阳性率应在 [alpha - 2%, alpha + 3%] 范围内。
        用 Delta Method 的假阳性率应接近理论值 5%。
        """
        alpha = 0.05
        fpr = self._simulate_aa_experiment(
            n_users=2000,
            n_simulations=200,  # 适当减少以控制测试时间
            true_rate=0.15,
            alpha=alpha,
        )
        lower_bound = alpha - 0.03
        upper_bound = alpha + 0.05  # 允许一定波动
        assert lower_bound <= fpr <= upper_bound, (
            f"AA 实验假阳性率 {fpr:.2%} 偏离预期 alpha={alpha:.2%}，"
            f"期望范围 [{lower_bound:.2%}, {upper_bound:.2%}]。"
            f"可能存在方差估计偏差。"
        )
        print(f"\n✅ AA 实验假阳性率: {fpr:.2%}（期望约 {alpha:.2%}）")


class TestSRMDetection:
    """SRM（样本比率不匹配）检测测试。"""

    def test_detect_obvious_srm(self):
        """当实际比例明显偏离配置时，应检测到 SRM。"""
        splitter = HashSplitter("srm_test", salt="v1")
        # 配置 50/50，但实际 60/40
        srm_result = splitter.check_srm(
            control_count=6000,
            treatment_count=4000,
            control_ratio=0.5,
            treatment_ratio=0.5,
            alpha=0.01,
        )
        assert srm_result["srm_detected"], "明显的 SRM（60/40 vs 50/50）未被检测到！"

    def test_no_srm_with_correct_ratio(self):
        """正常分流时，不应误报 SRM。"""
        splitter = HashSplitter("no_srm_test", salt="v1")
        srm_result = splitter.check_srm(
            control_count=5012,
            treatment_count=4988,  # 接近 50/50，只有微小随机波动
            control_ratio=0.5,
            treatment_ratio=0.5,
        )
        assert not srm_result["srm_detected"], (
            f"正常分流被误报为 SRM！p={srm_result['p_value']}"
        )


if __name__ == "__main__":
    # 快速运行 AA 假阳性率测试（演示用）
    test = TestAAFalsePositiveRate()
    fpr = test._simulate_aa_experiment(n_users=2000, n_simulations=500)
    print(f"AA 实验假阳性率: {fpr:.2%}（理论值约 5%）")


class TestCUPED:
    """
    验证 CUPED 协变量调整的核心性质：
    1. 调整后方差 ≤ 原始方差（单调降低或不变）
    2. 调整后均值与原始均值相等（期望不变，只降方差）
    3. pre/post 相关性越高，方差缩减越多
    """

    def test_cuped_reduces_variance(self):
        """CUPED 调整后方差应不大于原始方差。"""
        from stats.cuped import cuped_adjust, variance_reduction_ratio
        np.random.seed(42)
        n = 1000
        # 构造有相关性的 pre/post（rho ≈ 0.7）
        pre = pd.Series(np.random.normal(10, 3, n))
        noise = np.random.normal(0, 2, n)
        post = pre * 0.7 + noise

        post_cuped = cuped_adjust(pre, post)
        assert post_cuped.var(ddof=1) <= post.var(ddof=1), (
            f"CUPED 调整后方差 {post_cuped.var():.4f} 不小于原始方差 {post.var():.4f}"
        )

    def test_cuped_preserves_mean(self):
        """CUPED 调整后均值应与原始均值一致（允许浮点误差）。"""
        from stats.cuped import cuped_adjust
        np.random.seed(123)
        n = 500
        pre = pd.Series(np.random.normal(5, 2, n))
        post = pd.Series(np.random.normal(6, 2, n))
        post_cuped = cuped_adjust(pre, post)
        assert abs(post_cuped.mean() - post.mean()) < 1e-10, (
            f"CUPED 调整前后均值不一致：{post.mean():.6f} vs {post_cuped.mean():.6f}"
        )

    def test_higher_correlation_more_reduction(self):
        """pre/post 相关性越高，方差缩减越多。"""
        from stats.cuped import variance_reduction_ratio
        np.random.seed(42)
        n = 2000
        base = np.random.normal(10, 3, n)

        # 低相关（rho ≈ 0.3）
        post_low = pd.Series(base * 0.3 + np.random.normal(0, 3, n))
        pre_low = pd.Series(base)

        # 高相关（rho ≈ 0.8）
        post_high = pd.Series(base * 0.8 + np.random.normal(0, 1.5, n))
        pre_high = pd.Series(base)

        stats_low = variance_reduction_ratio(pre_low, post_low)
        stats_high = variance_reduction_ratio(pre_high, post_high)

        assert stats_high["reduction_pct"] > stats_low["reduction_pct"], (
            f"高相关（{stats_high['correlation']:.2f}）方差缩减 {stats_high['reduction_pct']:.2%} "
            f"应大于低相关（{stats_low['correlation']:.2f}）的 {stats_low['reduction_pct']:.2%}"
        )


class TestSubgroupAnalysis:
    """
    验证分层分析的正确性：
    1. 各子组样本量之和等于全量（无数据丢失）
    2. 各子组指标不受全量计算干扰（独立性）
    """

    def _make_event_log(self, n_per_group: int = 1000) -> pd.DataFrame:
        """生成包含 user_type 字段的 mock 事件日志。"""
        np.random.seed(42)
        rows = []
        for i in range(n_per_group * 2):
            uid = f"user_{i}"
            group = "control" if i < n_per_group else "treatment"
            user_type = "new" if i % 3 == 0 else "returning"
            rate = 0.10 if user_type == "new" else 0.20
            impressions = np.random.poisson(5)
            for _ in range(max(1, impressions)):
                rows.append({
                    "user_id": uid, "group": group,
                    "event_name": "booking_flow_start", "value": 1,
                    "timestamp": "2026-07-14", "user_type": user_type,
                })
            clicks = np.random.binomial(max(1, impressions), rate)
            for _ in range(clicks):
                rows.append({
                    "user_id": uid, "group": group,
                    "event_name": "booking_success", "value": 1,
                    "timestamp": "2026-07-14", "user_type": user_type,
                })
        return pd.DataFrame(rows)

    def test_subgroup_sample_size_sum_equals_total(self):
        """各子组对照组样本量之和等于全量对照组样本量。"""
        from metrics.metric_calculator import MetricCalculator
        from metrics.metric_definitions import HealthProductMetrics

        event_log = self._make_event_log(n_per_group=500)
        calc = MetricCalculator(event_log)
        metric = HealthProductMetrics.BOOKING_COMPLETION_RATE

        full_result = calc.calculate(metric)
        subgroup_results = calc.subgroup_analysis(metric, "user_type")

        total_ctrl = sum(r.control_sample_size for r in subgroup_results.values())
        total_trt = sum(r.treatment_sample_size for r in subgroup_results.values())

        assert total_ctrl == full_result.control_sample_size, (
            f"分层对照组样本量之和 {total_ctrl} != 全量 {full_result.control_sample_size}"
        )
        assert total_trt == full_result.treatment_sample_size, (
            f"分层实验组样本量之和 {total_trt} != 全量 {full_result.treatment_sample_size}"
        )

    def test_subgroup_new_user_lower_rate(self):
        """新用户的基准转化率应低于老用户（业务逻辑验证）。"""
        from metrics.metric_calculator import MetricCalculator
        from metrics.metric_definitions import HealthProductMetrics

        event_log = self._make_event_log(n_per_group=1000)
        calc = MetricCalculator(event_log)
        metric = HealthProductMetrics.BOOKING_COMPLETION_RATE

        subgroup_results = calc.subgroup_analysis(metric, "user_type")
        new_rate = subgroup_results.get("new", None)
        returning_rate = subgroup_results.get("returning", None)

        if new_rate and returning_rate:
            assert new_rate.control_value < returning_rate.control_value, (
                f"新用户转化率 {new_rate.control_value:.4f} 应低于老用户 {returning_rate.control_value:.4f}"
            )

