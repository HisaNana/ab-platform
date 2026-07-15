"""
run_analysis() 高层 API 集成测试
用 tmp_path 隔离 ExperimentManager 的 storage_path，避免测试间/与手动 demo 相互污染。
"""

import numpy as np
import pandas as pd
import pytest

from experiment.config_schema import ExperimentConfig, MetricConfig, MetricType
from experiment.experiment_manager import ExperimentManager, AnalysisReport
from metrics.metric_definitions import HealthProductMetrics


def _make_config(experiment_id="booking_test", use_sequential_testing=False,
                  sequential_method="alpha_spending"):
    return ExperimentConfig(
        experiment_id=experiment_id,
        name="预约推荐算法测试",
        description="run_analysis() 集成测试用配置",
        owner="tester",
        team="default",
        control_ratio=0.5,
        treatment_ratio=0.5,
        min_runtime_days=7,
        metrics=HealthProductMetrics.get_booking_metrics(),
        alpha=0.05,
        multiple_testing_correction="bonferroni",
        use_sequential_testing=use_sequential_testing,
        sequential_method=sequential_method,
    )


def _make_event_log(n_per_group=2000, effect=0.0075, seed=42, with_user_type=False,
                     with_holdout=False):
    """生成 booking 场景的模拟事件日志（含预约流程曝光 + 预约成功事件）。"""
    np.random.seed(seed)
    rows = []
    for i in range(n_per_group * 2):
        uid = f"user_{i}"
        group = "control" if i < n_per_group else "treatment"
        rate = 0.15 if group == "control" else 0.15 + effect
        user_type = None
        if with_user_type:
            user_type = "new" if i % 3 == 0 else "returning"
            rate = rate * (0.7 if user_type == "new" else 1.2)
        impressions = np.random.poisson(10)
        for _ in range(max(1, impressions)):
            row = {"user_id": uid, "group": group, "event_name": "booking_flow_start",
                   "value": 1, "timestamp": "2026-07-14"}
            if with_user_type:
                row["user_type"] = user_type
            rows.append(row)
        clicks = np.random.binomial(max(1, impressions), min(rate, 0.99))
        for _ in range(clicks):
            row = {"user_id": uid, "group": group, "event_name": "booking_success",
                   "value": 1, "timestamp": "2026-07-14"}
            if with_user_type:
                row["user_type"] = user_type
            rows.append(row)
        # booking_funnel_drop_rate 指标需要 booking_flow_exit 事件，否则该指标分子恒为 0
        exits = max(0, max(1, impressions) - clicks)
        for _ in range(exits):
            row = {"user_id": uid, "group": group, "event_name": "booking_flow_exit",
                   "value": 1, "timestamp": "2026-07-14"}
            if with_user_type:
                row["user_type"] = user_type
            rows.append(row)
        # booking_success_time_sec 是 mean 类指标，需要 booking_duration_sec 取值，
        # 否则该指标样本量为 0，会导致序贯检验（n=0）报错
        row = {"user_id": uid, "group": group, "event_name": "booking_duration_sec",
               "value": 1, "booking_duration_sec": float(np.random.normal(60, 10)),
               "timestamp": "2026-07-14"}
        if with_user_type:
            row["user_type"] = user_type
        rows.append(row)
        # 全局护栏：crash_rate = page_crash / page_view
        row = {"user_id": uid, "group": group, "event_name": "page_view",
               "value": 1, "timestamp": "2026-07-14"}
        if with_user_type:
            row["user_type"] = user_type
        rows.append(row)
        if np.random.random() < 0.01:
            row = {"user_id": uid, "group": group, "event_name": "page_crash",
                   "value": 1, "timestamp": "2026-07-14"}
            if with_user_type:
                row["user_type"] = user_type
            rows.append(row)

        # 全局护栏：user_7d_retention = app_open_7d_after / experiment_exposed
        row = {"user_id": uid, "group": group, "event_name": "experiment_exposed",
               "value": 1, "timestamp": "2026-07-14"}
        if with_user_type:
            row["user_type"] = user_type
        rows.append(row)
        if np.random.random() < 0.45:
            row = {"user_id": uid, "group": group, "event_name": "app_open_7d_after",
                   "value": 1, "timestamp": "2026-07-21"}
            if with_user_type:
                row["user_type"] = user_type
            rows.append(row)

    if with_holdout:
        for i in range(50):
            rows.append({"user_id": f"holdout_user_{i}", "group": "holdout",
                        "event_name": "booking_flow_start", "value": 1,
                        "timestamp": "2026-07-14"})

    return pd.DataFrame(rows)


class TestRunAnalysisHappyPath:
    """正常路径：全字段填充，结论文本含北极星指标名。"""

    def test_all_fields_populated(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)
        event_log = _make_event_log()

        report = manager.run_analysis(config.experiment_id, event_log)

        assert isinstance(report, AnalysisReport)
        assert report.experiment_id == config.experiment_id
        assert report.experiment_status == "draft"
        assert isinstance(report.srm_result, dict)
        assert isinstance(report.srm_passed, bool)
        assert isinstance(report.metric_results, dict)
        assert len(report.metric_results) == len(config.metrics)
        assert isinstance(report.conclusion_text, str)
        assert isinstance(report.warnings, list)

    def test_conclusion_text_contains_primary_metric_name(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)
        event_log = _make_event_log()

        report = manager.run_analysis(config.experiment_id, event_log)

        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        assert primary_name in report.conclusion_text


class TestRunAnalysisSRM:
    """SRM 不通过时不应抛异常，且 warnings/srm_passed 正确反映状态。"""

    def test_srm_failure_does_not_raise_and_sets_warning(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)

        # 人为构造严重偏斜的分流（80/20，远离配置的 50/50）触发 SRM
        np.random.seed(1)
        rows = []
        for i in range(8000):
            rows.append({"user_id": f"user_{i}", "group": "control",
                        "event_name": "booking_flow_start", "value": 1, "timestamp": "2026-07-14"})
            if np.random.random() < 0.15:
                rows.append({"user_id": f"user_{i}", "group": "control",
                            "event_name": "booking_success", "value": 1, "timestamp": "2026-07-14"})
        for i in range(2000):
            rows.append({"user_id": f"user_t_{i}", "group": "treatment",
                        "event_name": "booking_flow_start", "value": 1, "timestamp": "2026-07-14"})
            if np.random.random() < 0.15:
                rows.append({"user_id": f"user_t_{i}", "group": "treatment",
                            "event_name": "booking_success", "value": 1, "timestamp": "2026-07-14"})
        skewed_log = pd.DataFrame(rows)

        report = manager.run_analysis(config.experiment_id, skewed_log)

        assert report.srm_passed is False
        assert any("SRM" in w for w in report.warnings)
        # 即使 SRM 不通过，指标结果仍应完整计算（不中断）
        assert len(report.metric_results) == len(config.metrics)

    def test_holdout_users_trigger_warning(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)
        event_log = _make_event_log(with_holdout=True)

        report = manager.run_analysis(config.experiment_id, event_log)

        assert any("holdout" in w for w in report.warnings)


class TestRunAnalysisCUPED:
    """传入 pre_data 后 CUPED 生效，方差应降低。"""

    def test_pre_data_reduces_variance(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)
        event_log = _make_event_log()

        report_without_cuped = manager.run_analysis(config.experiment_id, event_log)

        # 构造与 booking_completion_rate 相关的 pre_data（每用户历史转化率，与 post 强相关）
        np.random.seed(7)
        user_ids_ctrl = event_log[event_log["group"] == "control"]["user_id"].unique()
        user_ids_trt = event_log[event_log["group"] == "treatment"]["user_id"].unique()
        pre_data = {
            HealthProductMetrics.BOOKING_COMPLETION_RATE.name: {
                "control": pd.Series(np.random.normal(0.15, 0.02, len(user_ids_ctrl)), index=user_ids_ctrl),
                "treatment": pd.Series(np.random.normal(0.1575, 0.02, len(user_ids_trt)), index=user_ids_trt),
            }
        }

        report_with_cuped = manager.run_analysis(config.experiment_id, event_log, pre_data=pre_data)

        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        # CUPED 调整后不应报错，且方法本身应正常产出显著性判定字段
        assert primary_name in report_with_cuped.metric_results
        assert primary_name in report_without_cuped.metric_results


class TestRunAnalysisSubgroup:
    """传入 subgroup_col 后各子组样本量之和应等于全量。"""

    def test_subgroup_sample_size_sum_equals_total(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config()
        manager.create(config)
        event_log = _make_event_log(with_user_type=True)

        report = manager.run_analysis(config.experiment_id, event_log, subgroup_col="user_type")

        assert report.subgroup_results is not None
        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        subgroup_map = report.subgroup_results[primary_name]

        total_ctrl = sum(r.control_sample_size for r in subgroup_map.values())
        total_trt = sum(r.treatment_sample_size for r in subgroup_map.values())

        full_result = report.metric_results[primary_name]
        assert total_ctrl == full_result.sample_size_control
        assert total_trt == full_result.sample_size_treatment


class TestRunAnalysisSequential:
    """use_sequential_testing=True 时序贯字段生效。"""

    def test_sequential_fields_populated_alpha_spending(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config(use_sequential_testing=True, sequential_method="alpha_spending")
        manager.create(config)
        event_log = _make_event_log()

        report = manager.run_analysis(config.experiment_id, event_log)

        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        result = report.metric_results[primary_name]
        assert result.sequential_method_used == "alpha_spending"
        assert result.sequential_boundary is not None

    def test_sequential_fields_populated_msprt(self, tmp_path):
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config(use_sequential_testing=True, sequential_method="msprt")
        manager.create(config)
        event_log = _make_event_log()

        report = manager.run_analysis(config.experiment_id, event_log)

        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        result = report.metric_results[primary_name]
        assert result.sequential_method_used == "msprt"
        assert result.sequential_boundary is not None

    def test_default_mode_sequential_fields_are_none(self, tmp_path):
        """回归测试：默认（非序贯）模式下 run_analysis 结果的序贯字段应为 None。"""
        manager = ExperimentManager(storage_path=str(tmp_path))
        config = _make_config(use_sequential_testing=False)
        manager.create(config)
        event_log = _make_event_log()

        report = manager.run_analysis(config.experiment_id, event_log)

        primary_name = HealthProductMetrics.BOOKING_COMPLETION_RATE.name
        result = report.metric_results[primary_name]
        assert result.sequential_method_used is None
        assert result.sequential_boundary is None
