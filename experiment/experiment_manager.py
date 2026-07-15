"""
实验生命周期管理器
负责实验的创建、启动、暂停、结论等状态流转，以及基本的 power analysis。
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from .config_schema import ExperimentConfig, ExperimentStatus, MetricType
import math


@dataclass
class AnalysisReport:
    """
    `ExperimentManager.run_analysis()` 的返回结果。

    把「读取配置 → SRM 检查 → 批量指标计算 → 假设检验 → 决策结论」
    的中间产物和最终结论打包成一个结构化对象，方便脚本化调用和自动化报表消费。
    """
    experiment_id: str
    experiment_status: str
    srm_result: dict
    srm_passed: bool
    metric_results: Dict[str, "TestResult"]
    conclusion_text: str
    subgroup_results: Optional[Dict[str, Dict[str, "MetricResult"]]] = None
    warnings: List[str] = field(default_factory=list)


class ExperimentManager:
    """
    实验管理器。

    状态流转：
        DRAFT → AA_TESTING → RUNNING → CONCLUDED
                                    ↘ PAUSED → RUNNING
                                    ↘ ABORTED

    关键设计原则（通用健康产品场景）：
        1. 必须先通过 AA 测试才能转为 RUNNING（避免带病上线）
        2. 设置最短运行时长，防止过早下结论（避免 peeking problem）
        3. SRM 检测集成在每次读取实验数据时
    """

    def __init__(self, storage_path: str = "./experiment_store"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._load_all()

    def _config_path(self, experiment_id: str) -> str:
        return os.path.join(self.storage_path, f"{experiment_id}.yaml")

    def _load_all(self):
        """从磁盘加载所有实验配置。"""
        for fname in os.listdir(self.storage_path):
            if fname.endswith(".yaml"):
                exp_id = fname.replace(".yaml", "")
                with open(os.path.join(self.storage_path, fname), "r", encoding="utf-8") as f:
                    self._experiments[exp_id] = ExperimentConfig.from_yaml(f.read())

    def _save(self, config: ExperimentConfig):
        """持久化单个实验配置。"""
        with open(self._config_path(config.experiment_id), "w", encoding="utf-8") as f:
            f.write(config.to_yaml())

    # ──────────────────────────────────────────
    # 实验生命周期操作
    # ──────────────────────────────────────────

    def create(self, config: ExperimentConfig) -> ExperimentConfig:
        """创建新实验（DRAFT 状态）。"""
        if config.experiment_id in self._experiments:
            raise ValueError(f"实验 {config.experiment_id} 已存在，请使用 update()。")
        config.status = ExperimentStatus.DRAFT
        self._experiments[config.experiment_id] = config
        self._save(config)
        return config

    def start_aa_test(self, experiment_id: str) -> ExperimentConfig:
        """
        启动 AA 测试。
        AA 测试阶段：使用真实流量，但两组均使用对照策略，
        目的是验证分流无偏，确保后续 AB 结论可信。
        """
        config = self._get_or_raise(experiment_id)
        if config.status != ExperimentStatus.DRAFT:
            raise ValueError(f"只有 DRAFT 状态的实验可以启动 AA 测试，当前状态: {config.status}")
        config.status = ExperimentStatus.AA_TESTING
        config.start_time = datetime.now().isoformat()
        self._save(config)
        print(f"✅ 实验 [{experiment_id}] 已进入 AA 测试阶段，建议运行 1-3 天后检查 AA 结果。")
        return config

    def pass_aa_and_start(self, experiment_id: str) -> ExperimentConfig:
        """AA 测试通过，正式启动实验。"""
        config = self._get_or_raise(experiment_id)
        if config.status != ExperimentStatus.AA_TESTING:
            raise ValueError(f"只有 AA_TESTING 状态才能转为 RUNNING，当前: {config.status}")
        config.aa_passed = True
        config.status = ExperimentStatus.RUNNING
        config.start_time = datetime.now().isoformat()  # 重置为正式启动时间
        self._save(config)
        print(f"🚀 实验 [{experiment_id}] AA 通过，正式启动！")
        return config

    def pause(self, experiment_id: str, reason: str = "") -> ExperimentConfig:
        """暂停实验（如发现 SRM 或线上告警）。"""
        config = self._get_or_raise(experiment_id)
        if config.status != ExperimentStatus.RUNNING:
            raise ValueError(f"只有 RUNNING 状态的实验可以暂停。")
        config.status = ExperimentStatus.PAUSED
        self._save(config)
        print(f"⏸️  实验 [{experiment_id}] 已暂停。原因: {reason or '未填写'}")
        return config

    def conclude(self, experiment_id: str, conclusion: str) -> ExperimentConfig:
        """
        得出实验结论。
        在结论前会检查最短运行时长，防止过早下结论。
        """
        config = self._get_or_raise(experiment_id)
        if config.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise ValueError("只有 RUNNING/PAUSED 状态的实验可以结论。")

        # 检查最短运行时长
        if config.start_time:
            start = datetime.fromisoformat(config.start_time)
            elapsed_days = (datetime.now() - start).days
            if elapsed_days < config.min_runtime_days:
                print(
                    f"⚠️  警告：实验仅运行了 {elapsed_days} 天，"
                    f"未达到最短 {config.min_runtime_days} 天要求，"
                    f"下结论存在 peeking 风险！"
                )

        config.status = ExperimentStatus.CONCLUDED
        self._save(config)
        print(f"📋 实验 [{experiment_id}] 已结论：{conclusion}")
        return config

    # ──────────────────────────────────────────
    # 功效分析（Sample Size Calculator）
    # ──────────────────────────────────────────

    def estimate_sample_size(
        self,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.8,
        ratio: float = 1.0,
        metric_type: str = "ratio",
        baseline_std: Optional[float] = None,
    ) -> Dict:
        """
        样本量估算，支持比率指标和均值指标两种公式。

        面试高频题：为什么要做 power analysis？
            - 样本不足 → 实验结论不可靠（假阴性，漏掉真实效果）
            - 提前告知需要多久能得出结论，帮助产品规划排期

        Args:
            baseline_rate: 对照组基准值（比率类传转化率；均值类传均值，如平均评分 3.8）
            mde:           最小可检测效应量（相对变化），如 0.05 表示 5% 相对提升
            alpha:         显著性水平，默认 0.05
            power:         统计功效，默认 0.8
            ratio:         treatment / control 用户比，默认 1.0（等比例）
            metric_type:   "ratio"（比率指标，用双比例 Z 检验）或
                           "mean"（均值指标，用 two-sample t-test）
            baseline_std:  均值指标必填，对照组历史标准差（metric_type="mean" 时使用）

        Returns:
            {
                "metric_type": str,
                "control_size": int,
                "treatment_size": int,
                "total_size": int,
                "tip": str
            }
        """
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        if metric_type == "ratio":
            p1 = baseline_rate
            p2 = baseline_rate * (1 + mde)
            # 双比例 Z 检验样本量公式
            p_bar = (p1 + ratio * p2) / (1 + ratio)
            n = (
                (z_alpha * math.sqrt((1 + 1 / ratio) * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2
                / (p1 - p2) ** 2
            )
            mde_absolute = round(p2 - p1, 5)
        elif metric_type == "mean":
            if baseline_std is None:
                raise ValueError(
                    "metric_type='mean' 时必须提供 baseline_std（对照组历史标准差）。"
                    "可从历史数据中用 df['value'].std() 获取。"
                )
            delta = baseline_rate * mde  # 绝对差值
            # two-sample t-test 样本量公式（等方差假设）
            # n = (z_alpha + z_beta)^2 * 2 * sigma^2 / delta^2
            n = (z_alpha + z_beta) ** 2 * 2 * baseline_std ** 2 / delta ** 2
            mde_absolute = round(delta, 5)
        else:
            raise ValueError(f"不支持的 metric_type: {metric_type}，请使用 'ratio' 或 'mean'")

        n_control = math.ceil(n)
        n_treatment = math.ceil(n * ratio)

        return {
            "metric_type": metric_type,
            "baseline_value": baseline_rate,
            "mde_relative": f"{mde:.1%}",
            "mde_absolute": mde_absolute,
            "control_size": n_control,
            "treatment_size": n_treatment,
            "total_size": n_control + n_treatment,
            "tip": f"需要 {n_control + n_treatment:,} 个用户，"
                   f"按日活 5 万计算约需 {math.ceil((n_control + n_treatment) / 50000)} 天"
        }

    # ──────────────────────────────────────────
    # run_analysis()：高层一体化分析 API
    # ──────────────────────────────────────────

    def run_analysis(
        self,
        experiment_id: str,
        event_log: pd.DataFrame,
        pre_data: Optional[Dict[str, Dict[str, "pd.Series"]]] = None,
        subgroup_col: Optional[str] = None,
    ) -> AnalysisReport:
        """
        高层一体化分析 API：一行调用完成
        「读取实验配置 → SRM 检查 → 批量指标计算 → 假设检验（含可选序贯检验路径）→ 生成决策结论」。

        相比手动串联 7 步（见 example_full_flow.py 第一段），适合脚本化调用、
        定时报表任务等不需要逐步查看中间产物的场景。

        SRM 不通过时的处理策略（与 Dashboard 行为保持一致）：
            不会中断计算，而是继续输出全部指标结果，并在返回的 `warnings` 中
            追加结构化告警，`srm_passed` 置为 False。是否据此阻断下游流程，
            交由调用方自行判断（如检查 `report.srm_passed`）。

        Args:
            experiment_id: 实验 ID，必须已通过 create() 注册。
            event_log:     事件日志 DataFrame（格式要求见 MetricCalculator）。
            pre_data:      可选，CUPED 协变量数据，格式为
                           {metric_name: {"control": pd.Series, "treatment": pd.Series}}。
            subgroup_col:  可选，分层分析字段名（如 "user_type"）。

        Returns:
            AnalysisReport
        """
        from metrics.metric_calculator import MetricCalculator
        from stats.t_test import HypothesisTester

        config = self._get_or_raise(experiment_id)
        warnings: List[str] = []

        # ── SRM 检查 ──────────────────────────────────────────
        from allocation.hash_splitter import HashSplitter

        group_counts = event_log.drop_duplicates("user_id").groupby("group")["user_id"].count()
        control_n = int(group_counts.get("control", 0))
        treatment_n = int(group_counts.get("treatment", 0))

        splitter = HashSplitter(config.experiment_id, salt=config.salt)
        srm_result = splitter.check_srm(
            control_n, treatment_n,
            control_ratio=config.control_ratio,
            treatment_ratio=config.treatment_ratio,
        )
        srm_passed = not srm_result["srm_detected"]
        if not srm_passed:
            warnings.append(
                f"SRM 检测未通过：{srm_result['conclusion']}（实际 treatment 占比 "
                f"{srm_result['actual_ratio']:.4f}，p={srm_result['p_value']}）。"
                "指标结果仍会计算，但结论可信度存疑，建议排查分流/日志采集问题。"
            )

        # holdout 组告警（与 MetricCalculator 内部告警保持一致，升级为结构化 warning）
        holdout_users = event_log[event_log["group"] == "holdout"]["user_id"].nunique() \
            if "group" in event_log.columns else 0
        if holdout_users > 0:
            warnings.append(f"event_log 中包含 {holdout_users} 个 holdout 组用户，不参与指标计算。")

        # ── 批量指标计算 ──────────────────────────────────────
        calculator = MetricCalculator(event_log)
        metric_results = {}
        subgroup_results = None
        for metric in config.metrics:
            metric_pre_data = pre_data.get(metric.name) if pre_data else None
            metric_results[metric.name] = calculator.calculate(metric, pre_data=metric_pre_data)

        if subgroup_col:
            subgroup_results = {
                metric.name: calculator.subgroup_analysis(metric, subgroup_col)
                for metric in config.metrics
            }

        # ── 假设检验（含可选序贯检验路径） ────────────────────
        tester = HypothesisTester(
            alpha=config.alpha,
            correction_method=config.multiple_testing_correction,
            use_sequential_testing=config.use_sequential_testing,
            sequential_method=config.sequential_method,
            sequential_spending_function=config.sequential_spending_function,
        )
        metric_configs = {m.name: m for m in config.metrics}

        if config.use_sequential_testing and config.sequential_method == "alpha_spending":
            # alpha_spending 模式下 test_multiple 无法传入 information_fraction，
            # 退化为逐指标调用 test()，用「当前样本量 / 计划样本量」近似信息量占比。
            # 这里没有独立的"计划总样本量"配置项，用 planned_analyses 作为查看次数的
            # 近似替代——把当前这次查看视为第 1 次查看（信息量占比=1/planned_analyses），
            # 这是一个简化近似，严肃场景建议由调用方显式传入真实的信息量占比。
            information_fraction = 1.0 / max(config.planned_analyses, 1)
            test_results = {
                name: tester.test(metric_results[name], metric_configs[name],
                                   information_fraction=information_fraction)
                for name in metric_results
            }
        else:
            test_results = tester.test_multiple(metric_results, metric_configs)

        conclusion_text = tester.make_experiment_conclusion(test_results)

        return AnalysisReport(
            experiment_id=experiment_id,
            experiment_status=config.status.value,
            srm_result=srm_result,
            srm_passed=srm_passed,
            metric_results=test_results,
            conclusion_text=conclusion_text,
            subgroup_results=subgroup_results,
            warnings=warnings,
        )

    # ──────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────

    def _get_or_raise(self, experiment_id: str) -> ExperimentConfig:
        if experiment_id not in self._experiments:
            raise KeyError(f"实验 [{experiment_id}] 不存在。")
        return self._experiments[experiment_id]

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentConfig]:
        exps = list(self._experiments.values())
        if status:
            exps = [e for e in exps if e.status == status]
        return exps

    def get(self, experiment_id: str) -> ExperimentConfig:
        return self._get_or_raise(experiment_id)
