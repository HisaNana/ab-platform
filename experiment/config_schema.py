"""
实验配置 Schema
面向通用健康产品场景定义实验配置的数据结构，支持 YAML/JSON 序列化。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import yaml
import json
from datetime import datetime


class ExperimentStatus(str, Enum):
    DRAFT = "draft"           # 草稿：配置未完成
    AA_TESTING = "aa_testing" # AA 测试中：验证分流是否无偏
    RUNNING = "running"       # 正式运行中
    PAUSED = "paused"         # 暂停（如发现 SRM 或线上问题）
    CONCLUDED = "concluded"   # 已结论
    ABORTED = "aborted"       # 提前终止


class MetricType(str, Enum):
    """
    指标类型分类（通用健康产品场景）

    北极星指标（Primary Metric）：
        实验最终的成功标准，通常只有 1 个，多重检验校正主要针对它。
        如：预约完成率、咨询启动转化率

    护栏指标（Guardrail Metric）：
        实验必须不能损害的指标，任何一个显著下降则实验失败。
        如：页面崩溃率、客诉率、用户 7 日留存

    探索性指标（Exploratory Metric）：
        了解实验对各类行为的影响，不作为决策依据，p-value 仅供参考。
        如：各内容分类点击分布、不同用户分层的互动率
    """
    PRIMARY = "primary"         # 北极星指标（主指标）
    GUARDRAIL = "guardrail"     # 护栏指标
    EXPLORATORY = "exploratory" # 探索性指标


@dataclass
class MetricConfig:
    """单个指标配置"""
    name: str
    metric_type: MetricType
    description: str
    # 计算方式：ratio（比率，如 CTR = 点击 / 曝光）或 mean（均值，如 停留时长均值）
    aggregation: str = "ratio"  # "ratio" | "mean" | "sum"
    numerator_event: Optional[str] = None   # 分子事件名（ratio 类型使用）
    denominator_event: Optional[str] = None  # 分母事件名（ratio 类型使用）
    value_field: Optional[str] = None        # 取值字段名（mean/sum 类型使用）
    # 方向：+1 表示越大越好，-1 表示越小越好
    direction: int = 1
    # 最小可检测效应量（MDE），用于实验功效分析
    min_detectable_effect: float = 0.01  # 1% 的相对变化


@dataclass
class ExperimentConfig:
    """
    完整的实验配置。

    通用健康产品典型实验场景举例：
        - 预约推荐算法 A/B：北极星=预约完成率，护栏=页面崩溃率+7日留存
        - 在线咨询入口样式 A/B：北极星=咨询启动率，护栏=整体 DAU
        - 内容 Feed 排序 A/B：北极星=深度阅读率(>30s)，护栏=负反馈率
    """
    # 基础信息
    experiment_id: str
    name: str
    description: str
    owner: str                    # 实验负责人（PM/算法工程师）
    team: str = "default"         # 所属团队

    # 流量配置
    traffic_key: str = "uid"      # 分流 key: "uid"（登录用户）| "did"（设备，含未登录）
    control_ratio: float = 0.5
    treatment_ratio: float = 0.5
    holdout_ratio: float = 0.0    # holdout 组比例（0 表示不设 holdout）
    salt: str = "v1"              # 哈希盐，重置分桶时递增

    # 分层配置（可选）
    stratify_by: Optional[str] = None  # 分层字段，如 "disease_type" | "city_tier"

    # 时间配置
    start_time: Optional[str] = None   # ISO 格式，如 "2026-07-14T10:00:00"
    end_time: Optional[str] = None
    min_runtime_days: int = 7          # 最短运行天数，防止过早下结论

    # 指标配置
    metrics: List[MetricConfig] = field(default_factory=list)

    # 统计配置
    alpha: float = 0.05          # 显著性水平
    power: float = 0.8           # 统计功效（1 - beta）
    use_sequential_testing: bool = False  # 是否使用序贯检验（允许中间停止）
    sequential_method: str = "alpha_spending"           # "alpha_spending" | "msprt"
    sequential_spending_function: str = "obrien_fleming"  # "obrien_fleming" | "pocock"（仅 alpha_spending 用）
    planned_analyses: int = 5                              # 计划总查看次数
    multiple_testing_correction: str = "bonferroni"  # "bonferroni" | "bh" | "none"

    # 实验状态
    status: ExperimentStatus = ExperimentStatus.DRAFT
    aa_passed: bool = False       # AA 测试是否通过

    def to_yaml(self) -> str:
        """序列化为 YAML（便于 Git 版本管理）"""
        data = {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "owner": self.owner,
            "team": self.team,
            "traffic": {
                "key": self.traffic_key,
                "control_ratio": self.control_ratio,
                "treatment_ratio": self.treatment_ratio,
                "holdout_ratio": self.holdout_ratio,
                "salt": self.salt,
                "stratify_by": self.stratify_by,
            },
            "schedule": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "min_runtime_days": self.min_runtime_days,
            },
            "metrics": [
                {
                    "name": m.name,
                    "type": m.metric_type.value,
                    "description": m.description,
                    "aggregation": m.aggregation,
                    "numerator_event": m.numerator_event,
                    "denominator_event": m.denominator_event,
                    "value_field": m.value_field,
                    "direction": m.direction,
                    "min_detectable_effect": m.min_detectable_effect,
                }
                for m in self.metrics
            ],
            "statistics": {
                "alpha": self.alpha,
                "power": self.power,
                "use_sequential_testing": self.use_sequential_testing,
                "sequential_method": self.sequential_method,
                "sequential_spending_function": self.sequential_spending_function,
                "planned_analyses": self.planned_analyses,
                "multiple_testing_correction": self.multiple_testing_correction,
            },
            "status": self.status.value,
            "aa_passed": self.aa_passed,
        }
        return yaml.dump(data, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ExperimentConfig":
        """从 YAML 字符串反序列化"""
        data = yaml.safe_load(yaml_str)
        metrics = [
            MetricConfig(
                name=m["name"],
                metric_type=MetricType(m["type"]),
                description=m.get("description", ""),
                aggregation=m.get("aggregation", "ratio"),
                numerator_event=m.get("numerator_event"),
                denominator_event=m.get("denominator_event"),
                value_field=m.get("value_field"),
                direction=m.get("direction", 1),
                min_detectable_effect=m.get("min_detectable_effect", 0.01),
            )
            for m in data.get("metrics", [])
        ]
        traffic = data.get("traffic", {})
        schedule = data.get("schedule", {})
        stats = data.get("statistics", {})
        return cls(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            team=data.get("team", "default"),
            traffic_key=traffic.get("key", "uid"),
            control_ratio=traffic.get("control_ratio", 0.5),
            treatment_ratio=traffic.get("treatment_ratio", 0.5),
            holdout_ratio=traffic.get("holdout_ratio", 0.0),
            salt=traffic.get("salt", "v1"),
            stratify_by=traffic.get("stratify_by"),
            start_time=schedule.get("start_time"),
            end_time=schedule.get("end_time"),
            min_runtime_days=schedule.get("min_runtime_days", 7),
            metrics=metrics,
            alpha=stats.get("alpha", 0.05),
            power=stats.get("power", 0.8),
            use_sequential_testing=stats.get("use_sequential_testing", False),
            sequential_method=stats.get("sequential_method", "alpha_spending"),
            sequential_spending_function=stats.get("sequential_spending_function", "obrien_fleming"),
            planned_analyses=stats.get("planned_analyses", 5),
            multiple_testing_correction=stats.get("multiple_testing_correction", "bonferroni"),
            status=ExperimentStatus(data.get("status", "draft")),
            aa_passed=data.get("aa_passed", False),
        )
