"""
指标定义模块 —— 通用健康产品场景
定义业务指标体系：北极星指标、护栏指标、探索性指标。

设计原则（产品视角）：
    一个实验的指标体系不是越多越好，而是：
    1. 有且只有 1 个北极星指标（实验判断的核心依据）
    2. 3-5 个护栏指标（任何一个显著下降则实验失败）
    3. 若干探索性指标（了解影响，不作判断依据）
"""

from dataclasses import dataclass
from typing import Dict, List
from experiment.config_schema import MetricConfig, MetricType


# ══════════════════════════════════════════════════════════════
# 通用健康产品业务场景指标定义
# ══════════════════════════════════════════════════════════════

class HealthProductMetrics:
    """
    通用健康产品标准指标库。

    业务场景分类：
        - 预约转化（Booking）：帮用户在线完成服务预约
        - 在线咨询（Consultation）：预约前的咨询引导
        - 内容推荐（Content Feed）：个性化健康内容推荐
    """

    # ── 预约转化场景 ─────────────────────────────────────────

    BOOKING_COMPLETION_RATE = MetricConfig(
        name="booking_completion_rate",
        metric_type=MetricType.PRIMARY,
        description="预约完成率 = 预约成功用户数 / 进入预约流程用户数（北极星）",
        aggregation="ratio",
        numerator_event="booking_success",
        denominator_event="booking_flow_start",
        direction=1,
        min_detectable_effect=0.03,  # 3% 相对提升视为有意义
    )

    BOOKING_FUNNEL_DROP_RATE = MetricConfig(
        name="booking_funnel_drop_rate",
        metric_type=MetricType.GUARDRAIL,
        description="预约漏斗流失率 = 中途退出用户数 / 进入流程用户数",
        aggregation="ratio",
        numerator_event="booking_flow_exit",
        denominator_event="booking_flow_start",
        direction=-1,  # 越低越好
        min_detectable_effect=0.05,
    )

    BOOKING_SUCCESS_TIME_SEC = MetricConfig(
        name="booking_success_time_sec",
        metric_type=MetricType.EXPLORATORY,
        description="预约完成耗时（秒，均值），反映预约流程效率",
        aggregation="mean",
        value_field="booking_duration_sec",
        direction=-1,
        min_detectable_effect=0.10,
    )

    # ── 在线咨询场景 ──────────────────────────────────────────

    CONSULTATION_START_RATE = MetricConfig(
        name="consultation_start_rate",
        metric_type=MetricType.PRIMARY,
        description="咨询启动率 = 点击咨询用户数 / 展示咨询入口用户数",
        aggregation="ratio",
        numerator_event="consultation_start",
        denominator_event="consultation_entry_show",
        direction=1,
        min_detectable_effect=0.05,
    )

    CONSULTATION_COMPLETION_RATE = MetricConfig(
        name="consultation_completion_rate",
        metric_type=MetricType.GUARDRAIL,
        description="咨询完成率，防止提升启动率但降低完成率的情况",
        aggregation="ratio",
        numerator_event="consultation_complete",
        denominator_event="consultation_start",
        direction=1,
        min_detectable_effect=0.05,
    )

    PROVIDER_SATISFACTION_SCORE = MetricConfig(
        name="provider_satisfaction_score",
        metric_type=MetricType.GUARDRAIL,
        description="服务提供者满意度评分（1-5 分均值，如营养师/护士/客服等角色），服务体验护栏",
        aggregation="mean",
        value_field="provider_rating",
        direction=1,
        min_detectable_effect=0.02,
    )

    # ── 内容推荐场景 ────────────────────────────────────────

    DEEP_READ_RATE = MetricConfig(
        name="deep_read_rate",
        metric_type=MetricType.PRIMARY,
        description="深度阅读率 = 停留 ≥ 30s 的阅读次数 / 总点击次数",
        aggregation="ratio",
        numerator_event="article_read_30s",
        denominator_event="article_click",
        direction=1,
        min_detectable_effect=0.03,
    )

    NEGATIVE_FEEDBACK_RATE = MetricConfig(
        name="negative_feedback_rate",
        metric_type=MetricType.GUARDRAIL,
        description="负反馈率 = 不感兴趣/屏蔽次数 / Feed 曝光次数，用户体验护栏",
        aggregation="ratio",
        numerator_event="negative_feedback",
        denominator_event="feed_impression",
        direction=-1,
        min_detectable_effect=0.02,
    )

    FEED_CTR = MetricConfig(
        name="feed_ctr",
        metric_type=MetricType.EXPLORATORY,
        description="Feed 点击率（CTR），探索性了解内容吸引力变化",
        aggregation="ratio",
        numerator_event="article_click",
        denominator_event="feed_impression",
        direction=1,
        min_detectable_effect=0.02,
    )

    # ── 全局护栏指标（任何实验都应包含） ────────────────────────

    CRASH_RATE = MetricConfig(
        name="crash_rate",
        metric_type=MetricType.GUARDRAIL,
        description="页面崩溃率（全局护栏，任何实验不得使其显著上升）",
        aggregation="ratio",
        numerator_event="page_crash",
        denominator_event="page_view",
        direction=-1,
        min_detectable_effect=0.001,  # 崩溃率 0.1% 的绝对变化就需关注
    )

    USER_7D_RETENTION = MetricConfig(
        name="user_7d_retention",
        metric_type=MetricType.GUARDRAIL,
        description="用户 7 日留存率（全局护栏，反映实验对长期用户价值的影响）",
        aggregation="ratio",
        numerator_event="app_open_7d_after",
        denominator_event="experiment_exposed",
        direction=1,
        min_detectable_effect=0.01,
    )

    @classmethod
    def get_standard_guardrails(cls) -> List[MetricConfig]:
        """获取所有实验应默认包含的全局护栏指标。"""
        return [cls.CRASH_RATE, cls.USER_7D_RETENTION]

    @classmethod
    def get_booking_metrics(cls) -> List[MetricConfig]:
        """预约转化场景推荐指标组合。"""
        return [
            cls.BOOKING_COMPLETION_RATE,    # 北极星
            cls.BOOKING_FUNNEL_DROP_RATE,   # 护栏
            cls.CRASH_RATE,                 # 全局护栏
            cls.USER_7D_RETENTION,          # 全局护栏
            cls.BOOKING_SUCCESS_TIME_SEC,   # 探索
        ]

    @classmethod
    def get_consultation_metrics(cls) -> List[MetricConfig]:
        """在线咨询场景推荐指标组合。"""
        return [
            cls.CONSULTATION_START_RATE,       # 北极星
            cls.CONSULTATION_COMPLETION_RATE,  # 护栏
            cls.PROVIDER_SATISFACTION_SCORE,   # 护栏
            cls.CRASH_RATE,                    # 全局护栏
            cls.USER_7D_RETENTION,             # 全局护栏
        ]

    @classmethod
    def get_feed_metrics(cls) -> List[MetricConfig]:
        """内容推荐场景推荐指标组合。"""
        return [
            cls.DEEP_READ_RATE,          # 北极星
            cls.NEGATIVE_FEEDBACK_RATE,  # 护栏
            cls.CRASH_RATE,              # 全局护栏
            cls.USER_7D_RETENTION,       # 全局护栏
            cls.FEED_CTR,                # 探索
        ]
