"""
使用示例 —— 完整的 AB 实验流程演示（通用健康产品·预约转化场景）

本文件包含两段演示：
    1. 手动串联 7 步（教学用，展示每一步在做什么）
    2. 使用 run_analysis() 一行调用达到同样效果（推荐用法）
"""
import pandas as pd
import numpy as np
from experiment.config_schema import ExperimentConfig, MetricType
from experiment.experiment_manager import ExperimentManager
from metrics.metric_definitions import HealthProductMetrics
from metrics.metric_calculator import MetricCalculator
from stats.t_test import HypothesisTester
from allocation.hash_splitter import HashSplitter


# ═══════════════════════════════════════════════════════════════
# 第一段：手动串联 7 步（教学对比，展示每一步的输入输出）
# ═══════════════════════════════════════════════════════════════

# ── Step 1: 创建实验配置 ───────────────────────────────────────
config = ExperimentConfig(
    experiment_id="booking_rec_v2",
    name="预约推荐算法 v2",
    description="测试新的个性化推荐模型，预期提升预约完成率 5%",
    owner="demo_owner",
    team="default",
    traffic_key="uid",
    control_ratio=0.5,
    treatment_ratio=0.5,
    min_runtime_days=7,
    metrics=HealthProductMetrics.get_booking_metrics(),
    alpha=0.05,
    multiple_testing_correction="bonferroni",
)

print("📋 实验配置 YAML：")
print(config.to_yaml())

# ── Step 2: 样本量估算 ─────────────────────────────────────────
manager = ExperimentManager(storage_path="/tmp/ab_experiments")
size_estimate = manager.estimate_sample_size(
    baseline_rate=0.15,  # 当前预约完成率 15%
    mde=0.05,            # 期望检测到 5% 相对提升
    alpha=0.05,
    power=0.8,
)
print("\n⚡ 样本量估算：")
for k, v in size_estimate.items():
    print(f"  {k}: {v}")

# ── Step 3: 模拟分流 ──────────────────────────────────────────
splitter = HashSplitter("booking_rec_v2", salt="v1")
user_ids = [f"uid_{i}" for i in range(20000)]
assignments = {uid: splitter.assign_group(uid) for uid in user_ids}

ctrl_users = sum(1 for g in assignments.values() if g == "control")
trt_users = sum(1 for g in assignments.values() if g == "treatment")
print(f"\n📊 分流结果：对照组 {ctrl_users}，实验组 {trt_users}")

# ── Step 4: SRM 检测 ──────────────────────────────────────────
srm = splitter.check_srm(ctrl_users, trt_users)
print(f"\n🔍 SRM 检测：{srm['conclusion']}")

# ── Step 5: 生成模拟事件日志（模拟实验运行 7 天后的数据）─────────
np.random.seed(42)
rows = []
for uid, group in list(assignments.items())[:10000]:
    rate = 0.15 if group == "control" else 0.1575  # 5% 相对提升
    impressions = np.random.poisson(10)
    for _ in range(max(1, impressions)):
        rows.append({"user_id": uid, "group": group,
                     "event_name": "booking_flow_start", "value": 1,
                     "timestamp": "2026-07-14"})
    clicks = np.random.binomial(max(1, impressions), rate)
    for _ in range(clicks):
        rows.append({"user_id": uid, "group": group,
                     "event_name": "booking_success", "value": 1,
                     "timestamp": "2026-07-14"})

event_log = pd.DataFrame(rows)

# ── Step 6: 计算指标 ──────────────────────────────────────────
calculator = MetricCalculator(event_log)
primary_result = calculator.calculate(HealthProductMetrics.BOOKING_COMPLETION_RATE)

print(f"\n📈 指标结果：")
print(f"  对照组预约完成率: {primary_result.control_value:.4f}")
print(f"  实验组预约完成率: {primary_result.treatment_value:.4f}")
print(f"  相对变化: {primary_result.relative_change:+.2%}")

# ── Step 7: 统计检验 ──────────────────────────────────────────
tester = HypothesisTester(alpha=0.05)
test_result = tester.test(primary_result, HealthProductMetrics.BOOKING_COMPLETION_RATE)

print(f"\n📊 检验结果：")
print(f"  z 统计量: {test_result.test_statistic}")
print(f"  p-value:  {test_result.p_value}")
print(f"  95% CI:   {test_result.confidence_interval}")
print(f"  结论: {test_result.conclusion}")


# ═══════════════════════════════════════════════════════════════
# 第二段：推荐用法 —— run_analysis() 一行调用
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("推荐用法：run_analysis() 一行完成「分流检查→指标计算→假设检验→决策结论」")
print("═" * 65)

manager.create(config)
report = manager.run_analysis(config.experiment_id, event_log)

print(f"\n📋 实验状态: {report.experiment_status}")
print(f"🔍 SRM 通过: {report.srm_passed}（{report.srm_result['conclusion']}）")
if report.warnings:
    print(f"⚠️  警告: {report.warnings}")
print(f"\n{report.conclusion_text}")

# 两种方式的核心结论方向应当一致：
same_direction = (
    (test_result.is_significant and test_result.direction_correct)
    == (report.metric_results[HealthProductMetrics.BOOKING_COMPLETION_RATE.name].is_significant
        and report.metric_results[HealthProductMetrics.BOOKING_COMPLETION_RATE.name].direction_correct)
)
print(f"\n✅ 手动流程与 run_analysis() 结论方向一致: {same_direction}")
