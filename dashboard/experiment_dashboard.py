"""
实验看板 —— 基于 Streamlit 的 AB 实验可视化界面
运行方式：streamlit run dashboard/experiment_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from allocation.hash_splitter import HashSplitter
from stats.t_test import HypothesisTester, TestResult
from stats.delta_method import compare_variance_methods
from experiment.config_schema import MetricConfig, MetricType


# ── 页面配置 ───────────────────────────────────────────────────
st.set_page_config(
    page_title="健康产品 AB 实验平台",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 健康产品 AB 实验评估平台")
st.caption("通用健康产品场景 · 实验数据分析看板")


# ── 侧边栏：实验配置 ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 实验参数")

    experiment_name = st.text_input("实验名称", value="预约推荐算法_v2")
    alpha = st.slider("显著性水平 α", 0.01, 0.10, 0.05, 0.01)
    correction = st.selectbox(
        "多重比较校正方法",
        ["bonferroni", "bh", "none"],
        help="Bonferroni：最保守；BH：控制 FDR；none：不校正"
    )

    st.divider()
    st.header("📊 模拟数据参数")
    n_users = st.slider("每组用户数", 1000, 50000, 10000, 1000)
    true_lift = st.slider("真实提升效果（相对）", -0.10, 0.20, 0.05, 0.01,
                          help="设为 0 模拟 AA 实验")
    base_rate = st.slider("对照组基准转化率", 0.05, 0.40, 0.15, 0.01)
    avg_impressions = st.slider("人均曝光次数", 3, 30, 10, 1)


# ── 生成模拟数据 ──────────────────────────────────────────────
@st.cache_data
def generate_mock_data(n_users, base_rate, true_lift, avg_impressions, seed=42):
    """生成通用健康产品·预约转化场景的模拟事件日志。"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    rows = []
    treatment_rate = base_rate * (1 + true_lift)

    # 生成 7 天内的随机日期（模拟真实实验周期）
    base_date = pd.Timestamp("2026-07-08")
    dates = [base_date + pd.Timedelta(days=d) for d in range(7)]

    for i in range(n_users * 2):
        uid = f"user_{i}"
        group = "control" if i < n_users else "treatment"
        rate = base_rate if group == "control" else treatment_rate
        # 新老用户：偶数 uid 为老用户，奇数为新用户（模拟新老用户效果差异）
        user_type = "returning" if i % 2 == 0 else "new"
        # 新用户转化率更低
        effective_rate = rate * (0.4 if user_type == "new" else 1.2)
        effective_rate = min(effective_rate, 0.99)

        impressions = np.random.poisson(avg_impressions)
        # 随机分配到某天
        event_date = random.choice(dates)

        # 预约流程曝光
        for _ in range(max(1, impressions)):
            rows.append({
                "user_id": uid, "group": group,
                "event_name": "booking_flow_start", "value": 1,
                "timestamp": event_date,
                "user_type": user_type,
            })
        # 预约完成（按转化率）
        clicks = np.random.binomial(max(1, impressions), effective_rate)
        for _ in range(clicks):
            rows.append({
                "user_id": uid, "group": group,
                "event_name": "booking_success", "value": 1,
                "timestamp": event_date,
                "user_type": user_type,
            })
        # 停留时长（均值 60s，实验组 +10%）
        duration = np.random.lognormal(4.0 + (0.1 if group == "treatment" else 0), 0.5)
        rows.append({
            "user_id": uid, "group": group,
            "event_name": "session",
            "value": round(duration, 1),
            "timestamp": event_date,
            "user_type": user_type,
        })

    return pd.DataFrame(rows)


event_log = generate_mock_data(n_users, base_rate, true_lift, avg_impressions)


# ── Tab 布局 ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📈 实验结果", "🔍 SRM 检测", "⚡ 样本量估算", "📚 Delta Method 演示",
     "🗂️ 实验层管理", "🔁 序贯检验演示", "🔄 实验生命周期"]
)

# ─────────────────────────────────────────────────────────────
# Tab 1: 实验结果
# ─────────────────────────────────────────────────────────────
with tab1:
    from metrics.metric_calculator import MetricCalculator
    from metrics.metric_definitions import HealthProductMetrics

    calculator = MetricCalculator(event_log)
    tester = HypothesisTester(alpha=alpha, correction_method=correction)

    primary_metric = HealthProductMetrics.BOOKING_COMPLETION_RATE
    guardrail_metrics = [HealthProductMetrics.CRASH_RATE, HealthProductMetrics.USER_7D_RETENTION]

    # ── SRM 前置检测 Banner（必须在统计结论之前展示） ────────────
    _user_groups = event_log.groupby(["user_id", "group"]).size().reset_index().groupby("group")["user_id"].count()
    _ctrl_n = int(_user_groups.get("control", n_users))
    _trt_n = int(_user_groups.get("treatment", n_users))
    _srm_banner = HashSplitter("dashboard_demo", salt="v1").check_srm(_ctrl_n, _trt_n)
    if _srm_banner["srm_detected"]:
        st.error(
            f"⚠️ **SRM 告警：检测到样本比率不匹配！**  \n"
            f"实际 treatment 占比 {_srm_banner['actual_ratio']:.2%}，"
            f"期望 {_srm_banner['expected_ratio']:.2%}，p={_srm_banner['p_value']:.4f}  \n"
            f"**实验统计结论不可信，请先排查分流/日志采集问题（见「SRM 检测」Tab）。**"
        )
    else:
        st.success(
            f"✅ SRM 检查通过（p={_srm_banner['p_value']:.4f}），分流比例正常，以下统计结论有效。"
        )

    # 计算北极星指标
    primary_result = calculator.calculate(primary_metric)
    primary_test = tester.test(primary_result, primary_metric)

    # ── 核心指标卡 ────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("对照组转化率", f"{primary_result.control_value:.2%}",
                  help="预约完成率（对照组）")
    with col2:
        delta_display = f"{primary_result.relative_change:+.2%}"
        st.metric("实验组转化率", f"{primary_result.treatment_value:.2%}",
                  delta=delta_display)
    with col3:
        significance = "✅ 显著" if primary_test.is_significant else "⚪ 不显著"
        st.metric("统计显著性", significance,
                  delta=f"p = {primary_test.p_value:.4f}")
    with col4:
        ci = primary_test.confidence_interval
        st.metric("95% 置信区间",
                  f"[{ci[0]:+.2%}, {ci[1]:+.2%}]",
                  help="相对变化的 95% 置信区间")

    st.divider()

    # ── 指标变化可视化 ────────────────────────────────────────
    st.subheader("指标对比")
    fig = go.Figure()
    groups = ["对照组 (Control)", "实验组 (Treatment)"]
    values = [primary_result.control_value, primary_result.treatment_value]
    colors = ["#636EFA", "#EF553B"]

    fig.add_trace(go.Bar(
        x=groups, y=values,
        marker_color=colors,
        text=[f"{v:.2%}" for v in values],
        textposition="outside",
        error_y=dict(
            type="data",
            array=[
                1.96 * np.sqrt(primary_result.control_variance),
                1.96 * np.sqrt(primary_result.treatment_variance),
            ],
            visible=True,
        )
    ))
    fig.update_layout(
        title="预约完成率对比（含 95% CI 误差棒）",
        yaxis_title="预约完成率",
        yaxis_tickformat=".1%",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── p-value 可视化 ────────────────────────────────────────
    st.subheader("检验统计量分布")
    z_vals = np.linspace(-4, 4, 400)
    from scipy.stats import norm
    y_vals = norm.pdf(z_vals)
    z_stat = primary_test.test_statistic

    fig2 = go.Figure()
    # 标准正态
    fig2.add_trace(go.Scatter(x=z_vals, y=y_vals, mode="lines",
                              line=dict(color="#636EFA", width=2),
                              name="标准正态分布 N(0,1)"))
    # 拒绝域
    z_crit = norm.ppf(1 - alpha / 2)
    mask_right = z_vals >= z_crit
    mask_left = z_vals <= -z_crit
    for mask, name in [(mask_right, "拒绝域"), (mask_left, "")]:
        fig2.add_trace(go.Scatter(
            x=z_vals[mask], y=y_vals[mask],
            fill="tozeroy", mode="lines",
            fillcolor="rgba(239,85,59,0.3)",
            line=dict(color="rgba(0,0,0,0)"),
            name=name, showlegend=(name != ""),
        ))
    # 当前 z 值
    fig2.add_vline(x=z_stat, line_dash="dash", line_color="green",
                   annotation_text=f"z = {z_stat:.3f}", annotation_position="top right")
    fig2.add_vline(x=-z_stat, line_dash="dash", line_color="green")

    fig2.update_layout(
        title=f"Z 检验统计量（α={alpha}，双尾）",
        xaxis_title="Z 值", yaxis_title="概率密度",
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── 结论（结构化决策框架） ─────────────────────────────────
    st.subheader("实验决策框架")

    # 计算护栏指标状态（使用 crash_rate 和 7d_retention 的 mock 结果作为演示）
    # ponytail: mock 护栏结果，真实场景从 calculator.calculate_all(guardrail_metrics) 获取
    guardrail_rows = []
    for gm in guardrail_metrics:
        try:
            gr = calculator.calculate(gm)
            gt = tester.test(gr, gm)
            failed = gt.is_significant and not gt.direction_correct
            guardrail_rows.append({
                "指标": gm.name,
                "类型": "护栏",
                "对照组": f"{gr.control_value:.4f}",
                "实验组": f"{gr.treatment_value:.4f}",
                "相对变化": f"{gr.relative_change:+.2%}",
                "p-value": f"{gt.p_value:.4f}",
                "状态": "❌ 告警" if failed else "✅ 通过",
            })
        except Exception:
            pass

    guardrail_rows.append({
        "指标": primary_metric.name,
        "类型": "北极星",
        "对照组": f"{primary_result.control_value:.4f}",
        "实验组": f"{primary_result.treatment_value:.4f}",
        "相对变化": f"{primary_result.relative_change:+.2%}",
        "p-value": f"{primary_test.p_value:.4f}",
        "状态": ("✅ 显著正向" if primary_test.is_significant and primary_test.direction_correct
                 else ("❌ 显著负向" if primary_test.is_significant else "⚪ 不显著")),
    })

    st.dataframe(pd.DataFrame(guardrail_rows), use_container_width=True, hide_index=True)

    # 最终推荐卡片
    guardrail_failed = any(r["状态"].startswith("❌") for r in guardrail_rows if r["类型"] == "护栏")
    north_star_ok = primary_test.is_significant and primary_test.direction_correct
    north_star_bad = primary_test.is_significant and not primary_test.direction_correct

    if guardrail_failed:
        st.error("**最终推荐：不推荐推全量** —— 护栏指标告警，需先排查问题再决策。")
    elif north_star_ok:
        st.success(
            f"**最终推荐：可推全量** —— 北极星指标显著正向提升 {primary_result.relative_change:+.2%}，"
            f"所有护栏通过。"
        )
    elif north_star_bad:
        st.error("**最终推荐：回滚实验** —— 北极星指标显著劣化，立即回滚。")
    else:
        st.warning(
            f"**最终推荐：延长观察** —— 北极星指标变化 {primary_result.relative_change:+.2%} 未达显著，"
            f"p={primary_test.p_value:.4f}。可延长实验或评估 MDE 是否合理。"
        )

    # ── CUPED 方差缩减演示 ────────────────────────────────────
    with st.expander("🔬 CUPED 协变量调整（降低方差，提升检验功效）"):
        st.info(
            "**CUPED 原理**：利用实验前的同指标（协变量）消除个体差异，"
            "可将方差降低 `rho²` 倍（rho 为前后指标相关系数）。\n\n"
            "等效于在不增加流量的情况下，让实验更灵敏地检测微小效果。\n\n"
            "面试说法：*我们用 CUPED 将方差降低了 XX%，等效于样本量提升了 X 倍。*"
        )
        use_cuped = st.checkbox("启用 CUPED 调整", value=False)

        # 用 session 数据模拟实验前协变量（pre_value = post_value + 噪声）
        np.random.seed(99)
        ctrl_events_cuped = event_log[event_log["group"] == "control"]
        ctrl_user_ids = ctrl_events_cuped["user_id"].unique()
        trt_events_cuped = event_log[event_log["group"] == "treatment"]
        trt_user_ids = trt_events_cuped["user_id"].unique()

        ctrl_post = ctrl_events_cuped[ctrl_events_cuped["event_name"] == "booking_success"]\
            .groupby("user_id")["value"].sum().reindex(ctrl_user_ids, fill_value=0)
        trt_post = trt_events_cuped[trt_events_cuped["event_name"] == "booking_success"]\
            .groupby("user_id")["value"].sum().reindex(trt_user_ids, fill_value=0)

        # 模拟实验前协变量：与 post 有 ~0.6 相关性
        ctrl_pre = ctrl_post + np.random.normal(0, ctrl_post.std() * 0.8, len(ctrl_post))
        trt_pre = trt_post + np.random.normal(0, trt_post.std() * 0.8, len(trt_post))

        from stats.cuped import variance_reduction_ratio
        cuped_stats = variance_reduction_ratio(ctrl_pre, ctrl_post)

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("pre/post 相关系数 ρ", f"{cuped_stats['correlation']:.3f}")
        with col_c2:
            st.metric("方差缩减比例", f"{cuped_stats['reduction_pct']:.1%}",
                      help="CUPED 后方差比原始方差低的百分比")
        with col_c3:
            st.metric("等效样本量提升", f"{cuped_stats['equivalent_sample_increase']:.2f}x",
                      help="CUPED 调整等效于样本量提升倍数 = 1/(1-ρ²)")

        if use_cuped:
            pre_data_dict = {"control": ctrl_pre.reset_index(drop=True),
                             "treatment": trt_pre.reset_index(drop=True)}
            result_cuped = calculator.calculate(primary_metric, pre_data=pre_data_dict)
            test_cuped = tester.test(result_cuped, primary_metric)
            cuped_se = np.sqrt(result_cuped.control_variance + result_cuped.treatment_variance)
            orig_se = np.sqrt(primary_result.control_variance + primary_result.treatment_variance)
            st.success(
                f"CUPED 已启用：SE {orig_se:.6f} → {cuped_se:.6f}（缩减 {1-cuped_se/orig_se:.1%}），"
                f"z 统计量 {primary_test.test_statistic:.3f} → {test_cuped.test_statistic:.3f}，"
                f"p-value {primary_test.p_value:.4f} → {test_cuped.p_value:.4f}"
            )

    st.divider()

    # ── 新老用户分层分析 ──────────────────────────────────────
    if "user_type" in event_log.columns:
        with st.expander("👥 新老用户分层分析（Subgroup Analysis）"):
            st.info(
                "**为什么要做分层分析？**\n\n"
                "新用户预约完成率基线远低于老用户（新用户约 5-8%，老用户约 18-22%）。\n"
                "若两组新老用户比例不一致，全量结果会受 **Simpson's Paradox** 影响。\n"
                "分层分析验证实验在各子群体中的效果是否一致（效果异质性检验）。"
            )
            subgroup_results = calculator.subgroup_analysis(primary_metric, "user_type")
            sg_data = []
            for sg_name, sg_result in subgroup_results.items():
                sg_test = tester.test(sg_result, primary_metric)
                sg_data.append({
                    "用户类型": "新用户" if sg_name == "new" else "老用户",
                    "对照组转化率": f"{sg_result.control_value:.2%}",
                    "实验组转化率": f"{sg_result.treatment_value:.2%}",
                    "相对变化": f"{sg_result.relative_change:+.2%}",
                    "p-value": f"{sg_test.p_value:.4f}",
                    "显著": "✅" if sg_test.is_significant else "—",
                    "对照组样本": sg_result.control_sample_size,
                    "实验组样本": sg_result.treatment_sample_size,
                })
            st.dataframe(pd.DataFrame(sg_data), use_container_width=True, hide_index=True)

            # 柱状图对比
            sg_names = [d["用户类型"] for d in sg_data]
            sg_ctrl = [float(d["对照组转化率"].rstrip("%")) / 100 for d in sg_data]
            sg_trt = [float(d["实验组转化率"].rstrip("%")) / 100 for d in sg_data]
            fig_sg = go.Figure(data=[
                go.Bar(name="对照组", x=sg_names, y=sg_ctrl,
                       text=[f"{v:.2%}" for v in sg_ctrl], textposition="outside",
                       marker_color="#636EFA"),
                go.Bar(name="实验组", x=sg_names, y=sg_trt,
                       text=[f"{v:.2%}" for v in sg_trt], textposition="outside",
                       marker_color="#EF553B"),
            ])
            fig_sg.update_layout(
                barmode="group", title="新老用户预约完成率对比",
                yaxis_tickformat=".1%", height=300,
            )
            st.plotly_chart(fig_sg, use_container_width=True)

    # ── Novelty Effect 时序分析 ───────────────────────────────
    with st.expander("📅 时间趋势 & Novelty Effect 检测"):
        st.info(
            "**Novelty Effect**：新功能上线初期，用户因好奇心点击行为异常，"
            "前几天效果虚高，后续衰减到真实水平。\n\n"
            "如果在 novelty 窗口内下结论，会高估实验收益。\n\n"
            "判断方法：前 3 天平均提升 vs 后 4 天平均提升，若后段 < 前段 70%，触发警告。"
        )
        ts_result = calculator.time_series_analysis(primary_metric)
        daily_data = ts_result["daily"]

        if len(daily_data) >= 2:
            ts_df = pd.DataFrame(daily_data)
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=ts_df["date"], y=ts_df["control_value"],
                mode="lines+markers", name="对照组",
                line=dict(color="#636EFA"),
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_df["date"], y=ts_df["treatment_value"],
                mode="lines+markers", name="实验组",
                line=dict(color="#EF553B"),
            ))
            fig_ts.update_layout(
                title="预约完成率按天趋势",
                yaxis_title="预约完成率", yaxis_tickformat=".1%",
                height=300,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            col_ts1, col_ts2 = st.columns(2)
            with col_ts1:
                st.metric("前 3 天平均提升", f"{ts_result['early_lift']:+.2%}")
            with col_ts2:
                st.metric("后段平均提升", f"{ts_result['late_lift']:+.2%}")

            if ts_result["novelty_warning"]:
                st.warning(
                    f"⚠️ **检测到 Novelty Effect！**\n\n"
                    f"前 3 天提升 {ts_result['early_lift']:+.2%}，"
                    f"后段提升 {ts_result['late_lift']:+.2%}，"
                    f"效果衰减明显。建议延长实验观察期，以后段趋势为准。"
                )
            else:
                st.success("✅ 无明显 Novelty Effect，效果相对稳定。")
        else:
            st.info("数据天数不足（需至少 2 天），请增加 mock 数据时间跨度。")


# Tab 2: SRM 检测
# ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 SRM（样本比率不匹配）检测")
    st.info(
        "**什么是 SRM？**\n\n"
        "如果实际进入实验的用户比例与配置不符（如配置 50/50 但实际 60/40），"
        "说明分流或数据采集存在 Bug，实验结论不可信。\n\n"
        "**SRM 应该在做统计检验之前先检查！**"
    )

    # 从数据中统计实际用户数
    user_groups = event_log.groupby(["user_id", "group"]).size().reset_index().groupby("group")["user_id"].count()
    ctrl_users = int(user_groups.get("control", n_users))
    trt_users = int(user_groups.get("treatment", n_users))

    splitter = HashSplitter("dashboard_demo", salt="v1")
    srm_result = splitter.check_srm(ctrl_users, trt_users)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("对照组用户数", f"{ctrl_users:,}")
    with col2:
        st.metric("实验组用户数", f"{trt_users:,}")
    with col3:
        srm_flag = "⚠️ 检测到 SRM" if srm_result["srm_detected"] else "✅ SRM 通过"
        st.metric("检测结果", srm_flag, delta=f"p = {srm_result['p_value']:.4f}")

    st.markdown(f"**{srm_result['conclusion']}**")

    # 分流比例可视化
    fig_srm = go.Figure(data=[
        go.Bar(name="实际比例", x=["对照组", "实验组"],
               y=[ctrl_users / (ctrl_users + trt_users),
                  trt_users / (ctrl_users + trt_users)],
               marker_color=["#636EFA", "#EF553B"]),
        go.Bar(name="期望比例（50%）", x=["对照组", "实验组"],
               y=[0.5, 0.5], marker_color=["#636EFA", "#EF553B"],
               opacity=0.3),
    ])
    fig_srm.update_layout(
        barmode="group",
        title="实际分流比例 vs 期望比例",
        yaxis_tickformat=".1%",
        height=300,
    )
    st.plotly_chart(fig_srm, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 3: 样本量估算
# ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("⚡ 实验样本量估算（Power Analysis）")
    st.info(
        "在实验开始前估算所需样本量，回答：**这个实验要跑多久才能得出结论？**\n\n"
        "面试问点：样本不足 → 假阴性（错过真实效果）；"
        "样本过多 → 浪费资源，且增加多重比较风险。"
    )

    col1, col2 = st.columns(2)
    with col1:
        calc_base_rate = st.number_input("基准转化率", 0.01, 0.99, 0.15, 0.01)
        mde_input = st.slider("最小可检测效应（MDE，相对变化）", 0.01, 0.30, 0.05, 0.01)
        calc_alpha = st.select_slider("显著性水平 α", [0.01, 0.05, 0.10], value=0.05)
        calc_power = st.select_slider("统计功效 (1-β)", [0.70, 0.80, 0.90], value=0.80)
        daily_traffic = st.number_input("日进入实验流量（人）", 100, 1000000, 50000, 1000)

    from experiment.experiment_manager import ExperimentManager
    import math
    from scipy.stats import norm

    p1 = calc_base_rate
    p2 = calc_base_rate * (1 + mde_input)
    z_alpha = norm.ppf(1 - calc_alpha / 2)
    z_beta = norm.ppf(calc_power)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
    n_per_group = math.ceil(n)
    total_n = n_per_group * 2
    days_needed = math.ceil(total_n / daily_traffic)

    with col2:
        st.metric("每组所需样本量", f"{n_per_group:,} 人")
        st.metric("总样本量（两组合计）", f"{total_n:,} 人")
        st.metric("预计实验周期", f"{days_needed} 天",
                  help=f"基于日流量 {daily_traffic:,} 人/天")
        st.metric("对照组基准率", f"{p1:.2%}")
        st.metric("实验组目标率", f"{p2:.2%}",
                  delta=f"+{mde_input:.1%} 相对变化")

    # MDE vs 样本量曲线
    mde_range = np.arange(0.01, 0.30, 0.01)
    sample_sizes = []
    for mde in mde_range:
        p2_tmp = calc_base_rate * (1 + mde)
        p_bar_tmp = (calc_base_rate + p2_tmp) / 2
        n_tmp = ((z_alpha * math.sqrt(2 * p_bar_tmp * (1 - p_bar_tmp)) +
                  z_beta * math.sqrt(calc_base_rate * (1 - calc_base_rate) + p2_tmp * (1 - p2_tmp))) ** 2
                 / (calc_base_rate - p2_tmp) ** 2)
        sample_sizes.append(math.ceil(n_tmp) * 2)

    fig_ss = px.line(
        x=mde_range * 100, y=sample_sizes,
        labels={"x": "MDE（最小可检测效应，%）", "y": "所需总样本量"},
        title="MDE 越小，所需样本量越大（检测微小效果代价更高）",
    )
    fig_ss.add_vline(x=mde_input * 100, line_dash="dash", line_color="red")
    fig_ss.add_annotation(
        x=mde_input * 100,
        y=max(sample_sizes),
        text=f"当前 MDE={mde_input:.0%}",
        showarrow=False,
        yshift=12,
        font=dict(color="red"),
    )
    st.plotly_chart(fig_ss, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 4: Delta Method 演示
# ─────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📚 Delta Method vs Naive 方法对比")
    st.info(
        "**为什么比率指标的方差不能直接用 p(1-p)/n？**\n\n"
        "当同一用户有多次事件（多次曝光、多次点击），"
        "用户内的事件是相关的，违反了二项分布的独立性假设。"
        "Delta Method 用泰勒展开在用户层面正确估计方差，"
        "Naive 方法低估方差，导致假阳性率虚高。"
    )

    # 提取对照组用户层数据
    ctrl_events = event_log[event_log["group"] == "control"]
    numerator = ctrl_events[ctrl_events["event_name"] == "booking_success"]\
        .groupby("user_id")["value"].sum().reindex(
            ctrl_events["user_id"].unique(), fill_value=0
        )
    denominator = ctrl_events[ctrl_events["event_name"] == "booking_flow_start"]\
        .groupby("user_id")["value"].sum().reindex(
            ctrl_events["user_id"].unique(), fill_value=1
        )

    comparison = compare_variance_methods(numerator, denominator)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("样本转化率", f"{comparison['sample_rate']:.4f}")
    with col2:
        st.metric("Delta Method SE", f"{comparison['delta_method_se']:.6f}",
                  help="推荐使用：正确的用户层方差估计")
    with col3:
        st.metric("Naive 二项 SE", f"{comparison['naive_binomial_se']:.6f}",
                  help="错误方法：忽略用户内部相关性")

    naive_inflation = comparison['delta_method_se'] / comparison['naive_binomial_se'] \
        if comparison['naive_binomial_se'] > 0 else 1
    st.markdown(
        f"**Delta Method SE 是 Naive 方法的 {naive_inflation:.2f}x。**\n\n"
        f"若使用 Naive 方法，标准误被低估 {(1-1/naive_inflation):.1%}，"
        f"z 统计量被虚高 {naive_inflation:.2f}x，导致假阳性率远超 {alpha:.0%}。"
    )

    st.caption(comparison["note"])


# ─────────────────────────────────────────────────────────────
# Tab 5: 实验层管理（Layer）
# ─────────────────────────────────────────────────────────────
with tab5:
    from allocation.experiment_layer import ExperimentLayer, LayerExperiment

    st.subheader("🗂️ 实验层管理（Layer System）")
    st.info(
        "**为什么需要实验层？**\n\n"
        "健康产品线上同时运行数十个实验。若不加管控，两个影响同一页面的实验会同时作用于用户，"
        "产生**实验间干扰**，导致归因错误。\n\n"
        "- **互斥层（Exclusive）**：同一 Layer 内用户只能进一个实验（如预约页 UI 改版）\n"
        "- **正交层（Orthogonal）**：不同 Layer 独立分流，用户可同时参与（如 UI 层 × 推荐算法层）"
    )

    # ── 演示：构建两个 Layer ──────────────────────────────────
    # 互斥层：预约页 UI 实验（ui_v1 和 ui_v2 互斥，占 60% 流量）
    ui_layer = ExperimentLayer("booking_ui_layer", layer_type="exclusive")
    ui_layer.add_experiment(LayerExperiment("ui_v1_button_color", traffic_fraction=0.3))
    ui_layer.add_experiment(LayerExperiment("ui_v2_step_simplify", traffic_fraction=0.3))

    # 正交层：Feed 推荐算法（与 UI 层正交，用户可同时命中）
    feed_layer = ExperimentLayer("feed_rank_layer", layer_type="orthogonal")
    feed_layer.add_experiment(LayerExperiment("rank_v2_personalized", traffic_fraction=0.5, control_ratio=0.5, treatment_ratio=0.5))

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**互斥层配置**")
        ui_summary = ui_layer.get_layer_summary()
        st.json(ui_summary)

    with col_l2:
        st.markdown("**正交层配置**")
        feed_summary = feed_layer.get_layer_summary()
        st.json(feed_summary)

    st.divider()
    st.subheader("用户分配演示（模拟 1000 个用户）")

    # 模拟分配
    demo_users = [f"user_{i}" for i in range(1000)]
    ui_assignments = {"ui_v1_button_color": 0, "ui_v2_step_simplify": 0, "未命中": 0}
    feed_hit = 0
    both_hit = 0

    for uid in demo_users:
        ui_result = ui_layer.assign(uid)
        feed_result = feed_layer.assign(uid)
        if ui_result:
            ui_assignments[ui_result["experiment_id"]] = ui_assignments.get(ui_result["experiment_id"], 0) + 1
        else:
            ui_assignments["未命中"] += 1
        if feed_result:
            feed_hit += 1
        if ui_result and feed_result:
            both_hit += 1

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.metric("命中 UI 实验 v1", f"{ui_assignments['ui_v1_button_color']} 人",
                  help="约 30% 流量")
    with col_d2:
        st.metric("命中 UI 实验 v2", f"{ui_assignments['ui_v2_step_simplify']} 人",
                  help="约 30% 流量")
    with col_d3:
        st.metric("UI 实验未命中", f"{ui_assignments['未命中']} 人",
                  help="剩余 40% 流量不进入任何 UI 实验")

    col_d4, col_d5 = st.columns(2)
    with col_d4:
        st.metric("命中 Feed 排序实验", f"{feed_hit} 人", help="约 50% 流量（与 UI 层正交）")
    with col_d5:
        st.metric("同时命中两个 Layer", f"{both_hit} 人",
                  help=f"理论值约 {0.6*0.5:.0%}×1000=300 人（正交性验证）")

    # 互斥性验证：同一用户不能同时命中 ui_v1 和 ui_v2
    collision = 0
    for uid in demo_users[:200]:
        r1 = ui_layer.assign(uid)
        r2 = ui_layer.assign(uid)
        if r1 and r2 and r1["experiment_id"] != r2["experiment_id"]:
            collision += 1  # 幂等性：同一用户两次应该返回相同结果

    st.success(
        f"互斥性验证：同一用户两次分配结果完全一致（碰撞次数 = {collision}，应为 0）。\n\n"
        f"正交性验证：同时命中两层 {both_hit} 人，理论值约 300 人，"
        f"{'✅ 正交性正常' if 250 <= both_hit <= 350 else '⚠️ 偏差较大，请检查'}"
    )


# ─────────────────────────────────────────────────────────────
# Tab 6: 序贯检验演示
# ─────────────────────────────────────────────────────────────
with tab6:
    from stats.sequential import obrien_fleming_boundary, mixture_sequential_probability_ratio

    st.subheader("🔁 序贯检验演示（解决 peeking problem）")
    st.info(
        "**为什么需要序贯检验？**\n\n"
        "固定样本量的 Z/t 检验只允许在预先计划好的样本量下检验一次。如果中途反复查看数据、"
        "一旦显著就停止（peeking），实际假阳性率会远超设定的 α。\n\n"
        "下面把左侧模拟数据按 20% / 40% / 60% / 80% / 100% 累计样本量模拟"
        "**\"多次查看\"**，对比：\n"
        "- 传统固定样本量 Z 检验（每次查看都直接用标准 z_alpha/2 判定，容易提前误判显著）\n"
        "- O'Brien-Fleming Alpha Spending 边界（早期严格，收窄到标准值）\n"
        "- mSPRT 似然比（anytime-valid，看阈值 1/α）"
    )

    from metrics.metric_calculator import MetricCalculator
    from metrics.metric_definitions import HealthProductMetrics

    primary_metric_seq = HealthProductMetrics.BOOKING_COMPLETION_RATE
    look_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    # 按累计样本量比例截取 event_log 的前 N 个用户（模拟"多次查看"）
    all_users_seq = event_log["user_id"].unique()
    n_total_users = len(all_users_seq)

    seq_rows = []
    for frac in look_fractions:
        n_cutoff = max(2, int(n_total_users * frac))
        cutoff_users = set(all_users_seq[:n_cutoff])
        partial_log = event_log[event_log["user_id"].isin(cutoff_users)]

        try:
            partial_calc = MetricCalculator(partial_log)
            partial_result = partial_calc.calculate(primary_metric_seq)
        except Exception:
            continue

        se_ctrl = np.sqrt(partial_result.control_variance)
        se_trt = np.sqrt(partial_result.treatment_variance)
        se_diff = np.sqrt(se_ctrl ** 2 + se_trt ** 2)
        diff = partial_result.treatment_value - partial_result.control_value
        z_stat = diff / se_diff if se_diff > 0 else 0.0

        of_boundary = obrien_fleming_boundary(information_fraction=frac, alpha=alpha)
        standard_z = 1.96  # alpha=0.05 双尾标准值，仅作参考线

        pooled_variance = partial_result.control_variance + partial_result.treatment_variance
        n_pooled = partial_result.control_sample_size + partial_result.treatment_sample_size
        msprt_result = mixture_sequential_probability_ratio(
            n=n_pooled, sample_mean_diff=diff, sample_variance=pooled_variance, alpha=alpha,
        )

        seq_rows.append({
            "信息量占比": f"{frac:.0%}",
            "累计样本量": n_pooled,
            "z 统计量": round(z_stat, 3),
            "O'Brien-Fleming 边界": round(of_boundary, 3),
            "传统检验是否显著(|z|>1.96)": abs(z_stat) > standard_z,
            "序贯检验是否可停止(|z|>边界)": abs(z_stat) > of_boundary,
            "mSPRT 似然比": msprt_result["likelihood_ratio"],
            "mSPRT 阈值(1/α)": msprt_result["threshold"],
            "mSPRT 是否显著": msprt_result["is_significant"],
        })

    if seq_rows:
        seq_df = pd.DataFrame(seq_rows)
        st.dataframe(seq_df, use_container_width=True)

        # 教学对比：找出"传统检验提前误判显著，但序贯方法未误判"的信息量占比
        premature_calls = seq_df[
            seq_df["传统检验是否显著(|z|>1.96)"] & (~seq_df["序贯检验是否可停止(|z|>边界)"])
        ]
        if not premature_calls.empty:
            st.warning(
                f"⚠️ 教学要点：信息量占比 {premature_calls.iloc[0]['信息量占比']} 时，"
                "传统固定样本量检验已判定显著（可能引发 peeking 导致的假阳性），"
                "但 O'Brien-Fleming 序贯边界认为还不能停止 —— 这正是序贯检验防止过早下结论的价值所在。"
            )
        else:
            st.success("当前模拟参数下，传统检验与序贯检验在各信息量占比下的显著性判定一致（未出现提前误判场景）。")

        # z 统计量走势 + O'Brien-Fleming 边界收窄曲线
        fig_seq = go.Figure()
        fig_seq.add_trace(go.Scatter(
            x=seq_df["信息量占比"], y=seq_df["z 统计量"],
            mode="lines+markers", name="z 统计量", line=dict(color="#1f77b4"),
        ))
        fig_seq.add_trace(go.Scatter(
            x=seq_df["信息量占比"], y=seq_df["O'Brien-Fleming 边界"],
            mode="lines+markers", name="O'Brien-Fleming 边界（上）", line=dict(color="#d62728", dash="dash"),
        ))
        fig_seq.add_trace(go.Scatter(
            x=seq_df["信息量占比"], y=-seq_df["O'Brien-Fleming 边界"],
            mode="lines+markers", name="O'Brien-Fleming 边界（下）", line=dict(color="#d62728", dash="dash"),
        ))
        fig_seq.add_hline(y=1.96, line_dash="dot", line_color="gray",
                          annotation_text="传统检验标准边界 z=1.96")
        fig_seq.add_hline(y=-1.96, line_dash="dot", line_color="gray")
        fig_seq.update_layout(
            title="z 统计量走势 vs O'Brien-Fleming 边界收窄曲线",
            xaxis_title="信息量占比（模拟第几次查看）", yaxis_title="z 值",
            height=400,
        )
        st.plotly_chart(fig_seq, use_container_width=True)

        # mSPRT 似然比 vs 1/alpha 阈值
        fig_msprt = go.Figure()
        fig_msprt.add_trace(go.Scatter(
            x=seq_df["信息量占比"], y=seq_df["mSPRT 似然比"],
            mode="lines+markers", name="mSPRT 似然比 Λ_n", line=dict(color="#2ca02c"),
        ))
        fig_msprt.add_hline(
            y=seq_df["mSPRT 阈值(1/α)"].iloc[0], line_dash="dot", line_color="gray",
            annotation_text=f"阈值 1/α={seq_df['mSPRT 阈值(1/α)'].iloc[0]:.1f}",
        )
        fig_msprt.update_layout(
            title="mSPRT 似然比走势 vs 1/α 阈值（anytime-valid，随时可停止判定）",
            xaxis_title="信息量占比（模拟第几次查看）", yaxis_title="似然比 Λ_n",
            height=400,
        )
        st.plotly_chart(fig_msprt, use_container_width=True)
    else:
        st.warning("当前模拟数据样本量过小，无法生成序贯检验演示数据，请调整左侧参数。")


# ─────────────────────────────────────────────────────────────
# Tab 7: 实验生命周期管理
# ─────────────────────────────────────────────────────────────
with tab7:
    from experiment.config_schema import ExperimentConfig, ExperimentStatus
    from experiment.experiment_manager import ExperimentManager
    from metrics.metric_definitions import HealthProductMetrics

    st.subheader("🔄 实验生命周期管理")
    st.info(
        "这里复用 ExperimentManager 的真实状态机和 YAML 持久化能力，"
        "用于演示实验从创建、AA 测试、正式运行到结论的完整流转。"
    )

    lifecycle_store = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiment_store_dashboard")
    manager = ExperimentManager(storage_path=lifecycle_store)

    def _scenario_metrics(scenario_name):
        if scenario_name == "预约转化":
            return HealthProductMetrics.get_booking_metrics()
        if scenario_name == "在线咨询":
            return HealthProductMetrics.get_consultation_metrics()
        return HealthProductMetrics.get_feed_metrics()

    def _status_label(status):
        labels = {
            ExperimentStatus.DRAFT: "草稿",
            ExperimentStatus.AA_TESTING: "AA 测试中",
            ExperimentStatus.RUNNING: "运行中",
            ExperimentStatus.PAUSED: "已暂停",
            ExperimentStatus.CONCLUDED: "已结论",
            ExperimentStatus.ABORTED: "已终止",
        }
        return labels.get(status, status.value if hasattr(status, "value") else str(status))

    def _format_time(value):
        return value if value else "-"

    create_col, list_col = st.columns([1, 1.45])

    with create_col:
        st.markdown("#### 创建实验")
        with st.form("create_experiment_form"):
            experiment_id_input = st.text_input("实验 ID", value="booking_rec_demo", key="lifecycle_experiment_id")
            experiment_name_input = st.text_input("实验名称", value="预约推荐算法实验", key="lifecycle_experiment_name")
            description_input = st.text_area(
                "实验描述", value="测试新的推荐策略对预约完成率的影响", height=88,
                key="lifecycle_description",
            )
            owner_input = st.text_input("负责人", value="demo_owner", key="lifecycle_owner")
            scenario_input = st.selectbox("场景模板", ["预约转化", "在线咨询", "内容推荐"], key="lifecycle_scenario")

            ratio_cols = st.columns(3)
            with ratio_cols[0]:
                control_ratio_input = st.number_input(
                    "Control 比例", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                    key="lifecycle_control_ratio",
                )
            with ratio_cols[1]:
                treatment_ratio_input = st.number_input(
                    "Treatment 比例", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                    key="lifecycle_treatment_ratio",
                )
            with ratio_cols[2]:
                holdout_ratio_input = st.number_input(
                    "Holdout 比例", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                    key="lifecycle_holdout_ratio",
                )

            runtime_cols = st.columns(3)
            with runtime_cols[0]:
                min_runtime_input = st.number_input(
                    "最短运行天数", min_value=1, max_value=60, value=7, step=1,
                    key="lifecycle_min_runtime",
                )
            with runtime_cols[1]:
                alpha_input = st.number_input(
                    "显著性水平 α", min_value=0.001, max_value=0.2, value=0.05, step=0.005,
                    key="lifecycle_alpha",
                )
            with runtime_cols[2]:
                power_input = st.number_input(
                    "统计功效", min_value=0.5, max_value=0.99, value=0.8, step=0.05,
                    key="lifecycle_power",
                )

            correction_input = st.selectbox(
                "多重检验校正", ["bonferroni", "bh", "none"], key="lifecycle_correction"
            )
            use_seq_input = st.checkbox("启用序贯检验", key="lifecycle_use_sequential")
            seq_cols = st.columns(2)
            with seq_cols[0]:
                seq_method_input = st.selectbox(
                    "序贯方法", ["alpha_spending", "msprt"], key="lifecycle_seq_method"
                )
            with seq_cols[1]:
                spending_input = st.selectbox(
                    "Alpha Spending 函数", ["obrien_fleming", "pocock"], key="lifecycle_spending"
                )

            submitted = st.form_submit_button("创建实验", use_container_width=True)

        if submitted:
            ratio_sum = control_ratio_input + treatment_ratio_input + holdout_ratio_input
            if not experiment_id_input.strip():
                st.error("实验 ID 不能为空。")
            elif not experiment_name_input.strip():
                st.error("实验名称不能为空。")
            elif abs(ratio_sum - 1.0) > 1e-6:
                st.error(f"流量比例之和必须等于 1，当前为 {ratio_sum:.2f}。")
            else:
                try:
                    config = ExperimentConfig(
                        experiment_id=experiment_id_input.strip(),
                        name=experiment_name_input.strip(),
                        description=description_input.strip(),
                        owner=owner_input.strip() or "demo_owner",
                        team="default",
                        control_ratio=control_ratio_input,
                        treatment_ratio=treatment_ratio_input,
                        holdout_ratio=holdout_ratio_input,
                        min_runtime_days=int(min_runtime_input),
                        metrics=_scenario_metrics(scenario_input),
                        alpha=alpha_input,
                        power=power_input,
                        use_sequential_testing=use_seq_input,
                        sequential_method=seq_method_input,
                        sequential_spending_function=spending_input,
                        multiple_testing_correction=correction_input,
                    )
                    manager.create(config)
                    st.success(f"实验 {config.experiment_id} 已创建，当前状态：草稿。")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with list_col:
        st.markdown("#### 实验列表")
        experiments = manager.list_experiments()
        status_counts = {status: 0 for status in ExperimentStatus}
        for exp in experiments:
            status_counts[exp.status] = status_counts.get(exp.status, 0) + 1

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("实验总数", len(experiments))
        with metric_cols[1]:
            st.metric("运行中", status_counts.get(ExperimentStatus.RUNNING, 0))
        with metric_cols[2]:
            st.metric("AA 测试中", status_counts.get(ExperimentStatus.AA_TESTING, 0))
        with metric_cols[3]:
            st.metric("已结论", status_counts.get(ExperimentStatus.CONCLUDED, 0))

        if experiments:
            rows = []
            for exp in experiments:
                rows.append({
                    "实验 ID": exp.experiment_id,
                    "实验名称": exp.name,
                    "负责人": exp.owner,
                    "状态": _status_label(exp.status),
                    "指标数": len(exp.metrics),
                    "Control": f"{exp.control_ratio:.0%}",
                    "Treatment": f"{exp.treatment_ratio:.0%}",
                    "Holdout": f"{exp.holdout_ratio:.0%}",
                    "开始时间": _format_time(exp.start_time),
                    "最短运行天数": exp.min_runtime_days,
                    "AA 通过": "是" if exp.aa_passed else "否",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            status_df = pd.DataFrame([
                {"状态": _status_label(status), "数量": count}
                for status, count in status_counts.items()
                if count > 0
            ])
            if not status_df.empty:
                fig_status = px.bar(status_df, x="状态", y="数量", text="数量", title="实验状态分布")
                fig_status.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_status, use_container_width=True)
        else:
            st.warning("当前还没有实验配置。请先在左侧创建一个实验。")

    st.divider()
    st.markdown("#### 状态操作与配置预览")

    experiments = manager.list_experiments()
    if experiments:
        exp_ids = [exp.experiment_id for exp in experiments]
        selected_exp_id = st.selectbox("选择实验", exp_ids, key="lifecycle_selected_experiment")
        selected_exp = manager.get(selected_exp_id)

        detail_cols = st.columns(4)
        with detail_cols[0]:
            st.metric("当前状态", _status_label(selected_exp.status))
        with detail_cols[1]:
            st.metric("最短运行天数", selected_exp.min_runtime_days)
        with detail_cols[2]:
            st.metric("指标数量", len(selected_exp.metrics))
        with detail_cols[3]:
            st.metric("AA 是否通过", "是" if selected_exp.aa_passed else "否")

        action_cols = st.columns(3)
        try:
            if selected_exp.status == ExperimentStatus.DRAFT:
                with action_cols[0]:
                    if st.button("启动 AA 测试", use_container_width=True, key="lifecycle_start_aa"):
                        manager.start_aa_test(selected_exp_id)
                        st.success("已启动 AA 测试。")
                        st.rerun()
            elif selected_exp.status == ExperimentStatus.AA_TESTING:
                with action_cols[0]:
                    if st.button("AA 通过并启动实验", use_container_width=True, key="lifecycle_pass_aa"):
                        manager.pass_aa_and_start(selected_exp_id)
                        st.success("实验已进入正式运行。")
                        st.rerun()
            elif selected_exp.status == ExperimentStatus.RUNNING:
                with action_cols[0]:
                    pause_reason = st.text_input(
                        "暂停原因", value="发现指标异常，暂停观察", key="lifecycle_pause_reason"
                    )
                    if st.button("暂停实验", use_container_width=True, key="lifecycle_pause"):
                        manager.pause(selected_exp_id, reason=pause_reason)
                        st.success("实验已暂停。")
                        st.rerun()
                with action_cols[1]:
                    conclusion_text = st.text_area(
                        "实验结论", value="实验达到预期，建议进入下一阶段。", height=90,
                        key="lifecycle_running_conclusion_text",
                    )
                    if st.button("实验结论", use_container_width=True, key="lifecycle_running_conclude"):
                        manager.conclude(selected_exp_id, conclusion_text)
                        st.success("实验已结论。")
                        st.rerun()
            elif selected_exp.status == ExperimentStatus.PAUSED:
                with action_cols[0]:
                    conclusion_text = st.text_area(
                        "实验结论", value="实验暂停后完成复盘，给出最终结论。", height=90,
                        key="lifecycle_paused_conclusion_text",
                    )
                    if st.button("实验结论", use_container_width=True, key="lifecycle_paused_conclude"):
                        manager.conclude(selected_exp_id, conclusion_text)
                        st.success("实验已结论。")
                        st.rerun()
                st.caption("当前 demo 后端未开放恢复运行操作，因此暂停状态只支持进入结论。")
            else:
                st.success("该实验已进入终态，当前没有可执行的状态操作。")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))

        with st.expander("查看实验 YAML 配置", expanded=False):
            st.code(selected_exp.to_yaml(), language="yaml")
    else:
        st.info("创建实验后，这里会显示状态操作和配置预览。")

