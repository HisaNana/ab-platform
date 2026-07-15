# AB 实验评估平台 —— 通用健康产品场景

> 一个从**分流 → 实验配置 → 指标计算 → 显著性检验 → 可视化看板**的完整 AB 实验框架。
> 面向通用健康产品场景设计（预约转化、在线咨询、内容推荐三类典型场景）。

---

## 项目架构

```
ab-platform/
├── allocation/
│   ├── hash_splitter.py        # 一致性哈希分流 + SRM 检测
│   ├── stratified_sampler.py   # 分层抽样（按用户关注类型/城市层级/活跃度）
│   └── experiment_layer.py     # 实验层（Layer）互斥/正交隔离
├── experiment/
│   ├── config_schema.py        # 实验配置 Schema（YAML/JSON 序列化）
│   └── experiment_manager.py   # 实验生命周期管理 + 样本量估算 + run_analysis() 高层 API
├── metrics/
│   ├── metric_definitions.py   # 通用健康产品标准指标库
│   └── metric_calculator.py    # 用户层指标聚合引擎
├── stats/
│   ├── t_test.py               # Z/t 检验 + 多重比较校正（Bonferroni/BH）
│   ├── delta_method.py         # Delta Method 比率指标方差估计（含原理注释）
│   ├── cuped.py                # CUPED 协变量方差缩减
│   └── sequential.py           # 序贯检验（Alpha Spending + mSPRT）
├── dashboard/
│   └── experiment_dashboard.py # Streamlit 实验看板
├── tests/
│   ├── test_aa_experiment.py   # AA 实验验证（假阳性率 + SRM 检测 + CUPED + 分层分析）
│   ├── test_sequential.py      # 序贯检验单测
│   └── test_run_analysis.py    # run_analysis() 高层 API 集成测试
├── example_full_flow.py        # 手动 7 步 vs run_analysis() 一行调用对比
├── .github/workflows/test.yml  # CI：push/PR 自动跑测试
└── requirements.txt
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 运行全部测试
pytest tests/ -v

# 运行完整流程演示（手动串联 vs run_analysis()）
python3 example_full_flow.py
```

### 一键启动看板

- macOS：双击 [start_dashboard.command](start_dashboard.command)
- Windows：双击 [start_dashboard.bat](start_dashboard.bat)
- 命令行方式：`python3 -m streamlit run dashboard/experiment_dashboard.py`

### 示例页面

- 静态示例 HTML：[docs/dashboard_example.html](docs/dashboard_example.html)
- 该页面用于快速预览项目能力，不依赖 Streamlit，可直接用浏览器打开。

### `run_analysis()` 一行上手

```python
from experiment.experiment_manager import ExperimentManager

manager = ExperimentManager(storage_path="./experiment_store")
manager.create(config)  # config 为 ExperimentConfig

report = manager.run_analysis(experiment_id="booking_rec_v2", event_log=event_log)

print(report.srm_passed)        # SRM 是否通过
print(report.warnings)          # 结构化告警列表（如 SRM 不通过、holdout 数据等）
print(report.conclusion_text)   # 综合决策结论（护栏 + 北极星）
print(report.metric_results)    # {metric_name: TestResult}
```

`run_analysis()` 内部完成：读取实验配置 → SRM 检查 → 批量指标计算 → 假设检验（含可选序贯检验路径）→ 生成决策结论。相比手动串联 7 步，适合脚本化调用和自动化报表场景。

---

## 核心设计决策

### 1. 为什么用 MD5 一致性哈希分流？

- **幂等性**：同一用户在相同实验配置下，永远进入相同分组
- **无状态**：不需要存储每个用户的分组，按需实时计算
- **隔离性**：`experiment_key + salt` 确保不同实验使用独立的哈希空间，避免用户重叠偏差

```python
raw = f"{experiment_key}:{salt}:{user_id}"
bucket = int(md5(raw)[:8], 16) % 10000  # 万分之一精度
```

### 2. 为什么需要 Delta Method 估计比率指标方差？

**错误做法（常见坑）：**
```python
# 直接用二项分布方差
var = p * (1 - p) / n  # 假设了事件独立，但同一用户的多次点击不独立！
```

**正确做法 —— Delta Method：**

设用户 $i$ 的分子为 $X_i$（点击数），分母为 $Y_i$（曝光数）。
比率 $R = \frac{\sum X_i}{\sum Y_i}$，用泰勒展开一阶近似：

$$\text{Var}(R) \approx \frac{\mu_Y^2 \sigma_X^2 + \mu_X^2 \sigma_Y^2 - 2\mu_X\mu_Y\text{Cov}(X,Y)}{n \cdot \mu_Y^4}$$

**影响**：忽略用户内部相关性会导致方差被低估，Z 统计量虚高，假阳性率远超 5%。

### 3. 指标体系分层设计

| 指标类型 | 作用 | 多重比较校正 | 判断逻辑 |
|---------|------|------------|---------|
| **北极星（Primary）** | 实验核心判断依据，只有 1 个 | Bonferroni | 显著正向 → 实验成功候选 |
| **护栏（Guardrail）** | 必须不能损害的体验基线 | 不校正（单独检验） | 任一显著下降 → 实验直接失败 |
| **探索性（Exploratory）** | 了解影响面，不作决策依据 | BH（宽松） | 仅供参考，需注明不确定性 |

### 4. SRM（样本比率不匹配）检测

SRM 是 AB 实验最常见的 data quality 问题，**必须在统计检验之前先做 SRM 检查**。

```python
# 使用卡方检验
chi2, p_value = chisquare(observed=[control_n, treatment_n],
                           expected=[total * 0.5, total * 0.5])
if p_value < 0.01:
    print("⚠️ 检测到 SRM，实验数据不可信！")
```

常见 SRM 原因：
- 客户端 SDK 对某类设备的日志采集有 Bug
- 某个服务端逻辑意外排除了部分实验组用户
- 缓存策略导致部分用户命中了旧的对照逻辑

### 5. 实验生命周期状态机

```
DRAFT → AA_TESTING → RUNNING → CONCLUDED
                              ↘ PAUSED → RUNNING
                              ↘ ABORTED
```

**AA 测试**是关键环节：在正式实验前，两组均使用对照策略运行 1-3 天，
验证分流无偏后才允许转为 RUNNING 状态。

### 6. 序贯检验：解决 peeking problem

固定样本量的 t/Z 检验只允许在预先计划好的样本量下检验一次。如果中途多次查看数据、
一旦显著就停止（peeking），实际假阳性率会远超设定的 α。序贯检验允许在任意时刻查看结果，
仍能保持统计有效性。本项目实现两种互补方法：

| 方法 | 原理 | 特点 |
|------|------|------|
| **Alpha Spending**（O'Brien-Fleming / Pocock） | 预先规划总查看次数，把 α 按信息量分配到各次检验，早期边界更严格 | 需提前定好总查看次数；O'Brien-Fleming 早期几乎不消耗 α，接近传统检验效力 |
| **mSPRT**（mixture Sequential Probability Ratio Test） | 正态-正态共轭混合似然比，构造 anytime-valid 的检验统计量 | 无需预先规划查看次数，可在任意时刻停止；对不同的效应量大小需设置 `tau_squared` 先验 |

```python
from stats.sequential import alpha_spending_test, mixture_sequential_probability_ratio

# Alpha Spending：需要提供当前信息量占比（如已积累样本 / 计划总样本）
result = alpha_spending_test(z_statistic=2.1, information_fraction=0.4,
                              spending_function="obrien_fleming")

# mSPRT：无需信息量占比，可在任意样本量下调用
result = mixture_sequential_probability_ratio(n=5000, sample_mean_diff=0.008,
                                                sample_variance=0.02)
```

---

## 通用健康产品业务场景说明

### 预约转化（Booking）
- **北极星指标**：预约完成率（`booking_completion_rate`）
- **护栏指标**：预约漏斗流失率、全局崩溃率、7 日留存
- **典型实验**：推荐算法优化、预约流程步骤简化

### 在线咨询（Consultation）
- **北极星指标**：咨询启动率（`consultation_start_rate`）
- **护栏指标**：咨询完成率（防止入口优化但体验下降）、服务提供者满意度
- **典型实验**：咨询入口位置/样式 A/B

### 内容推荐（Content Feed）
- **北极星指标**：深度阅读率（停留 ≥ 30s 的点击占比）
- **护栏指标**：负反馈率（不感兴趣/屏蔽）
- **典型实验**：个性化排序算法迭代

---

## 面试问答速查

**Q: 为什么设置 min_runtime_days？**
A: 防止 peeking problem。当你多次查看结果并在显著时停止，
相当于做了多次检验，实际假阳性率远高于 α。
最短运行时长确保数据覆盖一个完整的用户行为周期（通常是 7 天，覆盖工作日+周末差异）。

**Q: Bonferroni 和 BH 校正怎么选？**
A: 取决于你更关心 FWER 还是 FDR。
Bonferroni 控制 FWER（任何假阳性的概率），适合少量重要指标。
BH 控制 FDR（假阳性占显著结果的比例），适合探索性的多指标分析，power 更高。
在本系统中，北极星指标用 Bonferroni（只有 1 个时无需校正），探索性用 BH。

**Q: AA 实验的假阳性率应该是多少？**
A: 理论上应接近 α（如 5%）。如果 AA 实验假阳性率显著高于 α，
说明方差估计有偏（如该用 Delta Method 但用了 Naive 方法），
或者分流本身有偏。本项目的测试用例会模拟 200 次 AA 实验验证这一点。

**Q: 序贯检验和 min_runtime_days 是不是重复了？**
A: 不是同一层面的手段。`min_runtime_days` 只是一个"硬性下限"，防止实验运行几小时就下结论，
但达到下限后如果继续每天查看仍然会引入 peeking 问题。序贯检验（Alpha Spending / mSPRT）
从统计层面解决了"任意时刻查看都不引入额外假阳性"的问题，两者可以叠加使用。

**Q: Alpha Spending 和 mSPRT 该怎么选？**
A: 如果能提前确定总共要查看几次（如每天一次，跑 14 天），Alpha Spending（尤其 O'Brien-Fleming）
是标准选择，前期边界严格，后期趋近传统检验，功效损失小。如果查看时刻不可预知（如业务方随时可能要求
出结果），mSPRT 的 anytime-valid 特性更合适，但需要对效应量量级有一个先验估计（`tau_squared`）。

**Q: run_analysis() 遇到 SRM 不通过会怎么处理？**
A: 不会中断计算，而是继续输出全部指标结果，并在 `report.warnings` 中追加结构化告警，
`report.srm_passed` 置为 `False`。这个设计与 Dashboard 的行为保持一致（先给出 SRM 红色告警，
但仍展示统计结论），把决策权交给调用方——脚本化调用时可以检查 `srm_passed` 决定是否阻断下游流程。

---

## 技术栈

Python · pandas · numpy · scipy · Streamlit · Plotly · PyYAML · pytest
