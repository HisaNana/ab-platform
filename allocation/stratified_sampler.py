"""
分层抽样分流模块
在人群分层后进行流量分配，确保各层用户在对照/实验组中比例一致。
通用健康产品场景：按用户关注类型、城市层级、活跃度等分层。
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class StratumConfig:
    """分层配置"""
    name: str                         # 分层名称，如 "tier1_city"
    keys: List[str]                   # 该层的用户 ID 列表
    treatment_ratio: float = 0.5      # 该层内实验组占比


class StratifiedSampler:
    """
    分层抽样器。

    使用场景（通用健康产品）：
        - 关注类型分层：慢病管理用户 / 健身用户 / 母婴关注用户
          → 确保实验组和对照组中各类关注人群比例一致
        - 城市分层：一线城市 / 新一线 / 三四线
          → 避免资源可达性差异造成的地域偏差
        - 活跃度分层：高活 DAU / 中活 / 低活
          → 避免 novelty effect 在高活用户中被稀释

    为什么需要分层抽样而不是纯随机：
        纯随机在小流量实验中可能导致某个关键子群（如高意向预约用户）
        在两组中比例失衡，从而污染实验结论。
    """

    def __init__(self, experiment_key: str, salt: str = "v1"):
        from .hash_splitter import HashSplitter
        self.splitter = HashSplitter(experiment_key, salt)
        self.strata: Dict[str, StratumConfig] = {}

    def add_stratum(self, config: StratumConfig):
        """注册一个分层。"""
        self.strata[config.name] = config

    def assign_all(self) -> Dict[str, Dict[str, str]]:
        """
        对所有分层用户执行分组。

        Returns:
            {stratum_name: {user_id: group_name}}
        """
        result = {}
        for stratum_name, config in self.strata.items():
            assignments = {}
            for uid in config.keys:
                group = self.splitter.assign_group(
                    user_id=f"{stratum_name}:{uid}",
                    control_ratio=1 - config.treatment_ratio,
                    treatment_ratio=config.treatment_ratio,
                )
                assignments[uid] = group
            result[stratum_name] = assignments
        return result

    def get_balance_report(self) -> List[Dict]:
        """
        输出各层分组平衡报告，用于实验启动前的 AA 检查。

        Returns:
            列表，每项包含层名称、期望比例、实际比例和是否平衡
        """
        all_assignments = self.assign_all()
        report = []
        for stratum_name, assignments in all_assignments.items():
            config = self.strata[stratum_name]
            total = len(assignments)
            treatment_count = sum(1 for g in assignments.values() if g == "treatment")
            actual_ratio = treatment_count / total if total > 0 else 0
            expected = config.treatment_ratio
            # 允许 ±2% 的偏差
            balanced = abs(actual_ratio - expected) <= 0.02
            report.append({
                "stratum": stratum_name,
                "total_users": total,
                "treatment_count": treatment_count,
                "control_count": total - treatment_count,
                "expected_treatment_ratio": expected,
                "actual_treatment_ratio": round(actual_ratio, 4),
                "balanced": balanced,
            })
        return report
