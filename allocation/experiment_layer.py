"""
实验层（Layer）管理 —— 互斥/正交实验隔离。

背景：
    健康产品线上同时运行数十个实验（预约 UI、Feed 排序、咨询入口……），
    必须明确哪些实验会相互影响（互斥），哪些可以叠加（正交）。

核心概念：
    - Layer（实验层）：一组具有相同业务域的实验容器
    - 互斥层（Exclusive）：同一用户在一个 Layer 内只能命中一个实验
      用途：同一页面的 UI 改版实验，不能让用户同时看到两版 UI
    - 正交层（Orthogonal）：不同 Layer 的实验独立分流，用户可同时参与多个 Layer
      用途：预约 UI Layer 与 Feed 推荐 Layer 互不影响

分流原理：
    互斥层：先将用户哈希到 Layer 空间，再在层内分配到具体实验
            → 保证层内每个用户只属于一个实验
    正交层：各实验使用不同的 experiment_key 独立哈希
            → 用户在不同实验中的分组相互独立（近似正交）

面试说法：
    "我们用 Layer 机制解决了实验间干扰问题：
     同一业务域的实验放在互斥层，保证用户体验一致性；
     不同业务域的实验放在正交层，最大化流量利用率。"
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LayerExperiment:
    """层内的单个实验配置"""
    experiment_id: str
    traffic_fraction: float   # 该实验占层内流量的比例（互斥层内所有实验之和 ≤ 1）
    control_ratio: float = 0.5
    treatment_ratio: float = 0.5


class ExperimentLayer:
    """
    实验层：管理一组互斥或正交实验的流量分配。

    互斥层示例（预约页 UI 实验）：
        layer = ExperimentLayer("booking_ui_layer", layer_type="exclusive")
        layer.add_experiment(LayerExperiment("ui_v1", traffic_fraction=0.3))
        layer.add_experiment(LayerExperiment("ui_v2", traffic_fraction=0.3))
        # 剩余 40% 流量不进入任何实验
        result = layer.assign("user_123")
        # → {"experiment_id": "ui_v1", "group": "treatment"} 或 None（未命中任何实验）

    正交层示例（Feed 推荐算法实验，与 UI 层正交）:
        feed_layer = ExperimentLayer("feed_rank_layer", layer_type="orthogonal")
        feed_layer.add_experiment(LayerExperiment("rank_v2", traffic_fraction=0.5))
        # 用户可同时命中 booking_ui_layer 和 feed_rank_layer 中的实验
    """

    TOTAL_BUCKETS = 10000

    def __init__(self, layer_id: str, layer_type: str = "exclusive"):
        """
        Args:
            layer_id:   层唯一标识，如 "booking_ui_layer"
            layer_type: "exclusive"（互斥）或 "orthogonal"（正交）
        """
        if layer_type not in ("exclusive", "orthogonal"):
            raise ValueError("layer_type 必须为 'exclusive' 或 'orthogonal'")
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.experiments: List[LayerExperiment] = []

    def add_experiment(self, exp: LayerExperiment) -> "ExperimentLayer":
        """添加实验到层，返回 self 支持链式调用。"""
        if self.layer_type == "exclusive":
            used = sum(e.traffic_fraction for e in self.experiments)
            if used + exp.traffic_fraction > 1.0 + 1e-6:
                raise ValueError(
                    f"互斥层 [{self.layer_id}] 流量超限："
                    f"已用 {used:.2%} + 新增 {exp.traffic_fraction:.2%} > 100%"
                )
        self.experiments.append(exp)
        return self

    def assign(self, user_id: str) -> Optional[Dict]:
        """
        为用户分配实验组。

        互斥层：
            1. 将用户哈希到层级桶（使用 layer_id 作为命名空间）
            2. 按各实验的 traffic_fraction 依次切分桶范围
            3. 用户落在哪个实验的桶范围内，再用该实验的哈希分配 control/treatment

        正交层：
            各实验独立哈希，互不影响，用户可同时命中多个实验。

        Returns:
            互斥层：{"experiment_id": str, "group": str} 或 None（未命中）
            正交层：{"experiment_id": {"group": str}, ...}（所有实验的分配）
        """
        if self.layer_type == "exclusive":
            return self._assign_exclusive(user_id)
        else:
            return self._assign_orthogonal(user_id)

    def _assign_exclusive(self, user_id: str) -> Optional[Dict]:
        # 层级哈希：决定用户落在层内哪个桶
        layer_raw = f"{self.layer_id}:{user_id}"
        layer_bucket = int(hashlib.md5(layer_raw.encode()).hexdigest()[:8], 16) % self.TOTAL_BUCKETS

        cursor = 0
        for exp in self.experiments:
            exp_end = cursor + int(exp.traffic_fraction * self.TOTAL_BUCKETS)
            if cursor <= layer_bucket < exp_end:
                # 命中该实验，再用实验自身哈希分配 control/treatment
                exp_raw = f"{exp.experiment_id}:v1:{user_id}"
                exp_bucket = int(hashlib.md5(exp_raw.encode()).hexdigest()[:8], 16) % self.TOTAL_BUCKETS
                control_end = int(exp.control_ratio * self.TOTAL_BUCKETS)
                group = "control" if exp_bucket < control_end else "treatment"
                return {"experiment_id": exp.experiment_id, "group": group}
            cursor = exp_end

        return None  # 未命中任何实验（流量不够分）

    def _assign_orthogonal(self, user_id: str) -> Dict:
        result = {}
        for exp in self.experiments:
            exp_raw = f"{exp.experiment_id}:v1:{user_id}"
            exp_bucket = int(hashlib.md5(exp_raw.encode()).hexdigest()[:8], 16) % self.TOTAL_BUCKETS
            total_traffic = int((exp.control_ratio + exp.treatment_ratio) * exp.traffic_fraction * self.TOTAL_BUCKETS)
            if exp_bucket >= total_traffic:
                continue  # 不在该实验流量内
            control_end = int(exp.control_ratio * exp.traffic_fraction * self.TOTAL_BUCKETS)
            group = "control" if exp_bucket < control_end else "treatment"
            result[exp.experiment_id] = {"group": group}
        return result

    def get_layer_summary(self) -> Dict:
        """返回层配置摘要，用于 Dashboard 展示。"""
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "traffic_fraction": f"{e.traffic_fraction:.0%}",
                    "split": f"{e.control_ratio:.0%}/{e.treatment_ratio:.0%}",
                }
                for e in self.experiments
            ],
            "total_traffic_allocated": f"{sum(e.traffic_fraction for e in self.experiments):.0%}",
        }
