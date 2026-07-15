"""
一致性哈希分流模块
基于 MD5 哈希实现用户流量分流，保证同一用户始终进入同一实验组。
通用健康产品场景：按用户 ID (uid) 或设备 ID (did) 进行分流。
"""

import hashlib
from typing import Optional


class HashSplitter:
    """
    一致性哈希分流器。

    分流逻辑：
        1. 拼接 experiment_key + salt + user_id → 计算 MD5
        2. 取 MD5 前 8 位转为整数，对 10000 取模，得到 [0, 9999] 的哈希桶
        3. 根据实验配置的流量比例，将哈希桶映射到对照组/实验组

    为什么用 experiment_key + salt：
        - 避免不同实验使用相同哈希空间，导致实验间存在用户重叠偏差
        - salt 可以在需要"打散"历史缓存时重置分桶
    """

    TOTAL_BUCKETS = 10000  # 万分之一精度

    def __init__(self, experiment_key: str, salt: str = "v1"):
        """
        Args:
            experiment_key: 实验唯一标识，如 "health_doc_recommend_v2"
            salt: 版本盐，当需要重置历史分桶时递增
        """
        self.experiment_key = experiment_key
        self.salt = salt

    def _hash_to_bucket(self, user_id: str) -> int:
        """将 user_id 哈希到 [0, TOTAL_BUCKETS) 的桶编号。"""
        raw = f"{self.experiment_key}:{self.salt}:{user_id}"
        md5 = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return int(md5[:8], 16) % self.TOTAL_BUCKETS

    def assign_group(
        self,
        user_id: str,
        control_ratio: float = 0.5,
        treatment_ratio: float = 0.5,
        holdout_ratio: float = 0.0,
    ) -> str:
        """
        将用户分配到实验组。

        Args:
            user_id:         用户唯一标识（uid / did）
            control_ratio:   对照组流量比例（0~1）
            treatment_ratio: 实验组流量比例（0~1）
            holdout_ratio:   holdout 组比例（不参与实验，用于长期影响评估）

        Returns:
            分组名称："control" | "treatment" | "holdout" | "excluded"

        Raises:
            ValueError: 当流量比例之和超过 1 时
        """
        total = control_ratio + treatment_ratio + holdout_ratio
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"流量比例之和 {total:.4f} 超过 1.0，请检查实验配置。"
            )

        bucket = self._hash_to_bucket(user_id)
        control_end = int(control_ratio * self.TOTAL_BUCKETS)
        treatment_end = control_end + int(treatment_ratio * self.TOTAL_BUCKETS)
        holdout_end = treatment_end + int(holdout_ratio * self.TOTAL_BUCKETS)

        if bucket < control_end:
            return "control"
        elif bucket < treatment_end:
            return "treatment"
        elif bucket < holdout_end:
            return "holdout"
        else:
            return "excluded"

    def check_srm(
        self,
        control_count: int,
        treatment_count: int,
        control_ratio: float = 0.5,
        treatment_ratio: float = 0.5,
        alpha: float = 0.01,
    ) -> dict:
        """
        SRM（样本比率不匹配，Sample Ratio Mismatch）检测。

        SRM 是 AB 实验中最常见的 data quality 问题：
        - 实际流量比例与配置比例不符，说明分流/数据采集存在 bug
        - 必须在统计检验之前先过 SRM 检查，SRM 不通过则实验结论无效

        方法：使用卡方检验（Chi-Square Test）判断实际比例是否显著偏离预期。

        Returns:
            {
                "srm_detected": bool,
                "chi2_stat": float,
                "p_value": float,
                "actual_ratio": float,    # 实际 treatment 占比
                "expected_ratio": float,  # 配置 treatment 占比
                "conclusion": str
            }
        """
        from scipy import stats

        total = control_count + treatment_count
        expected_control = total * (control_ratio / (control_ratio + treatment_ratio))
        expected_treatment = total * (treatment_ratio / (control_ratio + treatment_ratio))

        observed = [control_count, treatment_count]
        expected = [expected_control, expected_treatment]

        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        srm_detected = p_value < alpha
        actual_ratio = treatment_count / total if total > 0 else 0

        return {
            "srm_detected": srm_detected,
            "chi2_stat": round(chi2, 4),
            "p_value": round(p_value, 6),
            "actual_ratio": round(actual_ratio, 4),
            "expected_ratio": round(treatment_ratio / (control_ratio + treatment_ratio), 4),
            "conclusion": (
                f"⚠️  检测到 SRM！实际 treatment 占比 {actual_ratio:.2%}，"
                f"期望 {treatment_ratio/(control_ratio+treatment_ratio):.2%}，"
                f"p={p_value:.4f} < {alpha}，实验数据不可信，请排查分流或日志采集问题。"
                if srm_detected
                else f"✅ SRM 检查通过，p={p_value:.4f} ≥ {alpha}，分流比例正常。"
            ),
        }
