"""
CUPED（Controlled-experiment Using Pre-Experiment Data）
利用实验前协变量降低实验指标方差，在相同样本量下提升统计功效（power）。

原理：
    设 Y 为实验期指标，X 为实验前同指标（协变量）。
    CUPED 调整后的指标：
        Y_cuped = Y - theta * X
        theta = Cov(Y, X) / Var(X)

    由于 E[X] 在对照/实验组理论上相同（实验前无干预），
    调整后 E[Y_cuped] = E[Y] - theta * E[X]，两组的期望差值不变，
    但方差降低：
        Var(Y_cuped) = Var(Y) * (1 - rho^2)
        其中 rho = Corr(Y, X)

    rho 越高，方差缩减越多。实践中 rho 通常在 0.5~0.8 之间，
    可将实验所需样本量减少 25%~64%。

参考：Deng et al., 2013, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data"
"""

import numpy as np
import pandas as pd
from typing import Tuple


def cuped_adjust(pre: pd.Series, post: pd.Series) -> pd.Series:
    """
    对实验指标做 CUPED 协变量调整。

    Args:
        pre:  实验前协变量（每用户），如上周的预约完成次数
        post: 实验期指标（每用户），如本周的预约完成次数

    Returns:
        Y_cuped：调整后的指标序列，与 post 等长，均值与 post 相同，
                 但方差更小（消除了协变量引入的个体差异）

    面试说法：
        "我们用 CUPED 降低了方差，等效于把实验敏感度提升了，
         同样的样本量可以检测到更小的效果差异。"
    """
    if len(pre) != len(post):
        raise ValueError("pre 和 post 序列长度必须相同（每用户一条）")

    pre = pd.Series(pre).reset_index(drop=True)
    post = pd.Series(post).reset_index(drop=True)

    var_pre = pre.var(ddof=1)
    if var_pre == 0:
        # 协变量无方差，CUPED 无效，直接返回原始值
        return post.copy()

    theta = pre.cov(post) / var_pre
    # ponytail: theta 全局估计，实践中应在对照组单独估计以避免干预污染
    post_cuped = post - theta * (pre - pre.mean())
    return post_cuped


def variance_reduction_ratio(pre: pd.Series, post: pd.Series) -> dict:
    """
    计算 CUPED 方差缩减比例，用于面试演示和 Dashboard 展示。

    Returns:
        {
            "theta": float,               # 协变量系数
            "correlation": float,         # pre/post 相关系数 rho
            "var_before": float,          # 原始方差
            "var_after": float,           # CUPED 后方差
            "reduction_pct": float,       # 方差缩减百分比（理论值 = rho^2）
            "equivalent_sample_increase": float  # 等效样本量提升倍数 1/(1-rho^2)
        }
    """
    pre = pd.Series(pre).reset_index(drop=True)
    post = pd.Series(post).reset_index(drop=True)

    post_cuped = cuped_adjust(pre, post)
    rho = pre.corr(post)

    var_before = post.var(ddof=1)
    var_after = post_cuped.var(ddof=1)
    reduction_pct = 1 - var_after / var_before if var_before > 0 else 0.0

    return {
        "theta": round(pre.cov(post) / pre.var(ddof=1) if pre.var(ddof=1) > 0 else 0, 4),
        "correlation": round(float(rho), 4),
        "var_before": round(var_before, 6),
        "var_after": round(var_after, 6),
        "reduction_pct": round(reduction_pct, 4),
        "equivalent_sample_increase": round(1 / (1 - rho ** 2), 2) if abs(rho) < 1 else float("inf"),
    }
