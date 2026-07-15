"""
Delta Method —— 比率指标方差估计
理论推导与实现，面试时重点讲这一块！
"""

import numpy as np
import pandas as pd
from typing import Tuple


"""
╔══════════════════════════════════════════════════════════════════╗
║              Delta Method 原理详解（面试必备）                    ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  问题背景：                                                        ║
║    比率指标 R = X / Y（如 CTR = 点击数 / 曝光数）                  ║
║    X 和 Y 都是随机变量，R 本身也是随机变量。                        ║
║    我们希望估计 Var(R)，以便做统计检验。                            ║
║                                                                  ║
║  错误做法（面试常见坑）：                                           ║
║    直接用 Var(X/Y) ≈ Var(sum X) / (sum Y)² * n                 ║
║    → 忽略了 X 和 Y 在用户层面的相关性，方差估计有偏。               ║
║                                                                  ║
║  正确做法：Delta Method（泰勒展开一阶近似）                         ║
║    设 μ_x = E[X_i]，μ_y = E[Y_i]（用户层均值）                    ║
║    R = sum(X_i) / sum(Y_i) ≈ μ_x / μ_y（样本比率）              ║
║                                                                  ║
║    对 f(X, Y) = X / Y 在 (μ_x, μ_y) 处泰勒展开：                 ║
║    f(X, Y) ≈ f(μ_x, μ_y)                                       ║
║             + (1/μ_y)(X - μ_x)                                  ║
║             - (μ_x/μ_y²)(Y - μ_y)                              ║
║                                                                  ║
║    Var(R) ≈ [σ²_x/μ_y² + μ_x²·σ²_y/μ_y⁴ - 2μ_x·Cov(X,Y)/μ_y³] / n ║
║                                                                  ║
║  核心洞察：                                                        ║
║    CTR 的 variance 不只是 p(1-p)/n，而是要考虑每个用户的           ║
║    click 数和 impression 数的联合分布。                            ║
║    这在用户点击行为存在差异时（高活用户点很多次）影响显著。           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def delta_method_variance(
    numerator: pd.Series,
    denominator: pd.Series,
) -> float:
    """
    用 Delta Method 估计用户层比率指标的方差。

    Args:
        numerator:   每个用户的分子（如点击次数）
        denominator: 每个用户的分母（如曝光次数）

    Returns:
        Var(R)，用于后续 Z 检验的标准误计算
    """
    n = len(numerator)
    if n < 2:
        return 0.0

    mu_x = numerator.mean()
    mu_y = denominator.mean()
    if mu_y == 0:
        return 0.0

    sigma2_x = numerator.var(ddof=1)
    sigma2_y = denominator.var(ddof=1)
    cov_xy = numerator.cov(denominator)

    # Delta Method 公式
    var_r = (
        sigma2_x / mu_y ** 2
        + mu_x ** 2 * sigma2_y / mu_y ** 4
        - 2 * mu_x * cov_xy / mu_y ** 3
    ) / n

    return max(var_r, 0.0)


def compare_variance_methods(
    numerator: pd.Series,
    denominator: pd.Series,
) -> dict:
    """
    对比三种方差估计方法，用于教学演示。

    Returns:
        {
            "delta_method_var": float,   推荐使用
            "naive_binomial_var": float, 忽略用户异质性（有偏）
            "bootstrap_var": float,      非参数基准
        }
    """
    # 1. Delta Method（推荐）
    delta_var = delta_method_variance(numerator, denominator)

    # 2. Naive 二项分布估计（常见错误：忽略用户层相关性）
    agg_rate = numerator.sum() / denominator.sum()
    n = len(numerator)
    naive_var = agg_rate * (1 - agg_rate) / denominator.sum()

    # 3. Bootstrap 估计（非参数，计算密集但无分布假设）
    bootstrap_rates = []
    np.random.seed(42)
    for _ in range(500):
        idx = np.random.choice(n, size=n, replace=True)
        b_num = numerator.iloc[idx].sum()
        b_den = denominator.iloc[idx].sum()
        if b_den > 0:
            bootstrap_rates.append(b_num / b_den)
    bootstrap_var = np.var(bootstrap_rates)

    return {
        "sample_rate": round(numerator.sum() / denominator.sum(), 6),
        "delta_method_var": round(delta_var, 8),
        "delta_method_se": round(np.sqrt(delta_var), 6),
        "naive_binomial_var": round(naive_var, 8),
        "naive_binomial_se": round(np.sqrt(naive_var), 6),
        "bootstrap_var": round(bootstrap_var, 8),
        "bootstrap_se": round(np.sqrt(bootstrap_var), 6),
        "note": "Delta Method 和 Bootstrap 应接近。Naive 方法通常低估方差，导致假阳性。",
    }
