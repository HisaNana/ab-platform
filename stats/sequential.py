"""
序贯检验模块（Sequential Testing / Always-Valid Inference）

解决的问题 —— peeking problem：
    固定样本量的 t/Z 检验只在预先设定的样本量下检验一次才有效。
    如果实验运行过程中反复查看数据、一旦显著就停止（peeking），
    多次检验相当于做了多次假设检验，实际假阳性率会远超设定的 alpha。

本模块提供两种互补的解决方案：

    1. Alpha Spending Function（O'Brien-Fleming / Pocock）
       预先规划总的查看次数，把总的 alpha 预算按"信息量"分配到每一次检验，
       早期查看的边界更严格（更难显著），到最后一次查看时收敛到标准 z 临界值。
       —— 需要提前确定总查看次数，属于"有计划的多看几眼"。

    2. mSPRT（mixture Sequential Probability Ratio Test）
       用正态-正态共轭先验构造混合似然比，得到 anytime-valid 的检验统计量：
       无论在哪个样本量、以任何频率查看，只要似然比未越过 1/alpha 阈值，
       就不会引入额外的假阳性膨胀。
       —— 不需要预先规划查看次数，属于"随时可以看，看多少次都不影响"。

面试高频对比：
    Alpha Spending 更适合"周期性汇报"场景（如每天看一次，跑够 14 天下结论）；
    mSPRT 更适合"业务方随时可能要求出结果"的场景，但需要对效应量量级
    有一个先验估计（tau_squared），估计不准会影响功效（不影响有效性）。
"""

import math
from typing import Optional

from scipy.stats import norm


# ──────────────────────────────────────────
# Alpha Spending Function
# ──────────────────────────────────────────

def obrien_fleming_boundary(information_fraction: float, alpha: float = 0.05) -> float:
    """
    O'Brien-Fleming 边界。

    公式：boundary(t) = z_{alpha/2} / sqrt(t)，t 为信息量占比（已积累样本 / 计划总样本），
    t ∈ (0, 1]。

    特点：
        - t 越小（早期），边界越大，越难被判定为显著 —— 几乎不消耗 alpha 预算；
        - t = 1（计划样本量点），边界收敛到标准 z_{alpha/2}，与传统固定样本量检验一致；
        - 整体检验效力（power）损失很小，是最常用的 alpha spending 方案。

    Args:
        information_fraction: 当前信息量占比，(0, 1] 区间。
        alpha: 总显著性水平。

    Returns:
        当前信息量占比下的 z 临界值边界。
    """
    if not (0 < information_fraction <= 1):
        raise ValueError(f"information_fraction 必须在 (0, 1] 区间内，收到: {information_fraction}")
    z_alpha_half = norm.ppf(1 - alpha / 2)
    return z_alpha_half / math.sqrt(information_fraction)


# Pocock 边界近似表：以 total_analyses（计划总查看次数 K）为 key，
# 存储的是使总体双侧 alpha=0.05 时每次查看所需的固定 z 临界值（近似值，非解析解）。
# 数值来源：Pocock (1977) 及后续文献中常见的近似边界表。
_POCOCK_APPROX_TABLE_ALPHA_05 = {
    1: 1.960,
    2: 2.178,
    3: 2.289,
    4: 2.361,
    5: 2.413,
    6: 2.453,
    7: 2.485,
    8: 2.512,
    9: 2.535,
    10: 2.555,
}


def pocock_boundary(
    information_fraction: float,
    alpha: float = 0.05,
    total_analyses: Optional[int] = None,
) -> float:
    """
    Pocock 边界（近似值）。

    与 O'Brien-Fleming 不同，Pocock 边界在整个检验过程中是**恒定的**——
    每次查看都使用相同的 z 临界值，不随信息量变化。

    注意：Pocock 边界没有像 O'Brien-Fleming 那样简洁的解析解，
    标准做法是通过数值积分或查近似表得到。本实现使用近似表
    （`_POCOCK_APPROX_TABLE_ALPHA_05`，仅覆盖 alpha=0.05、total_analyses<=10 的常见场景），
    超出覆盖范围时退化为通过 Bonferroni 思路给出保守估计，
    **不是精确解**，工程/教学场景够用，严肃统计分析建议使用专门的序贯设计软件核实。

    特点：
        - 边界恒定，早期比 O'Brien-Fleming 更容易显著；
        - 早期"偷看"消耗的 alpha 更多，整体功效通常略低于 O'Brien-Fleming；
        - 优点是解释简单："每次都用同一个临界值"。

    Args:
        information_fraction: 当前信息量占比（Pocock 边界本身不用它计算数值，
            仅用于保持与 obrien_fleming_boundary 一致的函数签名，便于上层统一调用）。
        alpha: 总显著性水平，目前近似表仅覆盖 0.05。
        total_analyses: 计划总查看次数，默认 5。

    Returns:
        当前查看点的 z 临界值边界（近似值，全程恒定）。
    """
    if not (0 < information_fraction <= 1):
        raise ValueError(f"information_fraction 必须在 (0, 1] 区间内，收到: {information_fraction}")
    k = total_analyses or 5

    if abs(alpha - 0.05) > 1e-9:
        # 表外 alpha：退化为 Bonferroni 近似（保守，非精确 Pocock 解）
        return norm.ppf(1 - alpha / (2 * k))

    if k in _POCOCK_APPROX_TABLE_ALPHA_05:
        return _POCOCK_APPROX_TABLE_ALPHA_05[k]

    # 超出表覆盖范围（k > 10）：用 Bonferroni 近似兜底，保守估计
    return norm.ppf(1 - alpha / (2 * k))


def alpha_spending_test(
    z_statistic: float,
    information_fraction: float,
    alpha: float = 0.05,
    spending_function: str = "obrien_fleming",
    total_analyses: Optional[int] = None,
) -> dict:
    """
    基于 Alpha Spending Function 的序贯检验。

    Args:
        z_statistic: 当前查看时刻计算出的 z 统计量。
        information_fraction: 当前信息量占比（已积累样本 / 计划总样本），(0, 1]。
        alpha: 总显著性水平。
        spending_function: "obrien_fleming" | "pocock"。
        total_analyses: 计划总查看次数（仅 pocock 需要）。

    Returns:
        {
            "boundary_z": 当前查看点的边界值,
            "z_statistic": 输入的 z 统计量,
            "is_significant": 是否越过边界（可判定显著）,
            "information_fraction": 信息量占比,
            "spending_function": 使用的方法名,
            "can_stop_early": 是否可以提前停止（等价于 is_significant）,
            "conclusion": 文字结论,
        }
    """
    if spending_function == "obrien_fleming":
        boundary = obrien_fleming_boundary(information_fraction, alpha)
    elif spending_function == "pocock":
        boundary = pocock_boundary(information_fraction, alpha, total_analyses)
    else:
        raise ValueError(f"不支持的 spending_function: {spending_function}，请使用 'obrien_fleming' 或 'pocock'")

    is_significant = abs(z_statistic) >= boundary

    if is_significant:
        conclusion = (
            f"|z|={abs(z_statistic):.4f} 已越过 {spending_function} 边界 {boundary:.4f}"
            f"（信息量占比 {information_fraction:.1%}），可提前停止实验，结论显著。"
        )
    else:
        conclusion = (
            f"|z|={abs(z_statistic):.4f} 未越过 {spending_function} 边界 {boundary:.4f}"
            f"（信息量占比 {information_fraction:.1%}），暂不显著，建议继续积累样本。"
        )

    return {
        "boundary_z": round(boundary, 4),
        "z_statistic": round(z_statistic, 4),
        "is_significant": is_significant,
        "information_fraction": information_fraction,
        "spending_function": spending_function,
        "can_stop_early": is_significant,
        "conclusion": conclusion,
    }


# ──────────────────────────────────────────
# mSPRT（mixture Sequential Probability Ratio Test）
# ──────────────────────────────────────────

def mixture_sequential_probability_ratio(
    n: int,
    sample_mean_diff: float,
    sample_variance: float,
    alpha: float = 0.05,
    tau_squared: Optional[float] = None,
) -> dict:
    """
    mSPRT —— 正态-正态共轭混合似然比检验（Johari et al., 2017 风格实现）。

    原理：
        假设组间均值差 Delta 服从正态先验 N(0, tau^2)，在 H0: Delta=0 下，
        用先验对似然做混合积分，得到关于 H0 的混合似然比：

            Lambda_n = sqrt(sigma^2 / (sigma^2 + n * tau^2))
                       * exp( n^2 * tau^2 * sample_mean_diff^2
                              / (2 * sigma^2 * (sigma^2 + n * tau^2)) )

        这里的 sample_variance 表示单个观测或等效观测的方差 sigma^2；
        sample_mean_diff 是当前样本均值差。Lambda_n 是一个关于时间 anytime-valid 的似然比过程：只要决策规则是
        "Lambda_n >= 1/alpha 时拒绝 H0"，无论查看多少次、在哪个 n 查看，
        假阳性率都不会超过 alpha（这是 mSPRT 相比固定样本 Z 检验的核心优势）。

    Args:
        n: 当前查看时刻的（单组或等效）样本量。
        sample_mean_diff: 实验组 - 对照组的样本均值差。
        sample_variance: 样本方差（通常为两组合并/Delta Method 估计的方差）。
        alpha: 显著性水平，决策阈值为 1/alpha。
        tau_squared: 效应量先验方差。未提供时按启发式默认值 `1e-3`。
            这个量级适合本项目中的比例/均值差异 demo（如 1%-5% 绝对差异）；
            严肃业务分析中应按 MDE 量级设定，例如 `tau_squared = mde_absolute ** 2`。

    Returns:
        {
            "likelihood_ratio": Lambda_n,
            "threshold": 1/alpha,
            "is_significant": Lambda_n >= 1/alpha,
            "can_stop_early": 同 is_significant，anytime-valid 特性下可随时停止,
            "alpha": alpha,
            "tau_squared": 实际使用的先验方差,
            "conclusion": 文字结论,
        }
    """
    if n <= 0:
        raise ValueError(f"n 必须为正整数，收到: {n}")
    if sample_variance < 0:
        raise ValueError(f"sample_variance 不能为负数，收到: {sample_variance}")

    if tau_squared is None:
        tau_squared = 1e-3
    if tau_squared <= 0:
        raise ValueError(f"tau_squared 必须为正数，收到: {tau_squared}")

    sigma_squared = sample_variance
    if sigma_squared <= 0:
        # 方差退化为 0（如样本量极小或数据无波动），无法构造有效似然比
        likelihood_ratio = 1.0
    else:
        denom = sigma_squared + n * tau_squared
        shrink_factor = math.sqrt(sigma_squared / denom)
        exponent = (n ** 2) * tau_squared * (sample_mean_diff ** 2) / (2 * sigma_squared * denom)
        # 指数项可能很大导致 exp 溢出，用 clip 保护，溢出即代表远超阈值，判定显著
        exponent = min(exponent, 700.0)  # exp(700) 已接近 float 上限，足够表示"远超阈值"
        likelihood_ratio = shrink_factor * math.exp(exponent)

    threshold = 1.0 / alpha
    is_significant = likelihood_ratio >= threshold

    if is_significant:
        conclusion = (
            f"似然比 Λ_n={likelihood_ratio:.4f} 已越过阈值 1/α={threshold:.2f}，"
            f"可在当前样本量（n={n}）下随时停止，结论显著（anytime-valid）。"
        )
    else:
        conclusion = (
            f"似然比 Λ_n={likelihood_ratio:.4f} 未越过阈值 1/α={threshold:.2f}，"
            f"暂不显著，可继续积累样本后再检验，不会因多次查看膨胀假阳性率。"
        )

    return {
        "likelihood_ratio": round(likelihood_ratio, 6),
        "threshold": round(threshold, 4),
        "is_significant": is_significant,
        "can_stop_early": is_significant,
        "alpha": alpha,
        "tau_squared": round(tau_squared, 6),
        "conclusion": conclusion,
    }
