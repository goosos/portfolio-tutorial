"""
portfolio.py — Part 8 of "Build Your Own Quant Research System".

Multi-strategy portfolios: combining strategies so the whole is
better than the sum of its parts. No new edge is created here —
diversification just lets you keep more of the edge you have.

Four building blocks:
  strategy_corr_matrix() — how similarly strategies behave
  equal_weight()         — 1/N, the robust default
  sharpe_weight()        — weight by (positive) Sharpe
  inverse_vol_weight()   — weight by 1/volatility
  combine_returns()      — weighted-sum strategy returns into one

Tutorial: https://goosos.com/multi-strategy-portfolios
Code: https://github.com/goosos/portfolio-tutorial
"""

import numpy as np
import pandas as pd


def strategy_corr_matrix(returns_dict):
    """
    Pairwise correlation matrix of strategy return series.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Strategy name -> daily returns (aligned on the same index;
        misaligned indexes are inner-joined).

    Returns
    -------
    pd.DataFrame — correlation matrix, 1.0 on the diagonal.

    Why it matters: correlation is the only input that makes a
    portfolio better than its best single strategy. Two strategies
    with correlation 0.9 add almost nothing to each other; two
    with correlation 0.2 can lift portfolio Sharpe well above
    either one alone.
    """
    frame = pd.DataFrame(returns_dict).dropna(how="all")
    return frame.corr()


def equal_weight(n):
    """
    1/N weights. Boring, robust, and embarrassingly hard to beat.

    Research keeps finding that equal weighting beats optimized
    weights out-of-sample, because optimized weights overfit the
    estimation window (see Part 4: overfitting).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.full(n, 1.0 / n)


def sharpe_weight(sharpes):
    """
    Weights proportional to (positive) Sharpe ratios.

    Negative Sharpes get zero weight — why allocate to something
    expected to lose? Normalized to sum to 1.
    """
    sharpes = np.asarray(sharpes, dtype=float)
    if len(sharpes) == 0:
        raise ValueError("sharpes must not be empty")
    clipped = np.clip(sharpes, 0, None)
    if clipped.sum() <= 0:
        # No positive Sharpe: fall back to equal weight rather than
        # concentrating on the least-bad loser.
        return equal_weight(len(sharpes))
    return clipped / clipped.sum()


def inverse_vol_weight(vols):
    """
    Weights proportional to 1/volatility ("risk parity lite").

    Equalizes each strategy's risk contribution. Calmer strategies
    get more capital; spikier ones get less. Normalized to sum to 1.
    """
    vols = np.asarray(vols, dtype=float)
    if len(vols) == 0:
        raise ValueError("vols must not be empty")
    if np.any(vols <= 0):
        raise ValueError(f"vols must all be > 0, got {vols}")
    inv = 1.0 / vols
    return inv / inv.sum()


def combine_returns(returns_dict, weights):
    """
    Weighted-sum strategy returns into a single portfolio series.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Strategy name -> daily returns. Inner-joined on the index.
    weights : array-like
        Portfolio weights, same order as returns_dict keys.
        Should sum to 1 (not enforced — your responsibility).

    Returns
    -------
    pd.Series — daily portfolio returns.

    Note: this assumes strategies can be scaled frictionlessly and
    rebalanced daily to target weights. Real portfolios drift and
    pay to rebalance (see Part 6: costs). Treat as a first-order
    approximation.
    """
    frame = pd.DataFrame(returns_dict).dropna(how="all").fillna(0.0)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != frame.shape[1]:
        raise ValueError(
            f"{len(weights)} weights for {frame.shape[1]} strategies"
        )
    return (frame * weights).sum(axis=1)
