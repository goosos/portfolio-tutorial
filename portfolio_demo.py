"""
portfolio_demo.py — Part 8: combine three strategies on SPY and
compare single-strategy results vs three portfolio weighting schemes.

Strategies (all on SPY daily, honest costs from Part 1's backtest):
  MA(20,50)  — the series' workhorse
  MA(10,30)  — faster, choppier sibling
  RSI(14)    — mean-reversion: buy oversold, sell overbought

Run: python portfolio_demo.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

import yfinance as yf
import vectorbt as vbt
from backtest import run_backtest, ma_crossover_signals
from portfolio import (
    strategy_corr_matrix,
    equal_weight,
    sharpe_weight,
    inverse_vol_weight,
    combine_returns,
)


def get_spy():
    df = yf.download("SPY", start="2023-10-09", end="2026-10-07",
                     auto_adjust=True, progress=False)
    price = df["Close"].iloc[:, 0] if hasattr(df["Close"], "iloc") else df["Close"]
    return pd.Series(np.asarray(price).ravel(), index=df.index, name="SPY")


def rsi_signals(price, window=14, oversold=30, overbought=70):
    """Mean-reversion: buy when RSI crosses up through oversold,
    exit when it crosses down through overbought. Shifted 1 bar."""
    rsi = vbt.RSI.run(price, window=window).rsi
    entries = rsi.vbt.crossed_above(oversold).vbt.signals.fshift(1)
    exits = rsi.vbt.crossed_below(overbought).vbt.signals.fshift(1)
    return entries, exits


def strategy_returns(price, entries, exits):
    """Daily returns of a strategy from its vectorbt portfolio."""
    pf = run_backtest(price, entries, exits)
    rets = pf.returns()
    return pd.Series(np.asarray(rets).ravel(), index=price.index)


def sharpe(rets):
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def main():
    price = get_spy()
    print(f"SPY daily: {len(price)} bars, "
          f"{price.index[0].date()} -> {price.index[-1].date()}")

    strategies = {}
    e, x = ma_crossover_signals(price, 20, 50)
    strategies["MA(20,50)"] = strategy_returns(price, e, x)
    e, x = ma_crossover_signals(price, 10, 30)
    strategies["MA(10,30)"] = strategy_returns(price, e, x)
    e, x = rsi_signals(price)
    strategies["RSI(14)"] = strategy_returns(price, e, x)

    print("\n--- Single strategies ---")
    sharpes, vols = {}, {}
    for name, rets in strategies.items():
        s = sharpe(rets)
        v = float(np.asarray(rets).std() * np.sqrt(252))
        sharpes[name] = s
        vols[name] = v
        tot = float((1 + rets.fillna(0)).prod() - 1)
        print(f"{name:12s} total {tot:+7.2%}  Sharpe {s:+.2f}  vol {v:.2%}")

    print("\n--- Correlation matrix ---")
    corr = strategy_corr_matrix(strategies)
    print(corr.round(2).to_string())

    names = list(strategies.keys())
    schemes = {
        "equal": equal_weight(len(names)),
        "sharpe-wtd": sharpe_weight([sharpes[n] for n in names]),
        "inv-vol": inverse_vol_weight([vols[n] for n in names]),
    }
    print("\n--- Portfolios ---")
    print(f"{'scheme':12s} {'weights':38s} {'Sharpe':>7s}")
    results = {}
    for sname, w in schemes.items():
        port = combine_returns(strategies, w)
        s = sharpe(port)
        results[sname] = (port, s)
        wstr = ", ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w))
        print(f"{sname:12s} {wstr:38s} {s:+.2f}")

    best_single = max(sharpes.values())
    best_port = max(s for _, s in results.values())
    print(f"\nBest single Sharpe: {best_single:+.2f}  ->  "
          f"best portfolio: {best_port:+.2f}  "
          f"(lift {best_port - best_single:+.2f})")

    # save series for the chart step
    out = pd.DataFrame({n: (1 + strategies[n].fillna(0)).cumprod()
                        for n in names})
    for sname, (port, _) in results.items():
        out[f"port:{sname}"] = (1 + port.fillna(0)).cumprod()
    out.to_csv("equity_curves.csv")
    corr.to_csv("corr.csv")
    print("\nSaved equity_curves.csv, corr.csv")


if __name__ == "__main__":
    main()
