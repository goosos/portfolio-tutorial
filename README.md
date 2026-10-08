# Multi-Strategy Portfolios Tutorial

Part 8 of [Build Your Own Quant Research System](https://goosos.com/tutorials/).
Article: https://goosos.com/multi-strategy-portfolios

One good strategy is fragile. Combine uncorrelated strategies and the
portfolio Sharpe beats every component — no new alpha, just combination.

## What this does

`portfolio_demo.py` runs three strategies on SPY (2023-10 → 2026-10)
and compares single-strategy results vs three portfolio weighting schemes:

| Strategy | Total return | Sharpe |
|---|---|---|
| MA(20,50) | +27.21% | +0.86 |
| MA(10,30) | +33.17% | +1.10 |
| RSI(14) | +36.10% | +1.01 |
| Portfolio (Sharpe-weighted) | — | **+1.35** |

The diversifier is RSI: ~0.10 correlation with both MAs (which correlate
0.81 with each other). One uncorrelated strategy beats three correlated ones.

## Files

- `portfolio.py` — correlation matrix, equal/Sharpe/inverse-vol weighting, `combine_returns()`
- `portfolio_demo.py` — full demo (downloads SPY, runs 3 strategies, prints tables)
- `backtest.py` — vendored from Part 1 (honest backtest defaults)
- `article.md` — the tutorial text

## Run it

```bash
pip install -r requirements.txt
python portfolio_demo.py
```

## The lesson

Weighting scheme barely matters (+1.33 / +1.34 / +1.35). What matters is
*which* strategies you combine: low correlation first, weighting second.
And correlations spike toward 1.0 in crises — size for the crisis, not the calm.

## Toolkit

This is a synthesis part — no new module. `portfolio.py` composes existing
[quant-toolkit](https://github.com/goosos/quant-toolkit) modules
(`backtest`, `metrics`, `costs`).
