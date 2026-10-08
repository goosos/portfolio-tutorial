"""make_charts.py — correlation heatmap + portfolio equity curves."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

eq = pd.read_csv("equity_curves.csv", index_col=0, parse_dates=True)
corr = pd.read_csv("corr.csv", index_col=0)

# 1. correlation heatmap
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
labels = list(corr.columns)
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=11, color="white" if v < 0.5 else "black",
                fontweight="bold")
ax.set_title("Strategy return correlations (daily, SPY 2023-10 → 2026-10)",
             fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.8, label="correlation")
fig.tight_layout()
fig.savefig("corr_heatmap.png", dpi=150)
plt.close(fig)

# 2. equity curves: three singles + best portfolio
fig, ax = plt.subplots(figsize=(9, 4.5))
for col in ["MA(20,50)", "MA(10,30)", "RSI(14)"]:
    ax.plot(eq.index, eq[col], lw=1.2, alpha=0.7, label=col)
ax.plot(eq.index, eq["port:sharpe-wtd"], lw=2.2, color="black",
        label="Portfolio (Sharpe-weighted)")
ax.set_title("Single strategies vs portfolio — same capital, smoother ride",
             fontsize=12)
ax.set_ylabel("Growth of $1")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("portfolio_equity.png", dpi=150)
plt.close(fig)
print("charts saved")
