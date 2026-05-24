#!/usr/bin/env python3
"""
Moving-average crossover backtest on SPY (~20 years of history via yfinance).

Change STRATEGY below to swap presets. Plots close price with MAs + buy/sell
markers and the equity curve (long SPY when bullish, cash otherwise).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# --- Swap strategy name here (see STRATEGIES dict below) ---
STRATEGY: str = "sma_crossover"

TICKER: str = "SPY"
START: str = "2006-01-01"  # yfinance + SPY: plenty of history for ~20yr window
INITIAL_CAPITAL: float = 10_000.0
DCA_AMOUNT: float = 500.0


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    description: str
    compute: Callable[[pd.DataFrame], pd.DataFrame]


def _prior_bool(bull: pd.Series) -> pd.Series:
    """Previous day's bull state; first row is False (no lookahead)."""
    out = pd.Series(False, index=bull.index, dtype=bool)
    if len(bull) > 1:
        out.iloc[1:] = bull.iloc[:-1].to_numpy(dtype=bool, copy=False)
    return out


def _sma_crossover(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    out = df.copy()
    out["fast_ma"] = out["Close"].rolling(fast, min_periods=fast).mean()
    out["slow_ma"] = out["Close"].rolling(slow, min_periods=slow).mean()
    bull = (out["fast_ma"] > out["slow_ma"]).fillna(False).astype(bool)
    out["position"] = bull.astype(float)
    prev_bull = _prior_bool(bull)
    buy = bull & ~prev_bull
    sell = ~bull & prev_bull
    out["signal"] = 0
    out.loc[buy, "signal"] = 1
    out.loc[sell, "signal"] = -1
    return out


def _ema_crossover(df: pd.DataFrame, fast: int = 21, slow: int = 55) -> pd.DataFrame:
    out = df.copy()
    out["fast_ma"] = out["Close"].ewm(span=fast, adjust=False).mean()
    out["slow_ma"] = out["Close"].ewm(span=slow, adjust=False).mean()
    bull = (out["fast_ma"] > out["slow_ma"]).fillna(False).astype(bool)
    out["position"] = bull.astype(float)
    prev_bull = _prior_bool(bull)
    buy = bull & ~prev_bull
    sell = ~bull & prev_bull
    out["signal"] = 0
    out.loc[buy, "signal"] = 1
    out.loc[sell, "signal"] = -1
    return out


def _triple_ema_trend(
    df: pd.DataFrame, fast: int = 12, mid: int = 26, slow: int = 50
) -> pd.DataFrame:
    """Long when EMA12 > EMA26 > EMA50 (stacked bullish); flat otherwise."""
    out = df.copy()
    e1 = out["Close"].ewm(span=fast, adjust=False).mean()
    e2 = out["Close"].ewm(span=mid, adjust=False).mean()
    e3 = out["Close"].ewm(span=slow, adjust=False).mean()
    out["fast_ma"] = e1
    out["slow_ma"] = e3
    out["mid_ma"] = e2
    bull = ((e1 > e2) & (e2 > e3)).fillna(False).astype(bool)
    out["position"] = bull.astype(float)
    prev_bull = _prior_bool(bull)
    buy = bull & ~prev_bull
    sell = ~bull & prev_bull
    out["signal"] = 0
    out.loc[buy, "signal"] = 1
    out.loc[sell, "signal"] = -1
    return out


def _sma_crossover_aapl_style(df: pd.DataFrame) -> pd.DataFrame:
    """Classic 50/200 SMA — good default for daily data."""
    return _sma_crossover(df, fast=50, slow=200)


STRATEGIES: dict[str, StrategyConfig] = {
    "sma_crossover": StrategyConfig(
        name="sma_crossover",
        description="50/200 SMA: long when fast > slow",
        compute=_sma_crossover_aapl_style,
    ),
    "ema_crossover": StrategyConfig(
        name="ema_crossover",
        description="21/55 EMA: long when fast > slow",
        compute=_ema_crossover,
    ),
    "triple_ema_trend": StrategyConfig(
        name="triple_ema_trend",
        description="12/26/50 stacked EMAs: long when 12>26>50",
        compute=_triple_ema_trend,
    ),
}


def equity_curve(
    close: pd.Series, position: pd.Series, initial: float
) -> pd.Series:
    """Long-only: earn daily return when position==1 (use prior day position)."""
    daily_ret = close.pct_change()
    pos_lagged = position.shift(1).fillna(0.0)
    strat_ret = pos_lagged * daily_ret
    return initial * (1 + strat_ret).cumprod()


def dca_invest_dates(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    """Boolean mask of DCA contribution dates (first trading day per period)."""
    if frequency == "weekly":
        periods = pd.Series(index.to_period("W-MON"), index=index)
    elif frequency == "monthly":
        periods = pd.Series(index.to_period("M"), index=index)
    else:
        raise ValueError("frequency must be 'weekly' or 'monthly'")
    return ~periods.duplicated()


def simulate_dca_equity(
    close: pd.Series,
    invest_mask: pd.Series,
    amount: float,
    position: pd.Series | None = None,
) -> tuple[pd.Series, float]:
    """
    Contribution-based equity simulator.
    - If position is None: always invest on contribution dates (pure DCA).
    - If position provided: invest only when position==1; otherwise hold cash.
    Returns (equity_curve, total_contributed).
    """
    shares = 0.0
    cash = 0.0
    total_contributed = 0.0
    equity_values: list[float] = []

    for dt, px in close.items():
        if bool(invest_mask.loc[dt]):
            total_contributed += amount
            if position is None or bool(position.loc[dt]):
                shares += amount / float(px)
            else:
                cash += amount
        equity_values.append(shares * float(px) + cash)
    return pd.Series(equity_values, index=close.index), total_contributed


def run(
    ticker: str, start: str, strategy_key: str, dca_amount: float, dca_frequency: str
) -> tuple[pd.DataFrame, StrategyConfig, float]:
    if strategy_key not in STRATEGIES:
        known = ", ".join(sorted(STRATEGIES))
        raise SystemExit(f"Unknown STRATEGY {strategy_key!r}. Choose one of: {known}")

    cfg = STRATEGIES[strategy_key]
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])

    out = cfg.compute(df)
    out["equity"] = equity_curve(out["Close"], out["position"], INITIAL_CAPITAL)
    out["buyh_equity"] = INITIAL_CAPITAL * (1 + out["Close"].pct_change().fillna(0)).cumprod()
    out["dca_date"] = dca_invest_dates(out.index, dca_frequency)
    out["dca_equity"], total_contributed = simulate_dca_equity(
        out["Close"], out["dca_date"], dca_amount
    )
    out["ma_dca_equity"], _ = simulate_dca_equity(
        out["Close"], out["dca_date"], dca_amount, out["position"]
    )
    return out, cfg, total_contributed


def plot_results(
    df: pd.DataFrame,
    ticker: str,
    cfg: StrategyConfig,
    output_path: str | None,
    dca_amount: float,
    dca_frequency: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax0 = axes[0]
    ax0.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.0, alpha=0.85)
    if "mid_ma" in df.columns:
        ax0.plot(df.index, df["fast_ma"], label="Fast EMA", linewidth=1.0)
        ax0.plot(df.index, df["mid_ma"], label="Mid EMA", linewidth=1.0)
        ax0.plot(df.index, df["slow_ma"], label="Slow EMA", linewidth=1.0)
    else:
        ax0.plot(df.index, df["fast_ma"], label="Fast MA", linewidth=1.0)
        ax0.plot(df.index, df["slow_ma"], label="Slow MA", linewidth=1.0)

    buys = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    ax0.scatter(buys.index, buys["Close"], marker="^", color="green", s=40, zorder=5, label="Buy")
    ax0.scatter(sells.index, sells["Close"], marker="v", color="red", s=40, zorder=5, label="Sell")
    ax0.set_ylabel("Price (USD)")
    ax0.set_title(f"{ticker} — {cfg.name}: {cfg.description}")
    ax0.legend(loc="upper left", fontsize=8, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(df.index, df["equity"], label="MA lump-sum equity", color="steelblue", linewidth=1.2)
    ax1.plot(df.index, df["dca_equity"], label=f"DCA ({dca_frequency}, ${dca_amount:,.0f})", color="darkgreen", linewidth=1.2)
    ax1.plot(
        df.index,
        df["ma_dca_equity"],
        label=f"MA-timed DCA ({dca_frequency})",
        color="darkorange",
        linewidth=1.2,
    )
    ax1.plot(df.index, df["buyh_equity"], label="Buy & hold lump-sum", color="gray", linewidth=0.9, alpha=0.75)
    ax1.set_ylabel("Equity ($)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return
    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        plt.close(fig)
        print(
            "Non-interactive backend (Agg): no window shown. "
            "Re-run with --output chart.png to save the figure.",
            file=sys.stderr,
        )
        return
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="MA crossover backtest (yfinance + pandas)")
    parser.add_argument("--ticker", default=TICKER, help="Symbol (default SPY)")
    parser.add_argument("--start", default=START, help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--strategy",
        default=STRATEGY,
        choices=sorted(STRATEGIES.keys()),
        help="Strategy preset",
    )
    parser.add_argument(
        "--output",
        metavar="PNG",
        default=None,
        help="Save figure to this path (recommended for non-interactive runs)",
    )
    parser.add_argument(
        "--dca-amount",
        type=float,
        default=DCA_AMOUNT,
        help="Fixed dollar contribution amount for DCA comparisons",
    )
    parser.add_argument(
        "--dca-frequency",
        choices=["weekly", "monthly"],
        default="monthly",
        help="Contribution schedule for DCA comparisons",
    )
    args = parser.parse_args()

    df, cfg, total_contributed = run(
        args.ticker, args.start, args.strategy, args.dca_amount, args.dca_frequency
    )
    # Console summary
    final_eq = df["equity"].iloc[-1]
    final_bh = df["buyh_equity"].iloc[-1]
    n_buys = (df["signal"] == 1).sum()
    n_sells = (df["signal"] == -1).sum()
    print(f"Ticker: {args.ticker} | Strategy: {cfg.name}")
    print(f"Rows: {len(df)} | Buys: {int(n_buys)} | Sells: {int(n_sells)}")
    print(f"Final MA lump-sum equity: ${final_eq:,.2f} | Buy & hold lump-sum: ${final_bh:,.2f}")
    print(
        f"DCA contributed: ${total_contributed:,.2f} | "
        f"Final DCA equity: ${df['dca_equity'].iloc[-1]:,.2f} | "
        f"Final MA-timed DCA equity: ${df['ma_dca_equity'].iloc[-1]:,.2f}"
    )
    plot_results(
        df, args.ticker, cfg, args.output, args.dca_amount, args.dca_frequency
    )


if __name__ == "__main__":
    main()
