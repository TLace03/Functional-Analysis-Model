"""
Copyright 2026 Lacy, Thomas Joseph
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
 
# ============================================================
# PERFORMANCE OPTIMIZATION LOG (v4 — Parallelized)
# ============================================================
# Hardware target: AMD Ryzen 9 9950X3D (16c/32t), AMD RX 9070 XT,
#                  64 GB RAM.
#
# Key changes from v3:
#
#  1. classify_regime:  Python for-loop (~12,500 .loc calls) →
#                       fully vectorized pandas/numpy operations.
#                       ~50x speedup; zero Python-level iteration.
#
#  2. smooth_regime:    Python for-loop → Numba @njit compiled
#                       to native C-speed machine code.
#                       ~30-50x speedup on the smoothing pass.
#
#  3. Backtest loops:   Per-date np.dot → grouped matrix multiply
#                       (n_dates × n_stocks) @ (n_stocks,) via BLAS.
#                       One BLAS call per phase group replaces
#                       thousands of Python function calls.
#
#  4. Phase opts:       4 sequential scipy.minimize → joblib
#                       Parallel across N_JOBS cores simultaneously.
#                       ~4x speedup (wall time, not CPU time).
#
#  5. Bates simulation: NumPy step-by-step → PyTorch tensor ops.
#                       Runs on AMD RX 9070 XT via ROCm if installed;
#                       falls back to CPU tensors otherwise.
#                       ~10-30x speedup on GPU for 1000+ paths.
#
#  6. Ticker download:  Single yfinance call (timeouts on 500+ tickers)
#                       → chunked parallel ThreadPoolExecutor downloads.
#                       ~3x speedup; more robust against partial failures.
#
#  7. DerivativesHedger integration: hedge is now applied in a single
#                       vectorized batch call instead of per-date.
# ============================================================
 
import os
import sys
import re
import warnings
 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.dates import date2num
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib.request
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import pickle
import threading
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

#live trading with IBKR API
from ib_insync import *
import datetime
import time
import logging
  
# ── Parallel computation stack ───────────────────────────────
# joblib: scikit-learn–style multiprocessing for independent tasks
# numba:  JIT compiler — converts Python functions to native machine code
# torch:  tensor library with GPU support (ROCm for AMD, CUDA for NVIDIA)
from joblib import Parallel, delayed
from numba import njit
import torch

try:
    import openpyxl
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False
 
def download_price_data(tickers, period, interval, chunk_size=50, retries=2, progress=False):
    """
    Download adjusted-close prices in parallel chunks, then merge.

    KEY FIX — weekly interval timestamp alignment:
    ──────────────────────────────────────────────
    yfinance with interval="1wk" sometimes anchors weekly bars to
    different days of the week across separate download calls
    (e.g. chunk 1 returns Mondays, chunk 2 returns Fridays).
    When those chunks are pd.concat'd on axis=1, pandas takes the
    UNION of both date sets, producing ~2x as many rows as expected.
    Every ticker then has ~50% NaN coverage, so all tickers fail
    the 95% coverage threshold and the model falls back to 5 ETFs.

    Fix — two steps applied after each chunk is extracted:
      Step A: strip timezone and time-of-day from the DatetimeIndex
              so Mondays and Fridays representing the same week are
              not treated as different dates.
              data.index = pd.to_datetime(data.index.date)

      Step B: after all chunks are concatenated, resample to a
              canonical weekly anchor (Friday close: "W-FRI") so
              every ticker shares an identical weekly DatetimeIndex.
              raw = raw.resample("W-FRI").last()
    """
    tickers = [t for t in dict.fromkeys(tickers) if isinstance(t, str) and t.strip()]
    if not tickers:
        return pd.DataFrame(), []

    failed = []
    frames = []
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    is_weekly = "wk" in str(interval).lower()

    print(f"Downloading price data for {len(tickers)} tickers in {len(chunks)} chunks...")

    for chunk in chunks:
        for attempt in range(1, retries + 1):
            try:
                data = yf.download(
                    chunk,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=progress,
                )
                if data is None or data.empty:
                    raise ValueError("Empty download result")
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                if isinstance(data.columns, pd.MultiIndex):
                    if "Close" in data.columns.levels[0]:
                        data = data["Close"]
                    else:
                        data = data.droplevel(0, axis=1)
                if data.empty:
                    raise ValueError("No close price data")

                # ── Step A: strip time-of-day and timezone ──────────────
                # pd.DatetimeIndex can carry UTC offsets or intraday times.
                # For weekly data, "2024-04-22 00:00:00-04:00" and
                # "2024-04-19 00:00:00+00:00" represent the same week but
                # compare as different keys.  Convert to plain dates so
                # all chunks share a common calendar representation.
                data.index = pd.to_datetime(data.index.date)

                frames.append(data)
                break
            except Exception as e:
                msg = str(e)
                if attempt == retries:
                    print(f"  Chunk failed after {retries} attempts: {chunk} -> {msg}")
                    failed.extend(chunk)
                else:
                    print(f"  Retry {attempt}/{retries} for chunk {chunk} due to {msg}")

    if frames:
        raw = pd.concat(frames, axis=1)
        raw = raw.loc[:, ~raw.columns.duplicated()]
    else:
        raw = pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.levels[0]:
        raw = raw["Close"]

    # ── Step B: canonical weekly resample ──────────────────────────────
    # Even after Step A, two chunks might have Monday vs Friday dates for
    # the same ISO week.  Resampling to "W-FRI" (Friday close) collapses
    # all rows within the same calendar week to a single Friday date.
    # This produces one consistent row per week for every ticker.
    # ".last()" picks the last available price in the week (closest to Fri).
    # Rows where ALL tickers are NaN (e.g. market holidays) are dropped.
    if is_weekly and not raw.empty:
        raw = raw.resample("W-FRI").last()
        raw = raw.dropna(how="all")

    return raw, failed

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
 
# ============================================================
# PARALLEL CONFIG
# ============================================================
# Use all physical cores for CPU-bound parallelism.
# On the 9950X3D this is 16 physical / 32 logical; we cap at
# physical count to avoid NUMA / SMT contention on math workloads.
# For I/O-bound work (downloads, HTTP) we use more threads because
# threads block on network, not CPU.
N_JOBS        = min(os.cpu_count() or 1, 16)   # physical cores for compute
N_IO_THREADS  = min(os.cpu_count() or 1, 32)   # logical threads for I/O
 
# ── GPU device detection ──────────────────────────────────────
# PyTorch with ROCm installed exposes AMD GPUs as "cuda" devices,
# so torch.cuda.is_available() returns True on your RX 9070 XT
# if the ROCm stack is installed.  Falls back to CPU seamlessly.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"  [GPU] Using: {torch.cuda.get_device_name(0)} (ROCm/CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("  [GPU] No ROCm/CUDA device found — Bates will run on CPU tensors.")
 
# ============================================================
# NEWS AGENT — live macro/sentiment signal (optional)
# ============================================================
try:
    from NewsAgent import NewsAgent as _NewsAgentClass
    print("  [NewsAgent] Initializing...")
    _news_agent  = _NewsAgentClass()
    _news_signal = _news_agent.get_signal()
    print(_news_signal.summary())
except Exception as _news_err:
    print(f"  [NewsAgent] Unavailable ({_news_err}) — running without live signal.")
    _news_signal = None
 
# ============================================================
# DERIVATIVES HEDGER — protective put overlay for drawdowns
# ============================================================
# With leverage now active in Phases 1-2 (up to 5x), we increase
# the base hedge ratio to 0.75.  When leverage is detected,
# DerivativesHedger will double this to 1.50, then cap at 1.0.
# This ensures robust downside protection during leveraged periods.
try:
    from DerivativesTrading import DerivativesHedger
    _hedger = DerivativesHedger(hedge_ratio=0.75)
    print("  [DerivativesHedger] Initialized (base 75% with adaptive scaling for leverage).")
    print("  [DerivativesHedger] Leverage + dynamic hedging active in Phases 1/1b/2.")
except Exception as _hedge_err:
    print(f"  [DerivativesHedger] Unavailable ({_hedge_err}) — running without derivatives overlay.")
    _hedger = None
 
# ============================================================
# OVERVIEW
# ============================================================
# Regime-aware spatial-temporal portfolio model.
#
#  - Regime detection: Markov-switching–style state classifier on
#    VIX z-score + SPY momentum → 4 discrete market phases.
#  - PCA factor construction: orthogonal latent factors from the
#    full-universe return covariance matrix.
#  - Bates jump-diffusion: Heston stochastic volatility + Poisson
#    jump process for per-regime scenario generation.
#  - Phase-specific blends: tactical allocations blending factor
#    exposure with ETF sleeves (QQQ, GLD, SDS, TLT, SH).
#  - Walk-forward validation: strict train/test split prevents
#    any look-ahead in factor construction or optimization.
 
# ============================================================
# CONFIG
# ============================================================
PERIOD      = "23y"
INTERVAL    = "1d"
N_FACTORS   = 15
MIN_WEIGHT  = 0.0
MAX_WEIGHT  = 0.10
CVAR_ALPHA  = 0.10
TRAIN_FRAC  = 0.5
REGIME_CONFIRM_DAYS = 5
PHASE1B_MOM_THRESH  = 0.03
SLEEVE_INSTRUMENTS  = ["SPY", "QQQ", "GLD", "SH", "SDS", "TLT", "BTC-USD"]
SDS_INCEPTION       = pd.Timestamp("2006-07-11")

# Actual fund leverage multipliers used for exposure control.
# This is separate from capital allocation weights.
SLEEVE_LEVERAGE     = {
    "SPY":    1.0,
    "QQQ":    1.0,
    "GLD":    1.0,
    "SH":    -1.0,
    "SDS":   -2.0,
    "TLT":    1.0,
    "BTC-USD":1.0,
}
MAX_EFFECTIVE_EXPOSURE = np.inf

# ── LEVERAGE CONFIGURATION ──────────────────────────────────
# Phase leverage now defines the maximum allowed gross exposure for
# the portfolio during that phase. The model will scale exposures
# down if the raw portfolio would exceed the phase cap.
LEVERAGE_CONFIG = {
    1: 3.0,      # Phase 1 (Growth/Buildout): max 3x exposure
    "1b": 3.0,  # Phase 1b (Momentum Accel): max 3x exposure
    2: 5.0,      # Phase 2 (Narrative/Momentum): max 5x exposure
    3: 1.0,      # Phase 3 (Unwind): max 1x exposure
    4: 1.5,      # Phase 4 (Reset): max 1.5x exposure
}

# ── Interval-aware bar count ────────────────────────────────────────────────
# BARS_PER_YEAR drives ALL time-scale-dependent constants in the model:
#   - Annualisation multipliers for Sharpe, Sortino, portfolio_performance
#   - Risk-free rate per bar  (RF_RATE)
#   - Lookback windows in the regime classifier
#
# Original code hardcoded 252 (trading days/year) everywhere.
# With INTERVAL="1wk", one bar = 1 week, so using 252 meant:
#   pct_change(63) = 63-WEEK lookback ≈ 14 months  (intended: 3 months)
#   np.sqrt(252)   = annualisation for daily data   (off by sqrt(252/52) ≈ 2.2×)
#   RF_RATE/252    = daily risk-free rate used weekly (hurdle 5× too small)
_interval_lower = INTERVAL.lower().replace(" ", "")
if _interval_lower in ("1m", "1min", "1minute"):
    BARS_PER_YEAR = 252 * 6.5 * 60   # 252 trading days × 6.5 hours/day × 60 minutes/hour
elif _interval_lower in ("5m", "5min", "5minute"):
    BARS_PER_YEAR = 252 * 6.5 * 12   # 252 trading days × 6.5 hours/day × 12 five-minute bars/hour
elif _interval_lower in ("15m", "15min", "15minute"):
    BARS_PER_YEAR = 252 * 6.5 * 4    # 252 trading days × 6.5 hours/day × 4 fifteen-minute bars/hour
elif _interval_lower in ("30m", "30min", "30minute"):
    BARS_PER_YEAR = 252 * 6.5 * 2    # 252 trading days × 6.5 hours/day × 2 thirty-minute bars/hour
elif _interval_lower in ("60m", "60min", "60minute", "1h", "1hr", "1hour"):
    BARS_PER_YEAR = 252 * 6.5        # 252 trading days × 6.5 hours/day × 1 sixty-minute bar/hour
elif _interval_lower in ("90m", "90min", "90minute"):
    BARS_PER_YEAR = 252 * 6.5 * (60 / 90)  # 252 trading days × 6.5 hours/day × (60/90) ninety-minute bars/hour
elif _interval_lower in ("4h", "4hr", "4hour"):
    BARS_PER_YEAR = 252 * 6.5 * (60 / 240) # 252 trading days × 6.5 hours/day × (60/240) four-hour bars/hour
elif  _interval_lower in ("1d",  "1day"):    BARS_PER_YEAR = 252
elif _interval_lower in ("1wk", "1w"):      BARS_PER_YEAR = 52
elif _interval_lower in ("1mo", "1month"):  BARS_PER_YEAR = 12
else:                                        BARS_PER_YEAR = 252   # safe default

# Risk-free rate per bar.  Always annual rate / bars per year.
RF_RATE = 0.03 / BARS_PER_YEAR

# Regime-classifier lookback windows scaled to the selected interval.
# Economic durations are preserved regardless of bar frequency:
#   3-month momentum  : BARS_PER_YEAR / 4   →  63 bars daily /  13 bars weekly
#   1-month momentum  : BARS_PER_YEAR / 12  →  21 bars daily /   4 bars weekly
#   Long VIX window   : BARS_PER_YEAR / 4   →  60 bars daily /  13 bars weekly
#   Short VIX window  : BARS_PER_YEAR / 12  →  20 bars daily /   4 bars weekly
MOM_SLOW_BARS  = max(2, int(BARS_PER_YEAR / 4))    # ~3-month momentum
MOM_FAST_BARS  = max(2, int(BARS_PER_YEAR / 12))   # ~1-month momentum
VIX_LONG_BARS  = max(4, int(BARS_PER_YEAR / 4))    # long VIX rolling window
VIX_SHORT_BARS = max(2, int(BARS_PER_YEAR / 12))   # short VIX rolling window
 
# ============================================================
# SECTION 1 — CONSTITUENT DOWNLOAD
# ============================================================
def get_html(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )}
    )
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")
 
def get_sp500():
    try:
        html    = get_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df      = pd.read_html(StringIO(html), header=0)[0]
        tickers = set(df["Symbol"].str.strip().tolist())
        print(f"  SP500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  SP500 fetch failed: {e}")
        return set()
 
def get_djia():
    try:
        html   = get_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
        tables = pd.read_html(StringIO(html), header=0)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "symbol" in cols or "ticker" in cols:
                col = [c for c in t.columns
                       if c.lower() in ("symbol", "ticker")][0]
                tickers = set(t[col].dropna().str.strip().tolist())
                print(f"  DJIA: {len(tickers)} tickers")
                return tickers
        print("  DJIA: no matching table found")
        return set()
    except Exception as e:
        print(f"  DJIA fetch failed: {e}")
        return set()
 
def get_nasdaq100():
    try:
        html   = get_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        tables = pd.read_html(StringIO(html), header=0)
        for t in tables:
            cols = [c.lower() for c in t.columns]
            if "ticker" in cols or "symbol" in cols:
                col = [c for c in t.columns
                       if c.lower() in ("ticker", "symbol")][0]
                tickers = set(t[col].dropna().str.strip().tolist())
                print(f"  NASDAQ100: {len(tickers)} tickers")
                return tickers
        print("  NASDAQ100: no matching table found")
        return set()
    except Exception as e:
        print(f"  NASDAQ100 fetch failed: {e}")
        return set()
 
print("Fetching constituent lists...")
sp500  = get_sp500()
djia   = get_djia()
nasdaq = get_nasdaq100()
 
all_tickers = sp500 | djia | nasdaq
all_tickers = {
    t.replace(".", "-") for t in all_tickers
    if isinstance(t, str) and len(t) <= 6
}
print(f"Total unique tickers: {len(all_tickers)}")
 
# ============================================================
# SECTION 2 — PRICE DOWNLOAD
# ============================================================
# We keep parallelism everywhere else (NewsAgent, optimization, backtest),


print("Downloading price data...")

raw, failed_tickers = download_price_data(
    list(all_tickers),
    period=PERIOD,
    interval=INTERVAL,
    chunk_size=60,
    retries=2,
    progress=True,
)

if failed_tickers:
    print(f"Failed downloads for {len(failed_tickers)} tickers: {failed_tickers}")

if raw.empty or raw.shape[1] == 0:
    fallback_universe = ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD"]
    print("Primary universe download returned zero valid tickers. Falling back to core ETFs:", fallback_universe)
    raw, fallback_failed = download_price_data(
        fallback_universe,
        period=PERIOD,
        interval=INTERVAL,
        chunk_size=6,
        retries=2,
        progress=False,
    )
    if fallback_failed:
        print(f"  Fallback failed for: {fallback_failed}")
    if raw.empty or raw.shape[1] == 0:
        raise ValueError(
            "No valid price data could be downloaded for the main or fallback universe. "
            "Check your internet connection and yfinance availability."
        )
    print(f"Using fallback tickers: {list(raw.columns)}")

threshold = 0.05
min_required = int(len(raw) * (1 - threshold))
coverage = raw.notna().sum(axis=0)
valid_columns = coverage[coverage >= min_required].index
if len(valid_columns) == 0:
    print(
        "No tickers met the coverage threshold for the primary universe. "
        "Falling back to core ETFs."
    )
    raw, fallback_failed = download_price_data(
        ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD"],
        period=PERIOD,
        interval=INTERVAL,
        chunk_size=6,
        retries=2,
        progress=False,
    )
    if fallback_failed:
        print(f"  Fallback failed for: {fallback_failed}")
    if raw.empty or raw.shape[1] == 0:
        raise ValueError(
            "No valid price data could be downloaded for the main or fallback universe. "
            "Check your internet connection and yfinance availability."
        )
    print(f"Using fallback tickers: {list(raw.columns)}")
    min_required = int(len(raw) * (1 - threshold))
    coverage = raw.notna().sum(axis=0)
    valid_columns = coverage[coverage >= min_required].index
    if len(valid_columns) == 0:
        raise ValueError(
            "Fallback universe also failed to meet the coverage threshold. "
            "Try a smaller interval, a lower threshold, or a different universe."
        )

raw = raw[valid_columns]
raw = raw.ffill().bfill().dropna()
if raw.shape[1] == 0:
    raise ValueError(
        "No price columns remain after filtering. "
        "Lower the threshold or use a smaller universe of more reliable tickers."
    )

print(f"Clean tickers after filtering: {raw.shape[1]}")
print(f"Trading days: {raw.shape[0]}")
 
# ============================================================
# SECTION 3 — RETURNS + SLEEVE INSTRUMENTS
# ============================================================
returns = raw.pct_change().dropna()
if isinstance(returns, pd.Series):
    returns = returns.to_frame()

if returns.empty:
    raise ValueError(
        "No return data available after price cleaning. "
        "Check the yfinance download, the dropna threshold, and the selected symbols."
    )

print("Downloading instrument sleeves...")
sleeve_raw = yf.download(
    SLEEVE_INSTRUMENTS,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True,
    progress=False
)["Close"]
 
if isinstance(sleeve_raw.columns, pd.MultiIndex):
    sleeve_raw.columns = sleeve_raw.columns.droplevel(1)
 
sleeve_raw = sleeve_raw.ffill()
# ── Why fillna(0.0) instead of dropna() ────────────────────────────────────
# SDS (launched 2006) have NaN returns for every
# date before their IPO.  dropna() on a multi-column DataFrame removes ANY
# row where even one column is NaN, so all pre-2010 dates get silently
# stripped.  The vectorized backtest then raises KeyError when it looks up
# those dates.  fillna(0.0) is semantically correct: a sleeve instrument that
# doesn't exist yet contributes 0% return for that period.
sleeve_returns = sleeve_raw.pct_change().fillna(0.0)
sleeve_returns = sleeve_returns.reindex(returns.index, fill_value=0.0)
print(f"Sleeve instruments loaded: {list(sleeve_returns.columns)}")
 
vix_raw = yf.download("^VIX", period=PERIOD, interval=INTERVAL,
                       auto_adjust=True, progress=False)["Close"]
hyg_raw = yf.download("HYG",  period=PERIOD, interval=INTERVAL,
                       auto_adjust=True, progress=False)["Close"]
 
vix = vix_raw.reindex(returns.index, method="ffill").squeeze()
hyg = hyg_raw.pct_change().reindex(returns.index, method="ffill").squeeze()
 
# ============================================================
# SECTION 4 — REGIME CLASSIFIER (fully vectorized) + SMOOTHING
# ============================================================
# ORIGINAL IMPLEMENTATION:
#   A Python for-loop iterated over every date and used
#   .loc[date] to read individual scalar values from the Series.
#   With 50 years × 252 days ≈ 12,600 dates, each loop iteration
#   performed ~4 pandas scalar lookups (each O(log n)) and several
#   Python-level float comparisons.  Total: ~50,000+ Python ops.
#
# OPTIMIZED IMPLEMENTATION:
#   All conditions are expressed as boolean mask arrays computed
#   over the full Series in one pass.  NumPy evaluates these
#   element-wise in compiled C code at memory bandwidth speed.
#   np.where and boolean indexing then assign regime labels in bulk.
#   Total Python-level iterations: 0.  ~50x wall-time speedup.
#
# FOUR PHASES:
#   Phase 1  — Buildout:   low vol, neutral/mild momentum
#   Phase 2  — Narrative:  strong momentum, VIX still low
#   Phase 3  — Unwind:     VIX z-score spike + negative momentum
#   Phase 4  — Reset:      elevated but falling VIX, weak momentum
#
# REGIME SMOOTHING: require REGIME_CONFIRM_DAYS consecutive
# days in a new phase before recording the switch.  Prevents
# whipsaw from single-day VIX spikes.  Zero lookahead.
 
def classify_regime(vix_series: pd.Series,
                    hyg_series: pd.Series,
                    spy_series: pd.Series):
    """
    Vectorized regime classifier.
 
    Formulas:
      spy_mom(t)      = SPY_t / SPY_{t-63}  - 1  (3-month momentum)
      spy_mom_fast(t) = SPY_t / SPY_{t-21}  - 1  (1-month momentum)
      vix_zscore(t)   = 0.5 * (VIX_t - μ_60) / σ_60
                      + 0.5 * (VIX_t - μ_20) / σ_20
 
    Phase priority (higher phase wins when multiple conditions true):
      Phase 2: VIX < 20 AND (3m-mom > 10% OR 1m-mom > 5%)
      Phase 4: VIX > 20 AND z < 0 AND 3m-mom < 5%
      Phase 3: z > 1.5 AND 3m-mom < -5%      ← highest priority
 
    Returns (regime_series, spy_mom_fast_series)
    """
    # ── Momentum and rolling vol statistics ──────────────────
    # Lookback windows are driven by MOM_SLOW_BARS / MOM_FAST_BARS /
    # VIX_LONG_BARS / VIX_SHORT_BARS, which are derived from BARS_PER_YEAR
    # in the CONFIG section.  This makes the classifier correct for both
    # daily (pct_change(63) = 3 months) and weekly (pct_change(13) = 3 months)
    # data without any manual re-tuning.
    spy_mom      = spy_series.pct_change(MOM_SLOW_BARS)
    spy_mom_fast = spy_series.pct_change(MOM_FAST_BARS)
    vix_sma60    = vix_series.rolling(VIX_LONG_BARS).mean()
    vix_sma20    = vix_series.rolling(VIX_SHORT_BARS).mean()
    vix_std60    = vix_series.rolling(VIX_LONG_BARS).std()
    vix_std20    = vix_series.rolling(VIX_SHORT_BARS).std()
 
    # Blended VIX z-score (equal weight of 60-day and 20-day lookbacks)
    vix_zscore = (
        (vix_series - vix_sma60) / (vix_std60 + 1e-10) * 0.5 +
        (vix_series - vix_sma20) / (vix_std20 + 1e-10) * 0.5
    )
 
    # ── Boolean masks (vectorized over entire index) ──────────
    # Each condition is a pandas boolean Series: True/False for every date.
    # No Python loop; pandas calls compiled NumPy ufuncs internally.
    cond_p2 = (vix_series < 20) & ((spy_mom > 0.10) | (spy_mom_fast > 0.05))
    cond_p4 = (vix_series > 20) & (vix_zscore < 0) & (spy_mom < 0.05)
    cond_p3 = (vix_zscore > 1.5) & (spy_mom < -0.05)
 
    # ── Assign phases by layered override (lowest → highest) ──
    # Start with Phase 1 everywhere (the default "calm" state).
    # Apply Phase 2, then Phase 4, then Phase 3 on top — later
    # assignments overwrite earlier ones so Phase 3 always wins
    # when its condition is true.
    regime = pd.Series(1.0, index=vix_series.index, dtype=float)
    regime[cond_p2] = 2.0
    regime[cond_p4] = 4.0
    regime[cond_p3] = 3.0      # Phase 3 is highest priority
 
    return regime, spy_mom_fast
 
 
# ── Numba JIT compiled regime smoother ───────────────────────
# OPTIMIZATION: The smoothing loop is inherently sequential — each
# output depends on the previous output (regime_smoothed_{t-1}).
# We cannot vectorize across the time axis.  Instead, we use
# Numba's @njit decorator to compile the function to native CPU
# machine code.  The first call triggers a one-time JIT compilation
# (~0.5s); every subsequent call runs at C speed (~30-50x faster
# than CPython for tight numeric loops).
#
# @njit = "no Python" — the function body must use only types that
# Numba understands (NumPy arrays, floats, ints).  Pandas objects
# are NOT allowed inside an @njit function, so we extract .values
# before calling and reconstruct the Series afterward.
 
@njit
def _smooth_regime_numba(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compiled regime smoothing kernel.
 
    Rule (backward-looking, zero lookahead):
      smoothed[i] = candidate     if values[i-window+1 : i+1] are all equal
                    smoothed[i-1] otherwise
 
    This prevents the classifier from reacting to single-day VIX spikes.
    `window` = REGIME_CONFIRM_DAYS (default 5 trading days).
 
    Parameters
    ----------
    values : 1-D float64 array  — raw (unsmoothed) regime labels
    window : int                — consecutive days required to confirm a switch
 
    Returns
    -------
    smoothed : 1-D float64 array — smoothed regime labels
    """
    n        = len(values)
    smoothed = values.copy()
 
    for i in range(window, n):
        candidate = values[i]
        # Check if the last `window` days all show the same regime
        all_same = True
        for j in range(i - window + 1, i + 1):
            if values[j] != candidate:
                all_same = False
                break
        # Only switch if confirmed; otherwise hold the previous label
        if not all_same:
            smoothed[i] = smoothed[i - 1]
 
    return smoothed
 
 
def smooth_regime(regime_series: pd.Series, window: int = REGIME_CONFIRM_DAYS) -> pd.Series:
    """
    Wrapper that extracts raw NumPy array, calls the Numba kernel,
    and returns a pandas Series with the original DatetimeIndex.
    """
    raw_values   = regime_series.values.astype(np.float64)
    smooth_vals  = _smooth_regime_numba(raw_values, window)
    return pd.Series(smooth_vals, index=regime_series.index)
 
 
labels = {1: "Buildout", 2: "Narrative", 3: "Unwind", 4: "Reset"}
 
spy_px = yf.download(
    "SPY", period=PERIOD, interval=INTERVAL,
    auto_adjust=True, progress=False
)["Close"].squeeze()
spy_px = spy_px.reindex(returns.index, method="ffill")
 
print("Classifying market regimes (vectorized)...")
regime_raw, spy_mom_fast = classify_regime(vix, hyg, spy_px)
 
print(f"Applying {REGIME_CONFIRM_DAYS}-day regime confirmation filter (Numba JIT)...")
# Numba compiles on first call — this warm-up message helps users
# understand the brief pause.
print("  (First run triggers Numba JIT compilation — ~0.5s one-time cost)")
regime = smooth_regime(regime_raw, window=REGIME_CONFIRM_DAYS)
 
spy_mom_fast = spy_mom_fast.reindex(returns.index, method="ffill")
 
phase_counts = regime.value_counts().sort_index()
print("\nREGIME DISTRIBUTION (after smoothing)")
print("=" * 35)
for phase, count in phase_counts.items():
    pct = count / len(regime) * 100
    print(f"  Phase {int(phase)} ({labels[int(phase)]:<12}): "
          f"{count:>4} days ({pct:.1f}%)")
print("=" * 35)
 
# ── Walk-forward split ────────────────────────────────────────
if len(returns) < 2:
    raise ValueError(
        "Insufficient return data for a train/test split. "
        f"Need at least 2 rows, got {len(returns)}."
    )

split_index = min(max(1, int(len(returns) * TRAIN_FRAC)), len(returns) - 1)
split_date = returns.index[split_index]
train_returns = returns.iloc[: split_index + 1]
test_returns = returns.iloc[split_index + 1 :]

if train_returns.empty or test_returns.empty:
    raise ValueError(
        "Walk-forward split produced an empty training or test set. "
        f"Train rows: {len(train_returns)}, Test rows: {len(test_returns)}, "
        f"split_date: {split_date}."
    )

if train_returns.shape[1] == 0:
    raise ValueError(
        "No asset columns remain after data cleaning. "
        "Ensure the downloaded price data contains valid tickers and that dropna thresholds are not too strict."
    )

max_components = min(train_returns.shape)
if max_components < 1:
    raise ValueError(
        "Insufficient training data for PCA factor extraction. "
        f"Training data shape is {train_returns.shape}."
    )

if N_FACTORS > max_components:
    print(
        f"Reducing N_FACTORS from {N_FACTORS} to {max_components} "
        f"because the training dataset only supports {max_components} PCA components."
    )
    N_FACTORS = max_components

print(f"\nWalk-forward split:")
print(f"  Train: {returns.index[0].date()} → {split_date.date()}")
print(f"  Test:  {split_date.date()} → {returns.index[-1].date()}")
 
# ============================================================
# SECTION 5 — PCA (fit on training data ONLY)
# ============================================================
print(f"\nRunning PCA — extracting {N_FACTORS} factors (training data only)...")
 
scaler               = StandardScaler()
train_returns_scaled = scaler.fit_transform(train_returns)
test_returns_scaled  = scaler.transform(test_returns)
 
pca           = PCA(n_components=N_FACTORS)
train_factors = pca.fit_transform(train_returns_scaled)
test_factors  = pca.transform(test_returns_scaled)
 
factor_df = pd.DataFrame(
    np.vstack([train_factors, test_factors]),
    index=pd.concat([train_returns, test_returns]).sort_index().index,
    columns=[f"F{i+1}" for i in range(N_FACTORS)]
)
 
explained  = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
 
print("\nPCA VARIANCE EXPLAINED")
print("=" * 40)
for i, (e, c) in enumerate(zip(explained, cumulative)):
    print(f"  Factor {i+1:>2}: {e*100:>5.2f}%  "
          f"(cumulative: {c*100:>5.2f}%)")
print("=" * 40)
 
# ============================================================
# SECTION 6 — FACTOR → STOCK WEIGHT MAPPING
# ============================================================
loadings = pd.DataFrame(
    pca.components_,
    columns=returns.columns,
    index=[f"F{i+1}" for i in range(N_FACTORS)]
)
 
def factor_weights_to_stock_weights(factor_weights: np.ndarray,
                                    loadings: pd.DataFrame) -> np.ndarray:
    """
    Project factor-level exposures into security-level weights.
 
      w_stock = clip(w_factors · loadings, min=0)  (long-only)
      w_stock /= sum(w_stock)                        (normalize to 1)
 
    The dot product (1xF) · (FxN) = (1xN) maps factor positions
    to stock positions.  Clipping to zero enforces the long-only
    constraint without short selling.
    """
    stock_weights = np.dot(factor_weights, loadings.values)
    stock_weights = np.clip(stock_weights, 0.0, None)
    total = stock_weights.sum()
    if total < 1e-10:
        stock_weights = np.ones(len(stock_weights)) / len(stock_weights)
    else:
        stock_weights /= total
    return stock_weights
 
# ============================================================
# SECTION 7 — BATES MODEL (PyTorch GPU-accelerated)
# ============================================================
# OPTIMIZATION: The original NumPy version ran a Python for-loop
# over each of the `steps` time points.  At each step it created
# fresh NumPy arrays for all `paths`.  The overhead of repeated
# array allocation + Python iteration adds up at 252 steps.
#
# PyTorch version:
#   1. Pre-allocate ALL random numbers for the entire simulation at
#      once: shape (steps, paths).  One allocation, one kernel call.
#   2. Use torch.cumsum / element-wise operations — these dispatch
#      to the GPU (RX 9070 XT via ROCm, or CUDA/CPU fallback).
#   3. The loop body is now ~6 tensor ops; Python overhead is
#      negligible compared to GPU kernel latency.
#
# Bates model equations (recap):
#   dv_t = κ(θ - v_t) dt + σ √v_t dW^v_t           (variance CIR process)
#   dS_t/S_t = -½v_t dt + √v_t dW^s_t + J dN_t     (price SDE + jumps)
#   N_t ~ Poisson(λ dt)                              (jump arrival)
#   log J ~ N(μ_j, σ_j²)                            (log-normal jump size)
 
def bates_simulate(S0: float, v0: float, kappa: float, theta: float,
                   sigma: float, rho: float, lam: float, mu_j: float,
                   sig_j: float, T: float = 1.0, steps: int = 252,
                   paths: int = 1000):
    """
    Simulate the Bates (1996) stochastic-vol + jump model using
    PyTorch tensors.  Runs on GPU if available; falls back to CPU.
 
    Parameters
    ----------
    S0     : float  — initial asset price
    v0     : float  — initial variance (e.g. 0.04 ≈ 20% annual vol)
    kappa  : float  — mean-reversion speed of variance
    theta  : float  — long-run variance (equilibrium level)
    sigma  : float  — vol-of-vol (variance diffusion coefficient)
    rho    : float  — correlation between price and variance Brownian motions
    lam    : float  — Poisson jump intensity (expected jumps per year)
    mu_j   : float  — mean of log-normal jump magnitude
    sig_j  : float  — std dev of log-normal jump magnitude
    T      : float  — time horizon in years
    steps  : int    — number of time steps (252 = trading days in 1 year)
    paths  : int    — number of Monte Carlo simulations
 
    Returns
    -------
    prices : (steps+1, paths) numpy array — simulated price paths
    vols   : (steps+1, paths) numpy array — simulated variance paths
    """
    dt = T / steps
 
    # ── Pre-allocate all random numbers at once ───────────────
    # Shape: (steps, paths).  This is ~1 VRAM allocation instead
    # of `steps` separate allocations.
    z1 = torch.randn(steps, paths, device=DEVICE, dtype=torch.float32)
 
    # Cholesky decomposition of the [1, rho; rho, 1] correlation matrix:
    #   z2 = rho*z1 + sqrt(1 - rho²)*ε   where ε ~ N(0,1) independent
    z2 = (rho * z1 +
          torch.sqrt(torch.tensor(1.0 - rho**2, device=DEVICE)) *
          torch.randn(steps, paths, device=DEVICE, dtype=torch.float32))
 
    # ── Tensor containers for paths ──────────────────────────
    prices = torch.zeros(steps + 1, paths, device=DEVICE, dtype=torch.float32)
    vols   = torch.zeros(steps + 1, paths, device=DEVICE, dtype=torch.float32)
    prices[0] = S0
    vols[0]   = v0
 
    dt_t       = torch.tensor(dt,    device=DEVICE, dtype=torch.float32)
    kappa_t    = torch.tensor(kappa, device=DEVICE, dtype=torch.float32)
    theta_t    = torch.tensor(theta, device=DEVICE, dtype=torch.float32)
    sigma_t    = torch.tensor(sigma, device=DEVICE, dtype=torch.float32)
    lam_t      = torch.tensor(lam,   device=DEVICE, dtype=torch.float32)
    mu_j_t     = torch.tensor(mu_j,  device=DEVICE, dtype=torch.float32)
    sig_j_t    = torch.tensor(sig_j, device=DEVICE, dtype=torch.float32)
 
    # ── Simulation loop (steps iterations, not paths × steps) ─
    # We still loop over time steps (252 for 1-year) but each
    # iteration operates on ALL paths simultaneously as a tensor op.
    # This is "vectorized over the paths dimension."
    for t in range(1, steps + 1):
        v_prev = vols[t - 1]
 
        # ── Variance update (Euler-Maruyama discretization of CIR) ──
        # v_t = |v_{t-1} + κ(θ - v_{t-1})dt + σ√(v_{t-1}·dt)·z2|
        # The absolute value is the "full truncation" scheme which
        # prevents variance from going negative.
        vols[t] = torch.abs(
            v_prev
            + kappa_t * (theta_t - v_prev) * dt_t
            + sigma_t * torch.sqrt(v_prev * dt_t) * z2[t - 1]
        )
 
        # ── Jump component ───────────────────────────────────────
        # N ~ Poisson(λ·dt): number of jumps in this interval
        n_jumps = torch.poisson(lam_t * dt_t * torch.ones(paths, device=DEVICE))
 
        # Log-normal jump size: J = exp(μ_j·N + σ_j·√N·ε) - 1
        # When N=0 this collapses to exp(0) - 1 = 0 (no jump).
        eps_j   = torch.randn(paths, device=DEVICE, dtype=torch.float32)
        jump    = torch.exp(mu_j_t * n_jumps +
                            sig_j_t * torch.sqrt(n_jumps) * eps_j) - 1.0
 
        # ── Price update (geometric Brownian motion + jump) ─────
        # log(S_t/S_{t-1}) = -½v_{t-1}dt  (Itô correction)
        #                   + √(v_{t-1}dt) · z1  (diffusion)
        #                   + log(1 + jump)       (jump contribution)
        prices[t] = prices[t - 1] * torch.exp(
            -0.5 * v_prev * dt_t
            + torch.sqrt(v_prev * dt_t) * z1[t - 1]
            + torch.log(1.0 + jump + 1e-10)
        )
 
    # Return as NumPy arrays for downstream matplotlib / NumPy code
    return prices.cpu().numpy(), vols.cpu().numpy()
 
 
bates_params = {
    1: dict(v0=0.04, kappa=2.0,  theta=0.04, sigma=0.3,
            rho=-0.7,  lam=0.5, mu_j=-0.02, sig_j=0.05),
    2: dict(v0=0.06, kappa=1.5,  theta=0.06, sigma=0.4,
            rho=-0.6,  lam=1.0, mu_j=-0.03, sig_j=0.07),
    3: dict(v0=0.15, kappa=1.0,  theta=0.10, sigma=0.6,
            rho=-0.8,  lam=3.0, mu_j=-0.08, sig_j=0.12),
    4: dict(v0=0.08, kappa=2.5,  theta=0.05, sigma=0.35,
            rho=-0.65, lam=0.8, mu_j=-0.03, sig_j=0.07),
}
 
current_vix    = float(vix.iloc[-1])
current_regime = int(regime.iloc[-1])
params         = bates_params[current_regime]
 
print(f"\nBates Model — Current Regime: Phase {current_regime} ({labels[current_regime]})")
print(f"Current VIX: {current_vix:.1f}")
 
spy_price = float(spy_px.iloc[-1])
sim_prices, sim_vols = bates_simulate(S0=spy_price, **params, paths=500)
 
# ============================================================
# SECTION 8 — METRICS
# ============================================================
def portfolio_performance(weights, ret_matrix):
    weights     = np.array(weights)
    port_return = np.dot(weights, ret_matrix.mean()) * BARS_PER_YEAR
    cov         = ret_matrix.cov() * BARS_PER_YEAR
    port_vol    = np.sqrt(weights @ cov @ weights)
    port_sharpe = (port_return - RF_RATE * BARS_PER_YEAR) / (port_vol + 1e-10)
    return port_return, port_vol, port_sharpe
 
def max_drawdown(cum_series):
    return (cum_series / cum_series.cummax() - 1).min()
 
def sharpe(ret_series):
    excess = ret_series - RF_RATE
    return (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(BARS_PER_YEAR)
 
def sortino(ret_series):
    excess   = ret_series - RF_RATE
    downside = excess[excess < 0]
    down_std = (downside.std() * np.sqrt(BARS_PER_YEAR)) if len(downside) > 1 else 1e-10
    return (excess.mean() * BARS_PER_YEAR) / (down_std + 1e-10)
 
def compute_cvar(port_rets, alpha=CVAR_ALPHA):
    """
    Conditional Value at Risk (CVaR / Expected Shortfall).
    Formula: mean of the worst alpha% daily losses.
      CVaR_α = mean( sorted(-r)[ -ceil(α·N) : ] )
    A better tail-risk measure than plain VaR because it averages
    the losses in the tail rather than just reporting the boundary.
    """
    if len(port_rets) == 0:
        return 0.0
    port_rets     = np.asarray(port_rets).flatten()
    losses        = -port_rets
    sorted_losses = np.sort(losses)
    n_tail        = max(int(np.ceil(alpha * len(losses))), 1)
    return np.mean(sorted_losses[-n_tail:])
 
# ============================================================
# SECTION 8b — OPTIMIZER
# ============================================================
def optimize_portfolio(ret_matrix, objective="sharpe", cvar_alpha=CVAR_ALPHA):
    """
    Constrained portfolio optimizer (SLSQP).
    Objectives: "sharpe" | "sortino" | "cvar" | fallback min-vol.
    Bounds: each weight in [MIN_WEIGHT, MAX_WEIGHT].
    Constraint: weights sum to 1.0.
    """
    n           = ret_matrix.shape[1]
    w0          = np.ones(n) / n
    bounds      = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
 
    if objective == "sharpe":
        def obj(w):
            _, _, s = portfolio_performance(w, ret_matrix)
            return -s
 
    elif objective == "sortino":
        def obj(w):
            w_arr     = np.array(w)
            port_rets = (ret_matrix.dot(w_arr)
                         if isinstance(ret_matrix, pd.DataFrame)
                         else np.dot(ret_matrix, w_arr))
            return -sortino(pd.Series(port_rets))
 
    elif objective == "cvar":
        def obj(w):
            w_arr     = np.array(w)
            port_rets = (ret_matrix.dot(w_arr)
                         if isinstance(ret_matrix, pd.DataFrame)
                         else np.dot(ret_matrix, w_arr))
            return compute_cvar(port_rets, alpha=cvar_alpha)
 
    else:
        def obj(w):
            _, v, _ = portfolio_performance(w, ret_matrix)
            return v
 
    result = minimize(
        obj, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        tol=1e-8, options={"maxiter": 500}
    )
 
    opt_w  = np.clip(np.array(result.x), MIN_WEIGHT, MAX_WEIGHT)
    opt_w /= (opt_w.sum() + 1e-10)
    return opt_w
 
# ============================================================
# SECTION 8c — PARALLEL PHASE-SPECIFIC OPTIMISATION
# ============================================================
# OPTIMIZATION: Previously, the four phase optimizations ran
# sequentially — each scipy.minimize finished before the next
# started.  Each optimization is completely independent of the
# others (they use different data slices and objectives), making
# this a textbook "embarrassingly parallel" workload.
#
# joblib.Parallel launches each job in a separate Python process
# (not thread) on the Ryzen 9 9950X3D cores.  Multiple processes
# sidestep the GIL — all 4 optimizations run truly simultaneously.
#
# Wall-time speedup ≈ min(4, N_JOBS) ≈ 4×
# (bounded by the slowest of the 4 jobs).
 
print("\nOptimizing portfolio on PCA factors (training data only)...")
 
train_factor_df = factor_df[factor_df.index <= split_date]
 
def subsample(df, step=5):
    """Subsample every `step` rows — 5× faster optimization with minimal quality loss."""
    return df.iloc[::step] if len(df) > 1000 else df
 
p1_train = subsample(train_factor_df[train_factor_df.index.isin(regime[regime == 1].index)])
p2_train = subsample(train_factor_df[train_factor_df.index.isin(regime[regime == 2].index)])
p3_train = subsample(train_factor_df[train_factor_df.index.isin(regime[regime == 3].index)])
p4_train = subsample(train_factor_df[train_factor_df.index.isin(regime[regime == 4].index)])
 
MIN_OBS = {1: 60, 2: 30, 3: 20, 4: 20}
 
def _run_phase4_custom(data):
    """
    Phase 4 uses a custom composite objective (Sharpe - 2×CVaR)
    that doesn't fit the generic optimize_portfolio interface.
    Defined as a standalone function so joblib can pickle it.
    (joblib requires all parallel-dispatched functions to be
    picklable; lambdas and inner functions are not picklable.)
    """
    if len(data) <= MIN_OBS[4]:
        return None   # signal caller to use fallback
 
    def phase4_obj(w):
        w_arr     = np.array(w)
        port_rets = data.dot(w_arr)
        _, _, s   = portfolio_performance(w_arr, data)
        return -s + 2.0 * compute_cvar(port_rets)
 
    n      = data.shape[1]
    result = minimize(
        phase4_obj,
        np.ones(n) / n,
        method="SLSQP",
        bounds=tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n)),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        tol=1e-8, options={"maxiter": 500}
    )
    w = np.clip(result.x, MIN_WEIGHT, MAX_WEIGHT)
    w /= (w.sum() + 1e-10)
    return w
 
print(f"  → Launching {min(4, N_JOBS)} parallel phase optimizations "
      f"(joblib, {N_JOBS} cores available)...")
 
# Each task: (function, args...) as a delayed call.
# Phases 1-3 use the generic optimizer; Phase 4 uses the custom obj.
parallel_results = Parallel(n_jobs=min(4, N_JOBS), prefer="processes")(
    [
        delayed(optimize_portfolio)(p1_train, "sharpe")
            if len(p1_train) > MIN_OBS[1]
            else delayed(lambda: np.ones(N_FACTORS) / N_FACTORS)(),
 
        delayed(optimize_portfolio)(p2_train, "sortino")
            if len(p2_train) > MIN_OBS[2]
            else delayed(lambda: None)(),
 
        delayed(optimize_portfolio)(p3_train, "cvar")
            if len(p3_train) > MIN_OBS[3]
            else delayed(lambda: np.ones(N_FACTORS) / N_FACTORS)(),
 
        delayed(_run_phase4_custom)(p4_train),
    ]
)
 
w_growth    = parallel_results[0]
w_momentum_ = parallel_results[1]
w_defensive = parallel_results[2]
w_recovery_ = parallel_results[3]
 
# Apply fallbacks for phases that lacked sufficient data
if w_growth is None:
    w_growth = np.ones(N_FACTORS) / N_FACTORS
if w_momentum_ is None:
    w_momentum_ = w_growth
if w_defensive is None:
    w_defensive = np.ones(N_FACTORS) / N_FACTORS
if w_recovery_ is None:
    w_recovery_ = 0.4 * w_defensive + 0.6 * w_growth
    w_recovery_ /= (w_recovery_.sum() + 1e-10)
 
w_momentum = w_momentum_
w_recovery = w_recovery_
 
print("  → Optimization complete — projecting to stock weights...")
 
stock_w_growth    = factor_weights_to_stock_weights(w_growth,    loadings)
stock_w_momentum  = factor_weights_to_stock_weights(w_momentum,  loadings)
stock_w_defensive = factor_weights_to_stock_weights(w_defensive, loadings)
stock_w_recovery  = factor_weights_to_stock_weights(w_recovery,  loadings)
 
phase_factor_weights = {
    1:    stock_w_growth,
    "1b": stock_w_momentum,
    2:    stock_w_momentum,
    3:    stock_w_defensive,
    4:    stock_w_recovery,
}
 
# ============================================================
# SECTION 9 — BLEND DEFINITIONS
# ============================================================
def validate_blend(blend: dict) -> dict:
    """Normalize phase blend weights to a true 100% capital allocation."""
    total = sum(blend.values())
    if abs(total - 1.0) > 1e-8 and total != 0.0:
        normalized = {k: v / total for k, v in blend.items()}
        print(f"  [Blend Validation] normalizing blend weights from {total:.4f} to 1.0000")
        return normalized
    if total == 0.0:
        raise ValueError("Phase blend weights sum to zero — cannot normalize blend.")
    return blend

phase_blend = {
    1: {"FACTOR": 0.75, "SPY": 0.00, "QQQ": 0.10, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.05, "TLT": 0.00, "BTC-USD": 0.10},
 
    "1b": {"FACTOR": 0.75, "SPY": 0.00, "QQQ": 0.10, "GLD": 0.00,
           "SH": 0.00, "SDS": 0.05, "TLT": 0.00, "BTC-USD": 0.10},
 
    2: {"FACTOR": 0.75, "SPY": 0.00, "QQQ": 0.10, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.00, "TLT": 0.00, "BTC-USD": 0.15},
 
    3: {"FACTOR": 0.20, "SPY": 0.00, "QQQ": 0.00, "GLD": 0.20,
        "SH": 0.10, "SDS": 0.35, "TLT": 0.15, "BTC-USD": 0.00},
 
    4: {"FACTOR": 0.75, "SPY": 0.00, "QQQ": 0.05, "GLD": 0.15,
        "SH": 0.00, "SDS": 0.00, "TLT": 0.00, "BTC-USD": 0.05},
}
phase_blend = {k: validate_blend(v) for k, v in phase_blend.items()}
 
# ============================================================
# SLEEVE + PHASE RESOLUTION HELPERS
# ============================================================
 
def _build_sleeve_weight_vector(blend: dict,
                                sleeve_cols: pd.Index,
                                pre_sds: bool = False) -> np.ndarray:
    """
    Build a 1-D weight vector aligned to sleeve_returns.columns.
 
    This is called once per (phase, date_class) combination and
    the result is reused for the entire vectorized block —
    replacing thousands of per-date dictionary lookups.
 
    Parameters
    ----------
    blend      : dict       — phase blend allocation dict
    sleeve_cols: Index      — ordered columns of sleeve_returns
    pre_sds    : bool       — if True, redirect SDS alloc → SH
                              (SDS did not exist before 2006-07-11)
    """
    alloc = np.array([blend.get(col, 0.0) for col in sleeve_cols])
    if pre_sds and "SDS" in sleeve_cols and "SH" in sleeve_cols:
        sds_i = sleeve_cols.get_loc("SDS")
        sh_i  = sleeve_cols.get_loc("SH")
        alloc[sh_i]  += alloc[sds_i]
        alloc[sds_i]  = 0.0
    return alloc
 
 
def resolve_phase(date, base_phase):
    """
    Return effective phase key for blend/weight lookup.
 
    Phase 1 upgrades to "1b" when 21-day SPY momentum > PHASE1B_MOM_THRESH.
    News signal may further override (live dates only; backtest = no-op).
    """
    if base_phase == 1:
        mom = float(spy_mom_fast.get(date, 0.0))
        if mom > PHASE1B_MOM_THRESH:
            base_phase = "1b"
    if _news_signal is not None:
        return _news_signal.adjust_phase(base_phase, date)
    return base_phase
 
 
def _save_portfolio_audit(audit_df: pd.DataFrame, label: str) -> None:
    """Save the audit trail for later inspection."""
    path = f"portfolio_audit_{label}.csv"
    audit_df.to_csv(path, index=False)
    print(f"Saved portfolio audit trail to {path}")

    if _HAS_OPENPYXL:
        excel_path = f"portfolio_audit_{label}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            audit_df.to_excel(writer, sheet_name="Audit", index=False)
            phase_summary = audit_df.groupby("phase").agg(
                days=("date", "size"),
                avg_gross_exposure=("effective_gross_exposure", "mean"),
                avg_net_exposure=("effective_net_exposure", "mean"),
                avg_scale=("exposure_scale", "mean"),
                avg_phase_cap=("phase_cap", "mean"),
                cap_exceed_days=("phase_cap_exceeded", "sum"),
                avg_leverage=("phase_leverage", "mean"),
                avg_unlevered_return=("return_before_leverage", "mean"),
                avg_levered_return=("return_after_leverage", "mean"),
                avg_hedge_pnl=("hedge_pnl", "mean"),
                hedge_days=("hedge_applied", "sum"),
            )
            phase_summary.to_excel(writer, sheet_name="Phase_Summary")
            key_columns = [
                "date", "phase", "phase_action", "phase_reason",
                "phase_cap", "phase_cap_exceeded", "raw_gross_exposure",
                "raw_net_exposure", "effective_gross_exposure",
                "effective_net_exposure", "factor_allocation",
                "alloc_SPY", "alloc_QQQ", "alloc_GLD", "alloc_SH",
                "alloc_SDS", "alloc_TLT", "alloc_BTC_USD",
                "hedge_pnl", "hedge_applied"
            ]
            key_columns = [c for c in key_columns if c in audit_df.columns]
            audit_df[key_columns].to_excel(writer, sheet_name="Key_Drivers", index=False)
        print(f"Saved portfolio audit workbook to {excel_path}")
    else:
        print("openpyxl unavailable; skipping Excel audit export.")
 
 
def _print_portfolio_audit_summary(audit_df: pd.DataFrame, label: str) -> None:
    """Print a compact exposure and contribution summary for the audit trail."""
    audit_df = audit_df.copy()
    if "date" not in audit_df.columns and audit_df.index.name == "date":
        audit_df = audit_df.reset_index()
    if "date" not in audit_df.columns:
        audit_df["date"] = pd.NaT
 
    print(f"\nPORTFOLIO AUDIT SUMMARY — {label}")
    print("=" * 80)
    phase_stats = audit_df.groupby("phase").agg(
        days=("date", "size"),
        avg_gross_exposure=("effective_gross_exposure", "mean"),
        avg_net_exposure=("effective_net_exposure", "mean"),
        avg_scale=("exposure_scale", "mean"),
        avg_phase_cap=("phase_cap", "mean"),
        cap_exceed_days=("phase_cap_exceeded", "sum"),
        avg_leverage=("phase_leverage", "mean"),
        avg_unlevered_return=("return_before_leverage", "mean"),
        avg_levered_return=("return_after_leverage", "mean"),
        avg_hedge_pnl=("hedge_pnl", "mean"),
        hedge_days=("hedge_applied", "sum"),
    )
    print(phase_stats.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("=" * 80)
 
 
# ============================================================
# VECTORIZED BACKTEST HELPER
# ============================================================
# OPTIMIZATION: The original backtest was a Python for-loop:
#
#   for date in dates:                                   ← Python loop
#       f_w = phase_factor_weights[eff_phase]
#       factor_ret = np.dot(f_w, returns.loc[date])     ← row-by-row dot
#       sleeve_ret = compute_sleeve_return(date, ...)   ← more .loc calls
#       portfolio_daily.loc[date] = ...
#
# With 30% of 50 years ≈ 3,750 test dates, this means:
#   • 3,750 Python loop iterations
#   • 3,750 × N_STOCKS (500+) scalar reads from .loc
#   • 3,750 individual np.dot calls (each dispatches to BLAS)
#
# Optimized version:
#   1. Pre-compute all effective phases with NO per-date Python loops
#      (vectorized momentum upgrade + tiny loop only for live days).
#   2. Group dates by effective phase (5 groups at most).
#   3. For each group: ONE matrix multiply
#          (n_dates × n_stocks) @ (n_stocks,) → (n_dates,)
#      This is a single BLAS DGEMV call — NumPy uses all available
#      BLAS threads (OpenBLAS/MKL) on your 9950X3D automatically.
#   4. ONE vectorized sleeve return: (n_dates × n_sleeves) @ (n_sleeves,)
#
# This reduces Python-level work from O(n_dates) to O(n_phases) ≈ 5.
# BLAS matrix-vector multiply fully utilizes all CPU cores internally.
 
def _compute_portfolio_returns_vectorized(
        ret_df: pd.DataFrame,
        regime_slice: pd.Series,
        sleeve_df: pd.DataFrame,
        spy_series: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Compute daily blended portfolio returns for an entire date range
    using grouped matrix operations.

    Returns
    -------
    portfolio_daily : pd.Series of daily portfolio returns
    portfolio_audit : pd.DataFrame of per-date exposure and contribution metrics
 
    Parameters
    ----------
    ret_df       : (dates × stocks) returns DataFrame
    regime_slice : base regime label per date (aligned to ret_df.index)
    sleeve_df    : (dates × sleeve_instruments) returns DataFrame
    spy_series   : SPY daily returns (for hedge overlay)
 
    Returns
    -------
    portfolio_daily : pd.Series of daily portfolio returns
    """
    dates = ret_df.index
    n     = len(dates)
 
    audit = {
        "date": np.array(dates, dtype="datetime64[ns]"),
        "phase": np.empty(n, dtype=object),
        "factor_return": np.zeros(n, dtype=np.float64),
        "sleeve_return": np.zeros(n, dtype=np.float64),
        "factor_contrib": np.zeros(n, dtype=np.float64),
        "sleeve_contrib": np.zeros(n, dtype=np.float64),
        "exposure_scale": np.ones(n, dtype=np.float64),
        "raw_gross_exposure": np.zeros(n, dtype=np.float64),
        "raw_net_exposure": np.zeros(n, dtype=np.float64),
        "effective_gross_exposure": np.zeros(n, dtype=np.float64),
        "effective_net_exposure": np.zeros(n, dtype=np.float64),
        "phase_cap": np.ones(n, dtype=np.float64),
        "phase_cap_exceeded": np.zeros(n, dtype=bool),
        "phase_reason": np.full(n, "", dtype=object),
        "phase_action": np.full(n, "", dtype=object),
        "factor_allocation": np.zeros(n, dtype=np.float64),
        "alloc_SPY": np.zeros(n, dtype=np.float64),
        "alloc_QQQ": np.zeros(n, dtype=np.float64),
        "alloc_GLD": np.zeros(n, dtype=np.float64),
        "alloc_SH": np.zeros(n, dtype=np.float64),
        "alloc_SDS": np.zeros(n, dtype=np.float64),
        "alloc_TLT": np.zeros(n, dtype=np.float64),
        "alloc_BTC_USD": np.zeros(n, dtype=np.float64),
        "phase_leverage": np.ones(n, dtype=np.float64),
        "return_before_leverage": np.zeros(n, dtype=np.float64),
        "return_after_leverage": np.zeros(n, dtype=np.float64),
        "hedge_pnl": np.zeros(n, dtype=np.float64),
        "return_after_hedge": np.zeros(n, dtype=np.float64),
        "hedge_applied": np.zeros(n, dtype=bool),
        "spy_return": np.zeros(n, dtype=np.float64),
        "qqq_return": np.zeros(n, dtype=np.float64),
    }
 
    # ── Step 1: Vectorized phase computation ─────────────────
    # Build a pandas Series of effective phase keys (1/"1b"/2/3/4)
    # without a Python loop.
    #
    # Momentum upgrade (Phase 1 → 1b):
    #   base == 1  AND  spy_mom_fast > PHASE1B_MOM_THRESH
    # All other phases pass through unchanged.
    base_phases  = regime_slice.reindex(dates, fill_value=1).astype(int)
    mom_aligned  = spy_mom_fast.reindex(dates, fill_value=0.0)
 
    # Start with base phases, then apply the 1 → 1b upgrade rule.
    # Using np.where on arrays — no Python loop.
    upgrade_mask = (base_phases.values == 1) & (mom_aligned.values > PHASE1B_MOM_THRESH)
    eff_arr      = np.where(upgrade_mask, "1b", base_phases.astype(str))
    # Correct: "1"→1, "2"→2, "3"→3, "4"→4, "1b"→"1b"
    eff_arr      = np.where(eff_arr == "1",  "1",  eff_arr)
    eff_arr      = np.where(eff_arr == "2",  "2",  eff_arr)
    eff_arr      = np.where(eff_arr == "3",  "3",  eff_arr)
    eff_arr      = np.where(eff_arr == "4",  "4",  eff_arr)
 
    eff_phases = pd.Series(eff_arr, index=dates)
 
    # News signal adjustment — only touches at most LIVE_SIGNAL_WINDOW_DAYS
    # dates, so this mini-loop is essentially free.
    if _news_signal is not None:
        live_cutoff = pd.Timestamp.now() - pd.Timedelta(days=5)
        live_dates  = dates[dates >= live_cutoff]
        for ld in live_dates:
            eff_phases[ld] = _news_signal.adjust_phase(eff_phases[ld], ld)
 
    # ── Step 2: Pre-compute sleeve weight vectors (one per phase) ─
    sleeve_cols = sleeve_df.columns
    sleeve_leverage = np.array(
        [SLEEVE_LEVERAGE.get(col, 1.0) for col in sleeve_cols],
        dtype=np.float64,
    )
    sleeve_vecs = {}
    for ph in eff_phases.unique():
        # Normalise key type: "1b" stays as string, others become int
        ph_key = ph if ph == "1b" else int(ph)
        blend  = phase_blend[ph_key]
 
        # Handle SDS inception — pre-2006 dates use SH as proxy
        if blend.get("SDS", 0) > 0:
            # Two vectors: one for pre-inception, one for post
            sleeve_vecs[(ph, "post")] = _build_sleeve_weight_vector(
                blend, sleeve_cols, pre_sds=False)
            sleeve_vecs[(ph, "pre")]  = _build_sleeve_weight_vector(
                blend, sleeve_cols, pre_sds=True)
        else:
            sleeve_vecs[(ph, "post")] = _build_sleeve_weight_vector(
                blend, sleeve_cols, pre_sds=False)
            sleeve_vecs[(ph, "pre")]  = sleeve_vecs[(ph, "post")]
 
    # ── Step 3: Output container ──────────────────────────────
    portfolio_arr = np.zeros(n, dtype=np.float64)
 
    # ── Step 4: One matrix multiply per effective phase ───────
    for ph in eff_phases.unique():
        ph_key   = ph if ph == "1b" else int(ph)
        blend    = phase_blend[ph_key]
        f_w      = phase_factor_weights[ph_key]
        mask     = eff_phases.values == ph
        ph_dates = dates[mask]
 
        if len(ph_dates) == 0:
            continue
 
        # (n_phase_dates × n_stocks) @ (n_stocks,) → (n_phase_dates,)
        # reindex instead of .loc[] for the same defensive reason as below.
        factor_rets = ret_df.reindex(ph_dates, fill_value=0.0).values @ f_w
 
        # (n_phase_dates × n_sleeves) @ (n_sleeves,) → (n_phase_dates,)
        # Pre-2006: SDS → SH substitution
        pre_mask  = ph_dates < SDS_INCEPTION
        post_mask = ~pre_mask
 
        sv_post = sleeve_vecs[(ph, "post")]
        sv_pre  = sleeve_vecs[(ph, "pre")]
        phase_cap = LEVERAGE_CONFIG.get(ph_key, 1.0)

        # ── reindex instead of .loc[] ────────────────────────────
        # .loc[] raises KeyError for dates absent from sleeve_df
        # (calendar mismatches between the main
        # returns index and the ETF trading calendar).
        # .reindex(..., fill_value=0.0) silently returns 0 for any
        # missing date — instrument not trading = 0% return.
        sleeve_rets = np.zeros(len(ph_dates))
        factor_contrib = blend["FACTOR"] * factor_rets
        exposure_abs = np.ones(len(ph_dates), dtype=np.float64) * blend["FACTOR"]
        exposure_net = np.ones(len(ph_dates), dtype=np.float64) * blend["FACTOR"]
        raw_exposure_abs = np.ones(len(ph_dates), dtype=np.float64) * blend["FACTOR"]
        raw_exposure_net = np.ones(len(ph_dates), dtype=np.float64) * blend["FACTOR"]
        scale_arr = np.ones(len(ph_dates), dtype=np.float64)
        phase_reason = np.full(len(ph_dates), "Within cap", dtype=object)
        sleeve_positions = np.zeros((len(ph_dates), len(sleeve_cols)), dtype=np.float64)

        if post_mask.any():
            sleeve_rets[post_mask] = (
                sleeve_df.reindex(ph_dates[post_mask], fill_value=0.0).values @ sv_post
            )
            effective_post = blend["FACTOR"] + np.dot(np.abs(sv_post), np.abs(sleeve_leverage))
            raw_exposure_abs[post_mask] = effective_post
            raw_exposure_net[post_mask] = blend["FACTOR"] + np.dot(sv_post, sleeve_leverage)
            sleeve_positions[post_mask, :] = np.outer(
                np.ones(post_mask.sum(), dtype=np.float64),
                sv_post * sleeve_leverage
            )
            exposure_abs[post_mask] = effective_post
            exposure_net[post_mask] = raw_exposure_net[post_mask]
            if effective_post > phase_cap:
                scale = phase_cap / effective_post
                scale_arr[post_mask] *= scale
                sleeve_rets[post_mask] *= scale
                factor_contrib[post_mask] *= scale
                exposure_abs[post_mask] *= scale
                exposure_net[post_mask] *= scale
                sleeve_positions[post_mask, :] *= scale
                phase_reason[post_mask] = "Phase cap exceeded"

        if pre_mask.any():
            sleeve_rets[pre_mask] = (
                sleeve_df.reindex(ph_dates[pre_mask], fill_value=0.0).values @ sv_pre
            )
            effective_pre = blend["FACTOR"] + np.dot(np.abs(sv_pre), np.abs(sleeve_leverage))
            raw_exposure_abs[pre_mask] = effective_pre
            raw_exposure_net[pre_mask] = blend["FACTOR"] + np.dot(sv_pre, sleeve_leverage)
            sleeve_positions[pre_mask, :] = np.outer(
                np.ones(pre_mask.sum(), dtype=np.float64),
                sv_pre * sleeve_leverage
            )
            exposure_abs[pre_mask] = effective_pre
            exposure_net[pre_mask] = raw_exposure_net[pre_mask]
            if effective_pre > phase_cap:
                scale = phase_cap / effective_pre
                scale_arr[pre_mask] *= scale
                sleeve_rets[pre_mask] *= scale
                factor_contrib[pre_mask] *= scale
                exposure_abs[pre_mask] *= scale
                exposure_net[pre_mask] *= scale
                sleeve_positions[pre_mask, :] *= scale
                phase_reason[pre_mask] = "Phase cap exceeded"

        if _news_signal is not None:
            live_cutoff = pd.Timestamp.now() - pd.Timedelta(days=5)
            for i, d in enumerate(ph_dates):
                if d >= live_cutoff:
                    adj_blend   = _news_signal.adjust_blend(blend, d)
                    sv_adj      = _build_sleeve_weight_vector(
                        adj_blend, sleeve_cols,
                        pre_sds=(d < SDS_INCEPTION))
                    adj_sleeve_ret = sleeve_df.reindex([d], fill_value=0.0).values[0] @ sv_adj
                    adj_exposure_abs = blend["FACTOR"] + np.dot(np.abs(sv_adj), np.abs(sleeve_leverage))
                    adj_exposure_net = blend["FACTOR"] + np.dot(sv_adj, sleeve_leverage)
                    adj_scale = 1.0
                    if adj_exposure_abs > phase_cap:
                        adj_scale = phase_cap / adj_exposure_abs
                    sleeve_rets[i] = adj_sleeve_ret * adj_scale
                    factor_contrib[i] *= adj_scale
                    exposure_abs[i] = adj_exposure_abs * adj_scale
                    exposure_net[i] = adj_exposure_net * adj_scale
                    scale_arr[i] *= adj_scale
                    sleeve_positions[i, :] = (sv_adj * sleeve_leverage) * adj_scale
                    phase_reason[i] = "News-adjusted blend"
 
        return_before_leverage = factor_contrib + sleeve_rets
        portfolio_arr[mask] = return_before_leverage
        audit["phase"][mask] = ph
        audit["factor_return"][mask] = factor_rets
        audit["sleeve_return"][mask] = sleeve_rets
        audit["factor_contrib"][mask] = factor_contrib
        audit["sleeve_contrib"][mask] = sleeve_rets
        audit["exposure_scale"][mask] = scale_arr
        audit["raw_gross_exposure"][mask] = raw_exposure_abs
        audit["raw_net_exposure"][mask] = raw_exposure_net
        audit["effective_gross_exposure"][mask] = exposure_abs
        audit["effective_net_exposure"][mask] = exposure_net
        audit["phase_cap"][mask] = phase_cap
        audit["phase_cap_exceeded"][mask] = raw_exposure_abs > phase_cap
        audit["phase_reason"][mask] = phase_reason
        audit["factor_allocation"][mask] = blend["FACTOR"] * scale_arr
        audit["alloc_SPY"][mask] = sleeve_positions[:, sleeve_cols.get_loc("SPY")]
        audit["alloc_QQQ"][mask] = sleeve_positions[:, sleeve_cols.get_loc("QQQ")]
        audit["alloc_GLD"][mask] = sleeve_positions[:, sleeve_cols.get_loc("GLD")]
        audit["alloc_SH"][mask] = sleeve_positions[:, sleeve_cols.get_loc("SH")]
        audit["alloc_SDS"][mask] = sleeve_positions[:, sleeve_cols.get_loc("SDS")]
        audit["alloc_TLT"][mask] = sleeve_positions[:, sleeve_cols.get_loc("TLT")]
        audit["alloc_BTC_USD"][mask] = sleeve_positions[:, sleeve_cols.get_loc("BTC-USD")]
        audit["phase_leverage"][mask] = exposure_abs
        audit["return_before_leverage"][mask] = return_before_leverage
 
    port_series = pd.Series(portfolio_arr, index=dates)
 
    # ── Step 4b: Phase leverage is now exposure-driven.
    # The model uses the portfolio's actual gross exposure and only
    # scales back when the phase cap is exceeded.
    audit["return_after_leverage"] = audit["return_before_leverage"]
 
    # ── Step 5: Derivatives hedge overlay (batch mode) ────────
    # OPTIMIZATION: Previously apply_hedge was called inside the
    # per-date loop, which downloaded VIX on every single date.
    # The new batch interface downloads VIX once and processes
    # all dates as arrays — see DerivativesTrading.py.
    if _hedger is not None:
        spy_aligned = spy_series.reindex(dates, fill_value=0.0)

        # Precompute QQQ proxy (you didn’t pass QQQ anywhere)
        qqq_proxy = spy_aligned * 1.2

        hedged_returns = []

        for i, date in enumerate(dates):
            current_return = float(port_series.iloc[i])
            spy_ret = float(spy_aligned.iloc[i])
            qqq_ret = float(qqq_proxy.iloc[i])
            hedged_ret = _hedger.apply_hedge(
                portfolio_daily_return = current_return,
                spy_daily_return       = spy_ret,
                qqq_daily_return       = qqq_ret,
                current_regime         = int(base_phases.iloc[i]),
                date                   = date,
                live_mode              = (date >= (pd.Timestamp.now() - pd.Timedelta(days=5)))
            )
            hedged_returns.append(hedged_ret)
            audit["spy_return"][i] = spy_ret
            audit["qqq_return"][i] = qqq_ret
            audit["hedge_pnl"][i] = hedged_ret - current_return
            audit["hedge_applied"][i] = abs(audit["hedge_pnl"][i]) > 1e-12
 
        port_series = pd.Series(hedged_returns, index=dates)
        audit["return_after_hedge"] = np.array(hedged_returns, dtype=np.float64)
    else:
        audit["return_after_hedge"] = port_series.values
 
    portfolio_audit = pd.DataFrame(audit)
    portfolio_audit.index = dates
    portfolio_audit["previous_phase"] = portfolio_audit["phase"].shift(1)
    portfolio_audit["phase_action"] = np.where(
        portfolio_audit["phase"] != portfolio_audit["previous_phase"],
        portfolio_audit["previous_phase"].fillna("start").astype(str) + "->" + portfolio_audit["phase"].astype(str),
        "Hold"
    )
    portfolio_audit.loc[portfolio_audit.index[0], "phase_action"] = "Initial allocation"
    return port_series, portfolio_audit

# ============================================================
# OUT-OF-SAMPLE BACKTEST
# ============================================================
test_returns_raw  = returns[returns.index > split_date]
test_regime_slice = regime[regime.index > split_date]
test_sleeve       = sleeve_returns[sleeve_returns.index > split_date]
spy_test          = spy_px[spy_px.index > split_date].pct_change().dropna()

print("\nRunning out-of-sample backtest (vectorized matrix operations)...")
portfolio_daily, portfolio_audit = _compute_portfolio_returns_vectorized(
    ret_df       = test_returns_raw,
    regime_slice = test_regime_slice,
    sleeve_df    = test_sleeve,
    spy_series   = spy_test,
)

portfolio_daily = portfolio_daily.reindex(spy_test.index).dropna()
spy_test        = spy_test.reindex(portfolio_daily.index).dropna()
portfolio_audit = portfolio_audit.reindex(portfolio_daily.index)
portfolio_audit["date"] = portfolio_daily.index
port_cum = (1 + portfolio_daily).cumprod()
spy_cum  = (1 + spy_test).cumprod()
_print_portfolio_audit_summary(portfolio_audit, "OUT-OF-SAMPLE")
_save_portfolio_audit(portfolio_audit, "out_of_sample")

print("=" * 80)
print(f"{'Phase':<24} {'Factor':>8} {'SPY':>6} {'QQQ':>6} "
      f"{'GLD':>6} {'SH':>6} {'SDS':>6} {'TLT':>6} {'BTC':>6}")
print("=" * 80)
blend_display_map = [
    ("1   (Buildout)",  phase_blend[1]),
    ("1b  (Accel)",     phase_blend["1b"]),
    ("2   (Narrative)", phase_blend[2]),
    ("3   (Unwind)",    phase_blend[3]),
    ("4   (Reset)",     phase_blend[4]),
]
for label_str, blend in blend_display_map:
    print(f"  Phase {label_str:<20} "
          f"{blend['FACTOR']*100:>7.0f}% "
          f"{blend['SPY']*100:>5.0f}% "
          f"{blend['QQQ']*100:>5.0f}% "
          f"{blend['GLD']*100:>5.0f}% "
          f"{blend['SH']*100:>5.0f}% "
          f"{blend.get('SDS', 0)*100:>5.0f}% "
          f"{blend.get('TLT', 0)*100:>5.0f}%"
          f"{blend.get('BTC-USD', 0)*100:>5.0f}%")
print("=" * 80)

print("\nEFFECTIVE EXPOSURE SUMMARY")
print("=" * 80)
print(f"{'Phase':<24} {'Factor':>8} {'Gross':>8} {'Net':>8}")
print("=" * 80)
for label_str, blend in blend_display_map:
    factor_exposure = blend['FACTOR']
    gross_sleeve = sum(abs(blend.get(inst, 0.0)) * abs(SLEEVE_LEVERAGE.get(inst, 1.0))
                       for inst in SLEEVE_INSTRUMENTS)
    net_sleeve   = sum(blend.get(inst, 0.0) * SLEEVE_LEVERAGE.get(inst, 1.0)
                       for inst in SLEEVE_INSTRUMENTS)
    total_gross  = abs(factor_exposure) + gross_sleeve
    total_net    = factor_exposure + net_sleeve
    print(f"  Phase {label_str:<20} "
          f"{factor_exposure:>8.2f} "
          f"{total_gross:>8.2f} "
          f"{total_net:>8.2f}")
print("=" * 80)

# ============================================================
# SECTION 10 — SCORECARD (out-of-sample)
# ============================================================
print("\n" + "=" * 60)
print("OUT-OF-SAMPLE PERFORMANCE (last 50% of data)")
print("=" * 60)
print(f"{'Metric':<25} {'Portfolio':>12}  {'SPY B&H':>10}")
print("=" * 60)
print(f"{'Total Return':<25} "
      f"{(port_cum.iloc[-1]-1)*100:>11.2f}%  "
      f"{(spy_cum.iloc[-1]-1)*100:>9.2f}%")
print(f"{'Sharpe Ratio':<25} "
      f"{sharpe(portfolio_daily):>12.3f}  "
      f"{sharpe(spy_test):>10.3f}")
print(f"{'Sortino Ratio':<25} "
      f"{sortino(portfolio_daily):>12.3f}  "
      f"{sortino(spy_test):>10.3f}")
print(f"{'Max Drawdown':<25} "
      f"{max_drawdown(port_cum)*100:>11.2f}%  "
      f"{max_drawdown(spy_cum)*100:>9.2f}%")
print("=" * 60)
 
test_df         = pd.DataFrame({"Portfolio": portfolio_daily, "SPY": spy_test})
test_df["Year"] = test_df.index.year
annual          = test_df.groupby("Year").apply(
    lambda x: pd.Series({
        "Portfolio": (1 + x["Portfolio"]).prod() - 1,
        "SPY":       (1 + x["SPY"]).prod() - 1,
    })
)
print("\nANNUAL RETURNS — OUT OF SAMPLE")
print("=" * 40)
print(f"{'Year':<8} {'Portfolio':>12} {'SPY':>10}")
print("=" * 40)
for year, row in annual.iterrows():
    print(f"{year:<8} "
          f"{row['Portfolio']*100:>11.2f}%"
          f"{row['SPY']*100:>10.2f}%")
print("=" * 40)
 
# ============================================================
# HOLDOUT VALIDATION (2024-present — vectorized)
# ============================================================
holdout_start   = pd.Timestamp("2024-01-01")
holdout_returns = returns[returns.index >= holdout_start]
holdout_regime  = regime[regime.index  >= holdout_start]
holdout_sleeve  = sleeve_returns[sleeve_returns.index >= holdout_start]
spy_holdout     = spy_px[spy_px.index >= holdout_start].pct_change().dropna()
 
print("\nRunning holdout validation (vectorized)...")
holdout_daily, holdout_audit = _compute_portfolio_returns_vectorized(
    ret_df       = holdout_returns,
    regime_slice = holdout_regime,
    sleeve_df    = holdout_sleeve,
    spy_series   = spy_holdout,
)
 
holdout_daily = holdout_daily.reindex(spy_holdout.index).dropna()
spy_holdout   = spy_holdout.reindex(holdout_daily.index).dropna()
holdout_audit = holdout_audit.reindex(holdout_daily.index)
holdout_audit["date"] = holdout_daily.index
holdout_cum   = (1 + holdout_daily).cumprod()
spy_h_cum     = (1 + spy_holdout).cumprod()
_print_portfolio_audit_summary(holdout_audit, "HOLDOUT")
_save_portfolio_audit(holdout_audit, "holdout")
 
print("\n" + "=" * 60)
print("HOLDOUT VALIDATION (2024 — present, never tuned on)")
print("=" * 60)
print(f"{'Metric':<25} {'Portfolio':>12}  {'SPY B&H':>10}")
print("=" * 60)
print(f"{'Total Return':<25} "
      f"{(holdout_cum.iloc[-1]-1)*100:>11.2f}%  "
      f"{(spy_h_cum.iloc[-1]-1)*100:>9.2f}%")
print(f"{'Sharpe Ratio':<25} "
      f"{sharpe(holdout_daily):>12.3f}  "
      f"{sharpe(spy_holdout):>10.3f}")
print(f"{'Sortino Ratio':<25} "
      f"{sortino(holdout_daily):>12.3f}  "
      f"{sortino(spy_holdout):>10.3f}")
print(f"{'Max Drawdown':<25} "
      f"{max_drawdown(holdout_cum)*100:>11.2f}%  "
      f"{max_drawdown(spy_h_cum)*100:>9.2f}%")
print("=" * 60)
 
holdout_df         = pd.DataFrame({"Portfolio": holdout_daily, "SPY": spy_holdout})
holdout_df["Year"] = holdout_df.index.year
holdout_annual     = holdout_df.groupby("Year").apply(
    lambda x: pd.Series({
        "Portfolio": (1 + x["Portfolio"]).prod() - 1,
        "SPY":       (1 + x["SPY"]).prod() - 1,
    })
)
 
print("\nHOLDOUT ANNUAL RETURNS")
print("=" * 40)
print(f"{'Year':<8} {'Portfolio':>12} {'SPY':>10}")
print("=" * 40)
for year, row in holdout_annual.iterrows():
    print(f"{year:<8} "
          f"{row['Portfolio']*100:>11.2f}%"
          f"{row['SPY']*100:>10.2f}%")
print("=" * 40)
 
# ============================================================
# SECTION 11 — ALL HOLDINGS BY PHASE
# ============================================================

def print_holdings(phase_key, stock_weights, ticker_list, min_weight_threshold=0.001):
    """
    Print all positions held in the portfolio for a given phase.
    
    Parameters
    ----------
    phase_key : int or str — phase identifier (1, 1b, 2, 3, 4)
    stock_weights : np.ndarray — weight for each stock
    ticker_list : np.ndarray — ticker symbols
    min_weight_threshold : float — only show positions above this threshold (default 0.1%)
    """
    label_str = "Momentum Accel" if phase_key == "1b" else labels[phase_key]
    
    # Sort all positions by weight (descending)
    sorted_idx = np.argsort(stock_weights)[::-1]
    
    # Filter for positions above minimum threshold
    significant_idx = sorted_idx[stock_weights[sorted_idx] >= min_weight_threshold]
    
    top_tickers = ticker_list[significant_idx]
    top_weights = stock_weights[significant_idx]
    num_positions = len(top_tickers)
    
    print(f"\n{'='*60}")
    print(f"ALL HOLDINGS — Phase {phase_key} ({label_str}) Portfolio")
    print(f"Total Positions: {num_positions} (weight ≥ {min_weight_threshold*100:.3f}%)")
    print(f"{'='*60}")
    print(f"{'Ticker':<10} {'Weight':>12} {'Cumulative %':>15}")
    print(f"{'-'*60}")
    
    cumulative = 0.0
    for ticker, weight in zip(top_tickers, top_weights):
        cumulative += weight
        print(f"{ticker:<10} {weight*100:>11.3f}% {cumulative*100:>14.2f}%")
    
    print(f"{'-'*60}")
    print(f"{'TOTAL':<10} {cumulative*100:>11.2f}%")
    print(f"{'='*60}")

phase_weights_display = {
    1: stock_w_growth, "1b": stock_w_momentum,
    2: stock_w_momentum, 3: stock_w_defensive, 4: stock_w_recovery,
}

for phase_key in [1, "1b", 2, 3, 4]:
    print_holdings(phase_key, phase_weights_display[phase_key],
                   returns.columns.values, min_weight_threshold=0.001)

current_phase = int(regime.iloc[-1])
eff_now       = resolve_phase(returns.index[-1], current_phase)
eff_label     = "Momentum Accel" if eff_now == "1b" else labels[current_phase]
print(f"\n>>> CURRENTLY ACTIVE: Phase {eff_now} ({eff_label}) <<<")

if _hedger is not None:
    today = pd.Timestamp.today().normalize()
    _hedger.apply_hedge(0.0, 0.0, current_phase, date=today, live_mode=True)
# ============================================================
# SECTION 12 — CHARTS
# ============================================================
# Detect live/headless mode early so plotting can be suppressed.
# Accept `--live` or a bare `live` token (e.g. `python FAMWithAIA.py -- live`).
is_live_mode = any(a.lower() in ("--live", "live") for a in sys.argv[1:])
if is_live_mode:
    import matplotlib
    matplotlib.use("Agg")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Spatial-Temporal Portfolio Model — v4 (Parallelized)", fontsize=14)
 
regime_colors = {1: "green", 2: "yellow", 3: "red", 4: "orange"}

# Create a single portfolio line with colors that change based on regime
# Align portfolio cumulative returns and regime to same index
port_cum_aligned = port_cum.reindex(regime.index).dropna()
regime_aligned = regime.reindex(port_cum_aligned.index)

if len(port_cum_aligned) > 1:
    # Convert datetime index to matplotlib's date format
    x_dates = date2num(port_cum_aligned.index)
    y_values = port_cum_aligned.values
    
    # Create line segments for LineCollection
    points = np.array([x_dates, y_values]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Assign colors to each segment based on regime
    colors = [regime_colors[int(regime_aligned.iloc[i])] for i in range(len(segments))]
    
    # Create and add LineCollection
    lc = LineCollection(segments, colors=colors, linewidths=1.5)
    axes[0, 0].add_collection(lc)
    axes[0, 0].autoscale()

axes[0, 0].plot(spy_cum.index,  spy_cum.values,
                label="SPY B&H",  color="cyan",    linewidth=1.5)
axes[0, 0].set_title("Cumulative Returns (Out of Sample)")
axes[0, 0].set_ylabel("Growth of $1")

# Add custom legend entry for portfolio line
legend_elements = [Line2D([0], [0], color="white", linewidth=1.5, label="Portfolio (colored by regime)"),
                   Line2D([0], [0], color="cyan", linewidth=1.5, label="SPY B&H")]
axes[0, 0].legend(handles=legend_elements, loc="upper left")

for phase in [1, 2, 3, 4]:
    mask = regime == phase
    axes[0, 1].scatter(
        regime.index[mask], spy_px.reindex(regime.index)[mask],
        c=regime_colors[phase], s=1, label=labels[phase]
    )
axes[0, 1].set_title("Spatial-Temporal Regime Map (smoothed)")
axes[0, 1].set_ylabel("SPY Price")
axes[0, 1].legend(markerscale=5)
 
axes[1, 0].plot(sim_prices[:, :50], alpha=0.1, color="red", linewidth=0.8)
axes[1, 0].plot(sim_prices.mean(axis=1), color="white",
                linewidth=2, label="Mean path")
axes[1, 0].set_title(
    f"Bates Model — Phase {current_regime} ({labels[current_regime]})")
axes[1, 0].set_ylabel("Price")
axes[1, 0].legend()
 
axes[1, 1].bar(range(1, N_FACTORS + 1), explained * 100,
               color="orange", alpha=0.7)
axes[1, 1].plot(range(1, N_FACTORS + 1), cumulative * 100,
                color="white", marker="o", linewidth=1.5, label="Cumulative")
axes[1, 1].set_title("PCA Variance Explained")
axes[1, 1].set_xlabel("Factor")
axes[1, 1].set_ylabel("Variance Explained (%)")
axes[1, 1].legend()
 
plt.tight_layout()
if not is_live_mode:
    plt.show()
else:
    plt.close("all")


# ============================================================
# LIVE / PAPER-TRADING HELPERS (IBKR TWS via ib_insync)
# ============================================================

def _ib_connect(host: str = "127.0.0.1", port: int = 7497, client_id: int = 101,
                live_data: bool = False, timeout: int = 10, max_attempts: int = 5) -> IB:
    ib = IB()
    market_data_type = 1 if live_data else 3
    for attempt in range(1, max_attempts + 1):
        try:
            ib.connect(host, port, clientId=client_id, timeout=timeout)
            # Request delayed market data type when possible (3 = delayed-frozen).
            try:
                ib.reqMarketDataType(market_data_type)
            except Exception as _md_err:
                print(f"[IB] reqMarketDataType warning: {_md_err}")
            print(f"[IB] Connected to {host}:{port} (clientId={client_id}) using market data type {market_data_type}")
            return ib
        except Exception as e:
            print(f"[IB] Connect attempt {attempt}/{max_attempts} failed: {e}")
            time.sleep(3)
    raise RuntimeError("[IB] Failed to connect to TWS/Gateway after retries")


def _get_account_net_liq_usd(ib: IB) -> float | None:
    try:
        for av in ib.accountValues():
            if getattr(av, "tag", None) == "NetLiquidation" and getattr(av, "currency", None) == "USD":
                try:
                    return float(av.value)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        for s in ib.accountSummary():
            if getattr(s, "tag", None) == "NetLiquidation" and getattr(s, "currency", None) == "USD":
                try:
                    return float(s.value)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _positions_dict(ib: IB) -> dict:
    out = {}
    try:
        for p in ib.positions():
            sym = getattr(p.contract, "symbol", None)
            if sym is None:
                continue
            out[sym] = out.get(sym, 0) + int(p.position)
    except Exception as e:
        print(f"[IB] positions() failed: {e}")
    return out


def _parse_int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        m = re.search(r"\d+", raw)
        if m:
            print(f"[Live] Parsed numeric {key}={m.group(0)} from '{raw}'")
            return int(m.group(0))
        print(f"[Live] Env var {key}='{raw}' invalid, falling back to {default}")
        return default


def _eastern_now() -> datetime.datetime:
    return datetime.datetime.utcnow() - datetime.timedelta(hours=4)


def _build_exit_checker() -> callable:
    try:
        import msvcrt

        def _check():
            try:
                if msvcrt.kbhit():
                    return msvcrt.getwch().lower() == "q"
            except Exception:
                return False
            return False

        return _check
    except Exception:
        try:
            import sys
            import select

            def _check():
                try:
                    dr, _, _ = select.select([sys.stdin], [], [], 0)
                    if dr:
                        return sys.stdin.read(1).lower() == "q"
                except Exception:
                    return False
                return False

            return _check
        except Exception:
            return lambda: os.path.exists("STOP_LIVE")


def _build_tradable_blend(blend: dict) -> dict:
    tradable_allowed = {"SPY", "QQQ", "QQQ", "GLD", "TLT", "SH", "SDS", "IWM", "DIA"}
    weights = {}
    for symbol, value in blend.items():
        if symbol == "FACTOR":
            weights["SPY"] = weights.get("SPY", 0.0) + float(value)
        elif symbol in tradable_allowed:
            weights[symbol] = weights.get(symbol, 0.0) + float(value)
        else:
            print(f"[Live] Skipping non-tradable sleeve: {symbol}")
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in weights.items()}


def _rebalance_portfolio(ib: IB, blend: dict, total_value: float,
                         tolerance: float = 0.001) -> None:
    target = _build_tradable_blend(blend)
    if not target:
        print("[Live] No tradable allocation after blend normalization.")
        return

    prices = _get_market_prices(ib, list(target.keys()))
    positions = _positions_dict(ib)

    orders = []
    for symbol, weight in target.items():
        price = prices.get(symbol)
        if price is None:
            print(f"[Live] No valid price for {symbol}, skipping.")
            continue

        target_value = float(total_value) * float(weight)
        desired_qty = int(round(target_value / price))
        current_qty = int(round(float(positions.get(symbol, 0))))
        delta = desired_qty - current_qty
        if abs(delta) * price < max(50.0, total_value * tolerance):
            print(f"[Live] {symbol}: delta {delta} shares below tolerance, skipping.")
            continue
        orders.append((symbol, delta, price))

    if not orders:
        print("[Live] No orders to place after tolerance check.")
        return

    for symbol, delta, price in orders:
        _place_market_order(ib, symbol, delta)


def _schedule_next_rebalances(today: datetime.date) -> set:
    return {(9, 35), (11, 30), (13, 30), (15, 45)}


def _get_market_prices(ib: IB, symbols: list[str]) -> dict:
    prices = {s: None for s in symbols}
    contracts = [Stock(s, "SMART", "USD") for s in symbols]

    try:
        tickers = ib.reqTickers(*contracts)
    except Exception as e:
        print(f"[IB] reqTickers failed: {e} — falling back to yfinance for prices")
        tickers = []

    # Map symbol -> ticker for quick lookup
    ticker_map = {}
    for t in tickers:
        try:
            sym = getattr(t.contract, "symbol", None)
            if sym:
                ticker_map[sym] = t
        except Exception:
            continue

    for s in symbols:
        price = None
        t = ticker_map.get(s)
        if t is not None:
            # Prefer live last trade if available
            try:
                if getattr(t, "last", None) is not None and float(t.last) > 0:
                    price = float(t.last)
            except Exception:
                price = None

            # If no live last, prefer IB delayed fields (delayedLast / delayedBid/delayedAsk / delayedClose)
            if price is None:
                try:
                    dlast = getattr(t, "delayedLast", None)
                    if dlast is not None and float(dlast) > 0:
                        price = float(dlast)
                        print(f"[IB] Using delayedLast for {s}: {price}")
                except Exception:
                    price = None

            if price is None:
                try:
                    dbid = getattr(t, "delayedBid", None)
                    dask = getattr(t, "delayedAsk", None)
                    if dbid is not None and dask is not None and float(dbid) > 0 and float(dask) > 0:
                        price = float((float(dbid) + float(dask)) / 2.0)
                        print(f"[IB] Using delayed Bid/Ask midpoint for {s}: {price}")
                except Exception:
                    price = None

            # fallback to delayed close if present
            if price is None:
                try:
                    dclose = getattr(t, "delayedClose", None)
                    if dclose is not None and float(dclose) > 0:
                        price = float(dclose)
                        print(f"[IB] Using delayedClose for {s}: {price}")
                except Exception:
                    price = None

            # If no delayed fields, try live Bid/Ask midpoint
            if price is None:
                try:
                    bid = getattr(t, "bid", None)
                    ask = getattr(t, "ask", None)
                    if bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0:
                        price = float((float(bid) + float(ask)) / 2.0)
                except Exception:
                    price = None

            # Last resort: ticker.close or contract close
            if price is None:
                try:
                    close = getattr(t, "close", None)
                    if close is not None and float(close) > 0:
                        price = float(close)
                except Exception:
                    price = None

        # Fallback: use yfinance last close when IB data is missing or invalid
        if price is None or not np.isfinite(price) or price <= 0:
            try:
                df = yf.download(s, period="5d", interval="1d", progress=False)
                if isinstance(df, pd.DataFrame) and "Close" in df.columns:
                    last_close = df["Close"].dropna().iloc[-1]
                    price = float(last_close)
            except Exception:
                price = None

        if price is None or not np.isfinite(price) or price <= 0:
            prices[s] = None
        else:
            prices[s] = float(price)

    # If all IB prices are missing, warn the user that market-data subscription
    # may be required; this is informative only — fallback to yfinance is used.
    if tickers and all(prices[s] is None for s in symbols):
        print("[IB] Market data appears delayed or unavailable for requested symbols; using yfinance fallback prices.")

    return prices


def _place_market_order(ib: IB, symbol: str, qty: int) -> None:
    if qty == 0:
        return
    contract = Stock(symbol, "SMART", "USD")
    side = "BUY" if qty > 0 else "SELL"
    order = MarketOrder(side, abs(int(qty)))
    try:
        trade = ib.placeOrder(contract, order)
        print(f"[IB] Placed {side} {abs(int(qty))} {symbol} (tradeId={getattr(trade, 'orderId', 'n/a')})")
    except Exception as e:
        print(f"[IB] Failed to place order for {symbol}: {e}")


def _rebalance_sleeves_to_blend(ib: IB, blend: dict, total_value: float, tolerance: float = 0.0005) -> None:
    # We map "FACTOR" to SPY as a tradable proxy for the factor sleeve.
    # Only attempt to trade common ETFs; skip crypto or exotic tickers.
    tradable_allowed = {"SPY", "QQQ", "GLD", "SH", "SDS", "TLT", "QQQ", "IWM", "DIA"}
    target = {}
    for k, v in blend.items():
        if k == "FACTOR":
            target["SPY"] = target.get("SPY", 0.0) + float(v)
        else:
            if k in tradable_allowed:
                target[k] = target.get(k, 0.0) + float(v)
            else:
                print(f"[Live] Skipping non-tradable or unsupported sleeve: {k}")

    if not target:
        print("[Live] No tradable target allocations found in blend — aborting rebalance.")
        return

    symbols = list(target.keys())
    prices = _get_market_prices(ib, symbols)
    pos = _positions_dict(ib)

    orders = []
    for sym in symbols:
        raw_price = prices.get(sym)
        try:
            price = float(raw_price) if raw_price is not None else None
        except Exception:
            price = None

        if price is None or not np.isfinite(price) or price <= 0:
            print(f"[Live] No valid price available for {sym} (raw={raw_price}), skipping")
            continue

        weight = float(target.get(sym, 0.0))
        desired_value = float(total_value) * weight

        # Avoid ZeroDivisionError / NaN rounding errors by guarding the division
        try:
            desired_shares = int(round(desired_value / price))
        except Exception:
            print(f"[Live] Failed to compute desired shares for {sym} (price={price}), skipping")
            continue

        try:
            current_shares = int(round(float(pos.get(sym, 0))))
        except Exception:
            current_shares = int(pos.get(sym, 0) or 0)

        delta = desired_shares - current_shares
        if abs(delta) * price / (total_value + 1e-12) < tolerance:
            print(f"[Live] {sym}: delta below tolerance — no trade (delta_shares={delta})")
            continue
        orders.append((sym, delta, price))

    if not orders:
        print("[Live] No orders required after tolerance check.")
        return

    for sym, delta, price in orders:
        _place_market_order(ib, sym, delta)


def run_live_paper_trading(rebalance_interval: int = 60, live_data: bool = False,
                           tolerance: float = 0.001) -> None:
    host = os.environ.get("IB_HOST", "127.0.0.1")
    # parse port with safe fallback
    try:
        port = int(os.environ.get("IB_PORT", "7497"))
    except Exception:
        print(f"[Live] IB_PORT env var '{os.environ.get('IB_PORT')}' invalid — using 7497")
        port = 7497
    # parse client id robustly; fall back to default 101 if not numeric
    client_id_env = os.environ.get("IB_CLIENT_ID", "DUQ210493")
    if client_id_env is None or client_id_env == "":
        client_id = 101
    else:
        try:
            client_id = int(client_id_env)
        except Exception:
            import re
            m = re.search(r"\d+", client_id_env)
            if m:
                client_id = int(m.group(0))
                print(f"[Live] Parsed numeric client id {client_id} from IB_CLIENT_ID '{client_id_env}'")
            else:
                client_id = 101
                print(f"[Live] IB_CLIENT_ID '{client_id_env}' not numeric — falling back to default client id {client_id}")

    ib = _ib_connect(host=host, port=port, client_id=client_id, live_data=live_data)

    try:
        net_liq = _get_account_net_liq_usd(ib) or 100000.0
        print(f"[Live] Using account net liquidation (USD): {net_liq:,.2f}")

        # Determine current effective phase and blend
        try:
            current_phase_live = int(regime.iloc[-1])
        except Exception:
            current_phase_live = int(current_regime)
        eff_now_live = resolve_phase(returns.index[-1], current_phase_live)
        ph_key_live = eff_now_live if eff_now_live == "1b" else int(eff_now_live)
        current_blend = phase_blend[ph_key_live]
        print(f"[Live] Current phase: {eff_now_live} — applying blend: {current_blend}")

        _rebalance_portfolio(ib, current_blend, float(net_liq), tolerance=tolerance)

        last_rebalance_date = pd.Timestamp.now().normalize()

        print("[Live] Entering headless loop — press 'q' to exit (Ctrl+C will be ignored).")
        exit_pressed = _build_exit_checker()

        try:
            while True:
                # Sleep in 1s increments so exit key is responsive
                for _ in range(max(1, int(rebalance_interval))):
                    time.sleep(1)
                    if exit_pressed():
                        print("[Live] Exit key detected — shutting down live loop.")
                        raise SystemExit

                now_date = pd.Timestamp.now().normalize()
                if now_date > last_rebalance_date:
                    try:
                        net_liq = _get_account_net_liq_usd(ib) or net_liq
                        try:
                            current_phase_live = int(regime.iloc[-1])
                        except Exception:
                            current_phase_live = int(current_regime)
                        eff_now_live = resolve_phase(returns.index[-1], current_phase_live)
                        ph_key_live = eff_now_live if eff_now_live == "1b" else int(eff_now_live)
                        current_blend = phase_blend[ph_key_live]
                        print(f"[Live] Daily rebalance — date {now_date.date()} phase {eff_now_live}")
                        _rebalance_portfolio(ib, current_blend, float(net_liq), tolerance=tolerance)
                        last_rebalance_date = now_date
                    except Exception as e:
                        print(f"[Live] Rebalance iteration failed: {e}")
        except SystemExit:
            print("[Live] Live loop exited by user request.")
        except KeyboardInterrupt:
            print("[Live] Ctrl+C detected — ignored. Use 'Ctrl + Q' to exit the live loop.")
        except Exception as e:
            print(f"[Live] Live loop terminated due to error: {e}")
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


# ============================================================
# FUND MANAGER CLASSES MERGED FROM FundManager.py
# ============================================================

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("FundManager")

IB_HOST = os.environ.get("IB_HOST", "127.0.0.1")
IB_PORT = int(os.environ.get("IB_PORT", "7497"))
IB_CLIENT_ID = _parse_int_env("IB_CLIENT_ID", 101)
IB_MARKET_DATA_TYPE = int(os.environ.get("IB_MKT_DATA_TYPE", "3"))

TRADABLE_UNIVERSE = {
    "FACTOR": "SPY",
    "SPY": "SPY",
    "QQQ": "QQQ",
    "GLD": "GLD",
    "TLT": "TLT",
    "SH": "SH",
    "SDS": "SDS",
    "TQQQ": "TQQQ",
    "IWM": "IWM",
    "DIA": "DIA",
    "BTC-USD": None,
}

DAILY_REBALANCE_TIMES = [
    (9, 35),
    (11, 30),
    (13, 30),
    (15, 45),
]

SIGNAL_CHECK_INTERVAL_SEC = 300
VWAP_MOMENTUM_THRESHOLD = 0.003
VWAP_MOMENTUM_THRESHOLD_STRONG = 0.008
MEAN_REVERSION_ZSCORE = 2.0
SIGNAL_CONFIDENCE_GATE = 0.55
SIGNAL_MAX_TRADE_PCT = 0.03

AI_AGENT_INTERVAL_SEC = 1800
AI_MODEL = "claude-sonnet-4-5-20251101"
AI_MAX_TOKENS = 800
AI_APPROVAL_REQUIRED = False

RISK_CHECK_INTERVAL_SEC = 60
MAX_DAILY_LOSS_PCT = 0.03
MAX_SINGLE_POSITION_PCT = 0.50
MIN_ORDER_VALUE = 50.0
ORDER_TOLERANCE_PCT = 0.001
LIMIT_ORDER_OFFSET_PCT = 0.0005


@dataclass
class MarketSnapshot:
    symbol: str
    last_price: float
    vwap: float
    open_price: float
    high_price: float
    low_price: float
    return_today: float
    return_1h: float
    rsi_14: float
    zscore_30m: float
    volume_ratio: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class StrategySignal:
    symbol: str
    direction: int
    confidence: float
    source: str
    size_pct: float
    reasoning: str = ""


@dataclass
class TradeRecord:
    timestamp: str
    symbol: str
    side: str
    qty: int
    price: float
    value: float
    source: str
    reasoning: str


@dataclass
class FundState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    net_liq: float = 100_000.0
    positions: dict = field(default_factory=dict)
    daily_pnl_pct: float = 0.0
    current_regime: int = 1
    current_phase: str = "1"
    current_blend: dict = field(default_factory=dict)
    regime_changed: bool = False
    latest_signals: list = field(default_factory=list)
    ai_conviction: float = 0.0
    ai_reasoning: str = ""
    circuit_breaker: bool = False
    trading_halted: bool = False
    rebalances_today: int = 0
    trades_today: int = 0
    session_date: datetime.date = field(default_factory=lambda: datetime.date.today())
    trade_log: list = field(default_factory=list)


class IBManager:
    def __init__(self, state: FundState):
        self.state = state
        self.ib = IB()
        self._lock = threading.Lock()

    def connect(self, max_attempts: int = 5) -> bool:
        for attempt in range(1, max_attempts + 1):
            try:
                self.ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID, timeout=15)
                self.ib.reqMarketDataType(IB_MARKET_DATA_TYPE)
                log.info(f"[IB] Connected to {IB_HOST}:{IB_PORT} (clientId={IB_CLIENT_ID})")
                log.info(f"[IB] Market data type set to {IB_MARKET_DATA_TYPE} ({'live' if IB_MARKET_DATA_TYPE == 1 else 'delayed (free)'})")
                return True
            except Exception as e:
                log.warning(f"[IB] Connect attempt {attempt}/{max_attempts}: {e}")
                time.sleep(5)
        log.error("[IB] Failed to connect after all attempts.")
        return False

    def ensure_connected(self) -> bool:
        if self.ib and self.ib.isConnected():
            return True
        log.warning("[IB] Connection lost — attempting reconnect.")
        return self.connect(max_attempts=3)

    def get_account_nlv(self) -> float:
        if not self.ensure_connected():
            return self.state.net_liq
        for av in (self.ib.accountValues() or []):
            if getattr(av, "tag", "") == "NetLiquidation" and getattr(av, "currency", "") == "USD":
                try:
                    return float(av.value)
                except Exception:
                    pass
        for s in (self.ib.accountSummary() or []):
            if getattr(s, "tag", "") == "NetLiquidation" and getattr(s, "currency", "") == "USD":
                try:
                    return float(s.value)
                except Exception:
                    pass
        return self.state.net_liq

    def get_positions(self) -> dict:
        pos = {}
        try:
            for p in (self.ib.positions() or []):
                sym = getattr(p.contract, "symbol", None)
                if sym:
                    pos[sym] = pos.get(sym, 0) + int(p.position)
        except Exception as e:
            log.warning(f"[IB] get_positions failed: {e}")
        return pos

    def get_prices(self, symbols: list) -> dict:
        if not self.ensure_connected():
            return self._prices_from_yfinance(symbols)
        prices = {s: None for s in symbols}
        contracts = [Stock(s, "SMART", "USD") for s in symbols if TRADABLE_UNIVERSE.get(s, s) is not None]
        try:
            tickers = self.ib.reqTickers(*contracts)
        except Exception as e:
            log.warning(f"[IB] reqTickers failed: {e} — using yfinance")
            return self._prices_from_yfinance(symbols)
        ticker_map = {}
        for t in tickers:
            sym = getattr(t.contract, "symbol", None)
            if sym:
                ticker_map[sym] = t
        for s in symbols:
            t = ticker_map.get(s)
            if t is None:
                continue
            for attr in ("last", "delayedLast", "close", "delayedClose"):
                try:
                    v = getattr(t, attr, None)
                    if v is not None and float(v) > 0:
                        prices[s] = float(v)
                        break
                except Exception:
                    continue
            if prices[s] is None:
                for bid_attr, ask_attr in [("delayedBid", "delayedAsk"), ("bid", "ask")]:
                    try:
                        bid = getattr(t, bid_attr, None)
                        ask = getattr(t, ask_attr, None)
                        if bid and ask and float(bid) > 0 and float(ask) > 0:
                            prices[s] = (float(bid) + float(ask)) / 2.0
                            break
                    except Exception:
                        continue
        missing = [s for s in symbols if prices[s] is None]
        if missing:
            yf_prices = self._prices_from_yfinance(missing)
            for s in missing:
                prices[s] = yf_prices.get(s)
        still_missing = [s for s in symbols if prices[s] is None]
        if still_missing:
            log.warning(f"[Prices] No price available for: {still_missing}")
        return prices

    def _prices_from_yfinance(self, symbols: list) -> dict:
        prices = {}
        for s in symbols:
            try:
                df = yf.download(s, period="5d", interval="1d", progress=False, auto_adjust=True)
                if not df.empty and "Close" in df.columns:
                    prices[s] = float(df["Close"].dropna().iloc[-1])
                else:
                    prices[s] = None
            except Exception:
                prices[s] = None
        return prices

    def get_intraday_bars(self, symbol: str, duration: str = "1 D", bar_size: str = "5 mins") -> pd.DataFrame:
        if not self.ensure_connected():
            return self._intraday_from_yfinance(symbol)
        try:
            contract = Stock(symbol, "SMART", "USD")
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
            if bars:
                df = util.df(bars)
                df.columns = [c.lower() for c in df.columns]
                df.index = pd.to_datetime(df["date"])
                return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            log.debug(f"[IB] get_intraday_bars({symbol}): {e}")
        return self._intraday_from_yfinance(symbol)

    def _intraday_from_yfinance(self, symbol: str) -> pd.DataFrame:
        try:
            df = yf.download(symbol, period="2d", interval="5m", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [c.lower() for c in df.columns]
            return df[["open", "high", "low", "close", "volume"]].dropna()
        except Exception:
            return pd.DataFrame()

    def place_limit_order(self, symbol: str, qty: int, price: float) -> bool:
        if not self.ensure_connected():
            return False
        if qty == 0:
            return True
        side = "BUY" if qty > 0 else "SELL"
        limit_px = round(price * (1 + LIMIT_ORDER_OFFSET_PCT), 2) if qty > 0 else round(price * (1 - LIMIT_ORDER_OFFSET_PCT), 2)
        try:
            contract = Stock(symbol, "SMART", "USD")
            order = LimitOrder(side, abs(qty), limit_px)
            trade = self.ib.placeOrder(contract, order)
            log.info(f"[Order] LIMIT {side} {abs(qty)} {symbol} @ ${limit_px:.2f} (orderId={getattr(trade, 'orderId', 'n/a')})")
            return True
        except Exception as e:
            log.error(f"[Order] Limit order failed {symbol}: {e}")
            return False

    def place_market_order(self, symbol: str, qty: int) -> bool:
        if not self.ensure_connected():
            return False
        if qty == 0:
            return True
        side = "BUY" if qty > 0 else "SELL"
        try:
            contract = Stock(symbol, "SMART", "USD")
            order = MarketOrder(side, abs(qty))
            trade = self.ib.placeOrder(contract, order)
            log.info(f"[Order] MARKET {side} {abs(qty)} {symbol} (orderId={getattr(trade, 'orderId', 'n/a')})")
            return True
        except Exception as e:
            log.error(f"[Order] Market order failed {symbol}: {e}")
            return False


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff().dropna()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_g = gains.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avg_l = losses.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    rs = avg_g / (avg_l + 1e-10)
    return float(100 - 100 / (1 + rs))


def build_market_snapshot(symbol: str, ibm: IBManager) -> Optional[MarketSnapshot]:
    bars = ibm.get_intraday_bars(symbol, duration="1 D", bar_size="5 mins")
    if bars.empty or len(bars) < 5:
        return None
    close = bars["close"]
    vol = bars["volume"]
    typical_px = (bars["high"] + bars["low"] + bars["close"]) / 3
    vwap = float((typical_px * vol).sum() / (vol.sum() + 1e-10))
    last_price = float(close.iloc[-1])
    open_price = float(bars["open"].iloc[0])
    roll_mean = close.rolling(6).mean()
    roll_std = close.rolling(6).std()
    last_z = float((close.iloc[-1] - roll_mean.iloc[-1]) / (roll_std.iloc[-1] + 1e-10))
    if len(close) >= 13:
        ret_1h = float(close.iloc[-1] / close.iloc[-13] - 1)
    else:
        ret_1h = 0.0
    try:
        prev_df = ibm.get_intraday_bars(symbol, duration="2 D", bar_size="1 day")
        if len(prev_df) >= 2:
            prev_close = float(prev_df["close"].iloc[-2])
            ret_today = (last_price - prev_close) / (prev_close + 1e-10)
        else:
            ret_today = (last_price - open_price) / (open_price + 1e-10)
    except Exception:
        ret_today = (last_price - open_price) / (open_price + 1e-10)
    try:
        daily_df = yf.download(symbol, period="25d", interval="1d", progress=False, auto_adjust=True)
        avg_vol_20d = float(daily_df["Volume"].dropna().tail(20).mean())
        cum_vol_today = float(vol.sum())
        vol_ratio = cum_vol_today / (avg_vol_20d + 1e-10)
    except Exception:
        vol_ratio = 1.0
    return MarketSnapshot(
        symbol=symbol,
        last_price=last_price,
        vwap=vwap,
        open_price=open_price,
        high_price=float(bars["high"].max()),
        low_price=float(bars["low"].min()),
        return_today=ret_today,
        return_1h=ret_1h,
        rsi_14=compute_rsi(close, period=14),
        zscore_30m=last_z,
        volume_ratio=vol_ratio,
    )


def momentum_signal(snap: MarketSnapshot) -> StrategySignal:
    if snap.vwap <= 0:
        return StrategySignal(snap.symbol, 0, 0.0, "momentum", 0.0, "No VWAP data")
    vwap_dev = (snap.last_price - snap.vwap) / (snap.vwap + 1e-10)
    if vwap_dev > VWAP_MOMENTUM_THRESHOLD_STRONG:
        direction = +1
        confidence = min(0.95, 0.55 + abs(vwap_dev) * 20)
        reason = f"Strong upside momentum: price {vwap_dev:+.2%} vs VWAP"
    elif vwap_dev > VWAP_MOMENTUM_THRESHOLD:
        direction = +1
        confidence = min(0.70, 0.50 + abs(vwap_dev) * 15)
        reason = f"Mild upside momentum: price {vwap_dev:+.2%} vs VWAP"
    elif vwap_dev < -VWAP_MOMENTUM_THRESHOLD_STRONG:
        direction = -1
        confidence = min(0.95, 0.55 + abs(vwap_dev) * 20)
        reason = f"Strong downside momentum: price {vwap_dev:+.2%} vs VWAP"
    elif vwap_dev < -VWAP_MOMENTUM_THRESHOLD:
        direction = -1
        confidence = min(0.70, 0.50 + abs(vwap_dev) * 15)
        reason = f"Mild downside momentum: price {vwap_dev:+.2%} vs VWAP"
    else:
        return StrategySignal(snap.symbol, 0, 0.0, "momentum", 0.0, f"Price within VWAP band: {vwap_dev:+.3%}")
    if snap.volume_ratio > 1.5:
        confidence = min(0.95, confidence * 1.15)
    elif snap.volume_ratio < 0.5:
        confidence *= 0.80
    if direction == +1 and snap.rsi_14 > 70:
        confidence *= 0.80
        reason += " (RSI overbought)"
    elif direction == -1 and snap.rsi_14 < 30:
        confidence *= 0.80
        reason += " (RSI oversold)"
    size_pct = min(SIGNAL_MAX_TRADE_PCT, SIGNAL_MAX_TRADE_PCT * confidence)
    return StrategySignal(snap.symbol, direction, confidence, "momentum", size_pct, reason)


def mean_reversion_signal(snap: MarketSnapshot) -> StrategySignal:
    z = snap.zscore_30m
    if z > MEAN_REVERSION_ZSCORE:
        direction = -1
        confidence = min(0.90, 0.50 + (z - MEAN_REVERSION_ZSCORE) * 0.15)
        reason = f"Mean reversion: z={z:.2f} (overbought)"
        if snap.rsi_14 > 75:
            confidence = min(0.95, confidence * 1.10)
            reason += " + RSI overbought"
    elif z < -MEAN_REVERSION_ZSCORE:
        direction = +1
        confidence = min(0.90, 0.50 + (abs(z) - MEAN_REVERSION_ZSCORE) * 0.15)
        reason = f"Mean reversion: z={z:.2f} (oversold)"
        if snap.rsi_14 < 25:
            confidence = min(0.95, confidence * 1.10)
            reason += " + RSI oversold"
    else:
        return StrategySignal(snap.symbol, 0, 0.0, "mean_reversion", 0.0, f"z={z:.2f} within normal range")
    size_pct = min(SIGNAL_MAX_TRADE_PCT * 0.75, SIGNAL_MAX_TRADE_PCT * 0.75 * confidence)
    return StrategySignal(snap.symbol, direction, confidence, "mean_reversion", size_pct, reason)


def regime_change_signal(state: FundState, target_blend: dict) -> list:
    if not state.regime_changed:
        return []
    log.info(f"[Signal] REGIME CHANGE detected → Phase {state.current_phase}")
    signals = []
    for sym, alloc in target_blend.items():
        tradable = TRADABLE_UNIVERSE.get(sym)
        if tradable is None:
            continue
        direction = +1 if alloc > 0 else 0
        signals.append(StrategySignal(
            symbol=tradable,
            direction=direction,
            confidence=0.95,
            source="regime",
            size_pct=float(alloc),
            reasoning=f"Regime changed to Phase {state.current_phase}",
        ))
    return signals


class AIAgent:
    def __init__(self, state: FundState):
        self.state = state
        self.client = anthropic.Anthropic() if ANTHROPIC_AVAILABLE else None
        self._last_call = datetime.datetime.min

    def _build_prompt(self, snapshots: dict, signals: list, news_summary: str) -> str:
        snapshot_data = {}
        for sym, snap in snapshots.items():
            if snap:
                snapshot_data[sym] = {
                    "last_price": round(snap.last_price, 2),
                    "vwap_dev_pct": round((snap.last_price / snap.vwap - 1) * 100, 2),
                    "rsi_14": round(snap.rsi_14, 1),
                    "return_today": f"{snap.return_today*100:+.2f}%",
                    "return_1h": f"{snap.return_1h*100:+.2f}%",
                    "zscore": round(snap.zscore_30m, 2),
                }
        signal_data = [{"symbol": s.symbol, "direction": s.direction, "confidence": round(s.confidence, 2), "source": s.source, "reasoning": s.reasoning} for s in signals]
        context = {
            "current_phase": self.state.current_phase,
            "current_blend": self.state.current_blend,
            "net_liq_usd": round(self.state.net_liq, 2),
            "daily_pnl_pct": f"{self.state.daily_pnl_pct*100:+.2f}%",
            "market_snapshots": snapshot_data,
            "active_signals": signal_data,
            "news_summary": news_summary or "No news signal available.",
            "trades_today": self.state.trades_today,
        }
        return (
            "You are a quantitative portfolio manager running a systematic ETF fund.\n\n"
            "Current portfolio context:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Your task is to analyze this context and return a JSON decision object.\n\n"
            "Consider:\n"
            "- Are the signals consistent or conflicting?\n"
            "- Does the news/macro backdrop support or contradict the technical signals?\n"
            "- What is the risk/reward given current regime and daily P&L?\n"
            "- Are we overtrading (trades_today is already high)?\n"
            "- Should position sizing be adjusted from default?\n\n"
            "Return ONLY a JSON object with exactly these fields (no markdown, no explanation):\n"
            "{\n"
            "  \"conviction\": <float -1.0 to 1.0, where -1=very bearish, 0=neutral, 1=very bullish>,\n"
            "  \"phase_override\": <null or \"1\" or \"1b\" or \"2\" or \"3\" or \"4\">,\n"
            "  \"sizing_multiplier\": <float 0.5 to 1.5>,\n"
            "  \"priority_action\": <\"rebalance\", \"trim_risk\", \"add_exposure\", \"hold\", \"halt\">,\n"
            "  \"symbols_to_increase\": [<list of symbols to increase allocation>],\n"
            "  \"symbols_to_decrease\": [<list of symbols to decrease allocation>],\n"
            "  \"reasoning\": <one sentence max 25 words explaining the key decision driver>\n"
            "}\n"
        )

    def run(self, snapshots: dict, signals: list, news_summary: str = "") -> dict:
        neutral = {
            "conviction": 0.0,
            "phase_override": None,
            "sizing_multiplier": 1.0,
            "priority_action": "hold",
            "symbols_to_increase": [],
            "symbols_to_decrease": [],
            "reasoning": "AI agent fallback — holding.",
        }
        if not ANTHROPIC_AVAILABLE or self.client is None:
            return neutral
        try:
            prompt = self._build_prompt(snapshots, signals, news_summary)
            response = self.client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text.strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw_text)
            decision["conviction"] = float(np.clip(decision.get("conviction", 0.0), -1.0, 1.0))
            decision["sizing_multiplier"] = float(np.clip(decision.get("sizing_multiplier", 1.0), 0.5, 1.5))
            with self.state.lock:
                self.state.ai_conviction = decision["conviction"]
                self.state.ai_reasoning = decision.get("reasoning", "")
            log.info(f"[AI Agent] conviction={decision['conviction']:+.2f}  action={decision['priority_action']}  reason: {decision.get('reasoning','')}")
            return decision
        except Exception as e:
            log.warning(f"[AI Agent] Call failed: {e} — using neutral hold.")
            return neutral


class RiskManager:
    def __init__(self, state: FundState, ibm: IBManager):
        self.state = state
        self.ibm = ibm
        self._open_nlv = None

    def record_open_nlv(self, nlv: float):
        self._open_nlv = nlv
        log.info(f"[Risk] Open NAV recorded: ${nlv:,.2f}")

    def check_daily_pnl(self, current_nlv: float) -> bool:
        if self._open_nlv is None or self._open_nlv <= 0:
            return True
        daily_pnl = (current_nlv - self._open_nlv) / self._open_nlv
        with self.state.lock:
            self.state.daily_pnl_pct = daily_pnl
        if daily_pnl < -MAX_DAILY_LOSS_PCT:
            log.warning(f"[Risk] CIRCUIT BREAKER: daily P&L {daily_pnl:+.2%} < -{MAX_DAILY_LOSS_PCT:.1%} — trading halted.")
            with self.state.lock:
                self.state.circuit_breaker = True
                self.state.trading_halted = True
            return False
        return True

    def is_order_allowed(self, symbol: str, qty: int, price: float, nlv: float) -> tuple:
        if self.state.circuit_breaker:
            return False, "Circuit breaker active — trading halted."
        value = abs(qty) * price
        if value < MIN_ORDER_VALUE:
            return False, f"Order value ${value:.2f} < minimum ${MIN_ORDER_VALUE}"
        positions = self.state.positions.copy()
        new_val = (positions.get(symbol, 0) + qty) * price
        if new_val / (nlv + 1e-10) > MAX_SINGLE_POSITION_PCT:
            return False, f"{symbol} would reach {new_val/nlv:.1%} of NAV > {MAX_SINGLE_POSITION_PCT:.0%} limit"
        return True, "OK"

    def new_trading_day(self, nlv: float):
        with self.state.lock:
            self.state.circuit_breaker = False
            self.state.trading_halted = False
            self.state.trades_today = 0
            self.state.rebalances_today = 0
            self.state.session_date = datetime.date.today()
        self.record_open_nlv(nlv)
        log.info("[Risk] New trading day — circuit breakers reset.")


class PortfolioRebalancer:
    def __init__(self, state: FundState, ibm: IBManager, risk: RiskManager):
        self.state = state
        self.ibm = ibm
        self.risk = risk

    def rebalance(self, blend: dict, nlv: float, urgent: bool = False, ai_sizing: float = 1.0, source: str = "scheduled") -> int:
        if self.state.trading_halted:
            log.warning("[Rebalancer] Trading halted — skipping rebalance.")
            return 0
        target_alloc = {}
        for key, alloc in blend.items():
            sym = TRADABLE_UNIVERSE.get(key)
            if sym is None:
                continue
            target_alloc[sym] = target_alloc.get(sym, 0.0) + float(alloc) * ai_sizing
        total = sum(target_alloc.values())
        if total > 1.0:
            target_alloc = {k: v / total for k, v in target_alloc.items()}
        symbols = list(target_alloc.keys())
        prices = self.ibm.get_prices(symbols)
        positions = self.ibm.get_positions()
        orders_placed = 0
        for sym in symbols:
            price = prices.get(sym)
            if price is None or price <= 0:
                log.warning(f"[Rebalancer] No price for {sym} — skipping.")
                continue
            target_value = nlv * target_alloc[sym]
            target_shares = int(round(target_value / price))
            current_shares = int(round(float(positions.get(sym, 0))))
            delta = target_shares - current_shares
            if abs(delta) * price / (nlv + 1e-10) < ORDER_TOLERANCE_PCT:
                log.debug(f"[Rebalancer] {sym}: delta {delta} below tolerance — skip")
                continue
            if abs(delta) * price < MIN_ORDER_VALUE:
                log.debug(f"[Rebalancer] {sym}: order value too small — skip")
                continue
            allowed, reason = self.risk.is_order_allowed(sym, delta, price, nlv)
            if not allowed:
                log.warning(f"[Rebalancer] {sym}: blocked — {reason}")
                continue
            success = self.ibm.place_market_order(sym, delta) if urgent else self.ibm.place_limit_order(sym, delta, price)
            if success:
                orders_placed += 1
                record = TradeRecord(
                    timestamp=datetime.datetime.now().isoformat(),
                    symbol=sym,
                    side="BUY" if delta > 0 else "SELL",
                    qty=abs(delta),
                    price=price,
                    value=abs(delta) * price,
                    source=source,
                    reasoning=f"Blend={target_alloc[sym]:.2%} AI_mult={ai_sizing:.2f}",
                )
                with self.state.lock:
                    self.state.trade_log.append(record)
                    self.state.trades_today += 1
        if orders_placed > 0:
            with self.state.lock:
                self.state.rebalances_today += 1
            log.info(f"[Rebalancer] {source}: placed {orders_placed} orders (rebalance #{self.state.rebalances_today} today)")
        return orders_placed


def load_model_state(pkl_path: str = "model_results.pkl") -> dict:
    defaults = {
        "current_regime": 1,
        "current_phase_eff": "1",
        "phase_blend": {
            1: {"FACTOR": 0.70, "SPY": 0.20, "QQQ": 0.00, "GLD": 0.05, "TLT": 0.05},
            "1b": {"FACTOR": 0.60, "SPY": 0.15, "QQQ": 0.20, "GLD": 0.05, "TLT": 0.00},
            2: {"FACTOR": 0.50, "SPY": 0.10, "QQQ": 0.35, "GLD": 0.05, "TLT": 0.00},
            3: {"FACTOR": 0.30, "SPY": 0.00, "QQQ": 0.00, "GLD": 0.15, "TLT": 0.15},
            4: {"FACTOR": 0.55, "SPY": 0.15, "QQQ": 0.00, "GLD": 0.15, "TLT": 0.15},
        },
        "news_signal": None,
        "regime": None,
    }
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        log.info(f"[State] Loaded model_results.pkl (generated {data.get('generated_at','?')})")
        return {
            "current_regime": int(data.get("current_regime", 1)),
            "current_phase_eff": str(data.get("current_phase_eff", "1")),
            "phase_blend": data.get("phase_blend", defaults["phase_blend"]),
            "news_signal": data.get("news_signal"),
            "regime": data.get("regime"),
        }
    except FileNotFoundError:
        log.warning(f"[State] {pkl_path} not found — using default Phase 1 state.")
        log.warning("        Run FAMWithAIA.py first to generate model_results.pkl")
        return defaults
    except Exception as e:
        log.error(f"[State] Failed to load pickle: {e} — using defaults.")
        return defaults


class FundManager:
    def __init__(self, approval_required: bool = False):
        self.state = FundState()
        self.ibm = IBManager(self.state)
        self.risk = RiskManager(self.state, self.ibm)
        self.rebal = PortfolioRebalancer(self.state, self.ibm, self.risk)
        self.ai_agent = AIAgent(self.state)
        self._approval_required = approval_required
        self._stop_event = threading.Event()
        self._monitored_syms = ["SPY", "QQQ", "GLD", "TLT"]

    def start(self):
        log.info("═" * 60)
        log.info("  FundManager starting...")
        log.info("  Strategy desks: ETF Sleeve + Intraday Signals + AI Agent")
        log.info(f"  Market data type: {IB_MARKET_DATA_TYPE} ({'live' if IB_MARKET_DATA_TYPE == 1 else '15-min delayed'})")
        log.info("═" * 60)
        if not self.ibm.connect():
            log.error("Cannot start without IB connection.")
            return
        self._refresh_model_state()
        nlv = self.ibm.get_account_nlv()
        with self.state.lock:
            self.state.net_liq = nlv
        self.risk.record_open_nlv(nlv)
        log.info(f"  Account NAV: ${nlv:,.2f}")
        self._do_scheduled_rebalance("startup")
        threads = [
            threading.Thread(target=self._thread_scheduled_rebalancer, name="ScheduledRebalancer", daemon=True),
            threading.Thread(target=self._thread_intraday_signals, name="IntradaySignals", daemon=True),
            threading.Thread(target=self._thread_ai_agent, name="AIAgent", daemon=True),
            threading.Thread(target=self._thread_risk_monitor, name="RiskMonitor", daemon=True),
            threading.Thread(target=self._thread_state_refresher, name="StateRefresher", daemon=True),
        ]
        for t in threads:
            t.start()
            log.info(f"  Thread started: {t.name}")
        log.info("\n[FundManager] All desks live.  STOP_LIVE file or Ctrl+C to exit.\n")
        try:
            while not self._stop_event.is_set():
                if os.path.exists("STOP_LIVE"):
                    log.info("[FundManager] STOP_LIVE sentinel detected — shutting down.")
                    break
                time.sleep(5)
        except KeyboardInterrupt:
            log.info("[FundManager] Keyboard interrupt — shutting down.")
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        try:
            self.ibm.ib.disconnect()
        except Exception:
            pass
        log.info("[FundManager] Disconnected. Goodbye.")

    def _thread_scheduled_rebalancer(self):
        fired_today = set()
        last_date = None
        while not self._stop_event.is_set():
            now_et = _eastern_now()
            if last_date != now_et.date():
                fired_today = set()
                last_date = now_et.date()
                nlv = self.ibm.get_account_nlv()
                self.risk.new_trading_day(nlv)
                with self.state.lock:
                    self.state.net_liq = nlv
            for hr, mn in DAILY_REBALANCE_TIMES:
                slot_key = f"{now_et.date()}-{hr:02d}{mn:02d}"
                if slot_key in fired_today:
                    continue
                slot_time = now_et.replace(hour=hr, minute=mn, second=0, microsecond=0)
                diff_sec = (now_et - slot_time).total_seconds()
                if 0 <= diff_sec <= 60:
                    log.info(f"[Scheduler] Firing scheduled rebalance: {hr:02d}:{mn:02d} ET")
                    self._do_scheduled_rebalance(f"sched_{hr:02d}{mn:02d}")
                    fired_today.add(slot_key)
            time.sleep(30)

    def _do_scheduled_rebalance(self, source: str):
        nlv = self.ibm.get_account_nlv()
        if nlv <= 0:
            log.warning("[Rebalancer] Cannot get valid NAV — skipping.")
            return
        with self.state.lock:
            self.state.net_liq = nlv
            blend = dict(self.state.current_blend)
            ai_sizing = float(np.clip(1.0 + self.state.ai_conviction * 0.25, 0.75, 1.25))
        self.rebal.rebalance(blend, nlv, urgent=False, ai_sizing=ai_sizing, source=source)

    def _thread_intraday_signals(self):
        while not self._stop_event.is_set():
            try:
                if self.state.trading_halted or not _is_market_hours():
                    time.sleep(SIGNAL_CHECK_INTERVAL_SEC)
                    continue
                snapshots = {}
                all_signals = []
                for sym in self._monitored_syms:
                    snap = build_market_snapshot(sym, self.ibm)
                    snapshots[sym] = snap
                    if snap is None:
                        continue
                    mom_sig = momentum_signal(snap)
                    rev_sig = mean_reversion_signal(snap)
                    for sig in (mom_sig, rev_sig):
                        if sig.direction != 0 and sig.confidence > SIGNAL_CONFIDENCE_GATE:
                            all_signals.append(sig)
                            log.info(f"[Signal] {sig.source.upper()} {sym} dir={sig.direction:+d} conf={sig.confidence:.2f} — {sig.reasoning}")
                regime_sigs = regime_change_signal(self.state, self.state.current_blend)
                if regime_sigs:
                    all_signals.extend(regime_sigs)
                    log.info("[Signal] REGIME CHANGE — executing immediate rebalance")
                    nlv = self.state.net_liq
                    self.rebal.rebalance(self.state.current_blend, nlv, urgent=True, source="regime_change")
                    with self.state.lock:
                        self.state.regime_changed = False
                with self.state.lock:
                    self.state.latest_signals = all_signals
                if all_signals and not regime_sigs:
                    self._execute_signal_trades(all_signals, snapshots)
            except Exception as e:
                log.error(f"[IntradaySignals] Error: {e}", exc_info=True)
            time.sleep(SIGNAL_CHECK_INTERVAL_SEC)

    def _execute_signal_trades(self, signals: list, snapshots: dict):
        nlv = self.state.net_liq
        prices = self.ibm.get_prices(self._monitored_syms)
        positions = self.ibm.get_positions()
        for sig in signals:
            if sig.confidence < SIGNAL_CONFIDENCE_GATE:
                continue
            if sig.symbol not in prices or prices[sig.symbol] is None:
                continue
            price = prices[sig.symbol]
            trade_val = nlv * sig.size_pct * (1.0 + self.state.ai_conviction * 0.20)
            trade_val = min(trade_val, nlv * SIGNAL_MAX_TRADE_PCT)
            shares = int(round(trade_val / price)) * sig.direction
            allowed, reason = self.risk.is_order_allowed(sig.symbol, shares, price, nlv)
            if not allowed:
                log.debug(f"[Intraday] {sig.symbol} blocked: {reason}")
                continue
            success = self.ibm.place_limit_order(sig.symbol, shares, price)
            if success:
                record = TradeRecord(
                    timestamp=datetime.datetime.now().isoformat(),
                    symbol=sig.symbol,
                    side="BUY" if shares > 0 else "SELL",
                    qty=abs(shares),
                    price=price,
                    value=abs(shares) * price,
                    source=f"intraday_{sig.source}",
                    reasoning=sig.reasoning,
                )
                with self.state.lock:
                    self.state.trade_log.append(record)
                    self.state.trades_today += 1

    def _thread_ai_agent(self):
        while not self._stop_event.is_set():
            try:
                if not _is_market_hours():
                    time.sleep(AI_AGENT_INTERVAL_SEC)
                    continue
                snapshots = {s: build_market_snapshot(s, self.ibm) for s in self._monitored_syms}
                news_summary = ""
                try:
                    from NewsAgent import NewsAgent
                    na = NewsAgent()
                    sig = na.get_signal()
                    news_summary = (
                        f"Macro sentiment: {sig.macro_sentiment:+.2f}  "
                        f"Risk-on: {sig.risk_on_score:.2f}  "
                        f"Recession prob (Polymarket): {sig.recession_prob:.1%}  "
                        f"Fed cut prob: {sig.fed_cut_prob:.1%}  "
                        f"Top risks: {', '.join(sig.top_risks[:2])}"
                    )
                except Exception:
                    pass
                with self.state.lock:
                    signals = list(self.state.latest_signals)
                decision = self.ai_agent.run(snapshots, signals, news_summary)
                if self._approval_required and decision["priority_action"] != "hold":
                    approved = self._request_approval(decision)
                    if not approved:
                        log.info("[AI Agent] Trade not approved — skipping.")
                        time.sleep(AI_AGENT_INTERVAL_SEC)
                        continue
                if decision["priority_action"] == "halt":
                    log.warning("[AI Agent] HALT signal — pausing all trading.")
                    with self.state.lock:
                        self.state.trading_halted = True
                elif decision["priority_action"] in ("rebalance", "trim_risk", "add_exposure"):
                    ai_sizing = decision["sizing_multiplier"]
                    nlv = self.state.net_liq
                    self.rebal.rebalance(self.state.current_blend, nlv, urgent=False, ai_sizing=ai_sizing, source=f"ai_{decision['priority_action']}")
            except Exception as e:
                log.error(f"[AIAgent] Error: {e}", exc_info=True)
            time.sleep(AI_AGENT_INTERVAL_SEC)

    def _thread_risk_monitor(self):
        while not self._stop_event.is_set():
            try:
                nlv = self.ibm.get_account_nlv()
                if nlv > 0:
                    with self.state.lock:
                        self.state.net_liq = nlv
                        self.state.positions = self.ibm.get_positions()
                    self.risk.check_daily_pnl(nlv)
            except Exception as e:
                log.error(f"[RiskMonitor] Error: {e}")
            time.sleep(RISK_CHECK_INTERVAL_SEC)

    def _thread_state_refresher(self):
        prev_regime = self.state.current_regime
        while not self._stop_event.is_set():
            time.sleep(300)
            try:
                model = load_model_state()
                new_regime = model["current_regime"]
                new_phase = str(model["current_phase_eff"])
                ph_key = new_phase if new_phase == "1b" else int(new_phase)
                new_blend = model["phase_blend"].get(ph_key, self.state.current_blend)
                with self.state.lock:
                    if new_regime != prev_regime:
                        log.info(f"[State] REGIME CHANGE: {prev_regime} → {new_regime}")
                        self.state.regime_changed = True
                    self.state.current_regime = new_regime
                    self.state.current_phase = new_phase
                    self.state.current_blend = new_blend
                    prev_regime = new_regime
                log.debug(f"[State] Refreshed: phase={new_phase} regime={new_regime}")
            except Exception as e:
                log.error(f"[StateRefresher] Error: {e}")

    def _refresh_model_state(self):
        model = load_model_state()
        ph = str(model["current_phase_eff"])
        ph_key = ph if ph == "1b" else int(ph)
        blend = model["phase_blend"].get(ph_key, {})
        with self.state.lock:
            self.state.current_regime = model["current_regime"]
            self.state.current_phase = ph
            self.state.current_blend = blend
        log.info(f"[State] Initial phase: {ph}  blend instruments: {[k for k,v in blend.items() if v>0]}")

    def _request_approval(self, decision: dict) -> bool:
        print(f"\n[AI AGENT APPROVAL REQUIRED]")
        print(f"  Action   : {decision['priority_action']}")
        print(f"  Sizing   : {decision['sizing_multiplier']:.2f}×")
        print(f"  Reasoning: {decision.get('reasoning','')}")
        try:
            ans = input("  Approve? (y/n): ").strip().lower()
            return ans == "y"
        except Exception:
            return False


def _eastern_now() -> datetime.datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        utc_now = datetime.datetime.utcnow()
        month = utc_now.month
        offset = -4 if 3 <= month <= 11 else -5
        return utc_now + datetime.timedelta(hours=offset)


def _is_market_hours() -> bool:
    now = _eastern_now()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def main():
    parser = argparse.ArgumentParser(
        description="FAMWithAIA live FundManager mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--live", action="store_true", help="Start the FundManager live trading loop")
    parser.add_argument("--approval", action="store_true", help="Require console approval for AI-driven trades")
    parser.add_argument("--live-data", action="store_true", help="Request live IB market data instead of delayed pricing")
    parser.add_argument("--no-ai", action="store_true", help="Disable the AI agent")
    parser.add_argument("--print-state", action="store_true", help="Print loaded model state and exit")
    raw_args = [a for a in sys.argv[1:] if a.lower() != "live"]
    if any(a.lower() == "live" for a in sys.argv[1:]):
        raw_args.append("--live")
    args, unknown = parser.parse_known_args(raw_args)

    if args.print_state:
        state = load_model_state()
        print(f"Loaded model state: {state}")
        return

    if args.live:
        if args.live_data:
            global IB_MARKET_DATA_TYPE
            IB_MARKET_DATA_TYPE = 1
            log.info("[Config] Live market data requested.")
        if args.no_ai:
            global ANTHROPIC_AVAILABLE
            ANTHROPIC_AVAILABLE = False
            log.info("[Config] AI agent disabled.")
        log.info(f"[Config] IB: {IB_HOST}:{IB_PORT} clientId={IB_CLIENT_ID}")
        log.info(f"[Config] Market data type: {IB_MARKET_DATA_TYPE}")
        log.info(f"[Config] Approval required: {args.approval}")
        log.info(f"[Config] Rebalance schedule: {DAILY_REBALANCE_TIMES} ET")
        manager = FundManager(approval_required=args.approval)
        manager.start()


# Simple CLI support: run headless live mode if `--live` or bare `live` present.
if __name__ == "__main__":
    main()
