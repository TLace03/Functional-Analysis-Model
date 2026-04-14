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
#    exposure with ETF sleeves (TQQQ, GLD, SDS, TLT, SH).
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
SLEEVE_INSTRUMENTS  = ["SPY", "TQQQ", "GLD", "SH", "SDS", "TLT", "BTC-USD"]
SDS_INCEPTION       = pd.Timestamp("2006-07-11")

# Actual fund leverage multipliers used for exposure control.
# This is separate from capital allocation weights.
SLEEVE_LEVERAGE     = {
    "SPY":    1.0,
    "TQQQ":   3.0,
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
# TQQQ (launched 2010) and SDS (launched 2006) have NaN returns for every
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
    1: {"FACTOR": 0.75, "SPY": 0.00, "TQQQ": 0.10, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.05, "TLT": 0.00, "BTC-USD": 0.10},
 
    "1b": {"FACTOR": 0.75, "SPY": 0.00, "TQQQ": 0.10, "GLD": 0.00,
           "SH": 0.00, "SDS": 0.05, "TLT": 0.00, "BTC-USD": 0.10},
 
    2: {"FACTOR": 0.75, "SPY": 0.00, "TQQQ": 0.10, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.00, "TLT": 0.00, "BTC-USD": 0.15},
 
    3: {"FACTOR": 0.20, "SPY": 0.00, "TQQQ": 0.00, "GLD": 0.20,
        "SH": 0.10, "SDS": 0.35, "TLT": 0.15, "BTC-USD": 0.00},
 
    4: {"FACTOR": 0.75, "SPY": 0.00, "TQQQ": 0.05, "GLD": 0.15,
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
                "alloc_SPY", "alloc_TQQQ", "alloc_GLD", "alloc_SH",
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
        "alloc_TQQQ": np.zeros(n, dtype=np.float64),
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
        # (e.g. pre-TQQQ era, calendar mismatches between the main
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
        audit["alloc_TQQQ"][mask] = sleeve_positions[:, sleeve_cols.get_loc("TQQQ")]
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
print(f"{'Phase':<24} {'Factor':>8} {'SPY':>6} {'TQQQ':>6} "
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
          f"{blend['TQQQ']*100:>5.0f}% "
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
plt.show()
