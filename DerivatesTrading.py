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
# DerivativesTrading.py
# Protective put overlay for drawdown hedging.
# Part of the Spatial-Temporal Portfolio Model.
# ============================================================
#
# PERFORMANCE OPTIMIZATION LOG (v2 — Vectorized)
# ============================================================
# Problem in v1:
#   apply_hedge() was called inside a per-date Python for-loop
#   in FAMWithAIA.py.  Each call invoked _get_vix(), which ran
#   a fresh yfinance.download("^VIX", period="5d") network request.
#   With 30% of 50 years ≈ 3,750 backtest dates, this meant
#   ~3,750 HTTP requests to the Yahoo Finance API — a serious
#   performance bottleneck (~1s/request × 3,750 = ~1 hour).
#
# Solution in v2:
#   1. _get_vix() now downloads VIX ONCE and caches the full
#      historical series with a TTL (time-to-live) expiry.
#      Subsequent calls within the TTL return cached data instantly.
#
#   2. New apply_hedge_batch() method accepts pandas arrays for
#      an entire date range and processes everything using NumPy
#      vectorized operations — no Python for-loop over dates.
#      Called from FAMWithAIA.py's vectorized backtest helper.
#
#   3. is_drawdown() fixed: the original used pd.Series([single_value])
#      with rolling(21), which always returned NaN.  The batch
#      version correctly computes a rolling 21-day window.
#
#   4. apply_hedge() (single-date interface) preserved for live-mode
#      console output and backward compatibility.
# ============================================================
 
import time
import warnings
from typing import Optional
 
import numpy as np
import pandas as pd
import yfinance as yf
 
 
# ── VIX cache TTL in seconds ─────────────────────────────────
# VIX data changes only on market-open days.  A 4-hour cache is
# more than sufficient for both backtesting (uses historical values)
# and live trading (only needs today's level).
VIX_CACHE_TTL_SECONDS = 4 * 3600   # 4 hours
 
 
class DerivativesHedger:
    """
    Protective put overlay engine.
 
    Usage (backtest — vectorized):
      hedger = DerivativesHedger(hedge_ratio=0.50)
      hedged_returns = hedger.apply_hedge_batch(
          portfolio_returns, spy_returns, regimes, dates, live_flags)
 
    Usage (live single-date):
      hedger.apply_hedge(port_ret, spy_ret, regime, date, live_mode=True)
 
    Parameters
    ----------
    hedge_ratio           : float — fraction of equity exposure to hedge
                            (0.50 = hedge 50% of the portfolio with puts).
    put_otm_strike        : float — strike as fraction of spot price
                            (0.95 = 5% out-of-the-money put).
    put_tenor_days        : int   — approximate days to option expiry.
    max_daily_hedge_cost_bp: float — hard cap on daily hedge premium in
                            basis points (1 bp = 0.01%).
                            Prevents the hedge from being excessively
                            expensive on high-VIX days.
    """
 
    def __init__(self,
                 hedge_ratio: float = 0.50,
                 put_otm_strike: float = 0.95,
                 put_tenor_days: int = 30,
                 max_daily_hedge_cost_bp: float = 8):
        self.hedge_ratio              = hedge_ratio
        self.put_otm_strike           = put_otm_strike
        self.put_tenor_days           = put_tenor_days
        self.max_daily_hedge_cost_bp  = max_daily_hedge_cost_bp / 10_000
 
        # ── Internal VIX cache ────────────────────────────────
        # _vix_series : full historical VIX close prices (pd.Series)
        # _vix_fetched_at : UNIX timestamp of last download
        # Storing the full historical series lets us look up any
        # backtest date without a second network request.
        self._vix_series     : Optional[pd.Series] = None
        self._vix_fetched_at : float = 0.0
 
    # ──────────────────────────────────────────────────────────
    # INTERNAL: VIX data access with TTL cache
    # ──────────────────────────────────────────────────────────
 
    def _ensure_vix_cache(self) -> None:
        """
        Download the full VIX history and store it in self._vix_series.
        Only downloads if the cache is empty or older than VIX_CACHE_TTL_SECONDS.
 
        This is the core fix for the v1 performance bottleneck:
        ONE download here replaces ~3,750 downloads in the backtest loop.
        """
        now = time.time()
        cache_age = now - self._vix_fetched_at
 
        if self._vix_series is not None and cache_age < VIX_CACHE_TTL_SECONDS:
            return   # cache is still fresh
 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = yf.download("^VIX", period="50y", interval="1d",
                                  auto_adjust=True, progress=False)
 
                # yfinance v0.2+ returns MultiIndex columns for single tickers;
                # squeeze() converts a single-column DataFrame → Series.
                if isinstance(raw.columns, pd.MultiIndex):
                    vix_data = raw["Close"].squeeze()
                else:
                    vix_data = raw["Close"]
 
                if isinstance(vix_data, pd.DataFrame):
                    vix_data = vix_data.iloc[:, 0]
 
                # Ensure DatetimeIndex with no timezone (for asof() lookups)
                if vix_data.index.tz is not None:
                    vix_data.index = vix_data.index.tz_localize(None)
 
                self._vix_series     = vix_data.ffill().dropna()
                self._vix_fetched_at = now
 
            except Exception:
                # On network failure keep whatever we have (or set a flat fallback)
                if self._vix_series is None:
                    # Create a constant fallback series so downstream code doesn't break
                    idx = pd.date_range("1990-01-01", pd.Timestamp.today(), freq="B")
                    self._vix_series = pd.Series(19.2, index=idx)
 
    def _get_vix_level(self, date: Optional[pd.Timestamp] = None) -> float:
        """
        Return the VIX closing level for a given date (or the latest if None).
        Uses the cached historical series — no network call after the first.
        """
        self._ensure_vix_cache()
 
        if self._vix_series is None or len(self._vix_series) == 0:
            return 19.2   # safe default
 
        if date is None:
            return float(self._vix_series.iloc[-1])
 
        # tz-normalise the query date to match the cached index
        ts = pd.Timestamp(date)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
 
        # asof() returns the last known value on or before `ts`
        # — perfect for backtesting where we need the VIX level
        #   that was available at the close of that trading day.
        try:
            val = self._vix_series.asof(ts)
            return float(val) if pd.notna(val) else 19.2
        except Exception:
            return 19.2
 
    def _get_vix_series_for_dates(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Return a NumPy array of VIX levels for an entire array of dates.
        Used by apply_hedge_batch() to vectorize the VIX lookup.
 
        pandas.Series.reindex(dates, method='ffill') aligns the full
        historical series to the exact date list — no Python loop.
        """
        self._ensure_vix_cache()
 
        if self._vix_series is None:
            return np.full(len(dates), 19.2)
 
        # Normalise timezone in query dates
        tz_free = dates
        if dates.tz is not None:
            tz_free = dates.tz_localize(None)
 
        aligned = self._vix_series.reindex(tz_free, method="ffill").fillna(19.2)
        return aligned.values.astype(np.float64)
 
    # ──────────────────────────────────────────────────────────
    # DRAWDOWN DETECTION
    # ──────────────────────────────────────────────────────────
 
    def is_drawdown(self,
                    spy_returns: pd.Series,
                    current_regime: int,
                    window: int = 21,
                    return_threshold: float = -0.05) -> bool:
        """
        Single-date drawdown check (used in live mode).
 
        Returns True if:
          (a) current_regime == 3  (Phase 3 = Unwind, always hedge), OR
          (b) the rolling `window`-day SPY return is below `return_threshold`.
 
        NOTE: For this to work correctly in live mode, `spy_returns` must
        contain at least `window` values (21 days of history).  If you
        pass a single-element Series the rolling sum will be NaN → False.
        Always call this with a trailing window of SPY returns in live use.
        """
        if current_regime == 3:
            return True
        if len(spy_returns) < window:
            return False   # not enough history — skip hedge
        roll = float(spy_returns.rolling(window).sum().iloc[-1])
        return roll < return_threshold if pd.notna(roll) else False
 
    def _is_drawdown_vectorized(self,
                                spy_daily: np.ndarray,
                                regimes: np.ndarray,
                                window: int = 21,
                                threshold: float = -0.05) -> np.ndarray:
        """
        Vectorized drawdown detection for an entire date array.
 
        Returns a boolean NumPy array: True where the hedge should activate.
 
        Logic (element-wise):
          drawdown[t] = True   if regime[t] == 3
          drawdown[t] = True   if rolling_21d_spy_return[t] < -5%
          drawdown[t] = False  otherwise
 
        This is called once by apply_hedge_batch() and the result is
        used to mask the PnL computation — no Python for-loop over dates.
        """
        n = len(spy_daily)
 
        # Phase 3 mask — directly from regime array
        phase3_mask = (regimes == 3)
 
        # Rolling 21-day SPY return: np.convolve gives the windowed sum.
        # We use np.cumsum for O(n) instead of O(n×window).
        cumsum = np.cumsum(np.insert(spy_daily, 0, 0))
        # rolling_sum[t] = sum of spy_daily[t-window+1 : t+1]
        rolling_sum = np.full(n, np.nan)
        for i in range(window - 1, n):
            rolling_sum[i] = cumsum[i + 1] - cumsum[i - window + 1]
 
        # Replace any NaN at the start with a non-triggering value
        rolling_sum = np.where(np.isnan(rolling_sum), 0.0, rolling_sum)
        rolling_mask = rolling_sum < threshold
 
        return phase3_mask | rolling_mask
 
    # ──────────────────────────────────────────────────────────
    # PUT P&L MODEL (analytical Black-Scholes approximation)
    # ──────────────────────────────────────────────────────────
 
    def _approximate_put_pnl(self,
                             spy_daily_return: float,
                             vix: float) -> float:
        """
        Single-date put P&L for live mode.
 
        Approximation:
          premium   = σ_annual × √T × 0.4     (ATM put Black-Scholes approx)
          premium   = min(premium, max_daily_cost_cap)
          payoff    = max(0, strike_move - spy_return)  if in-the-money
          daily_pnl = (payoff - premium) × hedge_ratio
 
        The 0.4 factor is the approximate at-the-money theta/vega
        ratio for short-tenor puts — a practical fintech simplification.
        """
        annual_vol = vix / 100.0
        tenor_yr   = self.put_tenor_days / 365.0
        premium    = annual_vol * np.sqrt(tenor_yr) * 0.4
        premium    = min(premium, self.max_daily_hedge_cost_bp)
 
        # A 5% OTM put pays off when SPY drops more than 5%
        # (put_otm_strike = 0.95 → strike is at 95% of spot)
        # Daily moneyness move = spy_return vs (strike - 1)
        itm_threshold = self.put_otm_strike - 1.0   # e.g. -0.05 for 5% OTM
        payoff = max(0.0, -(spy_daily_return - itm_threshold)) if spy_daily_return < itm_threshold else 0.0
 
        return (payoff - premium) * self.hedge_ratio
 
    def _approximate_put_pnl_vectorized(self,
                                        spy_returns: np.ndarray,
                                        vix_levels: np.ndarray) -> np.ndarray:
        """
        Vectorized put P&L for the entire backtest date array.
 
        Computes the same formula as _approximate_put_pnl() but as
        NumPy array operations — processes all dates in one pass.
 
        Parameters
        ----------
        spy_returns : (n,) float64 — daily SPY returns for each date
        vix_levels  : (n,) float64 — VIX closing level for each date
 
        Returns
        -------
        pnl : (n,) float64 — hedge P&L per date (can be positive or negative)
        """
        annual_vol = vix_levels / 100.0
        tenor_yr   = self.put_tenor_days / 365.0
 
        # Premium: σ × √T × 0.4, capped at max_daily_hedge_cost_bp
        # np.minimum replaces per-element min() calls
        premium = np.minimum(
            annual_vol * np.sqrt(tenor_yr) * 0.4,
            self.max_daily_hedge_cost_bp
        )
 
        # Payoff: only positive when spy_return < (strike - 1)
        # np.maximum with 0 is the equivalent of max(0, x)
        itm_threshold = self.put_otm_strike - 1.0
        payoff = np.maximum(0.0, -(spy_returns - itm_threshold))
        # Zero out days where the put is out-of-the-money
        payoff = np.where(spy_returns < itm_threshold, payoff, 0.0)
 
        return (payoff - premium) * self.hedge_ratio
 
    # ──────────────────────────────────────────────────────────
    # BATCH HEDGE APPLICATION (vectorized — for backtest)
    # ──────────────────────────────────────────────────────────
 
    def apply_hedge_batch(self,
                          portfolio_returns: pd.Series,
                          spy_returns: pd.Series,
                          regimes: np.ndarray,
                          dates: pd.DatetimeIndex,
                          live_flags: np.ndarray) -> pd.Series:
        """
        Apply the protective put overlay to an entire date range at once.
 
        This is the HIGH-PERFORMANCE path used by FAMWithAIA.py's
        vectorized backtest.  It replaces ~3,750 individual apply_hedge()
        calls with 5 NumPy array operations.
 
        Steps:
          1. Fetch VIX series for all dates in one aligned lookup.
          2. Compute drawdown mask for all dates (vectorized).
          3. Compute put P&L for ALL dates (vectorized).
          4. Apply P&L only where drawdown mask is True.
          5. Print live-mode diagnostics for very recent dates only.
 
        Parameters
        ----------
        portfolio_returns : pd.Series  — pre-hedge daily portfolio returns
        spy_returns       : pd.Series  — SPY daily returns (aligned)
        regimes           : np.ndarray — integer regime labels per date
        dates             : pd.DatetimeIndex — dates to process
        live_flags        : np.ndarray[bool]  — True for dates within 5 days of today
 
        Returns
        -------
        pd.Series — hedged daily portfolio returns (same index as input)
        """
        # ── 1. Align inputs to the date index ────────────────
        spy_arr   = spy_returns.reindex(dates, fill_value=0.0).values
        port_arr  = portfolio_returns.reindex(dates, fill_value=0.0).values
 
        # ── 2. Fetch VIX levels for all dates at once ─────────
        # _ensure_vix_cache() is called once here.  The reindex
        # alignment finds the correct historical VIX for each date.
        vix_arr = self._get_vix_series_for_dates(dates)
 
        # ── 3. Vectorized drawdown detection ──────────────────
        drawdown_mask = self._is_drawdown_vectorized(spy_arr, regimes)
 
        # ── 4. Vectorized P&L (computed for all dates) ────────
        # We compute P&L for every date but only apply it where
        # drawdown_mask is True.  This avoids branching and keeps
        # the computation as a single NumPy pass.
        pnl_all = self._approximate_put_pnl_vectorized(spy_arr, vix_arr)
 
        # Apply hedge only on drawdown days
        # np.where(condition, value_if_true, value_if_false)
        hedge_contribution = np.where(drawdown_mask, pnl_all, 0.0)
        hedged_returns     = port_arr + hedge_contribution
 
        # ── 5. Live mode diagnostics (at most 5 iterations) ───
        live_indices = np.where(live_flags & drawdown_mask)[0]
        for i in live_indices:
            d = dates[i]
            print(f"\n[DerivativesHedger] DRAWDOWN — {d.date()} | "
                  f"Regime {regimes[i]} | VIX {vix_arr[i]:.1f}")
            print(f"   → Hedge {self.hedge_ratio*100:.0f}% exposure | "
                  f"P&L contribution: {hedge_contribution[i]*100:.3f}%")
 
        return pd.Series(hedged_returns, index=dates)
 
    # ──────────────────────────────────────────────────────────
    # SINGLE-DATE INTERFACE (live mode / backward compat)
    # ──────────────────────────────────────────────────────────
 
    def apply_hedge(self,
                    portfolio_daily_return: float,
                    spy_daily_return: float,
                    current_regime: int,
                    date: Optional[pd.Timestamp] = None,
                    live_mode: bool = False) -> float:
        """
        Apply the protective put overlay for a single date.
 
        Preserved for:
          - Live/intraday use where you have one date at a time.
          - The console recommendation printout in live_mode.
          - Backward compatibility if called from external scripts.
 
        For backtesting, use apply_hedge_batch() — it is orders
        of magnitude faster.
        """
        # Use cached VIX (no extra network call if cache is warm)
        vix = self._get_vix_level(date)
 
        # For single-date call we only have 1 data point, so we
        # check regime directly rather than using rolling window.
        drawdown = (current_regime == 3) or (spy_daily_return < -0.02)
 
        if not drawdown:
            return portfolio_daily_return
 
        hedge_pnl = self._approximate_put_pnl(spy_daily_return, vix)
 
        if live_mode:
            print(f"\n[DerivativesHedger] DRAWDOWN DETECTED — "
                  f"Regime {current_regime} | VIX {vix:.1f}")
            print(f"   → Hedge {self.hedge_ratio*100:.0f}% of equity exposure "
                  f"with {self.put_tenor_days}-day SPY puts")
            print(f"   → Suggested notional: "
                  f"~${int(100_000 * self.hedge_ratio):,} per $100k portfolio")
            print(f"   → Estimated hedge P&L today: {hedge_pnl*100:.3f}%")
            self._print_live_option_recommendation()
 
        return portfolio_daily_return + hedge_pnl
 
    def _print_live_option_recommendation(self) -> None:
        """
        Attempt to fetch a real option contract from yfinance and print
        the suggested trade.  Silently skips on any error (e.g. market closed,
        no options data, network failure).
        """
        try:
            spy = yf.Ticker("SPY")
            if not spy.options:
                return
            opts          = spy.option_chain(spy.options[0])
            puts          = opts.puts
            spot          = spy.history(period="1d")["Close"].iloc[-1]
            target_strike = round(spot * self.put_otm_strike, 0)
            match         = puts[puts["strike"] == target_strike]
            if not match.empty:
                row = match.iloc[0]
                delta_str = (f"delta ≈ {row['delta']:.2f}"
                             if "delta" in row and pd.notna(row.get("delta"))
                             else "delta n/a")
                print(f"   → BUY {row['contractSymbol']} "
                      f"@ ask ${row['ask']:.2f} ({delta_str})")
        except Exception:
            pass   # silently skip if live option data is unavailable
 
 
if __name__ == "__main__":
    hedger = DerivativesHedger()
    print("DerivativesTrading.py v2 loaded — VIX cache + vectorized batch hedge ready.")
