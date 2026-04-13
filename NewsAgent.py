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


#NewsAgent.py — Real-time macro/micro news sentiment and prediction market monitor
 
#Part of the Spatial-Temporal Portfolio Model project.
#Copyright 2026 Lacy, Thomas Joseph — Apache License 2.0
 
#─────────────────────────────────────────────────────────────────────────────
#OVERVIEW
#─────────────────────────────────────────────────────────────────────────────
#This module provides a live NewsSignal that FunctionalAnalysisModel.py imports
#to optionally adjust the current-day regime phase and blend allocations based
#on real-time information from four source tiers:
 
#  Tier 1 — Polymarket (prediction markets)
#    Prices in prediction markets represent capital-weighted collective probability
#    estimates with real financial skin in the game — faster and less noisy
#    than news sentiment. Fetched from the public Gamma REST API (no auth).
 
#  Tier 2 — RSS news feeds (no API key required)
#    MarketWatch, CNBC Economy, CNBC Finance, Federal Reserve press releases,
#    and Yahoo Finance. Parsed with Python stdlib (urllib + xml), no third-
#    party parsers required.
 
#  Tier 3 — FOREX snapshot (via yfinance — already a project dependency)
#    5-day return on DXY (US Dollar Index), EUR/USD, USD/JPY, USD/CNY.
#    A surging dollar is a risk-off signal; a weakening dollar into falling
#    rates is historically risk-on.
 
#  Tier 4 — LLM headline scoring (optional — requires Anthropic API key)
#    If ANTHROPIC_API_KEY is set in the environment, headlines are batched
#    and sent to claude-haiku for structured sentiment scoring. If the key
#    is absent or the call fails, a deterministic keyword-based fallback
#    produces equivalent (lower-confidence) scores. The model never blocks
#    on this tier.
 
#─────────────────────────────────────────────────────────────────────────────
#IMPORTANT: BACKTEST SAFETY
#─────────────────────────────────────────────────────────────────────────────
#adjust_phase() and adjust_blend() are no-ops for historical dates older than
#LIVE_SIGNAL_WINDOW_DAYS. They only affect today's or very recent dates, so
#the full out-of-sample backtest is NEVER contaminated with live news data.
 
#─────────────────────────────────────────────────────────────────────────────
#INSTALLATION
#─────────────────────────────────────────────────────────────────────────────
#All required packages are already in FunctionalAnalysisModel.py.
#Optional LLM scoring:
#  pip install anthropic
 
#─────────────────────────────────────────────────────────────────────────────
#USAGE IN FunctionalAnalysisModel.py
#─────────────────────────────────────────────────────────────────────────────
#Add three blocks to the main file:
 
#  # ── 1. At the top, after imports ──────────────────────────────────────
#  try:
#      from NewsAgent import NewsAgent
#      _news_agent  = NewsAgent()
#      _news_signal = _news_agent.get_signal()
#      print(_news_signal.summary())
#  except Exception as _e:
#      print(f"  NewsAgent unavailable ({_e}) — running without live signal.")
#      _news_signal = None
 
#  # ── 2. Replace resolve_phase() with the news-aware version ────────────
#  def resolve_phase(date, base_phase):
#      if base_phase == 1:
#          mom = float(spy_mom_fast.get(date, 0.0))
#          if mom > PHASE1B_MOM_THRESH:
#              base_phase = "1b"
#      if _news_signal is not None:
#          return _news_signal.adjust_phase(base_phase, date)
#      return base_phase
 
#  # ── 3. After blend = phase_blend[eff_phase] in both loops ─────────────
#  if _news_signal is not None:
#      blend = _news_signal.adjust_blend(blend, date)
 
# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# All required packages are stdlib or already in FunctionalAnalysisModel.py.
# anthropic is imported lazily inside _score_headlines_llm() so its absence
# never raises an ImportError at module load time.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import copy
import warnings
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET
from typing import Optional
 
import numpy as np
import pandas as pd
import yfinance as yf
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# All tunable parameters live here. Values below are conservative defaults
# calibrated to avoid overriding the core regime model without strong evidence.
# ─────────────────────────────────────────────────────────────────────────────
 
# ── Caching ──────────────────────────────────────────────────────────────────
# The agent runs all four tiers once and caches the result for this many hours.
# Refreshing on every script run is wasteful; 4 hours is a good live-trading
# balance between freshness and API rate limits.
CACHE_TTL_HOURS = 4
 
# ── Backtest safety window ────────────────────────────────────────────────────
# adjust_phase() and adjust_blend() are complete no-ops for dates older than
# this many calendar days from today. Historical backtest data is never touched.
LIVE_SIGNAL_WINDOW_DAYS = 5
 
# ── Polymarket ────────────────────────────────────────────────────────────────
# Maximum number of active Polymarket markets to fetch (sorted by volume DESC).
# Higher limits improve keyword matching but slow the fetch.
POLYMARKET_FETCH_LIMIT = 300
 
# Minimum dollar volume for a Polymarket contract to be considered signal-worthy.
# Low-volume markets have unreliable prices; filtering them reduces noise.
POLYMARKET_MIN_VOLUME = 50_000
 
# ── Phase-override thresholds ────────────────────────────────────────────────
# These govern when the news signal is strong enough to override the regime.
# Deliberately conservative — the core classifier always has first say.
 
# If Polymarket prices a recession above this probability, the model will not
# enter Phase 1b (leveraged acceleration) and will treat Phase 1 as Phase 4
# (cautious re-entry posture) regardless of momentum signals.
RECESSION_PROB_PHASE_BRAKE   = 0.60   # suppress Phase 1b above this
 
# If Polymarket prices a recession above this probability AND the base regime
# is Phase 1 or Phase 2, force a Phase 3 (Unwind / protective) posture.
RECESSION_PROB_PHASE3_FORCE  = 0.75   # force Phase 3 above this
 
# If composite macro sentiment is below this threshold and the base phase is
# Phase 1b (leveraged), demote back to Phase 1 (unleveraged).
SENTIMENT_DEMOTE_1B_THRESHOLD = -0.35
 
# ── Blend-adjustment caps ─────────────────────────────────────────────────────
# The news signal can nudge blend allocations but cannot dominate them.
# Each cap limits how many percentage points a single factor can be shifted.
MAX_GLD_NUDGE   = 0.06   # max +6pp to GLD when geopolitical risk is elevated
MAX_TLT_NUDGE   = 0.05   # max +5pp to TLT when Fed-cut probability is high
MAX_TQQQ_NUDGE  = 0.05   # max +5pp to TQQQ when risk-on score is very strong
 
# ── Geopolitical risk threshold for GLD nudge ─────────────────────────────────
GEO_RISK_GLD_THRESHOLD  = 0.50   # geo_risk_score above this → add GLD
# ── Fed-cut probability threshold for TLT nudge ──────────────────────────────
FED_CUT_TLT_THRESHOLD   = 0.65   # fed_cut_prob above this → add TLT
# ── Risk-on threshold for TQQQ nudge (Phase 1 only) ──────────────────────────
RISK_ON_TQQQ_THRESHOLD  = 0.80   # risk_on_score above this → nudge TQQQ in Ph1
 
# ── HTTP timeout (seconds) for all external requests ─────────────────────────
HTTP_TIMEOUT = 10
 
# ── RSS feed URLs ─────────────────────────────────────────────────────────────
# Chosen for reliability and financial relevance. All are free, no auth.
RSS_FEEDS = {
    "MarketWatch Top Stories":      "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Economy":                 "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Finance":                 "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Federal Reserve Press":        "https://www.federalreserve.gov/feeds/press_all.xml",
    "Yahoo Finance":                "https://finance.yahoo.com/rss/topfinstories",
}
 
# ── Polymarket keyword filters ────────────────────────────────────────────────
# Markets whose questions contain any of these keywords (case-insensitive)
# are fetched and parsed for probability signals.
POLYMARKET_MACRO_KEYWORDS = [
    "recession", "gdp", "federal reserve", "fed rate", "rate cut", "rate hike",
    "inflation", "cpi", "unemployment", "tariff", "trade war", "debt ceiling",
    "treasury", "dollar", "default", "oil price", "crude oil",
    "china", "russia", "ukraine", "taiwan", "israel", "nato",
    "war", "ceasefire", "sanctions", "stock market", "s&p", "nasdaq",
    "bank failure", "credit", "yield curve", "soft landing", "hard landing",
]
 
# ── FOREX tickers (via yfinance) ──────────────────────────────────────────────
FOREX_TICKERS = {
    "DXY":    "DX-Y.NYB",   # US Dollar Index — primary risk-off gauge
    "EURUSD": "EURUSD=X",   # Euro/Dollar
    "USDJPY": "JPY=X",      # Dollar/Yen — yen strength = risk-off
    "USDCNY": "CNY=X",      # Dollar/Yuan — China trade risk
}
FOREX_RETURN_WINDOW = 5   # trading days over which to measure momentum
 
# ── Keyword sentiment weights (fallback when LLM is unavailable) ──────────────
# Positive values = risk-on / bullish; negative values = risk-off / bearish.
# Weights are on a common scale; absolute value indicates signal strength.
SENTIMENT_KEYWORDS = {
    # ── Strong bearish ──
    "recession":        -2.0, "default":          -2.0, "collapse":       -2.0,
    "crisis":           -1.8, "crash":             -1.8, "war escalat":    -1.8,
    "military action":  -1.5, "sanctions":         -1.5, "bank failure":   -1.8,
    "debt ceiling":     -1.2, "hyperinflation":    -1.5, "stagflation":    -1.5,
    "bear market":      -1.5, "sell-off":          -1.2, "panic":          -1.5,
    "downgrade":        -1.2, "tariff hike":       -1.2, "trade war":      -1.2,
    "rate hike":        -1.0, "hawkish":           -1.0, "tightening":     -0.8,
    "layoffs":          -0.8, "unemployment rise": -0.8, "deficit":        -0.6,
    "contraction":      -1.2, "inverted yield":    -1.2, "hard landing":   -1.3,
    # ── Moderate bearish ──
    "uncertainty":      -0.5, "volatility":        -0.4, "concern":        -0.4,
    "warning":          -0.5, "risk":              -0.3, "slowing":        -0.5,
    # ── Moderate bullish ──
    "growth":            0.5,  "recovery":          0.6,  "expansion":      0.7,
    "employment":        0.5,  "upgrade":           0.6,  "beat expectation": 0.7,
    "soft landing":      0.8,  "stability":         0.4,  "optimism":       0.5,
    # ── Strong bullish ──
    "rate cut":          1.2,  "stimulus":          1.2,  "rally":          1.0,
    "bull market":       1.2,  "trade deal":        1.0,  "ceasefire":      1.2,
    "gdp growth":        1.0,  "earnings beat":     0.8,  "strong jobs":    0.8,
    "dovish":            0.8,  "quantitative":      0.5,  "fed pivot":      1.2,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class NewsSignal:
    """
    Structured output produced by NewsAgent.get_signal().
 
    Scores are normalised to consistent ranges:
      macro_sentiment : -1.0 (very bearish) → +1.0 (very bullish)
      risk_on_score   :  0.0 (full risk-off) → 1.0 (full risk-on)
      geo_risk_score  :  0.0 (calm) → 1.0 (extreme geopolitical stress)
 
    Polymarket probabilities are raw market prices (0.0–1.0) from the
    highest-volume matching contracts.
 
    confidence reflects data availability:
      1.0 = all four tiers available and returned data
      0.0 = all tiers failed; signal is purely neutral defaults
    """
    # ── Composite scores ──────────────────────────────────────────────────
    macro_sentiment : float = 0.0    # -1.0 → +1.0
    risk_on_score   : float = 0.5    #  0.0 → 1.0
    geo_risk_score  : float = 0.0    #  0.0 → 1.0
 
    # ── Polymarket probabilities ──────────────────────────────────────────
    recession_prob  : float = 0.0    # "Will there be a US recession in [year]?"
    fed_cut_prob    : float = 0.5    # "Will the Fed cut rates at [next meeting]?"
    war_escalation_prob : float = 0.0  # highest-volume active war/conflict market
 
    # ── FOREX signals ─────────────────────────────────────────────────────
    dxy_5d_return   : float = 0.0    # DXY 5-day return; positive = dollar strong (risk-off)
 
    # ── Headlines ─────────────────────────────────────────────────────────
    headlines       : list  = field(default_factory=list)   # top 20 relevant headlines
    top_risks       : list  = field(default_factory=list)   # key risk factors (LLM or keyword)
    top_tailwinds   : list  = field(default_factory=list)   # key tailwind factors
 
    # ── Metadata ──────────────────────────────────────────────────────────
    confidence      : float = 0.0    # 0.0 → 1.0; how much data was available
    scored_by_llm   : bool  = False  # True if Anthropic API was used for scoring
    timestamp       : datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_counts   : dict  = field(default_factory=dict)   # {"rss": N, "polymarket": N, ...}
 
    # ─────────────────────────────────────────────────────────────────────
    def is_live_date(self, date) -> bool:
        """
        Returns True if `date` is recent enough for the news signal to apply.
        Historical backtest dates always return False — the signal never
        contaminates out-of-sample results.
        """
        try:
            ts = pd.Timestamp(date)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=LIVE_SIGNAL_WINDOW_DAYS)
            return ts >= cutoff
        except Exception:
            return False
 
    # ─────────────────────────────────────────────────────────────────────
    def adjust_phase(self, base_phase, date=None):
        """
        Optionally override the regime phase based on the news signal.
 
        Rules (applied in priority order):
          1. Recession prob > RECESSION_PROB_PHASE3_FORCE and base in {1, "1b", 2}
             → Force Phase 3 (Unwind / full protection).
          2. Recession prob > RECESSION_PROB_PHASE_BRAKE and base == "1b"
             → Demote to Phase 1 (no leveraged acceleration when recession
               is priced above 60%).
          3. macro_sentiment < SENTIMENT_DEMOTE_1B_THRESHOLD and base == "1b"
             → Demote to Phase 1 (news too bearish for TQQQ acceleration).
          4. All other cases → return base_phase unchanged.
 
        Only applied for dates within LIVE_SIGNAL_WINDOW_DAYS of today.
        """
        if date is not None and not self.is_live_date(date):
            return base_phase   # never touch historical backtest dates
 
        if self.confidence < 0.1:
            return base_phase   # too little data to override
 
        # Rule 1: forced Phase 3 on extreme recession pricing
        if (self.recession_prob >= RECESSION_PROB_PHASE3_FORCE
                and base_phase in (1, "1b", 2)):
            return 3
 
        # Rule 2: suppress leveraged acceleration on moderate recession pricing
        if self.recession_prob >= RECESSION_PROB_PHASE_BRAKE and base_phase == "1b":
            return 1
 
        # Rule 3: demote Phase 1b on strongly bearish news
        if self.macro_sentiment < SENTIMENT_DEMOTE_1B_THRESHOLD and base_phase == "1b":
            return 1
 
        return base_phase
 
    # ─────────────────────────────────────────────────────────────────────
    def adjust_blend(self, blend: dict, date=None) -> dict:
        """
        Apply marginal, news-driven nudges to an existing phase blend.
 
        The returned blend is a shallow copy — original is never mutated.
        Adjustments are always small (capped by MAX_*_NUDGE constants) so
        the core regime model's allocation intent is preserved.
 
        Rules:
          1. geo_risk_score > GEO_RISK_GLD_THRESHOLD
             → Shift up to MAX_GLD_NUDGE from FACTOR into GLD.
          2. fed_cut_prob > FED_CUT_TLT_THRESHOLD
             → Shift up to MAX_TLT_NUDGE from FACTOR/SPY into TLT.
          3. risk_on_score > RISK_ON_TQQQ_THRESHOLD (Phase 1 only)
             → Shift up to MAX_TQQQ_NUDGE from SPY into TQQQ.
 
        All nudges are proportional to signal strength so there are no
        cliff-edge discontinuities. The blend is renormalised to sum to 1.0
        after adjustments.
 
        Only applied for dates within LIVE_SIGNAL_WINDOW_DAYS of today.
        """
        if date is not None and not self.is_live_date(date):
            return blend
 
        if self.confidence < 0.1:
            return blend
 
        b = copy.deepcopy(blend)
 
        # ── Rule 1: geopolitical risk → add GLD ──────────────────────────
        if self.geo_risk_score > GEO_RISK_GLD_THRESHOLD:
            strength  = (self.geo_risk_score - GEO_RISK_GLD_THRESHOLD) / (1 - GEO_RISK_GLD_THRESHOLD)
            gld_add   = round(MAX_GLD_NUDGE * strength, 4)
            donor     = "FACTOR"
            if b.get(donor, 0) >= gld_add:
                b[donor]  = round(b.get(donor, 0) - gld_add, 4)
                b["GLD"]  = round(b.get("GLD", 0) + gld_add, 4)
 
        # ── Rule 2: high Fed-cut probability → add TLT ───────────────────
        if self.fed_cut_prob > FED_CUT_TLT_THRESHOLD:
            strength = (self.fed_cut_prob - FED_CUT_TLT_THRESHOLD) / (1 - FED_CUT_TLT_THRESHOLD)
            tlt_add  = round(MAX_TLT_NUDGE * strength, 4)
            # take from FACTOR first, then SPY
            for donor in ("FACTOR", "SPY"):
                available = b.get(donor, 0)
                take      = min(tlt_add, available)
                if take > 0:
                    b[donor] = round(available - take, 4)
                    b["TLT"] = round(b.get("TLT", 0) + take, 4)
                    tlt_add  -= take
                if tlt_add <= 0:
                    break
 
        # ── Rule 3: strong risk-on → nudge TQQQ (Phase 1 base only) ──────
        if (self.risk_on_score > RISK_ON_TQQQ_THRESHOLD
                and b.get("SPY", 0) > 0 and b.get("TQQQ", 0) == 0):
            strength  = (self.risk_on_score - RISK_ON_TQQQ_THRESHOLD) / (1 - RISK_ON_TQQQ_THRESHOLD)
            tqqq_add  = round(MAX_TQQQ_NUDGE * strength, 4)
            if b.get("SPY", 0) >= tqqq_add:
                b["SPY"]  = round(b.get("SPY", 0) - tqqq_add, 4)
                b["TQQQ"] = round(b.get("TQQQ", 0) + tqqq_add, 4)
 
        # ── Renormalise to ensure exact sum = 1.0 ────────────────────────
        total = sum(b.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            b = {k: round(v / total, 6) for k, v in b.items()}
 
        return b
 
    # ─────────────────────────────────────────────────────────────────────
    def summary(self) -> str:
        """Human-readable one-block summary for terminal output."""
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        llm_tag = " [LLM-scored]" if self.scored_by_llm else " [keyword-scored]"
        lines = [
            "",
            "=" * 60,
            f"NEWS AGENT SIGNAL  —  {ts_str}{llm_tag}",
            "=" * 60,
            f"  Macro Sentiment     {self.macro_sentiment:>+.3f}   "
            f"({'bullish' if self.macro_sentiment > 0.1 else 'bearish' if self.macro_sentiment < -0.1 else 'neutral'})",
            f"  Risk-On Score       {self.risk_on_score:>6.3f}   "
            f"({'risk-on' if self.risk_on_score > 0.6 else 'risk-off' if self.risk_on_score < 0.4 else 'neutral'})",
            f"  Geo-Risk Score      {self.geo_risk_score:>6.3f}   "
            f"({'elevated' if self.geo_risk_score > GEO_RISK_GLD_THRESHOLD else 'contained'})",
            f"  Recession Prob      {self.recession_prob:>6.1%}   (Polymarket)",
            f"  Fed Cut Prob        {self.fed_cut_prob:>6.1%}   (Polymarket)",
            f"  DXY 5d Return       {self.dxy_5d_return:>+6.2%}   "
            f"({'strengthening' if self.dxy_5d_return > 0.005 else 'weakening' if self.dxy_5d_return < -0.005 else 'stable'})",
            f"  Signal Confidence   {self.confidence:>6.1%}",
            f"  Sources             {self.source_counts}",
        ]
        if self.top_risks:
            lines.append(f"  Top Risks           {' | '.join(self.top_risks[:3])}")
        if self.top_tailwinds:
            lines.append(f"  Top Tailwinds       {' | '.join(self.top_tailwinds[:3])}")
        if self.headlines:
            lines.append("  Recent Headlines:")
            for h in self.headlines[:5]:
                lines.append(f"    • {h[:90]}")
        lines.append("=" * 60)
        return "\n".join(lines)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — POLYMARKET (prediction market probabilities)
# ─────────────────────────────────────────────────────────────────────────────
 
def _fetch_polymarket_signals() -> dict:
    """
    Fetch active Polymarket markets sorted by volume and filter for
    macro-relevant contracts.
 
    Returns a dict with keys:
      "recession_prob"       : float (highest-volume recession market "Yes" price)
      "fed_cut_prob"         : float (next-meeting Fed cut "Yes" price)
      "war_escalation_prob"  : float (highest-volume war/conflict market "Yes" price)
      "matched_markets"      : list of (question, yes_price, volume) tuples
      "count"                : int (number of matching markets found)
 
    Polymarket Gamma API endpoint (public, no auth required):
      GET https://gamma-api.polymarket.com/markets
          ?active=true&limit=N&order=volume&ascending=false
    """
    result = {
        "recession_prob":      0.0,
        "fed_cut_prob":        0.5,
        "war_escalation_prob": 0.0,
        "matched_markets":     [],
        "count":               0,
    }
 
    try:
        url = (
            f"https://gamma-api.polymarket.com/markets"
            f"?active=true&limit={POLYMARKET_FETCH_LIMIT}"
            f"&order=volume&ascending=false"
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/1.0)",
                "Accept":     "application/json",
            }
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
 
        markets = raw if isinstance(raw, list) else raw.get("markets", [])
 
        recession_candidates     = []
        fed_cut_candidates       = []
        war_escalation_candidates = []
 
        for m in markets:
            question = (m.get("question") or "").lower()
            volume   = float(m.get("volume") or 0)
 
            if volume < POLYMARKET_MIN_VOLUME:
                continue
 
            # Parse outcome prices — field is a JSON-encoded string
            try:
                outcome_prices = json.loads(m.get("outcomePrices", "[]"))
                outcomes       = json.loads(m.get("outcomes", "[]"))
            except (json.JSONDecodeError, TypeError):
                continue
 
            if not outcome_prices or not outcomes:
                continue
 
            # Map "Yes" outcome to its price (probability)
            yes_price = None
            for i, out in enumerate(outcomes):
                if str(out).lower() == "yes" and i < len(outcome_prices):
                    try:
                        yes_price = float(outcome_prices[i])
                    except (ValueError, TypeError):
                        pass
                    break
 
            # If no "Yes" outcome, use first price as proxy
            if yes_price is None:
                try:
                    yes_price = float(outcome_prices[0])
                except (ValueError, TypeError, IndexError):
                    continue
 
            # Check if any macro keyword matches the market question
            if not any(kw in question for kw in POLYMARKET_MACRO_KEYWORDS):
                continue
 
            entry = (m.get("question", ""), yes_price, volume)
            result["matched_markets"].append(entry)
 
            # Classify into signal buckets by question content
            if any(kw in question for kw in ["recession", "gdp contraction", "hard landing"]):
                recession_candidates.append((yes_price, volume))
 
            if any(kw in question for kw in ["rate cut", "fed cut", "federal reserve cut",
                                              "cut rate", "bps cut"]):
                fed_cut_candidates.append((yes_price, volume))
 
            if any(kw in question for kw in ["war", "ceasefire", "military", "invasion",
                                              "escalat", "attack", "conflict"]):
                war_escalation_candidates.append((yes_price, volume))
 
        # Volume-weighted average for each signal bucket
        def vol_weighted_avg(candidates):
            if not candidates:
                return None
            prices  = np.array([c[0] for c in candidates])
            volumes = np.array([c[1] for c in candidates])
            return float(np.average(prices, weights=volumes))
 
        rec_p = vol_weighted_avg(recession_candidates)
        fed_p = vol_weighted_avg(fed_cut_candidates)
        war_p = vol_weighted_avg(war_escalation_candidates)
 
        if rec_p is not None:
            result["recession_prob"] = np.clip(rec_p, 0.0, 1.0)
        if fed_p is not None:
            result["fed_cut_prob"] = np.clip(fed_p, 0.0, 1.0)
        if war_p is not None:
            result["war_escalation_prob"] = np.clip(war_p, 0.0, 1.0)
 
        result["count"] = len(result["matched_markets"])
 
    except Exception as e:
        # Polymarket fetch failed — return safe defaults silently
        result["_error"] = str(e)
 
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — RSS NEWS FEEDS
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_rss_feed(name: str, url: str) -> list:
    """
    Fetch and parse a single RSS 2.0 or Atom feed using only Python stdlib.
    Returns a list of headline strings. Returns [] on any error.
 
    Handles both RSS 2.0 (<item><title>…</title></item>) and
    Atom (<entry><title>…</title></entry>) feed formats.
    """
    headlines = []
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/1.0)",
                "Accept":     "application/rss+xml, application/xml, text/xml, */*",
            }
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            content = resp.read()
 
        root = ET.fromstring(content)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
 
        # RSS 2.0 items
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            if title:
                headlines.append(title)
 
        # Atom entries (fallback)
        if not headlines:
            for entry in root.findall(".//atom:entry", ns):
                t = entry.find("atom:title", ns)
                if t is not None and t.text:
                    headlines.append(t.text.strip())
 
            # Try without namespace prefix
            if not headlines:
                for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                    t = entry.find("{http://www.w3.org/2005/Atom}title")
                    if t is not None and t.text:
                        headlines.append(t.text.strip())
 
    except Exception:
        pass   # silently skip any feed that fails
 
    return headlines
 
 
def _fetch_all_headlines() -> dict:
    """
    Fetch headlines from all configured RSS feeds in parallel (sequential
    fallback if threading is unavailable).
 
    Returns:
      {
        "headlines": list[str],   # deduplicated, all sources combined
        "count":     int,
        "sources":   dict         # {feed_name: headline_count}
      }
    """
    all_headlines = []
    sources       = {}
 
    for name, url in RSS_FEEDS.items():
        hl = _parse_rss_feed(name, url)
        sources[name] = len(hl)
        all_headlines.extend(hl)
 
    # Deduplicate while preserving order
    seen  = set()
    dedup = []
    for h in all_headlines:
        key = h.lower().strip()
        if key not in seen:
            seen.add(key)
            dedup.append(h)
 
    return {
        "headlines": dedup,
        "count":     len(dedup),
        "sources":   sources,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — FOREX SNAPSHOT (via yfinance)
# ─────────────────────────────────────────────────────────────────────────────
 
def _fetch_forex_snapshot() -> dict:
    """
    Compute 5-day returns for DXY and key currency pairs via yfinance.
 
    A surging US dollar (positive DXY return) is historically a risk-off
    signal — capital is fleeing to safety. A weakening dollar alongside
    falling real rates is risk-on.
 
    Returns:
      {
        "dxy_5d_return":    float,
        "eurusd_5d_return": float,
        "usdjpy_5d_return": float,
        "usdcny_5d_return": float,
        "composite_risk_off": float  # 0.0 (risk-on) → 1.0 (risk-off)
      }
    """
    result = {
        "dxy_5d_return":      0.0,
        "eurusd_5d_return":   0.0,
        "usdjpy_5d_return":   0.0,
        "usdcny_5d_return":   0.0,
        "composite_risk_off": 0.5,
    }
 
    try:
        window   = FOREX_RETURN_WINDOW + 5    # fetch extra buffer
        tickers  = list(FOREX_TICKERS.values())
        raw      = yf.download(
            tickers, period=f"{window}d", interval="1d",
            auto_adjust=True, progress=False
        )["Close"]
 
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
 
        raw = raw.ffill().dropna()
 
        returns_5d = {}
        for name, ticker in FOREX_TICKERS.items():
            if ticker in raw.columns and len(raw) >= FOREX_RETURN_WINDOW + 1:
                r = float(raw[ticker].iloc[-1] / raw[ticker].iloc[-FOREX_RETURN_WINDOW - 1] - 1)
                returns_5d[name] = r
 
        result["dxy_5d_return"]    = returns_5d.get("DXY",    0.0)
        result["eurusd_5d_return"] = returns_5d.get("EURUSD", 0.0)
        result["usdjpy_5d_return"] = returns_5d.get("USDJPY", 0.0)
        result["usdcny_5d_return"] = returns_5d.get("USDCNY", 0.0)
 
        # Composite risk-off score from FOREX signals.
        # Positive DXY = risk-off. Stronger JPY (negative USD/JPY return) = risk-off.
        dxy_signal = np.clip(returns_5d.get("DXY", 0.0) * 20, -1.0, 1.0)   # scale ±5% → ±1
        jpy_signal = np.clip(-returns_5d.get("USDJPY", 0.0) * 20, -1.0, 1.0)
        composite  = (dxy_signal * 0.6 + jpy_signal * 0.4)                  # weighted blend
        result["composite_risk_off"] = float(np.clip((composite + 1) / 2, 0.0, 1.0))
 
    except Exception as e:
        result["_error"] = str(e)
 
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 4a — LLM HEADLINE SCORING (optional, via Anthropic API)
# ─────────────────────────────────────────────────────────────────────────────
 
def _score_headlines_llm(headlines: list) -> dict:
    """
    Send the top N headlines to claude-haiku for structured macro sentiment
    scoring. Returns a dict with all fields or None if the call fails.
 
    The API key is read from the ANTHROPIC_API_KEY environment variable.
    If the key is missing or the call errors, None is returned and the
    keyword fallback (_score_headlines_keywords) is used instead.
 
    Model: claude-haiku-4-5-20251001 — fastest and lowest cost per token.
    Output: JSON object with standardised keys (see prompt below).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
 
    try:
        import anthropic   # lazy import — only required if key is set
    except ImportError:
        return None
 
    # Use the top 30 headlines to stay well within token budget
    sample = headlines[:30]
    if not sample:
        return None
 
    headlines_text = "\n".join(f"- {h}" for h in sample)
 
    prompt = f"""You are a quantitative macro risk analyst. Analyze these recent financial news headlines and return ONLY a JSON object with exactly these fields (no markdown, no preamble):
 
{{
  "macro_sentiment": <float -1.0 to 1.0, where -1 = strongly bearish, 0 = neutral, 1 = strongly bullish>,
  "risk_on_score": <float 0.0 to 1.0, where 0 = full risk-off/defensive, 1 = full risk-on/growth>,
  "geo_risk_score": <float 0.0 to 1.0, where 0 = geopolitically calm, 1 = extreme geopolitical stress>,
  "top_risks": [<up to 3 concise risk factor strings, max 6 words each>],
  "top_tailwinds": [<up to 3 concise tailwind factor strings, max 6 words each>],
  "reasoning": <one sentence, max 20 words>
}}
 
Headlines to analyze:
{headlines_text}
 
Return only valid JSON. No other text."""
 
    try:
        client   = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 500,
            messages   = [{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text.strip()
 
        # Strip any accidental markdown fences
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed   = json.loads(raw_text)
 
        return {
            "macro_sentiment": float(np.clip(parsed.get("macro_sentiment", 0.0), -1.0, 1.0)),
            "risk_on_score":   float(np.clip(parsed.get("risk_on_score",   0.5),  0.0, 1.0)),
            "geo_risk_score":  float(np.clip(parsed.get("geo_risk_score",  0.0),  0.0, 1.0)),
            "top_risks":       parsed.get("top_risks",      []),
            "top_tailwinds":   parsed.get("top_tailwinds",  []),
            "scored_by_llm":   True,
        }
 
    except Exception:
        return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 4b — KEYWORD SENTIMENT FALLBACK (no API required)
# ─────────────────────────────────────────────────────────────────────────────
 
def _score_headlines_keywords(headlines: list) -> dict:
    """
    Deterministic, keyword-based sentiment scoring.
    Used when the Anthropic API is unavailable or returns an error.
 
    Computes a weighted sentiment score from SENTIMENT_KEYWORDS hits
    across all headlines. Normalises to the standard -1.0 → +1.0 range.
 
    Also derives geo_risk_score from war/conflict keyword density,
    and risk_on_score as a simple linear transform of macro_sentiment.
    """
    if not headlines:
        return {
            "macro_sentiment": 0.0,
            "risk_on_score":   0.5,
            "geo_risk_score":  0.0,
            "top_risks":       [],
            "top_tailwinds":   [],
            "scored_by_llm":   False,
        }
 
    combined_text = " ".join(headlines).lower()
 
    # Weighted keyword scan
    raw_score    = 0.0
    hit_positive = []
    hit_negative = []
 
    for keyword, weight in SENTIMENT_KEYWORDS.items():
        count = combined_text.count(keyword.lower())
        if count > 0:
            raw_score += weight * count
            if weight > 0:
                hit_positive.append((keyword, weight * count))
            else:
                hit_negative.append((keyword, weight * count))
 
    # Normalise: sigmoid-like compression into -1 to +1
    # Divide by number of headlines to remove length bias
    n            = max(len(headlines), 1)
    norm_score   = np.tanh(raw_score / n)
    macro_sent   = float(np.clip(norm_score, -1.0, 1.0))
    risk_on      = float(np.clip((macro_sent + 1) / 2, 0.0, 1.0))
 
    # Geo-risk: density of conflict keywords
    geo_keywords = ["war", "military", "invasion", "attack", "missile", "drone",
                    "ceasefire", "escalat", "nuclear", "troops", "sanctions"]
    geo_hits     = sum(combined_text.count(kw) for kw in geo_keywords)
    geo_score    = float(np.clip(geo_hits / (n * 2), 0.0, 1.0))
 
    # Top risks / tailwinds from highest-weight hits
    hit_negative.sort(key=lambda x: x[1])   # most negative first
    hit_positive.sort(key=lambda x: -x[1])  # most positive first
 
    top_risks      = [k for k, _ in hit_negative[:3]]
    top_tailwinds  = [k for k, _ in hit_positive[:3]]
 
    return {
        "macro_sentiment": macro_sent,
        "risk_on_score":   risk_on,
        "geo_risk_score":  geo_score,
        "top_risks":       top_risks,
        "top_tailwinds":   top_tailwinds,
        "scored_by_llm":   False,
    }
 
 
# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_composite(rss_data: dict, poly_data: dict,
                     forex_data: dict, sentiment: dict) -> NewsSignal:
    """
    Merge outputs from all four tiers into a single NewsSignal.
 
    Weighting logic:
      - Polymarket recession_prob overrides sentiment if very high (> 0.65)
        because prediction-market capital commitment is a stronger signal
        than news tone.
      - FOREX composite_risk_off adjusts the final risk_on_score by up to
        ±15pp based on dollar/yen momentum.
      - Confidence score is proportional to the number of live data sources
        that returned usable data.
 
    Each field is computed independently and clipped to its valid range
    before being stored in the NewsSignal.
    """
    # ── Base sentiment from RSS + LLM/keyword tier ────────────────────────
    macro_sent  = sentiment.get("macro_sentiment", 0.0)
    risk_on     = sentiment.get("risk_on_score",   0.5)
    geo_risk    = sentiment.get("geo_risk_score",  0.0)
 
    # ── Polymarket override / blend ───────────────────────────────────────
    rec_prob    = poly_data.get("recession_prob",       0.0)
    fed_prob    = poly_data.get("fed_cut_prob",         0.5)
    war_prob    = poly_data.get("war_escalation_prob",  0.0)
 
    # If Polymarket strongly prices recession, pull macro_sentiment bearish
    if rec_prob > 0.5:
        pull_strength = (rec_prob - 0.5) / 0.5    # 0.0 → 1.0 as rec_prob goes 0.5 → 1.0
        macro_sent    = macro_sent * (1 - pull_strength * 0.5) - pull_strength * 0.5
        risk_on       = risk_on   * (1 - pull_strength * 0.4)
 
    # If Polymarket prices active war escalation, elevate geo_risk
    if war_prob > 0.3:
        geo_risk = max(geo_risk, float(np.clip(war_prob * 0.8, 0.0, 1.0)))
 
    # ── FOREX adjustment to risk_on ───────────────────────────────────────
    forex_risk_off = forex_data.get("composite_risk_off", 0.5)
    # FOREX modulates risk_on by up to ±15pp
    forex_adjustment = (0.5 - forex_risk_off) * 0.3   # positive when risk-on forex
    risk_on = float(np.clip(risk_on + forex_adjustment, 0.0, 1.0))
 
    # ── Final clip and round ──────────────────────────────────────────────
    macro_sent = float(np.clip(macro_sent, -1.0, 1.0))
    geo_risk   = float(np.clip(geo_risk,   0.0,  1.0))
 
    # ── Confidence score ──────────────────────────────────────────────────
    # Each tier contributes 0.25 to confidence if it returned usable data.
    rss_ok     = rss_data.get("count", 0) > 0
    poly_ok    = poly_data.get("count", 0) > 0
    forex_ok   = "_error" not in forex_data
    sent_ok    = sentiment.get("macro_sentiment") is not None
 
    confidence = sum([rss_ok, poly_ok, forex_ok, sent_ok]) / 4.0
 
    # ── Source count summary ──────────────────────────────────────────────
    source_counts = {
        "rss_headlines":     rss_data.get("count", 0),
        "polymarket_markets": poly_data.get("count", 0),
        "forex_ok":          forex_ok,
        "llm_scored":        sentiment.get("scored_by_llm", False),
    }
 
    return NewsSignal(
        macro_sentiment      = round(macro_sent, 4),
        risk_on_score        = round(risk_on,    4),
        geo_risk_score       = round(geo_risk,   4),
        recession_prob       = round(rec_prob,   4),
        fed_cut_prob         = round(fed_prob,   4),
        war_escalation_prob  = round(war_prob,   4),
        dxy_5d_return        = round(forex_data.get("dxy_5d_return", 0.0), 5),
        headlines            = rss_data.get("headlines", [])[:20],
        top_risks            = sentiment.get("top_risks",     []),
        top_tailwinds        = sentiment.get("top_tailwinds", []),
        confidence           = round(confidence, 4),
        scored_by_llm        = sentiment.get("scored_by_llm", False),
        timestamp            = datetime.now(timezone.utc),
        source_counts        = source_counts,
    )
 
 
# ─────────────────────────────────────────────────────────────────────────────
# NewsAgent — main entry point
# ─────────────────────────────────────────────────────────────────────────────
 
class NewsAgent:
    """
    Orchestrates all four data tiers and returns a cached NewsSignal.
 
    Usage:
      agent  = NewsAgent()
      signal = agent.get_signal()   # fetches all tiers, caches result
      print(signal.summary())
 
      # Force a fresh fetch (ignore cache):
      signal = agent.refresh()
    """
 
    def __init__(self):
        self._cached_signal    : Optional[NewsSignal] = None
        self._cache_expires_at : Optional[datetime]   = None
 
    def _is_cache_valid(self) -> bool:
        if self._cached_signal is None or self._cache_expires_at is None:
            return False
        return datetime.now(timezone.utc) < self._cache_expires_at
 
    def get_signal(self) -> NewsSignal:
        """
        Return a NewsSignal, using a cached result if it is still within
        CACHE_TTL_HOURS. Fetches fresh data otherwise.
 
        This is the primary method to call from FunctionalAnalysisModel.py.
        Gracefully degrades: if all external calls fail, returns a neutral
        low-confidence NewsSignal rather than raising an exception.
        """
        if self._is_cache_valid():
            return self._cached_signal
        return self.refresh()
 
    def refresh(self) -> NewsSignal:
        """
        Force a fresh fetch from all four tiers regardless of cache state.
        Updates the cache and returns the new NewsSignal.
        """
        print("  [NewsAgent] Fetching Polymarket signals...", end=" ", flush=True)
        poly_data  = _fetch_polymarket_signals()
        print(f"found {poly_data.get('count', 0)} relevant markets.")
 
        print("  [NewsAgent] Fetching RSS headlines...",      end=" ", flush=True)
        rss_data   = _fetch_all_headlines()
        print(f"found {rss_data.get('count', 0)} headlines.")
 
        print("  [NewsAgent] Fetching FOREX snapshot...",     end=" ", flush=True)
        forex_data = _fetch_forex_snapshot()
        dxy_str    = f"DXY 5d={forex_data.get('dxy_5d_return', 0):.2%}"
        print(f"{dxy_str}.")
 
        print("  [NewsAgent] Scoring headlines...",           end=" ", flush=True)
        # Try LLM first, fall back to keyword scoring
        sentiment  = _score_headlines_llm(rss_data.get("headlines", []))
        if sentiment is None:
            sentiment = _score_headlines_keywords(rss_data.get("headlines", []))
            print("keyword fallback.")
        else:
            print("LLM scored.")
 
        signal = _build_composite(rss_data, poly_data, forex_data, sentiment)
 
        # Cache the result
        self._cached_signal    = signal
        self._cache_expires_at = datetime.now(timezone.utc) + timedelta(hours=CACHE_TTL_HOURS)
 
        return signal
