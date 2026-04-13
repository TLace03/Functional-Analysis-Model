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
 
# NewsAgent.py — Real-time macro/micro news sentiment and prediction market monitor
# Part of the Spatial-Temporal Portfolio Model project.
# Copyright 2026 Lacy, Thomas Joseph — Apache License 2.0
 
# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# Provides a live NewsSignal imported by FAMWithAIA.py to optionally adjust
# the current-day regime phase and blend allocations.
#
# Four data source tiers:
#   Tier 1 — Polymarket:  prediction-market probabilities (recession, Fed, war)
#   Tier 2 — RSS feeds:   MarketWatch, CNBC, Federal Reserve, Yahoo Finance
#   Tier 3 — FOREX:       DXY, EUR/USD, USD/JPY, USD/CNY (via yfinance)
#   Tier 4 — LLM scoring: claude-haiku headline sentiment (keyword fallback)
#
# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE OPTIMIZATION LOG (v2 — Concurrent I/O)
# ─────────────────────────────────────────────────────────────────────────────
# Problem in v1:
#   All four tiers were fetched SEQUENTIALLY in refresh():
#     poly_data  = _fetch_polymarket_signals()    # ~0.5–2s
#     rss_data   = _fetch_all_headlines()         # 5 feeds × ~0.5s = 2.5s
#     forex_data = _fetch_forex_snapshot()        # ~1s
#     sentiment  = _score_headlines_llm(...)      # ~0.3–2s
#   Total wall time: ~5–8 seconds per refresh.
#
#   Within _fetch_all_headlines(), the 5 RSS feeds were also
#   fetched sequentially, compounding the latency.
#
# Solution in v2:
#   All four tiers launch simultaneously using ThreadPoolExecutor.
#   Since every tier is pure network I/O (HTTP requests, yfinance
#   download), they all block on the network — not the CPU.
#   Python threads release the GIL during I/O, so 4 threads waiting
#   on network responses truly run concurrently in wall-clock time.
#
#   Similarly, the 5 RSS feeds are fetched in parallel within
#   _fetch_all_headlines(), reducing RSS fetch time from ~2.5s
#   (sequential) to ~0.5s (the single slowest feed).
#
#   Expected wall-time reduction: ~5-8s → ~1-2s (the slowest
#   single tier, because all others finish simultaneously).
#
# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST SAFETY — CRITICAL
# ─────────────────────────────────────────────────────────────────────────────
# adjust_phase() and adjust_blend() are guaranteed no-ops for any
# date older than LIVE_SIGNAL_WINDOW_DAYS.  The full out-of-sample
# backtest is NEVER touched by live news data.
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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
 
import numpy as np
import pandas as pd
import yfinance as yf
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
 
CACHE_TTL_HOURS = 4
 
# Only apply live news signal to dates within this many calendar days of now.
# Guarantees the historical backtest is never contaminated.
LIVE_SIGNAL_WINDOW_DAYS = 5
 
POLYMARKET_FETCH_LIMIT = 300
POLYMARKET_MIN_VOLUME  = 50_000
 
# Phase-override thresholds
RECESSION_PROB_PHASE_BRAKE   = 0.60
RECESSION_PROB_PHASE3_FORCE  = 0.75
SENTIMENT_DEMOTE_1B_THRESHOLD = -0.35
 
# Blend-adjustment caps
MAX_GLD_NUDGE  = 0.06
MAX_TLT_NUDGE  = 0.05
MAX_TQQQ_NUDGE = 0.05
 
GEO_RISK_GLD_THRESHOLD = 0.50
FED_CUT_TLT_THRESHOLD  = 0.65
RISK_ON_TQQQ_THRESHOLD = 0.80
 
HTTP_TIMEOUT = 10
 
# Number of parallel threads for I/O operations.
# RSS feeds: 5 concurrent HTTP requests.
# Tier parallelism: up to 4 concurrent requests (one per tier).
# I/O threads don't compete for CPU, so using more threads is safe.
N_IO_THREADS = 8
 
RSS_FEEDS = {
    "MarketWatch Top Stories": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Economy":            "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Finance":            "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Federal Reserve Press":   "https://www.federalreserve.gov/feeds/press_all.xml",
    "Yahoo Finance":           "https://finance.yahoo.com/rss/topfinstories",
}
 
POLYMARKET_MACRO_KEYWORDS = [
    "recession", "gdp", "federal reserve", "fed rate", "rate cut", "rate hike",
    "inflation", "cpi", "unemployment", "tariff", "trade war", "debt ceiling",
    "treasury", "dollar", "default", "oil price", "crude oil",
    "china", "russia", "ukraine", "taiwan", "israel", "nato",
    "war", "ceasefire", "sanctions", "stock market", "s&p", "nasdaq",
    "bank failure", "credit", "yield curve", "soft landing", "hard landing",
]
 
FOREX_TICKERS = {
    "DXY":    "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "USDCNY": "CNY=X",
}
FOREX_RETURN_WINDOW = 5
 
SENTIMENT_KEYWORDS = {
    "recession":        -2.0, "default":          -2.0, "collapse":       -2.0,
    "crisis":           -1.8, "crash":             -1.8, "war escalat":    -1.8,
    "military action":  -1.5, "sanctions":         -1.5, "bank failure":   -1.8,
    "debt ceiling":     -1.2, "hyperinflation":    -1.5, "stagflation":    -1.5,
    "bear market":      -1.5, "sell-off":          -1.2, "panic":          -1.5,
    "downgrade":        -1.2, "tariff hike":       -1.2, "trade war":      -1.2,
    "rate hike":        -1.0, "hawkish":           -1.0, "tightening":     -0.8,
    "layoffs":          -0.8, "unemployment rise": -0.8, "deficit":        -0.6,
    "contraction":      -1.2, "inverted yield":    -1.2, "hard landing":   -1.3,
    "uncertainty":      -0.5, "volatility":        -0.4, "concern":        -0.4,
    "warning":          -0.5, "risk":              -0.3, "slowing":        -0.5,
    "growth":            0.5, "recovery":           0.6, "expansion":       0.7,
    "employment":        0.5, "upgrade":            0.6, "beat expectation": 0.7,
    "soft landing":      0.8, "stability":          0.4, "optimism":        0.5,
    "rate cut":          1.2, "stimulus":           1.2, "rally":           1.0,
    "bull market":       1.2, "trade deal":         1.0, "ceasefire":       1.2,
    "gdp growth":        1.0, "earnings beat":      0.8, "strong jobs":     0.8,
    "dovish":            0.8, "quantitative":       0.5, "fed pivot":       1.2,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class NewsSignal:
    """
    Structured output from NewsAgent.get_signal().
 
    Score ranges:
      macro_sentiment : -1.0 (very bearish) → +1.0 (very bullish)
      risk_on_score   :  0.0 (full risk-off) → 1.0 (full risk-on)
      geo_risk_score  :  0.0 (calm)           → 1.0 (extreme stress)
 
    confidence reflects data availability across all four tiers:
      1.0 = all four tiers returned data
      0.0 = all tiers failed; signal defaults to neutral
    """
    macro_sentiment     : float = 0.0
    risk_on_score       : float = 0.5
    geo_risk_score      : float = 0.0
    recession_prob      : float = 0.0
    fed_cut_prob        : float = 0.5
    war_escalation_prob : float = 0.0
    dxy_5d_return       : float = 0.0
    headlines           : list  = field(default_factory=list)
    top_risks           : list  = field(default_factory=list)
    top_tailwinds       : list  = field(default_factory=list)
    confidence          : float = 0.0
    scored_by_llm       : bool  = False
    timestamp           : datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_counts       : dict  = field(default_factory=dict)
 
    def is_live_date(self, date) -> bool:
        """
        True only if `date` is recent enough for the live signal to apply.
        Historical backtest dates always return False — full backtest safety.
        """
        try:
            ts = pd.Timestamp(date)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=LIVE_SIGNAL_WINDOW_DAYS)
            return ts >= cutoff
        except Exception:
            return False
 
    def adjust_phase(self, base_phase, date=None):
        """
        Optionally override the regime phase.
 
        Priority order (highest wins):
          1. recession_prob ≥ RECESSION_PROB_PHASE3_FORCE  → force Phase 3
          2. recession_prob ≥ RECESSION_PROB_PHASE_BRAKE
             AND base == "1b"                              → demote to Phase 1
          3. macro_sentiment < SENTIMENT_DEMOTE_1B_THRESHOLD
             AND base == "1b"                              → demote to Phase 1
          4. No change.
 
        Only applies to dates within LIVE_SIGNAL_WINDOW_DAYS — all historical
        backtest dates pass through unchanged.
        """
        if date is not None and not self.is_live_date(date):
            return base_phase
        if self.confidence < 0.1:
            return base_phase
 
        if (self.recession_prob >= RECESSION_PROB_PHASE3_FORCE
                and base_phase in (1, "1b", 2)):
            return 3
        if self.recession_prob >= RECESSION_PROB_PHASE_BRAKE and base_phase == "1b":
            return 1
        if self.macro_sentiment < SENTIMENT_DEMOTE_1B_THRESHOLD and base_phase == "1b":
            return 1
 
        return base_phase
 
    def adjust_blend(self, blend: dict, date=None) -> dict:
        """
        Apply marginal news-driven nudges to an existing phase blend.
 
        Rules (proportional to signal strength, capped by MAX_*_NUDGE):
          1. geo_risk_score > GEO_RISK_GLD_THRESHOLD  → shift FACTOR → GLD
          2. fed_cut_prob > FED_CUT_TLT_THRESHOLD     → shift FACTOR/SPY → TLT
          3. risk_on_score > RISK_ON_TQQQ_THRESHOLD
             AND Phase 1 base only                    → shift SPY → TQQQ
 
        Returns a shallow copy — original blend is never mutated.
        Renormalized to sum = 1.0 after adjustments.
        Only applied to live dates.
        """
        if date is not None and not self.is_live_date(date):
            return blend
        if self.confidence < 0.1:
            return blend
 
        b = copy.deepcopy(blend)
 
        if self.geo_risk_score > GEO_RISK_GLD_THRESHOLD:
            strength = (self.geo_risk_score - GEO_RISK_GLD_THRESHOLD) / (1 - GEO_RISK_GLD_THRESHOLD)
            gld_add  = round(MAX_GLD_NUDGE * strength, 4)
            if b.get("FACTOR", 0) >= gld_add:
                b["FACTOR"] = round(b.get("FACTOR", 0) - gld_add, 4)
                b["GLD"]    = round(b.get("GLD",    0) + gld_add, 4)
 
        if self.fed_cut_prob > FED_CUT_TLT_THRESHOLD:
            strength = (self.fed_cut_prob - FED_CUT_TLT_THRESHOLD) / (1 - FED_CUT_TLT_THRESHOLD)
            tlt_add  = round(MAX_TLT_NUDGE * strength, 4)
            for donor in ("FACTOR", "SPY"):
                available = b.get(donor, 0)
                take      = min(tlt_add, available)
                if take > 0:
                    b[donor] = round(available - take, 4)
                    b["TLT"] = round(b.get("TLT", 0) + take, 4)
                    tlt_add -= take
                if tlt_add <= 0:
                    break
 
        if (self.risk_on_score > RISK_ON_TQQQ_THRESHOLD
                and b.get("SPY", 0) > 0 and b.get("TQQQ", 0) == 0):
            strength = (self.risk_on_score - RISK_ON_TQQQ_THRESHOLD) / (1 - RISK_ON_TQQQ_THRESHOLD)
            tqqq_add = round(MAX_TQQQ_NUDGE * strength, 4)
            if b.get("SPY", 0) >= tqqq_add:
                b["SPY"]  = round(b.get("SPY",  0) - tqqq_add, 4)
                b["TQQQ"] = round(b.get("TQQQ", 0) + tqqq_add, 4)
 
        total = sum(b.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            b = {k: round(v / total, 6) for k, v in b.items()}
 
        return b
 
    def summary(self) -> str:
        ts_str  = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        llm_tag = " [LLM-scored]" if self.scored_by_llm else " [keyword-scored]"
        lines   = [
            "", "=" * 60,
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
# TIER 1 — POLYMARKET
# ─────────────────────────────────────────────────────────────────────────────
 
def _fetch_polymarket_signals() -> dict:
    """
    Fetch active Polymarket markets and extract macro probability signals.
 
    API: GET https://gamma-api.polymarket.com/markets?active=true&order=volume
    No authentication required.  Markets are sorted by volume descending
    so we get the highest-conviction signals in the first POLYMARKET_FETCH_LIMIT
    results.
 
    Returns probabilities for: recession, Fed rate cut, war escalation.
    All probabilities are volume-weighted averages across matching contracts.
    """
    result = {
        "recession_prob": 0.0, "fed_cut_prob": 0.5,
        "war_escalation_prob": 0.0, "matched_markets": [], "count": 0,
    }
 
    try:
        url = (f"https://gamma-api.polymarket.com/markets"
               f"?active=true&limit={POLYMARKET_FETCH_LIMIT}"
               f"&order=volume&ascending=false")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/1.0)",
                     "Accept":     "application/json"}
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
            if not any(kw in question for kw in POLYMARKET_MACRO_KEYWORDS):
                continue
 
            try:
                outcome_prices = json.loads(m.get("outcomePrices", "[]"))
                outcomes       = json.loads(m.get("outcomes", "[]"))
            except (json.JSONDecodeError, TypeError):
                continue
 
            if not outcome_prices or not outcomes:
                continue
 
            yes_price = None
            for i, out in enumerate(outcomes):
                if str(out).lower() == "yes" and i < len(outcome_prices):
                    try:
                        yes_price = float(outcome_prices[i])
                    except (ValueError, TypeError):
                        pass
                    break
            if yes_price is None:
                try:
                    yes_price = float(outcome_prices[0])
                except (ValueError, TypeError, IndexError):
                    continue
 
            entry = (m.get("question", ""), yes_price, volume)
            result["matched_markets"].append(entry)
 
            if any(kw in question for kw in ["recession", "gdp contraction", "hard landing"]):
                recession_candidates.append((yes_price, volume))
            if any(kw in question for kw in ["rate cut", "fed cut", "federal reserve cut", "cut rate", "bps cut"]):
                fed_cut_candidates.append((yes_price, volume))
            if any(kw in question for kw in ["war", "ceasefire", "military", "invasion", "escalat", "attack", "conflict"]):
                war_escalation_candidates.append((yes_price, volume))
 
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
        result["_error"] = str(e)
 
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — RSS NEWS FEEDS (parallel fetch)
# ─────────────────────────────────────────────────────────────────────────────
 
def _parse_rss_feed(name: str, url: str) -> list:
    """
    Fetch and parse a single RSS 2.0 / Atom feed using only Python stdlib.
    Returns a list of headline strings.  Returns [] on any error.
 
    Handles:
      RSS 2.0: <item><title>…</title></item>
      Atom:    <entry><title>…</title></entry>  (with and without namespace prefix)
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
 
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            if title:
                headlines.append(title)
 
        if not headlines:
            for entry in root.findall(".//atom:entry", ns):
                t = entry.find("atom:title", ns)
                if t is not None and t.text:
                    headlines.append(t.text.strip())
 
        if not headlines:
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                t = entry.find("{http://www.w3.org/2005/Atom}title")
                if t is not None and t.text:
                    headlines.append(t.text.strip())
 
    except Exception:
        pass
 
    return headlines
 
 
def _fetch_all_headlines() -> dict:
    """
    Fetch headlines from all configured RSS feeds IN PARALLEL.
 
    OPTIMIZATION: v1 fetched each feed sequentially:
      feed_1 (~0.4s) → feed_2 (~0.5s) → feed_3 (~0.6s) → ... = ~2.5s total
 
    v2 launches all 5 feeds simultaneously:
      All 5 feeds run concurrently — total time ≈ slowest single feed (~0.6s).
 
    Implementation:
      ThreadPoolExecutor submits one task per RSS feed URL.
      as_completed() collects results as each feed finishes — fast feeds
      don't wait for slow ones.
 
    Returns deduplicated headline list with per-source counts.
    """
    all_headlines = []
    sources       = {}
 
    # ── Parallel RSS fetch ─────────────────────────────────────
    # Each thread runs _parse_rss_feed() independently.
    # Network I/O releases the GIL, so all 5 threads truly run
    # at the same time (not just interleaved).
    with ThreadPoolExecutor(max_workers=min(len(RSS_FEEDS), N_IO_THREADS)) as executor:
        # Submit all feed fetches simultaneously
        future_to_name = {
            executor.submit(_parse_rss_feed, name, url): name
            for name, url in RSS_FEEDS.items()
        }
 
        # Collect results as each feed completes (not in submission order)
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                headlines = future.result()
            except Exception:
                headlines = []
            sources[name] = len(headlines)
            all_headlines.extend(headlines)
 
    # Deduplicate while preserving order
    seen  = set()
    dedup = []
    for h in all_headlines:
        key = h.lower().strip()
        if key not in seen:
            seen.add(key)
            dedup.append(h)
 
    return {"headlines": dedup, "count": len(dedup), "sources": sources}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — FOREX SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
 
def _fetch_forex_snapshot() -> dict:
    """
    Compute 5-day returns for DXY and key FX pairs via yfinance.
 
    Risk-off indicators:
      Rising DXY (positive return)  → capital fleeing to USD safety
      Strengthening JPY (negative USD/JPY return) → classic risk-off
 
    composite_risk_off:
      0.0 = strongly risk-on (weak dollar, weak yen)
      1.0 = strongly risk-off (strong dollar, strong yen)
    """
    result = {
        "dxy_5d_return": 0.0, "eurusd_5d_return": 0.0,
        "usdjpy_5d_return": 0.0, "usdcny_5d_return": 0.0,
        "composite_risk_off": 0.5,
    }
 
    try:
        window  = FOREX_RETURN_WINDOW + 5
        tickers = list(FOREX_TICKERS.values())
        raw     = yf.download(tickers, period=f"{window}d", interval="1d",
                              auto_adjust=True, progress=False)["Close"]
 
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
 
        # Composite risk-off: scale ±5% move → ±1 signal
        dxy_signal = np.clip(returns_5d.get("DXY", 0.0) * 20, -1.0, 1.0)
        jpy_signal = np.clip(-returns_5d.get("USDJPY", 0.0) * 20, -1.0, 1.0)
        composite  = dxy_signal * 0.6 + jpy_signal * 0.4
        result["composite_risk_off"] = float(np.clip((composite + 1) / 2, 0.0, 1.0))
 
    except Exception as e:
        result["_error"] = str(e)
 
    return result
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 4a — LLM HEADLINE SCORING (optional)
# ─────────────────────────────────────────────────────────────────────────────
 
def _score_headlines_llm(headlines: list) -> Optional[dict]:
    """
    Send top 30 headlines to claude-haiku for structured macro sentiment.
 
    Requires ANTHROPIC_API_KEY environment variable.
    Returns None on any error → caller uses keyword fallback.
 
    The LLM returns a JSON object with standardized keys.
    Model: claude-haiku-4-5-20251001 (fastest, lowest cost per token).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
 
    try:
        import anthropic
    except ImportError:
        return None
 
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
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        parsed   = json.loads(raw_text)
 
        return {
            "macro_sentiment": float(np.clip(parsed.get("macro_sentiment", 0.0), -1.0, 1.0)),
            "risk_on_score":   float(np.clip(parsed.get("risk_on_score",   0.5),  0.0, 1.0)),
            "geo_risk_score":  float(np.clip(parsed.get("geo_risk_score",  0.0),  0.0, 1.0)),
            "top_risks":       parsed.get("top_risks",     []),
            "top_tailwinds":   parsed.get("top_tailwinds", []),
            "scored_by_llm":   True,
        }
    except Exception:
        return None
 
 
# ─────────────────────────────────────────────────────────────────────────────
# TIER 4b — KEYWORD SENTIMENT FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
 
def _score_headlines_keywords(headlines: list) -> dict:
    """
    Deterministic keyword-based sentiment scoring.
    Used when the Anthropic API is unavailable.
 
    Method:
      1. Concatenate all headlines into one lowercase string.
      2. Count occurrences of each SENTIMENT_KEYWORDS entry.
      3. Weighted sum: raw_score = Σ(weight × count).
      4. Normalize: macro_sentiment = tanh(raw_score / n_headlines).
         tanh compresses to [-1, +1] and handles long/short
         article sets without bias.
      5. Derive risk_on and geo_risk from macro_sentiment and
         conflict keyword density.
    """
    if not headlines:
        return {"macro_sentiment": 0.0, "risk_on_score": 0.5,
                "geo_risk_score": 0.0, "top_risks": [], "top_tailwinds": [],
                "scored_by_llm": False}
 
    combined_text = " ".join(headlines).lower()
 
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
 
    n          = max(len(headlines), 1)
    norm_score = np.tanh(raw_score / n)
    macro_sent = float(np.clip(norm_score, -1.0, 1.0))
    risk_on    = float(np.clip((macro_sent + 1) / 2, 0.0, 1.0))
 
    geo_keywords = ["war", "military", "invasion", "attack", "missile", "drone",
                    "ceasefire", "escalat", "nuclear", "troops", "sanctions"]
    geo_hits  = sum(combined_text.count(kw) for kw in geo_keywords)
    geo_score = float(np.clip(geo_hits / (n * 2), 0.0, 1.0))
 
    hit_negative.sort(key=lambda x: x[1])
    hit_positive.sort(key=lambda x: -x[1])
 
    return {
        "macro_sentiment": macro_sent,
        "risk_on_score":   risk_on,
        "geo_risk_score":  geo_score,
        "top_risks":       [k for k, _ in hit_negative[:3]],
        "top_tailwinds":   [k for k, _ in hit_positive[:3]],
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
      - Polymarket recession_prob overrides sentiment if > 0.5 because
        real-money prediction markets are a stronger signal than tone.
      - FOREX composite_risk_off adjusts final risk_on by ±15pp.
      - Confidence = fraction of tiers that returned usable data.
    """
    macro_sent = sentiment.get("macro_sentiment", 0.0)
    risk_on    = sentiment.get("risk_on_score",   0.5)
    geo_risk   = sentiment.get("geo_risk_score",  0.0)
 
    rec_prob = poly_data.get("recession_prob",      0.0)
    fed_prob = poly_data.get("fed_cut_prob",        0.5)
    war_prob = poly_data.get("war_escalation_prob", 0.0)
 
    if rec_prob > 0.5:
        pull_strength = (rec_prob - 0.5) / 0.5
        macro_sent    = macro_sent * (1 - pull_strength * 0.5) - pull_strength * 0.5
        risk_on       = risk_on   * (1 - pull_strength * 0.4)
 
    if war_prob > 0.3:
        geo_risk = max(geo_risk, float(np.clip(war_prob * 0.8, 0.0, 1.0)))
 
    forex_risk_off = forex_data.get("composite_risk_off", 0.5)
    forex_adj      = (0.5 - forex_risk_off) * 0.3
    risk_on        = float(np.clip(risk_on + forex_adj, 0.0, 1.0))
    macro_sent     = float(np.clip(macro_sent, -1.0, 1.0))
    geo_risk       = float(np.clip(geo_risk,   0.0,  1.0))
 
    rss_ok   = rss_data.get("count", 0) > 0
    poly_ok  = poly_data.get("count", 0) > 0
    forex_ok = "_error" not in forex_data
    sent_ok  = sentiment.get("macro_sentiment") is not None
 
    confidence = sum([rss_ok, poly_ok, forex_ok, sent_ok]) / 4.0
 
    source_counts = {
        "rss_headlines":      rss_data.get("count", 0),
        "polymarket_markets": poly_data.get("count", 0),
        "forex_ok":           forex_ok,
        "llm_scored":         sentiment.get("scored_by_llm", False),
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
 
    OPTIMIZATION: v1 ran all four tiers sequentially (~5-8s total).
    v2 launches Tiers 1, 2, 3 as concurrent threads and runs Tier 4
    (LLM/keyword scoring) after RSS headlines are available — since
    scoring requires the headlines as input.
 
    Execution timeline:
      t=0:       Tier 1 (Polymarket) ──────┐
      t=0:       Tier 2 (RSS feeds)  ──────┤ all 3 start simultaneously
      t=0:       Tier 3 (FOREX)      ──────┘
      t≈0.5s:    RSS done → Tier 4 (LLM/keywords) starts
      t≈1-2s:    all tiers done → _build_composite()
 
    Wall time: ~1-2s (down from ~5-8s in v1)
 
    Usage:
      agent  = NewsAgent()
      signal = agent.get_signal()   # cached for CACHE_TTL_HOURS
      print(signal.summary())
      signal = agent.refresh()      # force re-fetch
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
        Return a NewsSignal, using cached data if still fresh.
        Fetches all tiers in parallel otherwise.
        Gracefully degrades to neutral defaults on any failure.
        """
        if self._is_cache_valid():
            return self._cached_signal
        return self.refresh()
 
    def refresh(self) -> NewsSignal:
        """
        Force a fresh parallel fetch from all four tiers.
 
        Concurrency design:
          Tiers 1 (Polymarket), 2 (RSS), and 3 (FOREX) are all pure
          network I/O with no inter-dependencies.  They are submitted
          as concurrent futures and we wait for ALL of them to complete
          (wait(..., return_when=ALL_COMPLETED)) before proceeding.
 
          Tier 4 (LLM/keyword scoring) depends on Tier 2's headlines,
          so it runs after RSS returns — but while Tiers 1 and 3 may
          still be running.  Using as_completed() for early RSS finish
          gets Tier 4 started as early as possible.
        """
        print("  [NewsAgent] Launching concurrent tier fetch...", flush=True)
 
        rss_data   = None
        poly_data  = None
        forex_data = None
 
        # ── Launch Tiers 1, 2, 3 simultaneously ──────────────────
        with ThreadPoolExecutor(max_workers=3) as tier_pool:
            poly_future  = tier_pool.submit(_fetch_polymarket_signals)
            rss_future   = tier_pool.submit(_fetch_all_headlines)
            forex_future = tier_pool.submit(_fetch_forex_snapshot)
 
            # Wait for ALL three to finish.
            # wall-time ≈ max(Tier1_time, Tier2_time, Tier3_time)
            # instead of  Tier1_time + Tier2_time + Tier3_time
            done, _ = wait(
                [poly_future, rss_future, forex_future],
                return_when=ALL_COMPLETED
            )
 
            for fut in done:
                if fut is poly_future:
                    poly_data = fut.result()
                    print(f"  [NewsAgent] Polymarket: "
                          f"{poly_data.get('count', 0)} markets", flush=True)
                elif fut is rss_future:
                    rss_data = fut.result()
                    print(f"  [NewsAgent] RSS: "
                          f"{rss_data.get('count', 0)} headlines", flush=True)
                elif fut is forex_future:
                    forex_data = fut.result()
                    dxy_str = f"{forex_data.get('dxy_5d_return', 0):.2%}"
                    print(f"  [NewsAgent] FOREX: DXY 5d={dxy_str}", flush=True)
 
        # Provide safe defaults if any tier failed
        if poly_data  is None: poly_data  = {"recession_prob": 0.0, "fed_cut_prob": 0.5, "war_escalation_prob": 0.0, "matched_markets": [], "count": 0}
        if rss_data   is None: rss_data   = {"headlines": [], "count": 0, "sources": {}}
        if forex_data is None: forex_data = {"dxy_5d_return": 0.0, "composite_risk_off": 0.5}
 
        # ── Tier 4: score the headlines ────────────────────────────
        # RSS must be done before we can score.  At this point it is,
        # because we waited for ALL three futures above.
        print("  [NewsAgent] Scoring headlines...", end=" ", flush=True)
        sentiment = _score_headlines_llm(rss_data.get("headlines", []))
        if sentiment is None:
            sentiment = _score_headlines_keywords(rss_data.get("headlines", []))
            print("keyword fallback.", flush=True)
        else:
            print("LLM scored.", flush=True)
 
        signal = _build_composite(rss_data, poly_data, forex_data, sentiment)
 
        self._cached_signal    = signal
        self._cache_expires_at = datetime.now(timezone.utc) + timedelta(hours=CACHE_TTL_HOURS)
 
        return signal
 
