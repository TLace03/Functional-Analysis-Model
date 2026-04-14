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

──────────────────────────────────────────────────────────────────────────────
NewsAgent.py · Real-time macro/micro intelligence layer · v2
──────────────────────────────────────────────────────────────────────────────

OVERVIEW:
This module provides live market intelligence that can optionally nudge the
current-day regime and blend allocations in FAMWithAIA.py.

FIVE DATA TIERS (new in v2: Tier 5 — FRED):
Tier 1 — Polymarket (prediction markets)
Tier 2 — RSS news feeds (no API key required)
Tier 3 — FOREX snapshot (via yfinance)
Tier 4 — LLM headline scoring (optional — needs ANTHROPIC_API_KEY)
Tier 5 — FRED Macroeconomic Data (NEW in v2)

BACKTEST SAFETY:
All phase/blend adjustments are no-ops for historical dates older than
LIVE_SIGNAL_WINDOW_DAYS. The backtest is NEVER contaminated with live data.
"""

import os
import io
import json
import copy
import time
import warnings
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CACHE_TTL_HOURS = 4
LIVE_SIGNAL_WINDOW_DAYS = 5
HTTP_TIMEOUT = 10
N_IO_THREADS = 8

# ── Polymarket ────────────────────────────────────────────────────────────
POLYMARKET_FETCH_LIMIT = 300
POLYMARKET_MIN_VOLUME = 50_000

# ── Phase override thresholds ─────────────────────────────────────────────
RECESSION_PROB_PHASE_BRAKE = 0.60
RECESSION_PROB_PHASE3_FORCE = 0.75
SENTIMENT_DEMOTE_1B_THRESHOLD = -0.35

# ── FRED: additional recession/stress signal ──────────────────────────────
FRED_CURVE_INVERSION_BRAKE = True
FRED_HY_SPREAD_BRAKE_BPS = 500
FRED_UNEMPLOYMENT_SURGE_PCT = 0.5

# ── Blend nudge caps ──────────────────────────────────────────────────────
MAX_GLD_NUDGE = 0.06
MAX_TLT_NUDGE = 0.05
MAX_TQQQ_NUDGE = 0.05

GEO_RISK_GLD_THRESHOLD = 0.50
FED_CUT_TLT_THRESHOLD = 0.65
RISK_ON_TQQQ_THRESHOLD = 0.80

# ── RSS feeds ─────────────────────────────────────────────────────────────
RSS_FEEDS = {
    "MarketWatch Top Stories": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "CNBC Finance": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "Federal Reserve Press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "Yahoo Finance": "https://finance.yahoo.com/rss/topfinstories",
}

# ── Polymarket keyword filters ────────────────────────────────────────────
POLYMARKET_MACRO_KEYWORDS = [
    "recession","gdp","federal reserve","fed rate","rate cut","rate hike",
    "inflation","cpi","unemployment","tariff","trade war","debt ceiling",
    "treasury","dollar","default","oil price","crude oil","china","russia",
    "ukraine","taiwan","israel","nato","war","ceasefire","sanctions",
    "stock market","s&p","nasdaq","bank failure","credit","yield curve",
    "soft landing","hard landing",
]

# ── FOREX ─────────────────────────────────────────────────────────────────
FOREX_TICKERS = {
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "USDCNY": "CNY=X",
}
FOREX_RETURN_WINDOW = 5

# ── FRED: series to fetch (no API key needed) ─────────────────────────────
FRED_SERIES = {
    "FEDFUNDS": ("Federal Funds Rate (%)", "rate"),
    "T10Y2Y": ("10Y-2Y Treasury Spread (pp)", "spread"),
    "UNRATE": ("Unemployment Rate (%)", "rate"),
    "CPIAUCSL": ("CPI All Items (level)", "level"),
    "BAMLH0A0HYM2": ("HY Credit Spread (OAS, bps)", "spread"),
    "DGS10": ("10Y Treasury Yield (%)", "rate"),
    "DGS2": ("2Y Treasury Yield (%)", "rate"),
    "UMCSENT": ("UMich Consumer Sentiment", "sentiment"),
    "DCOILWTICO": ("WTI Crude Oil ($/barrel)", "level"),
}

# ── Sentiment keywords ────────────────────────────────────────────────────
SENTIMENT_KEYWORDS = {
    "recession": -2.0, "default": -2.0, "collapse": -2.0,
    "crisis": -1.8, "crash": -1.8, "war escalat": -1.8,
    "military action": -1.5, "sanctions": -1.5, "bank failure": -1.8,
    "debt ceiling": -1.2, "hyperinflation": -1.5, "stagflation": -1.5,
    "bear market": -1.5, "sell-off": -1.2, "panic": -1.5,
    "downgrade": -1.2, "tariff hike": -1.2, "trade war": -1.2,
    "rate hike": -1.0, "hawkish": -1.0, "tightening": -0.8,
    "layoffs": -0.8, "unemployment rise": -0.8, "deficit": -0.6,
    "contraction": -1.2, "inverted yield": -1.2, "hard landing": -1.3,
    "uncertainty": -0.5, "volatility": -0.4, "concern": -0.4,
    "warning": -0.5, "risk": -0.3, "slowing": -0.5,
    "growth": 0.5, "recovery": 0.6, "expansion": 0.7,
    "employment": 0.5, "upgrade": 0.6, "beat expectation": 0.7,
    "soft landing": 0.8, "stability": 0.4, "optimism": 0.5,
    "rate cut": 1.2, "stimulus": 1.2, "rally": 1.0,
    "bull market": 1.2, "trade deal": 1.0, "ceasefire": 1.2,
    "gdp growth": 1.0, "earnings beat": 0.8, "strong jobs": 0.8,
    "dovish": 0.8, "fed pivot": 1.2,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class FredSignal:
    """Processed signals from FRED macroeconomic data."""
    yield_curve_bps: float = 10.0
    is_inverted: bool = False
    hy_spread_bps: float = 350.0
    hy_spread_trend: str = "stable"
    fed_funds_rate: float = 3.0
    fed_funds_trend: str = "stable"
    cpi_yoy_pct: float = 2.5
    cpi_trend: str = "stable"
    unemployment_rate: float = 4.0
    unemployment_trend: str = "stable"
    consumer_sentiment: float = 80.0
    crude_oil_price: float = 70.0
    fred_risk_score: float = 0.5
    series_loaded: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def summary_lines(self) -> list:
        lines = [
            " ── FRED Macro Dashboard ──────────────────────────────",
            f" Yield Curve (T10-T2) {self.yield_curve_bps:>+7.1f}bps "
            f"{'⚠ INVERTED' if self.is_inverted else ' normal'}",
            f" HY Credit Spread {self.hy_spread_bps:>8.0f}bps "
            f"({self.hy_spread_trend})",
            f" Fed Funds Rate {self.fed_funds_rate:>8.2f}% "
            f"({self.fed_funds_trend})",
            f" CPI YoY {self.cpi_yoy_pct:>8.2f}% "
            f"({self.cpi_trend})",
            f" Unemployment {self.unemployment_rate:>8.2f}% "
            f"({self.unemployment_trend})",
            f" Consumer Sentiment {self.consumer_sentiment:>8.1f} "
            f"(UMich index)",
            f" WTI Crude ${self.crude_oil_price:>7.1f} /barrel",
            f" FRED Risk Score {self.fred_risk_score:>8.3f} "
            f"({self.series_loaded} series loaded)",
        ]
        return lines


@dataclass
class NewsSignal:
    """Structured output from NewsAgent.get_signal()."""
    macro_sentiment: float = 0.0
    risk_on_score: float = 0.5
    geo_risk_score: float = 0.0
    recession_prob: float = 0.0
    fed_cut_prob: float = 0.5
    war_escalation_prob: float = 0.0
    dxy_5d_return: float = 0.0
    fred: FredSignal = field(default_factory=FredSignal)
    headlines: list = field(default_factory=list)
    top_risks: list = field(default_factory=list)
    top_tailwinds: list = field(default_factory=list)
    confidence: float = 0.0
    scored_by_llm: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_counts: dict = field(default_factory=dict)

    def is_live_date(self, date) -> bool:
        try:
            ts = pd.Timestamp(date)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=LIVE_SIGNAL_WINDOW_DAYS)
            return ts >= cutoff
        except Exception:
            return False

    def adjust_phase(self, base_phase, date=None):
        if date is not None and not self.is_live_date(date):
            return base_phase
        if self.confidence < 0.1:
            return base_phase

        if (self.recession_prob >= RECESSION_PROB_PHASE3_FORCE and
                base_phase in (1, "1b", 2)):
            return 3

        if (FRED_CURVE_INVERSION_BRAKE and self.fred.is_inverted and
                self.fred.hy_spread_trend == "widening" and base_phase == "1b"):
            return 1

        if self.fred.hy_spread_bps > FRED_HY_SPREAD_BRAKE_BPS and base_phase == "1b":
            return 1

        if self.recession_prob >= RECESSION_PROB_PHASE_BRAKE and base_phase == "1b":
            return 1

        if self.macro_sentiment < SENTIMENT_DEMOTE_1B_THRESHOLD and base_phase == "1b":
            return 1

        return base_phase

    def adjust_blend(self, blend: dict, date=None) -> dict:
        if date is not None and not self.is_live_date(date):
            return blend
        if self.confidence < 0.1:
            return blend

        b = copy.deepcopy(blend)

        # Rule 1: geo risk → add GLD
        if self.geo_risk_score > GEO_RISK_GLD_THRESHOLD:
            strength = (self.geo_risk_score - GEO_RISK_GLD_THRESHOLD) / (1 - GEO_RISK_GLD_THRESHOLD)
            gld_add = round(MAX_GLD_NUDGE * strength, 4)
            if b.get("FACTOR", 0) >= gld_add:
                b["FACTOR"] = round(b.get("FACTOR", 0) - gld_add, 4)
                b["GLD"] = round(b.get("GLD", 0) + gld_add, 4)

        # Rule 2: high Fed-cut probability → add TLT
        if self.fed_cut_prob > FED_CUT_TLT_THRESHOLD:
            strength = (self.fed_cut_prob - FED_CUT_TLT_THRESHOLD) / (1 - FED_CUT_TLT_THRESHOLD)
            tlt_add = round(MAX_TLT_NUDGE * strength, 4)
            for donor in ("FACTOR", "SPY"):
                available = b.get(donor, 0)
                take = min(tlt_add, available)
                if take > 0:
                    b[donor] = round(available - take, 4)
                    b["TLT"] = round(b.get("TLT", 0) + take, 4)
                    tlt_add -= take
                if tlt_add <= 0:
                    break

        # Rule 3: strong risk-on → nudge TQQQ (Phase 1 only)
        if (self.risk_on_score > RISK_ON_TQQQ_THRESHOLD and
                b.get("SPY", 0) > 0 and b.get("TQQQ", 0) == 0):
            strength = (self.risk_on_score - RISK_ON_TQQQ_THRESHOLD) / (1 - RISK_ON_TQQQ_THRESHOLD)
            tqqq_add = round(MAX_TQQQ_NUDGE * strength, 4)
            if b.get("SPY", 0) >= tqqq_add:
                b["SPY"] = round(b.get("SPY", 0) - tqqq_add, 4)
                b["TQQQ"] = round(b.get("TQQQ", 0) + tqqq_add, 4)

        # Rule 4 (FRED): inverted curve + Fed cutting → extra TLT
        if (self.fred.is_inverted and self.fred.fed_funds_trend == "cutting" and
                b.get("TLT", 0) < 0.20):
            tlt_add = round(min(0.03, b.get("FACTOR", 0) * 0.05), 4)
            if b.get("FACTOR", 0) >= tlt_add and tlt_add > 0:
                b["FACTOR"] = round(b.get("FACTOR", 0) - tlt_add, 4)
                b["TLT"] = round(b.get("TLT", 0) + tlt_add, 4)

        # Rule 5 (FRED): rising unemployment → small GLD
        if (self.fred.unemployment_trend == "rising" and b.get("GLD", 0) < 0.10):
            gld_add = round(min(0.02, b.get("FACTOR", 0) * 0.03), 4)
            if b.get("FACTOR", 0) >= gld_add and gld_add > 0:
                b["FACTOR"] = round(b.get("FACTOR", 0) - gld_add, 4)
                b["GLD"] = round(b.get("GLD", 0) + gld_add, 4)

        total = sum(b.values())
        if abs(total - 1.0) > 1e-6 and total > 0:
            b = {k: round(v / total, 6) for k, v in b.items()}

        return b

    def summary(self) -> str:
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        llm_tag = " [LLM-scored]" if self.scored_by_llm else " [keyword-scored]"
        lines = [
            "", "=" * 62,
            f"NEWS AGENT SIGNAL — {ts_str}{llm_tag}",
            "=" * 62,
            f" Macro Sentiment {self.macro_sentiment:>+.3f} "
            f"({'bullish' if self.macro_sentiment > 0.1 else 'bearish' if self.macro_sentiment < -0.1 else 'neutral'})",
            f" Risk-On Score {self.risk_on_score:>6.3f} "
            f"({'risk-on' if self.risk_on_score > 0.6 else 'risk-off' if self.risk_on_score < 0.4 else 'neutral'})",
            f" Geo-Risk Score {self.geo_risk_score:>6.3f} "
            f"({'elevated' if self.geo_risk_score > GEO_RISK_GLD_THRESHOLD else 'contained'})",
            f" Recession Prob {self.recession_prob:>6.1%} (Polymarket)",
            f" Fed Cut Prob {self.fed_cut_prob:>6.1%} (Polymarket)",
            f" DXY 5d Return {self.dxy_5d_return:>+6.2%} "
            f"({'strengthening' if self.dxy_5d_return > 0.005 else 'weakening' if self.dxy_5d_return < -0.005 else 'stable'})",
            f" Signal Confidence {self.confidence:>6.1%}",
            f" Sources {self.source_counts}",
        ]
        lines.extend(self.fred.summary_lines())
        if self.top_risks:
            lines.append(f" Top Risks {' | '.join(self.top_risks[:3])}")
        if self.top_tailwinds:
            lines.append(f" Top Tailwinds {' | '.join(self.top_tailwinds[:3])}")
        if self.headlines:
            lines.append(" Recent Headlines:")
            for h in self.headlines[:5]:
                lines.append(f" • {h[:90]}")
        lines.append("=" * 62)
        return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 5 — FRED MACROECONOMIC DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_fred_series(series_id: str) -> Optional[pd.Series]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/2.0)", "Accept": "text/csv"}
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), parse_dates=["DATE"], index_col="DATE")
        series = pd.to_numeric(df["VALUE"], errors="coerce").dropna()
        series.index = pd.DatetimeIndex(series.index)
        return series.sort_index()
    except Exception:
        return None


def _compute_trend(series: pd.Series, window: int = 3) -> str:
    if series is None or len(series) < window * 2:
        return "stable"
    recent = series.iloc[-window:].mean()
    prior = series.iloc[-window*2:-window].mean()
    if pd.isna(recent) or pd.isna(prior):
        return "stable"
    pct_change = (recent - prior) / (abs(prior) + 1e-10)
    if pct_change > 0.03:
        return "rising"
    elif pct_change < -0.03:
        return "falling"
    return "stable"


def _fetch_fred_signals() -> FredSignal:
    signal = FredSignal()
    series_loaded = 0

    # T10Y2Y
    s = _fetch_fred_series("T10Y2Y")
    if s is not None and len(s) > 0:
        latest = float(s.iloc[-1])
        signal.yield_curve_bps = latest * 100
        signal.is_inverted = latest < 0.0
        series_loaded += 1

    # FEDFUNDS
    s = _fetch_fred_series("FEDFUNDS")
    if s is not None and len(s) > 0:
        signal.fed_funds_rate = float(s.iloc[-1])
        trend = _compute_trend(s, window=3)
        signal.fed_funds_trend = "hiking" if trend == "rising" else ("cutting" if trend == "falling" else "stable")
        series_loaded += 1

    # UNRATE
    s = _fetch_fred_series("UNRATE")
    if s is not None and len(s) > 0:
        signal.unemployment_rate = float(s.iloc[-1])
        signal.unemployment_trend = _compute_trend(s, window=3)
        series_loaded += 1

    # CPIAUCSL
    s = _fetch_fred_series("CPIAUCSL")
    if s is not None and len(s) >= 13:
        yoy = float(s.iloc[-1] / s.iloc[-13] - 1) * 100
        signal.cpi_yoy_pct = yoy
        trend = _compute_trend(s, window=3)
        signal.cpi_trend = "accelerating" if trend == "rising" else ("decelerating" if trend == "falling" else "stable")
        series_loaded += 1

    # BAMLH0A0HYM2
    s = _fetch_fred_series("BAMLH0A0HYM2")
    if s is not None and len(s) > 0:
        signal.hy_spread_bps = float(s.iloc[-1]) * 100
        trend = _compute_trend(s.iloc[-30:] if len(s) >= 30 else s, window=5)
        signal.hy_spread_trend = "widening" if trend == "rising" else ("tightening" if trend == "falling" else "stable")
        series_loaded += 1

    # UMCSENT
    s = _fetch_fred_series("UMCSENT")
    if s is not None and len(s) > 0:
        signal.consumer_sentiment = float(s.iloc[-1])
        series_loaded += 1

    # DCOILWTICO
    s = _fetch_fred_series("DCOILWTICO")
    if s is not None and len(s) > 0:
        signal.crude_oil_price = float(s.iloc[-1])
        series_loaded += 1

    # Composite risk score
    risk = 0.0
    if signal.is_inverted:
        risk += 0.30
    if signal.hy_spread_bps > FRED_HY_SPREAD_BRAKE_BPS:
        risk += 0.25
    elif signal.hy_spread_bps > 400:
        risk += 0.12
    if signal.unemployment_trend == "rising":
        risk += 0.20
    if signal.cpi_trend == "accelerating" and signal.cpi_yoy_pct > 4.0:
        risk += 0.15
    if signal.fed_funds_trend == "hiking":
        risk += 0.10
    signal.fred_risk_score = float(np.clip(risk, 0.0, 1.0))
    signal.series_loaded = series_loaded
    signal.timestamp = datetime.now(timezone.utc)
    return signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 1 — POLYMARKET
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _fetch_polymarket_signals() -> dict:
    result = {"recession_prob": 0.0, "fed_cut_prob": 0.5, "war_escalation_prob": 0.0,
              "matched_markets": [], "count": 0}
    try:
        url = (f"https://gamma-api.polymarket.com/markets"
               f"?active=true&limit={POLYMARKET_FETCH_LIMIT}"
               f"&order=volume&ascending=false")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/2.0)",
            "Accept": "application/json"
        })
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            markets = json.loads(resp.read().decode("utf-8"))
        if isinstance(markets, dict):
            markets = markets.get("markets", [])

        recession_cands = []
        fed_cut_cands = []
        war_cands = []

        for m in markets:
            question = (m.get("question") or "").lower()
            volume = float(m.get("volume") or 0)
            if volume < POLYMARKET_MIN_VOLUME:
                continue
            try:
                outcome_prices = json.loads(m.get("outcomePrices", "[]"))
                outcomes = json.loads(m.get("outcomes", "[]"))
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

            if not any(kw in question for kw in POLYMARKET_MACRO_KEYWORDS):
                continue

            result["matched_markets"].append((m.get("question", ""), yes_price, volume))

            if any(kw in question for kw in ["recession", "gdp contraction", "hard landing"]):
                recession_cands.append((yes_price, volume))
            if any(kw in question for kw in ["rate cut", "fed cut", "federal reserve cut", "bps cut"]):
                fed_cut_cands.append((yes_price, volume))
            if any(kw in question for kw in ["war", "ceasefire", "military", "invasion", "escalat", "attack"]):
                war_cands.append((yes_price, volume))

        def vol_avg(cands):
            if not cands:
                return None
            p = np.array([c[0] for c in cands])
            v = np.array([c[1] for c in cands])
            return float(np.average(p, weights=v))

        r = vol_avg(recession_cands)
        f = vol_avg(fed_cut_cands)
        w = vol_avg(war_cands)
        if r is not None:
            result["recession_prob"] = np.clip(r, 0.0, 1.0)
        if f is not None:
            result["fed_cut_prob"] = np.clip(f, 0.0, 1.0)
        if w is not None:
            result["war_escalation_prob"] = np.clip(w, 0.0, 1.0)

        result["count"] = len(result["matched_markets"])
    except Exception as e:
        result["_error"] = str(e)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 2 — RSS HEADLINES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _parse_rss_feed(name: str, url: str) -> list:
    headlines = []
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioAgent/2.0)",
                     "Accept": "application/rss+xml, application/xml, text/xml, */*"}
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for item in root.findall(".//item"):
            t = item.findtext("title", "").strip()
            if t:
                headlines.append(t)
        if not headlines:
            for entry in root.findall(".//atom:entry", ns):
                t = entry.find("atom:title", ns)
                if t is not None and t.text:
                    headlines.append(t.text.strip())
    except Exception:
        pass
    return headlines


def _fetch_all_headlines() -> dict:
    all_hl = []
    sources = {}
    with ThreadPoolExecutor(max_workers=min(len(RSS_FEEDS), N_IO_THREADS)) as executor:
        future_to_name = {executor.submit(_parse_rss_feed, name, url): name
                          for name, url in RSS_FEEDS.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                hl = future.result()
            except Exception:
                hl = []
            sources[name] = len(hl)
            all_hl.extend(hl)
    seen, dedup = set(), []
    for h in all_hl:
        k = h.lower().strip()
        if k not in seen:
            seen.add(k)
            dedup.append(h)
    return {"headlines": dedup, "count": len(dedup), "sources": sources}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 3 — FOREX SNAPSHOT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _fetch_forex_snapshot() -> dict:
    result = {"dxy_5d_return": 0.0, "eurusd_5d_return": 0.0,
              "usdjpy_5d_return": 0.0, "usdcny_5d_return": 0.0,
              "composite_risk_off": 0.5}
    try:
        window = FOREX_RETURN_WINDOW + 5
        tickers = list(FOREX_TICKERS.values())
        raw = yf.download(tickers, period=f"{window}d", interval="1d",
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.ffill().dropna()
        returns_5d = {}
        for name, ticker in FOREX_TICKERS.items():
            if ticker in raw.columns and len(raw) >= FOREX_RETURN_WINDOW + 1:
                returns_5d[name] = float(raw[ticker].iloc[-1] / raw[ticker].iloc[-FOREX_RETURN_WINDOW - 1] - 1)
        result["dxy_5d_return"] = returns_5d.get("DXY", 0.0)
        result["eurusd_5d_return"] = returns_5d.get("EURUSD", 0.0)
        result["usdjpy_5d_return"] = returns_5d.get("USDJPY", 0.0)
        result["usdcny_5d_return"] = returns_5d.get("USDCNY", 0.0)
        dxy_sig = np.clip(returns_5d.get("DXY", 0.0) * 20, -1.0, 1.0)
        jpy_sig = np.clip(-returns_5d.get("USDJPY", 0.0) * 20, -1.0, 1.0)
        composite = dxy_sig * 0.6 + jpy_sig * 0.4
        result["composite_risk_off"] = float(np.clip((composite + 1) / 2, 0.0, 1.0))
    except Exception as e:
        result["_error"] = str(e)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 4a — LLM SCORING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _score_headlines_llm(headlines: list) -> Optional[dict]:
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
  "macro_sentiment": <float -1.0 to 1.0>,
  "risk_on_score": <float 0.0 to 1.0>,
  "geo_risk_score": <float 0.0 to 1.0>,
  "top_risks": [<up to 3 concise strings, max 6 words each>],
  "top_tailwinds": [<up to 3 concise strings, max 6 words each>],
  "reasoning": <one sentence max 20 words>
}}

Headlines:
{headlines_text}

Return only valid JSON."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            "macro_sentiment": float(np.clip(parsed.get("macro_sentiment", 0.0), -1.0, 1.0)),
            "risk_on_score": float(np.clip(parsed.get("risk_on_score", 0.5), 0.0, 1.0)),
            "geo_risk_score": float(np.clip(parsed.get("geo_risk_score", 0.0), 0.0, 1.0)),
            "top_risks": parsed.get("top_risks", []),
            "top_tailwinds": parsed.get("top_tailwinds", []),
            "scored_by_llm": True,
        }
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TIER 4b — KEYWORD SENTIMENT FALLBACK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _score_headlines_keywords(headlines: list) -> dict:
    if not headlines:
        return {"macro_sentiment": 0.0, "risk_on_score": 0.5, "geo_risk_score": 0.0,
                "top_risks": [], "top_tailwinds": [], "scored_by_llm": False}
    combined = " ".join(headlines).lower()
    raw_score = 0.0
    hit_pos, hit_neg = [], []
    for kw, weight in SENTIMENT_KEYWORDS.items():
        cnt = combined.count(kw.lower())
        if cnt > 0:
            raw_score += weight * cnt
            (hit_pos if weight > 0 else hit_neg).append((kw, weight * cnt))
    n = max(len(headlines), 1)
    macro = float(np.clip(np.tanh(raw_score / n), -1.0, 1.0))
    risk_on = float(np.clip((macro + 1) / 2, 0.0, 1.0))
    geo_kws = ["war", "military", "invasion", "attack", "missile", "drone",
               "ceasefire", "escalat", "nuclear", "troops", "sanctions"]
    geo_hits = sum(combined.count(kw) for kw in geo_kws)
    geo_score = float(np.clip(geo_hits / (n * 2), 0.0, 1.0))
    hit_neg.sort(key=lambda x: x[1])
    hit_pos.sort(key=lambda x: -x[1])
    return {
        "macro_sentiment": macro,
        "risk_on_score": risk_on,
        "geo_risk_score": geo_score,
        "top_risks": [k for k, _ in hit_neg[:3]],
        "top_tailwinds": [k for k, _ in hit_pos[:3]],
        "scored_by_llm": False,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPOSITE BUILDER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def _build_composite(rss_data: dict, poly_data: dict,
                     forex_data: dict, sentiment: dict,
                     fred_data: FredSignal) -> NewsSignal:
    macro_sent = sentiment.get("macro_sentiment", 0.0)
    risk_on = sentiment.get("risk_on_score", 0.5)
    geo_risk = sentiment.get("geo_risk_score", 0.0)

    rec_prob = poly_data.get("recession_prob", 0.0)
    fed_prob = poly_data.get("fed_cut_prob", 0.5)
    war_prob = poly_data.get("war_escalation_prob", 0.0)

    if rec_prob > 0.5:
        pull_strength = (rec_prob - 0.5) / 0.5
        macro_sent = macro_sent * (1 - pull_strength * 0.5) - pull_strength * 0.5
        risk_on = risk_on * (1 - pull_strength * 0.4)

    if war_prob > 0.3:
        geo_risk = max(geo_risk, float(np.clip(war_prob * 0.8, 0.0, 1.0)))

    forex_risk_off = forex_data.get("composite_risk_off", 0.5)
    forex_adj = (0.5 - forex_risk_off) * 0.3
    risk_on = float(np.clip(risk_on + forex_adj, 0.0, 1.0))
    macro_sent = float(np.clip(macro_sent, -1.0, 1.0))
    geo_risk = float(np.clip(geo_risk, 0.0, 1.0))

    rss_ok = rss_data.get("count", 0) > 0
    poly_ok = poly_data.get("count", 0) > 0
    forex_ok = "_error" not in forex_data
    sent_ok = sentiment.get("macro_sentiment") is not None
    fred_ok = fred_data.series_loaded > 0

    confidence = sum([rss_ok, poly_ok, forex_ok, sent_ok, fred_ok]) / 5.0

    source_counts = {
        "rss_headlines": rss_data.get("count", 0),
        "polymarket_markets": poly_data.get("count", 0),
        "forex_ok": forex_ok,
        "llm_scored": sentiment.get("scored_by_llm", False),
        "fred_ok": fred_ok,
    }

    return NewsSignal(
        macro_sentiment= round(macro_sent, 4),
        risk_on_score= round(risk_on, 4),
        geo_risk_score= round(geo_risk, 4),
        recession_prob= round(rec_prob, 4),
        fed_cut_prob= round(fed_prob, 4),
        war_escalation_prob= round(war_prob, 4),
        dxy_5d_return= round(forex_data.get("dxy_5d_return", 0.0), 5),
        fred= fred_data,
        headlines= rss_data.get("headlines", [])[:20],
        top_risks= sentiment.get("top_risks", []),
        top_tailwinds= sentiment.get("top_tailwinds", []),
        confidence= round(confidence, 4),
        scored_by_llm= sentiment.get("scored_by_llm", False),
        timestamp= datetime.now(timezone.utc),
        source_counts= source_counts,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NEWSAGENT — MAIN ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class NewsAgent:
    def __init__(self):
        self._cached_signal = None
        self._cache_expires_at = None

    def _is_cache_valid(self) -> bool:
        if self._cached_signal is None or self._cache_expires_at is None:
            return False
        return datetime.now(timezone.utc) < self._cache_expires_at

    def get_signal(self) -> NewsSignal:
        if self._is_cache_valid():
            return self._cached_signal
        return self.refresh()

    def refresh(self) -> NewsSignal:
        print("  [NewsAgent] Launching concurrent tier fetch...", flush=True)

        # Launch Tiers 1, 2, 3, 5 simultaneously
        with ThreadPoolExecutor(max_workers=4) as tier_pool:
            poly_future = tier_pool.submit(_fetch_polymarket_signals)
            rss_future = tier_pool.submit(_fetch_all_headlines)
            forex_future = tier_pool.submit(_fetch_forex_snapshot)
            fred_future = tier_pool.submit(_fetch_fred_signals)

            done, _ = wait([poly_future, rss_future, forex_future, fred_future],
                           return_when=ALL_COMPLETED)

            poly_data = poly_future.result() if poly_future in done else None
            rss_data = rss_future.result() if rss_future in done else None
            forex_data = forex_future.result() if forex_future in done else None
            fred_data = fred_future.result() if fred_future in done else FredSignal()

            if poly_data:
                print(f"  [NewsAgent] Polymarket: {poly_data.get('count', 0)} markets", flush=True)
            if rss_data:
                print(f"  [NewsAgent] RSS: {rss_data.get('count', 0)} headlines", flush=True)
            if forex_data:
                dxy_str = f"{forex_data.get('dxy_5d_return', 0):.2%}"
                print(f"  [NewsAgent] FOREX: DXY 5d={dxy_str}", flush=True)
            print(f"  [NewsAgent] FRED: {fred_data.series_loaded} series loaded", flush=True)

        # Safe defaults
        if poly_data is None:
            poly_data = {"recession_prob": 0.0, "fed_cut_prob": 0.5, "war_escalation_prob": 0.0,
                         "matched_markets": [], "count": 0}
        if rss_data is None:
            rss_data = {"headlines": [], "count": 0, "sources": {}}
        if forex_data is None:
            forex_data = {"dxy_5d_return": 0.0, "composite_risk_off": 0.5}

        # Tier 4: scoring (depends on RSS)
        print("  [NewsAgent] Scoring headlines...", end=" ", flush=True)
        sentiment = _score_headlines_llm(rss_data.get("headlines", []))
        if sentiment is None:
            sentiment = _score_headlines_keywords(rss_data.get("headlines", []))
            print("keyword fallback.", flush=True)
        else:
            print("LLM scored.", flush=True)

        signal = _build_composite(rss_data, poly_data, forex_data, sentiment, fred_data)

        self._cached_signal = signal
        self._cache_expires_at = datetime.now(timezone.utc) + timedelta(hours=CACHE_TTL_HOURS)
        return signal


if __name__ == "__main__":
    agent = NewsAgent()
    signal = agent.get_signal()
    print(signal.summary())
    print("NewsAgent v2 loaded successfully.")
 
