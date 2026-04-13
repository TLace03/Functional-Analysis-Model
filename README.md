# Functional-Analysis-Model
A proprietary quantitative analysis model that utilizes my proprietary framework I call "Spatial-Temporal map of Market Volatility"
Technical Overview: Spatial-Temporal Quantitative FrameworkFunctionalAnalysisModel.py is a high-performance quantitative trading and portfolio management framework. It utilizes a regime-switching architecture to dynamically rotate between factor-driven stock portfolios and defensive ETF "sleeves." The model integrates Principal Component Analysis (PCA) for dimensionality reduction and the Bates Jump-Diffusion Model for forward-looking risk assessment.
Core Components
1. Automated Universe Construction & CleaningThe script dynamically scrapes the current constituents of the S&P 500, Dow Jones Industrial Average, and NASDAQ-100 from Wikipedia.Deduplication: Ensures no overlapping tickers distort allocation.Survivorship Bias Mitigation: While using current constituents, the model filters out tickers with >5% missing data over a 50-year lookback period.Preprocessing: Implements standard scaling and forward-filling to ensure a clean return matrix for PCA fitting.
2. Spatial-Temporal Regime ClassifierThe model segments market history into four distinct phases based on VIX Volatility Z-Scores, SPY Momentum, and High-Yield (HYG) trends:Phase 1 (Buildout): Low volatility, stable positive momentum.Phase 2 (Narrative): High momentum "blow-off" phases; typically tech/growth-heavy (QQQ tilt).Phase 3 (Unwind): Spiking VIX and negative momentum. Triggers aggressive defensive shifts into inverse ETFs (SDS, SH) and safe havens (GLD, TLT).Phase 4 (Reset): High but stabilizing VIX; tactical re-entry with bond convexity.
3. PCA-Driven Factor OptimizationInstead of optimizing on hundreds of individual stocks, the model performs Principal Component Analysis to extract the top 15 Eigen-factors that explain the majority of market variance.Leakage Prevention: Crucially, the PCA scaler and transformation are fit only on the training data (first 70%) to ensure the out-of-sample backtest is untainted by future information.Long-Only Mapping: Factor weights are projected back into stock space using a clipped loading matrix, ensuring the resulting portfolio is long-only and executable.
4. Advanced Portfolio Optimization (Sharpe & CVaR)The model employs two distinct objective functions depending on the market regime:Sharpe Maximization: Used in Phases 1 and 2 to capture risk-adjusted returns during expansion.Conditional Value at Risk (CVaR): Used in Phase 3 (Unwind) to minimize the expected loss in the worst 5% of outcomes (the "tail").Blended Optimization: Phase 4 uses a custom objective function that penalizes tail risk while simultaneously targeting recovery-driven Sharpe ratios.
5. Stochastic Volatility: The Bates ModelFor forward-looking projections, the script implements the Bates Model—an extension of the Heston model that incorporates Poisson Jump-Diffusion.Jump Dynamics: Unlike standard Monte Carlo, this accounts for "black swan" jumps in price, calibrated to the current market regime’s jump intensity ($\lambda$) and mean jump size ($\mu_j$).Regime Calibration: Parameters like $\kappa$ (mean reversion speed) and $\rho$ (volatility/price correlation) are updated in real-time based on the active market phase.
6. Walk-Forward Backtesting & Holdout ValidationThe model includes a rigorous evaluation suite:Out-of-Sample (OOS): The final 30% of historical data.Holdout Validation (2024–Present): A "blind" test on recent data that was never seen during the model's logic-tuning phase.Instrument Sleeves: Integration of leveraged/inverse instruments (SDS, SH) with an "inception guard" that falls back to non-leveraged instruments for dates preceding ETF launches (e.g., pre-2006).
7. Performance Metrics & VisualizationThe framework generates a four-pane diagnostic dashboard:Equity Curve: Compares the blended portfolio against a "Buy & Hold" SPY benchmark.Regime Map: A scatter plot of SPY price points colored by the classified market phase.Monte Carlo Simulation: A 500-path Bates projection showing the mean expected path and variance.Scree Plot: Visualizes the variance explained by each PCA factor, confirming the efficiency of the dimensionality reduction.

April 11, 2026

Refactoring Changes: 
N_FACTORS: 15 → 20
The previous 15 factors only captured 54.79% of cross-sectional variance. Adding 5 more factors will capture roughly 60%+ — the optimizer now has more signal to work with when distinguishing growth vs. defensive vs. momentum stocks, particularly in Phase 1 where the factor portfolio does most of its heavy lifting.

MAX_WEIGHT: 0.15 → 0.10
Single-stock concentration is one of the cleanest levers for Sharpe improvement. Tighter caps force wider diversification, which reduces idiosyncratic variance without reducing expected return — the numerator stays the same, denominator shrinks.

Regime transition smoothing (REGIME_CONFIRM_DAYS = 5)
The original regime switched on a single day's signal. A one-day VIX spike could flip the model into Phase 3 (30% SDS exposure) and then back out the next day, generating large one-day returns that spike the variance calculation. Requiring 5 consecutive days of agreement before recording a phase change eliminates this noise. Completely backward-looking — zero lookahead.

Phase 1b — Momentum Acceleration sub-phase
This is the primary Sharpe driver. Phase 1 is 80% of all trading days, and it's where the model was leaving returns on the table in 2013, 2017, and 2024. When the base regime is Phase 1 AND 21-day SPY return exceeds 3%, resolve_phase() upgrades to Phase 1b: the blend shifts to 40% factor / 20% SPY / 40% QQQ using the momentum-optimized stock weights. The 3% threshold is grounded in momentum factor literature — it's roughly equivalent to 18% annualized — not fitted to specific years.

Phase 2 optimizer: Sharpe → Sortino
Phase 2 days have positively skewed returns — that's the whole point of riding a momentum bull. Sharpe penalizes upside variance the same as downside, which actively discourages selecting the stocks that occasionally spike hard upward. Sortino only penalizes downside deviation, so the Phase 2 (and Phase 1b) factor portfolio now explicitly seeks asymmetric upside.

Sortino ratio added to scorecard
Given the model's deliberate asymmetric structure (lose less in Phase 3, gain more in Phase 2), Sortino is actually a more honest measure of risk-adjusted performance than Sharpe. Both are now displayed.

Added Apache 2.0 Copyright License (April 11, 2026 @ 5:47PM CST)

Added comments to each section that describe the code in detail.

April 13, 2026

Changed QQQ to TQQQ in Sleeve_Instruments 

CONFIG block — 6 changes:

MAX_WEIGHT comment: updated to explain the 0.12 middle-ground rationale
REGIME_CONFIRM_DAYS removed — replaced by three named constants below it
Added CONFIRM_ENTER_PHASE3 = 2, CONFIRM_EXIT_PHASE3 = 5, CONFIRM_DEFAULT = 4 with explanatory comments for each
PHASE1B_MOM_THRESH: 0.03 → 0.04, comment updated to explain why (false triggers in choppy markets)
Added TQQQ_INCEPTION = pd.Timestamp("2010-02-11")

Section 3 comment: updated to name TQQQ and explain its role alongside SDS/GLD/TLT

Section 4 comment block: rewritten to describe asymmetric smoothing, explain the three windows, and document the state machine logic — the old symmetric description was replaced entirely

smooth_regime function: replaced with smooth_regime_asymmetric — the state machine implementation with the three-window logic and a full docstring explaining both failure modes it prevents

Print statement: updated from "Applying N-day regime confirmation filter..." to the asymmetric window summary

Regime distribution header: updated from "(after smoothing)" to "(after asymmetric smoothing)"

Section 8 metrics comment: added Calmar = annualised return / |max drawdown| to the formula list

calmar function: added with docstring explaining why it's more informative than Sharpe for leveraged strategies

Section 9 comment: QQQ → TQQQ references updated throughout, threshold updated to 4%

compute_sleeve_return docstring: added TQQQ inception guard documentation and the TQQQ → SPY redirect logic in the function body

Both scorecards: calmar(...) line added after Sortino in both out-of-sample and holdout blocks

Chart title and regime map subtitle: both updated to v4 (TQQQ / Asymmetric Smoothing)

Created and pushed FAMWithAIA.py

NewsAgent.py — Architecture
Four source tiers, in priority order:
Tier 1 — Polymarket (gamma-api.polymarket.com, public REST, no auth): Fetches the top 300 active markets by volume, filters for macro-relevant keywords, and produces volume-weighted probability estimates for three signal buckets — recession_prob, fed_cut_prob, and war_escalation_prob. A market at 70% "Yes" for recession is more actionable than any number of news articles because that price represents capital committed to the view.

Tier 2 — RSS feeds (stdlib urllib + xml.etree): MarketWatch, CNBC Economy, CNBC Finance, Federal Reserve press releases, and Yahoo Finance. No third-party parsers — zero new dependencies.

Tier 3 — FOREX snapshot (yfinance, already a dependency): 5-day returns on DXY, EUR/USD, USD/JPY, USD/CNY. A surging dollar + strengthening yen is a composite risk-off signal that adjusts risk_on_score by up to ±15 percentage points.

Tier 4 — Headline scoring: If ANTHROPIC_API_KEY is set, claude-haiku-4-5-20251001 scores the top 30 headlines into structured JSON (macro_sentiment, risk_on_score, geo_risk_score, top risks, top tailwinds). If the key is absent or the call fails, a deterministic weighted-keyword fallback using np.tanh normalisation runs instead. The model never blocks or fails on this tier.

Changes to FunctionalAnalysisModel.py used to create FAMWithAIA.py — exactly four edits:
Block 1 (after warnings.filterwarnings): The import + initialisation block. Wrapped in try/except — if NewsAgent.py isn't in the same directory, the model runs exactly as before. _news_signal = None is the fallback.

Block 2 (resolve_phase): After the existing Phase 1→1b momentum check, _news_signal.adjust_phase(base_phase, date) is called. The adjust_phase method has three rules: force Phase 3 if Polymarket recession >75%, demote Phase 1b to Phase 1 if recession >60%, and demote Phase 1b if news sentiment is strongly bearish. All three are no-ops for historical dates.

Block 3 & 4 (backtest loop + holdout loop): After blend = phase_blend[eff_phase], _news_signal.adjust_blend(blend, date) is called. This applies marginal nudges — up to +6pp GLD for elevated geopolitical risk, +5pp TLT when Fed-cut probability is high, +5pp TQQQ when risk-on score is very strong in Phase 1 — each proportional to signal strength with no cliff edges. Again, no-op for historical dates.
_________________________________________________________________
How to Use:

# Clone the repository
git clone https://github.com/TLace03/Functional-Analysis-Model.git

# Install dependencies
pip install numpy pandas yfinance matplotlib scipy scikit-learn anthropic

# Run the model
python FunctionalAnalysisModel.py

For NewsAgent.py
***Install the optional LLM tier:***

pip install anthropic

set ANTHROPIC_API_KEY=your_key_here   # Windows
# or
export ANTHROPIC_API_KEY=your_key_here  # Mac/Linux
