import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import urllib.request
import sys
import warnings
from io import StringIO
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
PERIOD      = "50y"
INTERVAL    = "1d"
RF_RATE     = 0.03 / 252
N_FACTORS   = 15        # PCA factors to extract (single definition)
N_SIM       = 5000      # Monte Carlo paths for Bates simulation
MIN_WEIGHT  = 0.0
MAX_WEIGHT  = 0.15      # No single stock > 15% of portfolio
CVAR_ALPHA  = 0.05      # Tail probability for CVaR (5% worst outcomes)
TRAIN_FRAC  = 0.70      # Walk-forward train/test split fraction
SLEEVE_INSTRUMENTS = ["SPY", "QQQ", "GLD", "SH", "SDS", "TLT"]

# ============================================================
# SECTION 1 — CONSTITUENT DOWNLOAD
# Pull S&P 500, DJIA, NASDAQ-100 from Wikipedia
# Deduplicate so no company appears twice
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

# ============================================================
# FETCH CONSTITUENTS
# ============================================================
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
print("Constituent fetch complete — starting download...")

# ============================================================
# SECTION 2 — PRICE DOWNLOAD
# Download adjusted close for all tickers
# Drop any ticker with more than 5% missing data
# ============================================================
print("Downloading price data (this will take 2-3 minutes)...")

raw = yf.download(
    list(all_tickers),
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True,
    progress=True
)["Close"]

# Drop columns with more than 5% NaN
threshold = 0.05
raw = raw.dropna(thresh=int(len(raw) * (1 - threshold)), axis=1)
raw = raw.ffill().dropna()

print(f"Clean tickers after filtering: {raw.shape[1]}")
print(f"Trading days: {raw.shape[0]}")

# ============================================================
# SECTION 3 — RETURNS + SLEEVE INSTRUMENTS
# ============================================================
returns = raw.pct_change().dropna()

print("Downloading instrument sleeves (including SDS + TLT for Phase 3/4 convexity)...")
sleeve_raw = yf.download(
    SLEEVE_INSTRUMENTS,
    period=PERIOD,
    interval=INTERVAL,
    auto_adjust=True,
    progress=False
)["Close"]

# Flatten MultiIndex if present
if isinstance(sleeve_raw.columns, pd.MultiIndex):
    sleeve_raw.columns = sleeve_raw.columns.droplevel(1)

sleeve_raw     = sleeve_raw.ffill().dropna()
sleeve_returns = sleeve_raw.pct_change().dropna()

# Align sleeve returns to same index as stock returns
sleeve_returns = sleeve_returns.reindex(returns.index).dropna()
print(f"Sleeve instruments loaded: {list(sleeve_returns.columns)}")

# Download VIX and HYG for regime detection
vix_raw = yf.download("^VIX", period=PERIOD, interval=INTERVAL,
                       auto_adjust=True, progress=False)["Close"]
hyg_raw = yf.download("HYG",  period=PERIOD, interval=INTERVAL,
                       auto_adjust=True, progress=False)["Close"]

vix = vix_raw.reindex(returns.index, method="ffill").squeeze()
hyg = hyg_raw.pct_change().reindex(returns.index, method="ffill").squeeze()

# ============================================================
# SECTION 4 — SPATIAL-TEMPORAL REGIME CLASSIFIER
#
# Phase 1 — Buildout:   Low vol, positive/neutral momentum
# Phase 2 — Narrative:  Rising momentum, still low vol
# Phase 3 — Unwind:     Spiking vol z-score, negative momentum
# Phase 4 — Reset:      Elevated VIX but falling, weak momentum
# ============================================================

def classify_regime(vix_series, hyg_series, spy_series):
    spy_mom      = spy_series.pct_change(63)   # 3-month momentum
    spy_mom_fast = spy_series.pct_change(21)   # 1-month momentum
    vix_sma60    = vix_series.rolling(60).mean()
    vix_sma20    = vix_series.rolling(20).mean()
    vix_std60    = vix_series.rolling(60).std()
    vix_std20    = vix_series.rolling(20).std()

    # Composite z-score blending 20-day and 60-day references
    vix_zscore = (
        (vix_series - vix_sma60) / (vix_std60 + 1e-10) * 0.5 +
        (vix_series - vix_sma20) / (vix_std20 + 1e-10) * 0.5
    )
    vix_change = vix_series.pct_change(5)

    regime = pd.Series(index=vix_series.index, dtype=float)

    for date in vix_series.index:
        try:
            v        = float(vix_series.loc[date])
            vz       = float(vix_zscore.loc[date])
            v_chg    = float(vix_change.loc[date])
            mom      = float(spy_mom.loc[date])
            mom_fast = float(spy_mom_fast.loc[date])

            # Phase 3 — Synchronous Unwind
            # VIX z-score > 1.5 AND momentum negative
            if vz > 1.5 and mom < -0.05:
                regime.loc[date] = 3

            # Phase 4 — Reset
            # VIX above 20 but z-score falling, momentum still weak
            elif v > 20 and vz < 0 and mom < 0.05:
                regime.loc[date] = 4

            # Phase 2 — Narrative Capture
            # VIX below 20 AND strong momentum (3-month OR 1-month)
            elif v < 20 and (mom > 0.10 or mom_fast > 0.05):
                regime.loc[date] = 2

            # Phase 1 — Buildout (default)
            else:
                regime.loc[date] = 1

        except Exception:
            regime.loc[date] = 1

    return regime

# ============================================================
# LABELS + SPY DOWNLOAD + REGIME CLASSIFICATION
# ============================================================
labels = {1: "Buildout", 2: "Narrative", 3: "Unwind", 4: "Reset"}

spy_px = yf.download(
    "SPY", period=PERIOD, interval=INTERVAL,
    auto_adjust=True, progress=False
)["Close"].squeeze()
spy_px = spy_px.reindex(returns.index, method="ffill")

print("Classifying market regimes...")
regime = classify_regime(vix, hyg, spy_px)

phase_counts = regime.value_counts().sort_index()
print("\nREGIME DISTRIBUTION")
print("=" * 35)
for phase, count in phase_counts.items():
    pct = count / len(regime) * 100
    print(f"  Phase {int(phase)} ({labels[int(phase)]:<12}): "
          f"{count:>4} days ({pct:.1f}%)")
print("=" * 35)

# ============================================================
# WALK-FORWARD SPLIT DATE
# Must be established BEFORE PCA so that the scaler and PCA
# are fit exclusively on training data — no lookahead leakage.
# ============================================================
split_date = returns.index[int(len(returns) * TRAIN_FRAC)]
print(f"\nWalk-forward split:")
print(f"  Train: {returns.index[0].date()} → {split_date.date()}")
print(f"  Test:  {split_date.date()} → {returns.index[-1].date()}")

train_returns = returns[returns.index <= split_date]
test_returns  = returns[returns.index >  split_date]

# ============================================================
# SECTION 5 — PCA  (fit on TRAINING DATA ONLY)
#
# FIX: Previously PCA was fit on the full returns matrix,
# contaminating out-of-sample periods with future information.
# Now the scaler and PCA are fit on train_returns only, then
# transform() is applied to both train and test separately.
# ============================================================
print(f"\nRunning PCA — extracting {N_FACTORS} factors (fit on training data only)...")

scaler              = StandardScaler()
train_returns_scaled = scaler.fit_transform(train_returns)   # fit + transform train
test_returns_scaled  = scaler.transform(test_returns)        # transform only (no fit)

pca = PCA(n_components=N_FACTORS)
train_factors = pca.fit_transform(train_returns_scaled)      # fit + transform train
test_factors  = pca.transform(test_returns_scaled)           # transform only (no fit)

# Build unified factor DataFrame for downstream use
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
# SECTION 6 — FACTOR TO STOCK WEIGHT MAPPING
#
# FIX: Previously used np.abs() which treated negative loadings
# identically to positive ones, effectively converting intended
# short positions into long ones and distorting the optimizer's
# intent.  We now clip to zero (long-only) so only stocks with
# genuinely positive alignment to the selected factors receive
# allocation.
# ============================================================
loadings = pd.DataFrame(
    pca.components_,
    columns=returns.columns,
    index=[f"F{i+1}" for i in range(N_FACTORS)]
)

def factor_weights_to_stock_weights(factor_weights, loadings):
    """
    Convert optimised factor weights → long-only stock weights.

    Steps:
      1. Project factor weights onto stock loadings.
      2. Clip negatives to zero  (long-only; no unintended short longs).
      3. Normalise to sum = 1.
    """
    stock_weights = np.dot(factor_weights, loadings.values)
    stock_weights = np.clip(stock_weights, 0.0, None)   # long-only clip

    total = stock_weights.sum()
    if total < 1e-10:
        # Fallback: equal-weight if all projections are non-positive
        stock_weights = np.ones(len(stock_weights)) / len(stock_weights)
    else:
        stock_weights = stock_weights / total

    return stock_weights

# ============================================================
# SECTION 7 — BATES MODEL
# Stochastic volatility with jump diffusion — used for the
# forward-looking Monte Carlo chart.  Parameters are set per
# regime to reflect realistic vol/jump dynamics.
# ============================================================

def bates_simulate(S0, v0, kappa, theta, sigma, rho,
                   lam, mu_j, sig_j, T=1.0, steps=252, paths=1000):
    dt     = T / steps
    prices = np.zeros((steps + 1, paths))
    vols   = np.zeros((steps + 1, paths))
    prices[0] = S0
    vols[0]   = v0

    for t in range(1, steps + 1):
        z1 = np.random.normal(0, 1, paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, paths)

        vols[t] = np.abs(
            vols[t-1]
            + kappa * (theta - vols[t-1]) * dt
            + sigma * np.sqrt(vols[t-1] * dt) * z2
        )

        n_jumps   = np.random.poisson(lam * dt, paths)
        jump_size = (
            np.exp(mu_j * n_jumps +
                   sig_j * np.sqrt(n_jumps) * np.random.normal(0, 1, paths))
            - 1
        )

        prices[t] = prices[t-1] * np.exp(
            -0.5 * vols[t-1] * dt
            + np.sqrt(vols[t-1] * dt) * z1
            + np.log(1 + jump_size + 1e-10)
        )

    return prices, vols

bates_params = {
    1: dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3,
            rho=-0.7,  lam=0.5, mu_j=-0.02, sig_j=0.05),
    2: dict(v0=0.06, kappa=1.5, theta=0.06, sigma=0.4,
            rho=-0.6,  lam=1.0, mu_j=-0.03, sig_j=0.07),
    3: dict(v0=0.15, kappa=1.0, theta=0.10, sigma=0.6,
            rho=-0.8,  lam=3.0, mu_j=-0.08, sig_j=0.12),
    4: dict(v0=0.08, kappa=2.5, theta=0.05, sigma=0.35,
            rho=-0.65, lam=0.8, mu_j=-0.03, sig_j=0.07),
}

current_vix    = float(vix.iloc[-1])
current_regime = int(regime.iloc[-1])
params         = bates_params[current_regime]

print(f"\nBates Model — Current Regime: "
      f"Phase {current_regime} ({labels[current_regime]})")
print(f"Current VIX: {current_vix:.1f}")

spy_price = float(spy_px.iloc[-1])
sim_prices, sim_vols = bates_simulate(
    S0=spy_price, **params, paths=500
)

# ============================================================
# SECTION 8 — PORTFOLIO METRICS + OPTIMIZER
#
# FIX (CVaR): Previously sorted_losses[:n_tail] selected the
# SMALLEST (best) losses.  CVaR requires the LARGEST (worst)
# alpha-tail losses.  Corrected to sorted_losses[-n_tail:].
# ============================================================

def portfolio_performance(weights, ret_matrix):
    weights     = np.array(weights)
    port_return = np.dot(weights, ret_matrix.mean()) * 252
    cov         = ret_matrix.cov() * 252
    port_vol    = np.sqrt(weights @ cov @ weights)
    port_sharpe = (port_return - RF_RATE * 252) / (port_vol + 1e-10)
    return port_return, port_vol, port_sharpe

def max_drawdown(cum_series):
    return (cum_series / cum_series.cummax() - 1).min()

def sharpe(ret_series):
    excess = ret_series - RF_RATE
    return (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252)

def compute_cvar(port_rets, alpha=CVAR_ALPHA):
    """
    Conditional Value at Risk: average loss in the worst alpha% of returns.

    FIX: previous implementation sorted ascending and took [:n_tail],
    which selected the BEST outcomes (smallest losses) rather than the
    worst.  Corrected to [-n_tail:] — the largest losses in the tail.
    """
    if len(port_rets) == 0:
        return 0.0
    port_rets    = np.asarray(port_rets).flatten()
    losses       = -port_rets                      # positive = loss
    sorted_losses = np.sort(losses)                # ascending
    n            = len(losses)
    n_tail       = max(int(np.ceil(alpha * n)), 1)
    cvar         = np.mean(sorted_losses[-n_tail:]) # WORST tail (largest losses)
    return cvar

def optimize_portfolio(ret_matrix, objective="sharpe", cvar_alpha=CVAR_ALPHA):
    n           = ret_matrix.shape[1]
    w0          = np.ones(n) / n
    bounds      = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if objective == "sharpe":
        def obj(w):
            _, _, s = portfolio_performance(w, ret_matrix)
            return -s   # maximise Sharpe

    elif objective == "cvar":
        def obj(w):
            w_arr    = np.array(w)
            port_rets = (ret_matrix.dot(w_arr)
                         if isinstance(ret_matrix, pd.DataFrame)
                         else np.dot(ret_matrix, w_arr))
            return compute_cvar(port_rets, alpha=cvar_alpha)  # minimise CVaR

    else:   # min-volatility fallback
        def obj(w):
            _, v, _ = portfolio_performance(w, ret_matrix)
            return v

    result = minimize(
        obj, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        tol=1e-8, options={"maxiter": 1000}
    )

    opt_w  = np.clip(np.array(result.x), MIN_WEIGHT, MAX_WEIGHT)
    opt_w /= (opt_w.sum() + 1e-10)
    return opt_w

# ============================================================
# SECTION 8b — PHASE-SPECIFIC FACTOR OPTIMISATION
#
# All optimisation is performed exclusively on TRAINING data.
# The scaler+PCA were already fit on training data in Section 5,
# so factor_df rows up to split_date are clean train factors.
#
# Phase 4 now has its own dedicated optimiser (previously it was
# just a hard-coded blend of defensive + growth weights).
# ============================================================
print("\nOptimizing portfolio on PCA factors (training data only)...")

train_factor_df = factor_df[factor_df.index <= split_date]

# Slice each phase's factor rows from training period only
p1_train = train_factor_df[train_factor_df.index.isin(
    regime[regime == 1].index)]
p2_train = train_factor_df[train_factor_df.index.isin(
    regime[regime == 2].index)]
p3_train = train_factor_df[train_factor_df.index.isin(
    regime[regime == 3].index)]
p4_train = train_factor_df[train_factor_df.index.isin(
    regime[regime == 4].index)]

MIN_OBS = {1: 60, 2: 30, 3: 20, 4: 20}   # minimum days before optimising

w_growth = (optimize_portfolio(p1_train, "sharpe")
            if len(p1_train) > MIN_OBS[1]
            else np.ones(N_FACTORS) / N_FACTORS)

w_momentum = (optimize_portfolio(p2_train, "sharpe")
              if len(p2_train) > MIN_OBS[2]
              else w_growth)

w_defensive = (optimize_portfolio(p3_train, "cvar")
               if len(p3_train) > MIN_OBS[3]
               else np.ones(N_FACTORS) / N_FACTORS)

# Phase 4 — dedicated optimiser (blend objective: Sharpe + CVaR)
# Uses Phase 4 training days.  Falls back to a 40/60 blend if
# there are insufficient Phase 4 observations in training.
if len(p4_train) > MIN_OBS[4]:
    # Custom blended objective: maximise Sharpe while penalising CVaR
    def phase4_obj(w):
        w_arr     = np.array(w)
        port_rets = p4_train.dot(w_arr)
        _, _, s   = portfolio_performance(w_arr, p4_train)
        cvar_pen  = compute_cvar(port_rets)
        return -s + 2.0 * cvar_pen  # maximise Sharpe, penalise tail risk

    p4_bounds      = tuple((MIN_WEIGHT, MAX_WEIGHT) for _ in range(N_FACTORS))
    p4_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    p4_result      = minimize(
        phase4_obj,
        np.ones(N_FACTORS) / N_FACTORS,
        method="SLSQP",
        bounds=p4_bounds,
        constraints=p4_constraints,
        tol=1e-8, options={"maxiter": 1000}
    )
    w_recovery = np.clip(p4_result.x, MIN_WEIGHT, MAX_WEIGHT)
    w_recovery /= (w_recovery.sum() + 1e-10)
else:
    w_recovery = 0.4 * w_defensive + 0.6 * w_growth

# Map factor weights → stock weights (long-only clipped projection)
stock_w_growth    = factor_weights_to_stock_weights(w_growth,    loadings)
stock_w_momentum  = factor_weights_to_stock_weights(w_momentum,  loadings)
stock_w_defensive = factor_weights_to_stock_weights(w_defensive, loadings)
stock_w_recovery  = factor_weights_to_stock_weights(w_recovery,  loadings)

# Phase weight lookup used in the backtest loop
phase_factor_weights = {
    1: stock_w_growth,
    2: stock_w_momentum,
    3: stock_w_defensive,
    4: stock_w_recovery,
}

# ============================================================
# SECTION 9 — WALK-FORWARD BACKTEST WITH BLENDED SLEEVES
#
# BLEND DEFINITION
# Each phase allocates between the factor stock portfolio and
# one or more ETF instrument sleeves.  All values in each row
# must sum exactly to 1.0.
#
# SDS (2× inverse S&P 500) launched July 2006.  Any Phase 3
# day before that date falls back to a doubled SH allocation
# to maintain equivalent short exposure without SDS.
# ============================================================

phase_blend = {
    1: {"FACTOR": 0.55, "SPY": 0.45, "QQQ": 0.00, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.00, "TLT": 0.00},
    # Buildout — factor core + broad market participation

    2: {"FACTOR": 0.30, "SPY": 0.00, "QQQ": 0.70, "GLD": 0.00,
        "SH": 0.00, "SDS": 0.00, "TLT": 0.00},
    # Narrative — ride QQQ tech/momentum wave

    3: {"FACTOR": 0.30, "SPY": 0.00, "QQQ": 0.00, "GLD": 0.20,
        "SH": 0.10, "SDS": 0.30, "TLT": 0.10},
    # Unwind — aggressive short (SDS+SH) + flight-to-safety (GLD+TLT)

    4: {"FACTOR": 0.45, "SPY": 0.10, "QQQ": 0.00, "GLD": 0.20,
        "SH": 0.05, "SDS": 0.00, "TLT": 0.20},
    # Reset — re-entry tilt + bond convexity + light short hedge
}

# Inception date guard for SDS (began trading 2006-07-11)
SDS_INCEPTION = pd.Timestamp("2006-07-11")

def compute_sleeve_return(date, blend, sleeve_df):
    """
    Compute the weighted sleeve return for a single day.

    FIX: The original code called this function AND then looped
    over SPY/QQQ/GLD/SH a second time, counting those four
    instruments twice.  This function is now the single source
    of truth for all sleeve return calculations; the redundant
    inner loop has been removed from the backtest.

    SDS guard: on dates before SDS inception, any SDS allocation
    is redirected to SH (same short direction, 1× leverage) to
    avoid using synthetic/missing data.
    """
    sleeve_ret = 0.0
    for instrument in SLEEVE_INSTRUMENTS:
        if instrument == "FACTOR":
            continue
        alloc = blend.get(instrument, 0.0)
        if alloc <= 0.0:
            continue

        # Redirect SDS to SH if date precedes SDS launch
        effective_instrument = instrument
        if instrument == "SDS" and date < SDS_INCEPTION:
            effective_instrument = "SH"

        if (effective_instrument in sleeve_df.columns
                and date in sleeve_df.index):
            sleeve_ret += alloc * sleeve_df.loc[date, effective_instrument]

    return sleeve_ret

# ============================================================
# BACKTEST LOOP — out-of-sample (test period)
# ============================================================
test_returns_raw  = returns[returns.index > split_date]
test_regime_slice = regime[regime.index > split_date]
test_sleeve       = sleeve_returns[sleeve_returns.index > split_date]

portfolio_daily = pd.Series(index=test_returns_raw.index, dtype=float)

for date in test_returns_raw.index:
    phase = int(test_regime_slice.get(date, 1))
    blend = phase_blend[phase]
    f_w   = phase_factor_weights[phase]

    # Factor sleeve return (stock portfolio)
    factor_ret = np.dot(f_w, test_returns_raw.loc[date].values)

    # Instrument sleeve return — single call, no double-counting
    sleeve_ret = compute_sleeve_return(date, blend, test_sleeve)

    # Blended daily portfolio return
    portfolio_daily.loc[date] = blend["FACTOR"] * factor_ret + sleeve_ret

# Align with SPY benchmark
spy_test        = spy_px[spy_px.index > split_date].pct_change().dropna()
spy_test        = spy_test.reindex(portfolio_daily.index).dropna()
portfolio_daily = portfolio_daily.reindex(spy_test.index).dropna()

port_cum = (1 + portfolio_daily).cumprod()
spy_cum  = (1 + spy_test).cumprod()

# ============================================================
# BLEND SUMMARY — show what each phase holds
# ============================================================
print("\nPHASE BLEND ALLOCATION SUMMARY (with inverse/derivative overlays)")
print("=" * 80)
print(f"{'Phase':<20} {'Factor':>8} {'SPY':>6} {'QQQ':>6} "
      f"{'GLD':>6} {'SH':>6} {'SDS':>6} {'TLT':>6}")
print("=" * 80)
for phase, blend in phase_blend.items():
    print(f"  Phase {phase} ({labels[phase]:<12}) "
          f"{blend['FACTOR']*100:>7.0f}% "
          f"{blend['SPY']*100:>5.0f}% "
          f"{blend['QQQ']*100:>5.0f}% "
          f"{blend['GLD']*100:>5.0f}% "
          f"{blend['SH']*100:>5.0f}% "
          f"{blend.get('SDS', 0)*100:>5.0f}% "
          f"{blend.get('TLT', 0)*100:>5.0f}%")
print("=" * 80)

# ============================================================
# SECTION 10 — SCORECARD (out-of-sample)
# ============================================================
print("\n" + "=" * 55)
print("OUT-OF-SAMPLE PERFORMANCE (last 30% of data)")
print("=" * 55)
print(f"{'Metric':<25} {'Portfolio':>12}  {'SPY B&H':>10}")
print("=" * 55)
print(f"{'Total Return':<25} "
      f"{(port_cum.iloc[-1]-1)*100:>11.2f}%  "
      f"{(spy_cum.iloc[-1]-1)*100:>9.2f}%")
print(f"{'Sharpe Ratio':<25} "
      f"{sharpe(portfolio_daily):>12.3f}  "
      f"{sharpe(spy_test):>10.3f}")
print(f"{'Max Drawdown':<25} "
      f"{max_drawdown(port_cum)*100:>11.2f}%  "
      f"{max_drawdown(spy_cum)*100:>9.2f}%")
print("=" * 55)

# Annual breakdown — out-of-sample
test_df        = pd.DataFrame({"Portfolio": portfolio_daily, "SPY": spy_test})
test_df["Year"] = test_df.index.year
annual         = test_df.groupby("Year").apply(
    lambda x: pd.Series({
        "Portfolio": (1 + x["Portfolio"]).prod() - 1,
        "SPY":       (1 + x["SPY"]).prod() - 1,
    })
)

print("\nANNUAL RETURNS — OUT OF SAMPLE")

# ============================================================
# HOLDOUT VALIDATION (2024-present — never tuned on)
# ============================================================
holdout_start   = pd.Timestamp("2024-01-01")
holdout_returns = returns[returns.index >= holdout_start]
holdout_regime  = regime[regime.index  >= holdout_start]
holdout_sleeve  = sleeve_returns[sleeve_returns.index >= holdout_start]

holdout_daily = pd.Series(index=holdout_returns.index, dtype=float)

for date in holdout_returns.index:
    phase      = int(holdout_regime.get(date, 1))
    blend      = phase_blend[phase]
    f_w        = phase_factor_weights[phase]

    factor_ret = np.dot(f_w, holdout_returns.loc[date].values)

    # Single call — no double-counting
    sleeve_ret = compute_sleeve_return(date, blend, holdout_sleeve)

    holdout_daily.loc[date] = blend["FACTOR"] * factor_ret + sleeve_ret

# Align with SPY benchmark
spy_holdout   = spy_px[spy_px.index >= holdout_start].pct_change().dropna()
spy_holdout   = spy_holdout.reindex(holdout_daily.index).dropna()
holdout_daily = holdout_daily.reindex(spy_holdout.index).dropna()

holdout_cum = (1 + holdout_daily).cumprod()
spy_h_cum   = (1 + spy_holdout).cumprod()

# Holdout scorecard
print("\n" + "=" * 55)
print("HOLDOUT VALIDATION (2024 — present, never tuned on)")
print("=" * 55)
print(f"{'Metric':<25} {'Portfolio':>12}  {'SPY B&H':>10}")
print("=" * 55)
print(f"{'Total Return':<25} "
      f"{(holdout_cum.iloc[-1]-1)*100:>11.2f}%  "
      f"{(spy_h_cum.iloc[-1]-1)*100:>9.2f}%")
print(f"{'Sharpe Ratio':<25} "
      f"{sharpe(holdout_daily):>12.3f}  "
      f"{sharpe(spy_holdout):>10.3f}")
print(f"{'Max Drawdown':<25} "
      f"{max_drawdown(holdout_cum)*100:>11.2f}%  "
      f"{max_drawdown(spy_h_cum)*100:>9.2f}%")
print("=" * 55)

# Holdout annual breakdown
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
for year, row in annual.iterrows():
    print(f"{year:<8} "
          f"{row['Portfolio']*100:>11.2f}%"
          f"{row['SPY']*100:>10.2f}%")
print("=" * 40)

# ============================================================
# SECTION 11 — TOP HOLDINGS BY PHASE
# ============================================================

def print_holdings(phase, stock_weights, ticker_list, top_n=30):
    top_idx     = np.argsort(stock_weights)[::-1][:top_n]
    top_tickers = ticker_list[top_idx]
    top_weights = stock_weights[top_idx]
    print(f"\nTOP {top_n} HOLDINGS — Phase {phase} "
          f"({labels[phase]}) Portfolio")
    print("=" * 35)
    for ticker, weight in zip(top_tickers, top_weights):
        print(f"  {ticker:<8} {weight*100:>6.2f}%")
    print("=" * 35)

phase_weights = {
    1: stock_w_growth,
    2: stock_w_momentum,
    3: stock_w_defensive,
    4: stock_w_recovery,
}

for phase in [1, 2, 3, 4]:
    print_holdings(phase, phase_weights[phase],
                   returns.columns.values, top_n=30)

current_phase = int(regime.iloc[-1])
print(f"\n>>> CURRENTLY ACTIVE: Phase {current_phase} "
      f"({labels[current_phase]}) <<<")

# ============================================================
# SECTION 12 — CHARTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Spatial-Temporal Portfolio Model — Corrected", fontsize=14)

# Chart 1: Cumulative returns
axes[0, 0].plot(port_cum.index, port_cum.values,
                label="Portfolio", color="magenta", linewidth=1.5)
axes[0, 0].plot(spy_cum.index,  spy_cum.values,
                label="SPY B&H",  color="cyan",    linewidth=1.5)
axes[0, 0].set_title("Cumulative Returns (Out of Sample)")
axes[0, 0].set_ylabel("Growth of $1")
axes[0, 0].legend()

# Chart 2: Regime map
regime_colors = {1: "green", 2: "yellow", 3: "red", 4: "orange"}
for phase in [1, 2, 3, 4]:
    mask = regime == phase
    axes[0, 1].scatter(
        regime.index[mask],
        spy_px.reindex(regime.index)[mask],
        c=regime_colors[phase], s=1, label=labels[phase]
    )
axes[0, 1].set_title("Spatial-Temporal Regime Map")
axes[0, 1].set_ylabel("SPY Price")
axes[0, 1].legend(markerscale=5)

# Chart 3: Bates Monte Carlo
axes[1, 0].plot(sim_prices[:, :50],
                alpha=0.1, color="red", linewidth=0.8)
axes[1, 0].plot(sim_prices.mean(axis=1),
                color="white", linewidth=2, label="Mean path")
axes[1, 0].set_title(
    f"Bates Model — Phase {current_regime} ({labels[current_regime]})"
)
axes[1, 0].set_ylabel("Price")
axes[1, 0].legend()

# Chart 4: PCA scree
axes[1, 1].bar(range(1, N_FACTORS + 1),
               explained * 100, color="orange", alpha=0.7)
axes[1, 1].plot(range(1, N_FACTORS + 1),
                cumulative * 100, color="white",
                marker="o", linewidth=1.5, label="Cumulative")
axes[1, 1].set_title("PCA Variance Explained")
axes[1, 1].set_xlabel("Factor")
axes[1, 1].set_ylabel("Variance Explained (%)")
axes[1, 1].legend()

plt.tight_layout()
plt.show()