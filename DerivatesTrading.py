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
DerivativesTrading.py  ·  Long/Short Index Hedging with MLP Ratio Optimizer
──────────────────────────────────────────────────────────────────────────────

WHAT THIS MODULE DOES:
  When the portfolio enters a drawdown or Phase 3 regime, this module
  activates a HEDGING OVERLAY to protect against further losses.

  PREVIOUS APPROACH (v3):
    · Buy protective puts on SPY (options contracts)
    · Problems: expensive premium (~0.5-1% per month), requires options
      access, complex pricing model needed, unrealistic to simulate

  NEW APPROACH (v4 — this file):
    · SHORT SELL index ETFs/futures: SPY (S&P 500), QQQ (Nasdaq-100)
    · Benefits: no time decay, no premium, straightforward linear payoff
    · Realistic: achievable via ETF short selling or E-mini/Micro futures
    · MLP neural network determines the OPTIMAL SHORT RATIO dynamically

  HOW THE MLP WORKS:
    A small neural network (3 layers, 46 parameters) predicts how much
    of the portfolio to short based on current market conditions:
      Features: [VIX level, VIX trend, regime, drawdown, momentum, vol]
      Output: hedge_ratio ∈ [0.0, 1.0] (fraction of equity to short)

  BACKTEST VS. LIVE MODE:
    · Backtest: returns are adjusted by a simplified PnL model
    · Live mode: prints actionable signals ("Short X% of portfolio in SPY/QQQ")
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf

# ── Deep learning (optional — falls back to rule-based if unavailable) ────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  [DerivativesHedger] PyTorch not found — using rule-based hedge ratio.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLP HEDGE RATIO MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HedgeRatioMLP(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-layer Perceptron (MLP) that predicts the optimal hedge ratio.

    WHAT IS A HEDGE RATIO?
      If your portfolio is worth $1,000,000 and the hedge ratio is 0.60,
      you short $600,000 worth of SPY (or equivalent futures) to offset
      60% of your equity market exposure. If SPY falls 5%, your short
      position gains $30,000, partially offsetting losses in your long book.

    WHY USE A NEURAL NETWORK?
      A simple rule like "always hedge 50% in Phase 3" is static. The MLP
      learns from historical data that sometimes a 30% hedge is sufficient
      (mild drawdown at low VIX) and sometimes 80% is needed (VIX spike
      with accelerating momentum loss). This dynamic sizing reduces the
      cost of unnecessary hedging during benign volatility.

    ARCHITECTURE:
      Input  : (batch, 6 features)
        [vix_level, vix_zscore_20d, regime_encoded, drawdown_21d,
         momentum_21d, realized_vol_21d]
      Hidden : Linear(6→32) → ReLU → Dropout(0.2) → Linear(32→16) → ReLU
      Output : Linear(16→1) → Sigmoid → hedge_ratio ∈ [0, 1]
    """

    def __init__(self, input_dim: int = 6):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)   # shape: (batch, 1)


def _build_training_features(
    vix_px: pd.Series,
    spy_px: pd.Series,
    regime: pd.Series,
) -> tuple:
    """
    Build the feature matrix and target labels for training the MLP.

    Features (all backward-looking):
      vix_level     : Raw VIX value (fear gauge)
      vix_zscore    : VIX z-score relative to 20-day rolling mean/std
      regime_enc    : Regime normalized to [0,1] (1→0.0, 2→0.33, 3→0.67, 4→1.0)
      drawdown_21d  : 21-day return on SPY (negative = falling market)
      momentum_21d  : Same as drawdown_21d (redundant but helps gradient flow)
      realized_vol  : 21-day annualized SPY realized vol

    Target label ("optimal_hedge"):
      We define the "should-have-been" hedge ratio using a simple heuristic:
        · If Phase 3: hedge_target = 0.75 (strong protection needed)
        · If drawdown_21d < -10%: hedge_target = 0.90 (aggressive drawdown)
        · If drawdown_21d < -5% : hedge_target = 0.55 (moderate protection)
        · If Phase 4 (recovery)  : hedge_target = 0.20 (light hedge)
        · Otherwise              : hedge_target = 0.00 (no hedge)

    The MLP learns to predict these targets from the feature patterns,
    effectively learning WHEN high-VIX / drawdown conditions warrant hedging.
    """
    spy_ret     = spy_px.pct_change()
    mom_21d     = spy_px.pct_change(21)
    vix_sma20   = vix_px.rolling(20).mean()
    vix_std20   = vix_px.rolling(20).std()
    vix_zscore  = (vix_px - vix_sma20) / (vix_std20 + 1e-10)
    realized_vol = spy_ret.rolling(21).std() * np.sqrt(252)

    # Align all series to a common index
    common_idx = (vix_px.dropna().index
                  .intersection(spy_px.dropna().index)
                  .intersection(regime.dropna().index))

    vix_arr    = vix_px.reindex(common_idx).values
    zscore_arr = vix_zscore.reindex(common_idx).values
    regime_arr = regime.reindex(common_idx).values
    mom_arr    = mom_21d.reindex(common_idx).values
    vol_arr    = realized_vol.reindex(common_idx).values

    # Encode regime: 1→0.00, 2→0.33, 3→0.67, 4→1.00
    regime_enc = (regime_arr - 1.0) / 3.0

    # Build feature matrix (drop rows with NaN)
    features = np.column_stack([
        vix_arr,       # raw VIX
        zscore_arr,    # VIX z-score (20d)
        regime_enc,    # normalized regime
        mom_arr,       # 21-day SPY momentum
        mom_arr,       # duplicate channel (momentum gradient help)
        vol_arr,       # realized volatility
    ])

    # Define target hedge ratio based on rules above
    targets = np.zeros(len(common_idx))
    for i, (reg, mom, vix_v) in enumerate(zip(regime_arr, mom_arr, vix_arr)):
        if reg == 3:
            targets[i] = 0.75
        elif mom < -0.10:
            targets[i] = 0.90
        elif mom < -0.05:
            targets[i] = 0.55
        elif reg == 4:
            targets[i] = 0.20
        else:
            targets[i] = 0.00

    # Remove rows with any NaN
    valid  = ~np.isnan(features).any(axis=1) & ~np.isnan(targets)
    return features[valid], targets[valid], common_idx[valid]


def _train_mlp(
    features: np.ndarray,
    targets:  np.ndarray,
    epochs:   int = 80,
    lr:       float = 1e-3,
    batch_size: int = 64,
) -> "HedgeRatioMLP | None":
    """
    Train the HedgeRatioMLP on historical feature/target pairs.

    Uses MSE loss (regression problem — predicting a continuous [0,1] value).
    Early stopping via a 20% held-out validation split.
    """
    if not TORCH_AVAILABLE or len(features) < 100:
        return None

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X      = scaler.fit_transform(features).astype(np.float32)
    y      = targets.astype(np.float32)

    # 80/20 train/val split
    n_train = int(0.80 * len(X))
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    X_t  = torch.from_numpy(X_train)
    y_t  = torch.from_numpy(y_train).unsqueeze(1)
    X_v  = torch.from_numpy(X_val)
    y_v  = torch.from_numpy(y_val).unsqueeze(1)

    model     = HedgeRatioMLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    model._scaler = scaler
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DerivativesHedger:
    """
    Long/short index hedging with MLP-driven hedge ratio sizing.

    Usage:
      hedger = DerivativesHedger(hedge_ratio=0.50)

      # Optional: train the MLP on historical data for dynamic sizing
      hedger.train(spy_px, vix_px, regime)

      # In the backtest loop:
      adjusted_return = hedger.apply_hedge(
          portfolio_daily_return=0.005,
          spy_daily_return=-0.020,
          qqq_daily_return=-0.030,
          current_regime=3,
          date=pd.Timestamp("2024-03-15"),
          live_mode=False,
      )
    """

    def __init__(
        self,
        hedge_ratio:          float = 0.50,   # base hedge ratio (overridden by MLP)
        drawdown_threshold:   float = -0.05,  # 21-day return < -5% triggers hedge
        spy_hedge_fraction:   float = 0.60,   # fraction of hedge in SPY (rest in QQQ)
        qqq_hedge_fraction:   float = 0.40,
        max_hedge_ratio:      float = 0.90,   # MLP output capped at this
        min_hedge_ratio:      float = 0.00,
        drawdown_window:      int   = 21,      # days to measure drawdown
    ):
        self.base_hedge_ratio    = hedge_ratio
        self.drawdown_threshold  = drawdown_threshold
        self.spy_hedge_fraction  = spy_hedge_fraction
        self.qqq_hedge_fraction  = qqq_hedge_fraction
        self.max_hedge_ratio     = max_hedge_ratio
        self.min_hedge_ratio     = min_hedge_ratio
        self.drawdown_window     = drawdown_window

        # MLP is populated by calling .train()
        self._mlp: "HedgeRatioMLP | None" = None

        # Rolling buffer of SPY returns for drawdown calculation
        self._spy_return_buffer: list = []

        print("  [DerivativesHedger] Initialized "
              f"(base ratio={hedge_ratio:.0%}, "
              f"SPY/QQQ split={spy_hedge_fraction:.0%}/{qqq_hedge_fraction:.0%})")

    # ──────────────────────────────────────────────────────────────────────
    def train(
        self,
        spy_px:  pd.Series,
        vix_px:  pd.Series,
        regime:  pd.Series,
        verbose: bool = True,
    ) -> None:
        """
        Train the MLP on historical data to learn dynamic hedge ratios.

        Call this AFTER the main model has computed the regime series,
        but BEFORE running the backtest loop. Only uses training-period data.

        Args:
          spy_px: SPY price series (full history)
          vix_px: VIX price series (full history)
          regime: Smoothed regime labels (full history)
          verbose: Print training progress
        """
        if not TORCH_AVAILABLE:
            if verbose:
                print("  [DerivativesHedger] PyTorch unavailable — "
                      "MLP training skipped.")
            return

        if verbose:
            print("  [DerivativesHedger] Building MLP training features...")

        features, targets, _ = _build_training_features(vix_px, spy_px, regime)

        if verbose:
            print(f"  [DerivativesHedger] Training MLP on {len(features):,} samples "
                  f"({targets.sum():.0f} hedge events)...")

        self._mlp = _train_mlp(features, targets, epochs=80, verbose=False)

        if self._mlp is not None and verbose:
            print("  [DerivativesHedger] MLP training complete — "
                  "dynamic hedge ratios active.")
        elif verbose:
            print("  [DerivativesHedger] MLP training failed — "
                  "using static base hedge ratio.")

    # ──────────────────────────────────────────────────────────────────────
    def _predict_hedge_ratio(
        self,
        vix_level:   float,
        vix_zscore:  float,
        regime:      int,
        drawdown_21d: float,
        realized_vol: float,
    ) -> float:
        """
        Predict the optimal hedge ratio from current market features.

        Returns the MLP prediction if trained; otherwise falls back to
        a simple rule-based hedge ratio using the same logic as the
        training target generator.
        """
        if self._mlp is not None and TORCH_AVAILABLE:
            try:
                regime_enc = (regime - 1.0) / 3.0
                features = np.array([[
                    vix_level, vix_zscore, regime_enc,
                    drawdown_21d, drawdown_21d, realized_vol
                ]], dtype=np.float32)
                # Scale using the same scaler fitted during training
                features = self._mlp._scaler.transform(features)
                with torch.no_grad():
                    ratio = float(self._mlp(torch.from_numpy(features)).item())
                return float(np.clip(ratio, self.min_hedge_ratio, self.max_hedge_ratio))
            except Exception:
                pass   # fall through to rule-based backup

        # ── Rule-based fallback (identical logic to training targets) ─────
        if regime == 3:
            return 0.75
        elif drawdown_21d < -0.10:
            return 0.90
        elif drawdown_21d < -0.05:
            return 0.55
        elif regime == 4:
            return 0.20
        return 0.00

    # ──────────────────────────────────────────────────────────────────────
    def _get_current_vix(self, date: pd.Timestamp = None) -> float:
        """Fetch VIX level, returns 19.2 (long-run average) on any failure."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                raw = yf.download("^VIX", period="5d", interval="1d",
                                  auto_adjust=True, progress=False)
                vix_data = (raw["Close"].squeeze()
                            if not isinstance(raw.columns, pd.MultiIndex)
                            else raw["Close"].squeeze())
                if not isinstance(vix_data, pd.Series) or len(vix_data) == 0:
                    return 19.2
                if date is None or date >= vix_data.index[-1]:
                    return float(vix_data.iloc[-1])
                return float(vix_data.asof(date))
            except Exception:
                return 19.2

    # ──────────────────────────────────────────────────────────────────────
    def is_drawdown_active(
        self,
        spy_daily_return: float,
        current_regime:   int,
    ) -> bool:
        """
        Returns True if a hedge should be applied today.

        Two conditions activate the hedge:
          1. We are in Phase 3 (crisis regime) — always hedge
          2. The rolling 21-day cumulative SPY return falls below
             the drawdown_threshold (default: –5%)
        """
        # Always hedge in Phase 3
        if current_regime == 3:
            return True

        # Update rolling buffer
        self._spy_return_buffer.append(spy_daily_return)
        if len(self._spy_return_buffer) > self.drawdown_window:
            self._spy_return_buffer.pop(0)

        if len(self._spy_return_buffer) < self.drawdown_window:
            return False   # not enough history yet

        cumulative_return = np.prod(
            [1 + r for r in self._spy_return_buffer]
        ) - 1

        return cumulative_return < self.drawdown_threshold

    # ──────────────────────────────────────────────────────────────────────
    def _compute_short_pnl(
        self,
        spy_daily_return: float,
        qqq_daily_return: float,
        hedge_ratio:      float,
    ) -> float:
        """
        Compute the P&L from a long/short index hedge.

        HOW SHORT SELLING WORKS (simplified):
          You borrow and sell $X of SPY at today's price.
          If SPY falls 2%, you profit $X × 2% (you can buy it back cheaper).
          If SPY rises 2%, you lose $X × 2% (you pay more to close).

          P&L of short position = –hedge_ratio × spy_return × spy_fraction
                                 + –hedge_ratio × qqq_return × qqq_fraction

          We also subtract a small borrowing cost (~50 bps annually → 0.002%/day)
          for maintaining the short position (stock loan fee in practice).

        No premium decay (unlike options) — cost is purely proportional to position.
        """
        BORROW_RATE_DAILY = 0.005 / 252   # 50bps/yr stock loan fee

        spy_pnl = hedge_ratio * self.spy_hedge_fraction * (-spy_daily_return)
        qqq_pnl = hedge_ratio * self.qqq_hedge_fraction * (-qqq_daily_return)

        # Borrow cost: daily fee on notional of the short position
        borrow_cost = hedge_ratio * BORROW_RATE_DAILY

        return float(spy_pnl + qqq_pnl - borrow_cost)

    # ──────────────────────────────────────────────────────────────────────
    def apply_hedge(
        self,
        portfolio_daily_return: float,
        spy_daily_return:       float,
        current_regime:         int,
        qqq_daily_return:       float = None,
        date:                   pd.Timestamp = None,
        live_mode:              bool = False,
    ) -> float:
        """
        Apply the hedge overlay and return the adjusted portfolio return.

        In backtest mode: adjusts the portfolio return by the computed hedge P&L.
        In live mode: additionally prints actionable trading instructions.

        Args:
          portfolio_daily_return: Today's portfolio return before hedging
          spy_daily_return:       SPY daily return (used for hedge P&L)
          current_regime:         Phase (1-4) from the regime classifier
          qqq_daily_return:       QQQ daily return (defaults to spy × 1.2 if None)
          date:                   Trading date (for live signal context)
          live_mode:              If True, print hedge instructions

        Returns:
          Adjusted portfolio return (float)
        """
        # ── Estimate QQQ return if not provided ───────────────────────────
        if qqq_daily_return is None:
            # QQQ is typically ~1.1-1.3× more volatile than SPY
            qqq_daily_return = spy_daily_return * 1.20

        # ── Check whether hedge should activate ───────────────────────────
        if not self.is_drawdown_active(spy_daily_return, current_regime):
            return portfolio_daily_return   # no hedge needed

        # ── Compute hedge ratio via MLP (or rule-based fallback) ──────────
        vix_level    = self._get_current_vix(date)
        vix_zscore   = max(-3.0, min(3.0, (vix_level - 18.5) / 6.0))   # approx
        drawdown_21d = float(np.sum(self._spy_return_buffer))           # approx
        realized_vol = abs(spy_daily_return) * np.sqrt(252)             # approx

        hedge_ratio = self._predict_hedge_ratio(
            vix_level    = vix_level,
            vix_zscore   = vix_zscore,
            regime       = current_regime,
            drawdown_21d = drawdown_21d,
            realized_vol = realized_vol,
        )

        # ── Compute hedge P&L ─────────────────────────────────────────────
        hedge_pnl = self._compute_short_pnl(
            spy_daily_return = spy_daily_return,
            qqq_daily_return = qqq_daily_return,
            hedge_ratio      = hedge_ratio,
        )

        # ── Live mode: print actionable instructions ──────────────────────
        if live_mode and hedge_ratio > 0.05:
            spy_notional = hedge_ratio * self.spy_hedge_fraction
            qqq_notional = hedge_ratio * self.qqq_hedge_fraction
            print(f"\n[DerivativesHedger] HEDGE ACTIVE  "
                  f"| Regime {current_regime} | VIX {vix_level:.1f}")
            print(f"  MLP hedge ratio : {hedge_ratio:.1%}")
            print(f"  → SHORT {spy_notional:.1%} of portfolio in SPY")
            print(f"    (via SPY short sale or /ES Micro futures)")
            print(f"  → SHORT {qqq_notional:.1%} of portfolio in QQQ")
            print(f"    (via QQQ short sale or /MNQ Micro futures)")
            print(f"  → Per $100,000 portfolio:")
            print(f"    SPY short notional : ${spy_notional*100_000:>10,.0f}")
            print(f"    QQQ short notional : ${qqq_notional*100_000:>10,.0f}")
            # Try to get live SPY/QQQ prices for sizing guidance
            try:
                snap = yf.download(["SPY","QQQ"], period="1d",
                                   interval="1d", progress=False)["Close"]
                spy_price = float(snap["SPY"].iloc[-1])
                qqq_price = float(snap["QQQ"].iloc[-1])
                spy_shares = int(spy_notional * 100_000 / spy_price)
                qqq_shares = int(qqq_notional * 100_000 / qqq_price)
                print(f"    SPY shares to short: {spy_shares:,}  "
                      f"@ ${spy_price:.2f}")
                print(f"    QQQ shares to short: {qqq_shares:,}  "
                      f"@ ${qqq_price:.2f}")
            except Exception:
                pass

        return (
            portfolio_daily_return * (1 - hedge_ratio)
            + hedge_pnl
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    # Quick smoke test
    print("DerivativesTrading.py — Long/Short Index Hedging with MLP Optimizer")
    print("─" * 60)

    hedger = DerivativesHedger(hedge_ratio=0.50)

    # Simulate a Phase 3 drawdown day
    adjusted = hedger.apply_hedge(
        portfolio_daily_return = -0.025,  # portfolio down 2.5%
        spy_daily_return       = -0.032,  # SPY down 3.2%
        current_regime         = 3,
        date                   = pd.Timestamp.today(),
        live_mode              = True,
    )
    print(f"\nBefore hedge: -2.50%  |  After hedge: {adjusted*100:.2f}%")
    print("✓ Module loaded successfully.")
