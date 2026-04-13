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

#
#DerivativesTrading.py
#Protective put overlay for drawdown hedging in the Spatial-Temporal Portfolio Model.
#

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

class DerivativesHedger:
    def __init__(self,
                 hedge_ratio: float = 0.50,
                 put_otm_strike: float = 0.95,
                 put_tenor_days: int = 30,
                 max_daily_hedge_cost_bp: float = 8):
        self.hedge_ratio = hedge_ratio
        self.put_otm_strike = put_otm_strike
        self.put_tenor_days = put_tenor_days
        self.max_daily_hedge_cost_bp = max_daily_hedge_cost_bp / 10000

    def _get_vix(self, date: pd.Timestamp = None) -> float:
        """Live or historical VIX level (used as implied-vol proxy)."""
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)["Close"]
        if date is None or date >= vix_data.index[-1]:
            return float(vix_data.iloc[-1])          # ← now guaranteed scalar
        try:
            return float(vix_data.asof(date))
        except:
            return 19.2  # fallback

    def is_drawdown(self, spy_returns: pd.Series, current_regime: int, window: int = 21,
                    return_threshold: float = -0.05) -> bool:
        if current_regime == 3:
            return True
        roll = spy_returns.rolling(window).sum().iloc[-1]
        return roll < return_threshold

    def _approximate_put_pnl(self, spy_daily_return: float, vix: float) -> float:
        annual_vol = vix / 100
        tenor_yr = self.put_tenor_days / 365.0
        premium = annual_vol * np.sqrt(tenor_yr) * 0.4
        premium = min(premium, self.max_daily_hedge_cost_bp)

        strike_level = self.put_otm_strike
        if spy_daily_return < (strike_level - 1.0):
            payoff = -(spy_daily_return - (strike_level - 1.0))
        else:
            payoff = 0.0

        daily_pnl = (payoff - premium) * self.hedge_ratio
        return daily_pnl

    def apply_hedge(self,
                    portfolio_daily_return: float,
                    spy_daily_return: float,
                    current_regime: int,
                    date: pd.Timestamp = None,
                    live_mode: bool = False) -> float:
        vix = self._get_vix(date)
        drawdown = self.is_drawdown(pd.Series([spy_daily_return]), current_regime)

        if not drawdown:
            return portfolio_daily_return

        hedge_pnl = self._approximate_put_pnl(spy_daily_return, vix)

        if live_mode:
            print(f"\n[DerivativesHedger] DRAW DOWN DETECTED — Regime {current_regime} | VIX {vix:.1f}")
            print(f"   → Hedge {self.hedge_ratio*100:.0f}% of equity exposure with 1-month SPY puts")
            print(f"   → Suggested notional: ~${int(100_000 * self.hedge_ratio):,} per $100k portfolio")
            try:
                spy = yf.Ticker("SPY")
                opts = spy.option_chain(spy.options[0])
                puts = opts.puts
                target_strike = round(spy.history(period="1d")["Close"].iloc[-1] * self.put_otm_strike, 0)
                candidate = puts[puts["strike"] == target_strike].iloc[0] if not puts.empty else None
                if candidate is not None:
                    print(f"   → BUY {candidate['contractSymbol']} @ ask {candidate['ask']:.2f} (delta ≈ {candidate['delta']:.2f})")
            except:
                pass

        return portfolio_daily_return + hedge_pnl

if __name__ == "__main__":
    hedger = DerivativesHedger()
    print("DerivativesTrading.py loaded — ready for integration into FAMWithAIA.py")
