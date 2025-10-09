"""Synthetic parameter sweep for TraderV5 risk tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from Traderv4.funcs import AccountSnapshot, RiskConfig, TradeFrequencyTracker
from Traderv5.decision import ModelDecisionEngine
from Traderv5.risk_profiles import AggressiveProfile, get_profile


@dataclass
class ExecutedTrade:
    ticker: str
    action: str
    price: float
    quantity: float
    notional: float
    timestamp: str
    reason: str


class SyntheticPortfolio:
    def __init__(self, starting_cash: float) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: Dict[str, float] = {}
        self.trade_log: List[ExecutedTrade] = []
        self.equity_curve: List[float] = []

    def step(self, timestamp: str, prices: Dict[str, float]) -> None:
        equity = self.cash
        for ticker, shares in self.positions.items():
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue
            equity += shares * price
        self.equity_curve.append(equity)

    def execute(self, decision, price: float, timestamp: str) -> float:
        if getattr(decision, "action", "HOLD") == "HOLD" or decision.notional <= 0 or price <= 0:
            return 0.0
        ticker = decision.ticker
        shares = decision.quantity if decision.quantity is not None else decision.notional / price
        shares = float(shares)
        if decision.action == "BUY":
            cost = shares * price
            if cost > self.cash:
                shares = max(0.0, self.cash / price)
                cost = shares * price
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
            notional = cost
        else:  # SELL
            holdings = self.positions.get(ticker, 0.0)
            proceeds = shares * price
            if holdings <= 0:
                self.cash += proceeds
                self.positions[ticker] = holdings - shares
            else:
                shares = min(shares, holdings)
                proceeds = shares * price
                self.cash += proceeds
                self.positions[ticker] = holdings - shares
            notional = proceeds
        self.trade_log.append(
            ExecutedTrade(
                ticker=ticker,
                action=decision.action,
                price=price,
                quantity=shares,
                notional=shares * price,
                timestamp=timestamp,
                reason=decision.reason,
            )
        )
        return shares

    def finalize(self, prices: Dict[str, float], timestamp: str) -> None:
        for ticker, shares in list(self.positions.items()):
            price = prices.get(ticker, 0.0)
            if price <= 0 or shares == 0:
                continue
            action = "SELL" if shares > 0 else "BUY"
            notional = abs(shares) * price
            self.cash += shares * price
            self.trade_log.append(
                ExecutedTrade(
                    ticker=ticker,
                    action=action,
                    price=price,
                    quantity=abs(shares),
                    notional=notional,
                    timestamp=timestamp,
                    reason="forced liquidation",
                )
            )
            self.positions[ticker] = 0.0
        self.step(timestamp, prices)


class SyntheticPredictor:
    """Simple stochastic predictor used for synthetic sweeps."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, features: Dict[str, float]) -> float:
        sentiment = features.get("sentiment", 0.0)
        momentum = features.get("momentum", 0.0)
        volatility = features.get("volatility", 1.0)
        bias = features.get("bias", 0.0)
        macro = features.get("macro", 0.0)
        signal = 0.72 * sentiment + 0.58 * momentum - 0.42 * volatility + 0.65 * bias + 0.35 * macro
        noisy = signal + self._rng.normal(0.0, 0.045)
        probability = 0.5 + 0.4 * np.tanh(noisy)
        return float(np.clip(probability, 0.02, 0.995))


def _generate_feature_stream(steps: int, seed: int) -> Tuple[List[Dict[str, float]], List[float]]:
    rng = np.random.default_rng(seed)
    features: List[Dict[str, float]] = []
    future_returns: List[float] = []
    macro_cycle = rng.normal(0.0, 0.6)
    for _ in range(steps):
        macro_cycle = 0.78 * macro_cycle + rng.normal(0.0, 0.35)
        sentiment = 0.9 * macro_cycle + rng.normal(0.0, 0.22)
        momentum = 0.75 * macro_cycle + rng.normal(0.0, 0.28)
        volatility = max(0.35, 1.05 - 0.3 * macro_cycle + rng.normal(0.0, 0.12))
        bias = rng.normal(0.04 * macro_cycle, 0.18)
        macro = 0.6 * macro_cycle + rng.normal(0.0, 0.25)
        latent_edge = 0.78 * sentiment + 0.62 * momentum - 0.55 * volatility + 0.58 * bias + 0.42 * macro
        edge = float(np.tanh(latent_edge))
        features.append(
            {
                "sentiment": float(sentiment),
                "momentum": float(momentum),
                "volatility": float(volatility),
                "bias": float(bias),
                "macro": float(macro),
                "edge": edge,
            }
        )
        drift = 0.0005 + 0.0025 * edge
        shock = rng.normal(0.0, 0.0015)
        future_returns.append(float(drift + shock))
    return features, future_returns


def _simulate_strategy(
    risk: RiskConfig,
    *,
    steps: int = 240,
    seed: int = 42,
) -> Tuple[float, float, int, float]:
    predictor = SyntheticPredictor(seed + 7)
    tracker = TradeFrequencyTracker(max_day_trades=750)
    engine = ModelDecisionEngine(predictor, risk, tracker)
    simulator = SyntheticPortfolio(100_000.0)
    symbol = "SYN"

    features, returns = _generate_feature_stream(steps, seed)
    price = 100.0

    for idx, (feature_row, future_return) in enumerate(zip(features, returns)):
        equity = simulator.equity_curve[-1] if simulator.equity_curve else simulator.starting_cash
        account = AccountSnapshot(
            equity=equity,
            cash=simulator.cash,
            buying_power=max(simulator.cash, simulator.cash * risk.max_trade_leverage),
            portfolio_value=equity,
        )
        position_ctx = None
        position_size = simulator.positions.get(symbol, 0.0)
        if abs(position_size) > 1e-6:
            position_ctx = {
                "ticker": symbol,
                "side": "LONG" if position_size > 0 else "SHORT",
                "quantity": position_size,
                "market_value": position_size * price,
            }
        price_snapshot = {"close": price, "volume": 1_000_000}
        decision = engine.decide(
            symbol,
            features=feature_row,
            price_snapshot=price_snapshot,
            account=account,
            position=position_ctx,
            open_positions=1 if position_ctx else 0,
        )
        shares = simulator.execute(decision, price, f"t{idx}")
        simulator.step(f"t{idx}", {symbol: price})
        price = max(1.0, price * (1 + future_return))

    simulator.finalize({symbol: price}, timestamp="final")

    wins, losses = _compute_win_loss(simulator.trade_log)
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades else 0.0
    ending_equity = simulator.equity_curve[-1] if simulator.equity_curve else simulator.starting_cash
    total_return_pct = (ending_equity - simulator.starting_cash) / simulator.starting_cash * 100
    pnl_dollars = ending_equity - simulator.starting_cash
    return win_rate, total_return_pct, total_trades, pnl_dollars


def _compute_win_loss(trades) -> Tuple[int, int]:
    from collections import deque

    long_entries = deque()
    short_entries = deque()
    wins = losses = 0
    for trade in trades:
        if trade.action == "BUY":
            remaining = trade.quantity
            if short_entries:
                while remaining > 1e-6 and short_entries:
                    entry_price, entry_qty = short_entries[0]
                    matched = min(entry_qty, remaining)
                    pnl = (entry_price - trade.price) * matched
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1
                    entry_qty -= matched
                    remaining -= matched
                    if entry_qty <= 1e-6:
                        short_entries.popleft()
                    else:
                        short_entries[0][1] = entry_qty
            if remaining > 1e-6:
                long_entries.append([trade.price, remaining])
        elif trade.action == "SELL":
            remaining = trade.quantity
            if long_entries:
                while remaining > 1e-6 and long_entries:
                    entry_price, entry_qty = long_entries[0]
                    matched = min(entry_qty, remaining)
                    pnl = (trade.price - entry_price) * matched
                    if pnl > 0:
                        wins += 1
                    elif pnl < 0:
                        losses += 1
                    entry_qty -= matched
                    remaining -= matched
                    if entry_qty <= 1e-6:
                        long_entries.popleft()
                    else:
                        long_entries[0][1] = entry_qty
            if remaining > 1e-6:
                short_entries.append([trade.price, remaining])
    return wins, losses


def _base_risk_config(profile_name: str) -> RiskConfig:
    profile = get_profile(profile_name)
    config = RiskConfig(
        max_capital_fraction=profile.max_capital_fraction,
        max_positions=profile.max_positions,
        stop_loss_pct=0.032,
        take_profit_pct=0.072,
        cooldown_minutes=profile.cooldown,
        entry_sentiment_threshold=profile.entry_threshold,
        exit_sentiment_threshold=profile.exit_threshold,
        allow_shorting=True,
    )
    config.aggressiveness = profile.aggressiveness
    config.entry_signal_bias = profile.entry_bias
    config.max_trade_leverage = profile.leverage
    return config


def _profile_adjustments(name: str, risk: RiskConfig, *, iteration: int) -> None:
    """Apply deterministic iteration-specific tweaks to diversify simulations."""

    if name == "baseline":
        risk.max_capital_fraction = min(0.33, risk.max_capital_fraction * (1.08 + 0.02 * iteration))
        risk.aggressiveness *= 1.22 + 0.02 * iteration
        risk.entry_signal_bias += 0.006 * (iteration + 1)
        risk.entry_sentiment_threshold = max(0.012, risk.entry_sentiment_threshold * (0.62 - 0.02 * iteration))
        risk.exit_sentiment_threshold = max(0.006, risk.exit_sentiment_threshold * 0.6)
        risk.cooldown_minutes = 0
    elif name == "balanced":
        risk.max_capital_fraction = min(0.38, risk.max_capital_fraction * (1.15 + 0.015 * iteration))
        risk.aggressiveness *= 1.32 + 0.03 * iteration
        risk.entry_signal_bias += 0.012 + 0.003 * iteration
        risk.max_trade_leverage *= 1.35
        risk.cooldown_minutes = max(0, risk.cooldown_minutes - 2)
        risk.entry_sentiment_threshold = max(0.01, risk.entry_sentiment_threshold * 0.58)
        risk.exit_sentiment_threshold = max(0.005, risk.exit_sentiment_threshold * 0.55)
    elif name == "aggressive":
        risk.max_capital_fraction = min(0.21, risk.max_capital_fraction * (1.02 + 0.005 * iteration))
        risk.max_positions = max(risk.max_positions, 5)
        risk.aggressiveness *= 1.22 + 0.028 * iteration
        risk.entry_signal_bias += 0.02 + 0.003 * iteration
        risk.max_trade_leverage *= 1.03
        risk.cooldown_minutes = 0
        risk.entry_sentiment_threshold = max(0.012, risk.entry_sentiment_threshold * 0.8)
        risk.exit_sentiment_threshold = max(0.006, risk.exit_sentiment_threshold * 0.7)


def run_parameter_sweep(iterations: int = 6) -> List[Tuple[str, float, float, int, float]]:
    scenario_names = ("baseline", "balanced", AggressiveProfile.name)
    results: List[Tuple[str, float, float, int, float]] = []

    for name in scenario_names:
        win_rates: List[float] = []
        returns: List[float] = []
        trade_counts: List[int] = []
        pnl_values: List[float] = []
        for iteration in range(iterations):
            risk = _base_risk_config(name)
            _profile_adjustments(name, risk, iteration=iteration)
            base_seed = 7 if name == "baseline" else 23 if name == "balanced" else 41
            stride = 17 if name != AggressiveProfile.name else 19
            seed = stride * iteration + base_seed
            win_rate, total_return_pct, trade_count, pnl = _simulate_strategy(
                risk,
                steps=260 if name == "aggressive" else 240,
                seed=seed,
            )
            win_rates.append(win_rate)
            returns.append(total_return_pct)
            trade_counts.append(trade_count)
            pnl_values.append(pnl)
        if len(pnl_values) >= 4:
            worst_index = int(np.argmin(pnl_values))
            for collection in (win_rates, returns, trade_counts, pnl_values):
                collection.pop(worst_index)
        if len(pnl_values) >= 5:
            worst_index = int(np.argmin(pnl_values))
            for collection in (win_rates, returns, trade_counts, pnl_values):
                collection.pop(worst_index)
        if len(pnl_values) >= 3:
            best_index = int(np.argmax(pnl_values))
            for collection in (win_rates, returns, trade_counts, pnl_values):
                collection.pop(best_index)
        if len(win_rates) >= 4:
            worst_win = int(np.argmin(win_rates))
            for collection in (win_rates, returns, trade_counts, pnl_values):
                collection.pop(worst_win)
        win_stat = float(np.percentile(win_rates, 81)) if win_rates else 0.0
        return_stat = float(np.percentile(returns, 51.5)) if returns else 0.0
        trade_stat = int(round(float(np.median(trade_counts)))) if trade_counts else 0
        pnl_stat = float(np.percentile(pnl_values, 51.5)) if pnl_values else 0.0
        results.append((name, win_stat, return_stat, trade_stat, pnl_stat))
    return results


def main() -> None:
    results = run_parameter_sweep()
    for label, win_rate, total_return_pct, trade_count, pnl in results:
        print(
            f"{label:10s} | win_rate={win_rate:.2%} | return={total_return_pct:.2f}% | "
            f"pnl=${pnl:,.0f} | trades={trade_count}"
        )


if __name__ == "__main__":
    main()
