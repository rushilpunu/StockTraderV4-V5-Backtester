# TraderV4 Package Guide

TraderV4 is split into a few focused modules. This guide walks through every
public class and function, explains what it does, and notes any important edge
cases. Use it as the reference for maintaining or extending the live trading
stack.

---

## `Traderv4/GDELT.py`

### `GDELTArticle`
Lightweight container for the key fields we keep from a GDELT news record:
title, source URL, published timestamp, sentiment `tone`, and an optional
ticker symbol.

### `GDELTClient`
* `BASE_URL` – Endpoint for the GDELT document API.
* `__init__(delay: float = 1.5)` – Stores a fixed throttling delay so we do not
  hammer the public API.
* `_request(params)` – Issues the HTTP GET with a 10-second timeout, sleeps for
  the configured delay, and converts the response to JSON. Bad HTTP status codes
  or JSON decode issues are surfaced as `RuntimeError`.
* `_resolve_ticker(entry, fallback)` – Normalises the symbols supplied by GDELT
  (themes, tags, etc.) and falls back to the originally requested ticker when no
  valid symbol is present.
* `fetch_recent_articles(ticker, minutes_back)` – Pulls up to 250 articles for
  the requested span, normalises each into a `GDELTArticle`, and returns the
  list. This runs on every trading cycle when the market is open.
* `fetch_sentiment_timeline(ticker, minutes_back)` – Convenience helper for the
  alternate `TimelineTone` endpoint. Currently unused by the live loop.

---

## `Traderv4/funcs.py`

### Dataclasses
* `ArticleSentiment` – Normalised sentiment datapoint (ticker, score, timestamp,
  source, metadata).
* `TickerSentimentSnapshot` – Aggregated window for a ticker containing average
  sentiment, delta, weighted score, and the raw articles used.
* `Alert` – Result of the volatility analyzer; includes trigger reason and
  metrics.
* `RiskConfig` – All runtime risk knobs (allocation, stop/take levels, cooldown,
  entry/exit sentiment buffers, per-position caps, optional shorting flag).
* `VolatilityThresholds` – Controls alert generation. Besides spike/extreme
  thresholds it includes an average sentiment floor and a single-article spike
  check so the engine can react to sparse news.
* `TradeDecision` – Output of the decision engine: action, notional/quantity,
  stops, and intent (`entry`, `exit`, or `hold`).
* `AccountSnapshot` – Normalised Alpaca account balances used for sizing.

### Market data utilities
* `YahooFinanceClient._fetch_chart` – Shared helper for the Yahoo chart API.
* `fetch_intraday_snapshot(ticker, period='5d', interval='15m')` – Provides the
  latest close/volume, or `None` values when data is missing.
* `fetch_volatility_baseline(ticker, lookback_days=20)` – Computes average
  volume and intraday range from the Yahoo daily candles.

### Sentiment processing
* `SentimentProcessor.normalize_gdelt_payload(payload)` – Converts a raw payload
  from the REST API into `ArticleSentiment` rows.
* `from_gdelt_articles(records)` – Converts already-parsed `GDELTArticle`
  objects into `ArticleSentiment`.
* `group_by_ticker(articles)` – Dict keyed by ticker containing all associated
  articles.
* `aggregate_sentiment(grouped, window)` – Filters to the requested time window
  and produces `TickerSentimentSnapshot` instances for downstream analysis.

### Volatility and frequency logic
* `VolatilityAnalyzer` – Applies the configured thresholds to a snapshot. The
  analyzer now supports:
  - traditional spike/extreme triggers,
  - an average-sentiment floor when the reading stays elevated, and
  - a single-article spike check for headline shocks.
* `TradeFrequencyTracker` – Tracks intraday trade counts, enforces PDT guard,
  and exposes `intraday_trades()`, `can_enter()`, and `next_reset_date()`.

### Decision pipeline
* `DecisionEngine` – Central brain for position management. The workflow is:
  1. Build a blended sentiment score (average tone + delta, boosted by article
     count) and maintain a short history per ticker.
  2. If a position already exists, inspect the trend score and exit thresholds
     before sending an unwind order; otherwise prefer holding to stay within PDT
     limits.
  3. When flat, adjust the entry requirement according to current PDT usage,
     size the order proportionally to the signal strength, and reject signals
     that cannot justify a minimum allocation.
  4. Emit a `TradeDecision` (`hold`, `entry`, or `exit`) with metadata that the
     dashboard/logs can display (score, trend, article boost, etc.).

### Trade execution
* `TradeExecutor.place_trade(decision)` – Translates a decision into an Alpaca
  order. Entries submit notional bracket orders for long trades only; exits send
  quantity-based market orders.
* `sync_trade_activity()` – Pulls recent fills to keep the frequency tracker in
  sync. Connection errors are logged and ignored so the loop keeps running.
* `get_open_positions()` – Normalises the Alpaca position payload into plain
  dictionaries.
* `get_account_snapshot()` – Fetches the latest account balances, falling back
  to zeroes on error.
* `get_market_clock()` – Reads the exchange clock (open/close/timestamps) so the
  main loop can skip trading outside market hours.
* `_parse_activity_time(raw)` – Utility used by the methods above to turn ISO
  strings into naive UTC datetimes.

### Event logging
`EventLogger` centralises the human-readable log lines for alerts, decisions,
trade-frequency stats, market-hour notifications, and end-of-cycle summaries.

---

## `Traderv4/state.py`

### Records
* `AlertRecord`, `DecisionRecord`, `PositionRecord`, `SentimentPoint` –
  Serialisable wrappers used by the FastAPI layer.
* `DayTradeStatus` – Shows current PDT usage and next reset timestamp.
* `MarketStatus` – Describes whether the market is open along with the next
  open/close times.
* `CycleDiagnostics` – Captures the last trading cycle’s diagnostics (start/end
  timestamps, counts of tickers/alerts/decisions, entry/exit/hold totals, and
  any errors that were caught without killing the loop).

### `TraderSnapshot`
Aggregates the records above so the dashboard can display one coherent view of
alerts, decisions, exposure, sentiment history, day-trade guard, market status,
and detailed cycle diagnostics.

### `StateStore`
Thread-safe in-memory cache. `update(...)` replaces the full snapshot each cycle
and `snapshot()` returns a deep copy for the API layer.

---

## `Traderv4/main.py`

### `TraderConfig`
Configuration bundle for the automation loop (watch list, GDELT lookback,
window size, pause between cycles, thresholds, and risk profile). The default
watch list now contains a diversified basket of large-cap equities; override it
by setting the environment variable `TRADERV4_TICKERS=SYM1,SYM2,...`.

### `NewsSentimentTrader`
* `__init__` – Wires together all collaborators (GDELT client, Yahoo data,
  sentiment processor, analyzer, decision engine, executor, state store).
* `run_cycle()` – One end-to-end pass:
  1. Sync recent fills and fetch an account snapshot.
  2. Pull the current market clock. When the exchange is closed, the loop skips
     sentiment/news processing, logs the planned next open, and simply refreshes
     state for the dashboard.
  3. When the market is open, fetch baselines, ingest GDELT news, aggregate
     sentiment, evaluate alerts, and produce decisions.
  4. Log every decision (including `HOLD`), send live orders for actionable
     entries/exits, emit a cycle summary, and update the shared `StateStore`.
* `run_forever(stop_event)` – Repeats `run_cycle` with the configured pause,
  catching unexpected exceptions so the loop keeps running.
* `get_settings()` / `update_settings(payload)` – Expose live configuration for
  the API and dashboard. The response now includes the extra sentiment
  thresholds described above.

### Module helpers
* `_load_alpaca_client()` – Creates the Alpaca REST client from environment
  variables.
* `_load_account_balance(alpaca)` – Simple helper used for legacy balance reads
  (kept for compatibility with older interfaces).
* CLI entry point – When the module is executed directly it installs signal
  handlers and starts `run_forever`, so the trader stays alive until you press
  Ctrl+C.

---

## `Traderv4/api.py` and `Traderv4/server.py`
* `create_app(...)` – Builds the FastAPI application that exposes snapshots,
  alerts, positions, risk settings, diagnostics, and backtester endpoints.
* `build_trader(config)` – Factory for the live `NewsSentimentTrader` wired to
  Alpaca. When no config is provided it uses the expanded default watch list
  described above.
* `server.run(...)` – Launches both the FastAPI app and the trading loop in a
  background thread so the dashboard reflects real-time data.

---

## Operational Notes

* The live loop only submits orders during regular market hours (as determined
  by Alpaca’s exchange clock). Outside the session it still updates the
  dashboard so you can monitor exposure.
* Connection issues to Alpaca or GDELT are logged but do not stop the loop; they
  simply result in empty cycles.
* When you adjust risk or threshold settings through the API/dashboard, the
  values propagate immediately to the next trading cycle.
* Use the new diagnostics block exposed through `/snapshot` to monitor how many
  tickers were scanned, how many signals generated trades, and whether any
  recoverable errors occurred during the loop.
