# 🏦 Hedge Fund Trading Bot

An autonomous, multi-signal AI trading bot running on **Alpaca paper trading** (zero-cost OCI Docker deployment). The bot uses a local **Ollama LLM (llama3.2:3b)** to synthesise market signals, reflect on its own mistakes, and improve over time.

---

## 📊 Implementation Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| **Phase 1** | Continuous Loop | ✅ **Done** | Infinite scheduler loop, market-hours gate, graceful shutdown |
| **Phase 2** | Richer Signals | ✅ **Done** | MACD, Bollinger Bands, volume spike, earnings flag, momentum score |
| **Phase 3** | Portfolio Risk Controller | ✅ **Done** | Daily drawdown halt, portfolio heat limit, risk snapshots |
| **Phase 4** | Post-Trade Reflection | ✅ **Done** | Stop-loss real-time reflection, EOD review, lesson injection |
| **Phase 5** | Multi-Strategy | 🔲 Backlog | Momentum, mean-reversion, pairs trading |
| **Phase 6** | Performance Attribution | 🔲 Next up | Signal accuracy tracking, Sharpe, alpha vs benchmark |
| **Phase 7** | Dashboard | ✅ Partial | Reflections viewer + risk status; needs attribution charts |
| **Phase 8** | Hardening | 🔲 Backlog | Tests, CI/CD, secrets vault, alert notifications |

### ➡️ **Next Phase to Work On: Phase 6 — Performance Attribution**

---

## ✅ What Has Been Implemented

### Phase 2 — Richer Signals
- `signals/technical.py` — complete rewrite, now computes a **full indicator suite**:
  - **RSI(14)**: oversold < 30 → BULLISH, overbought > 70 → BEARISH
  - **MACD(12,26,9)**: histogram direction + fresh crossover detection → BULLISH / BEARISH / NEUTRAL
  - **Bollinger Bands(20, 2σ)**: price below lower band → BULLISH, above upper → BEARISH
  - **Volume spike**: current volume ≥ 2× 20-day average → SPIKE_UP / SPIKE_DOWN
  - **Momentum score**: composite −3 to +3 (RSI ±1, MACD ±1, BBands ±1, Volume ±0.5)
  - `get_technical_signals(ticker) → dict` returns all indicators plus a compact `summary` string
- `signals/earnings.py` (new) — **zero-cost earnings proximity detection**:
  - Uses NewsAPI to scan the last 4 days of headlines for upcoming-earnings language
  - Returns `NEAR` (event risk, be cautious), `SAFE`, or `UNKNOWN` (no API key)
  - Distinguishes "upcoming" from "already reported" via separate regex pattern sets
- Pre-trade LLM prompt now shows **all 6 technical dimensions** with inline explanations; the LLM reasons about earnings risk, volume spikes, and momentum direction
- **Five new DB columns** on the `trades` table: `macd_signal`, `bbands_signal`, `volume_signal`, `earnings_flag`, `momentum_score`
- Dashboard trade-history table now shows all new signal columns

### Phase 1 — Continuous Loop
- `main.py` is now an **infinite scheduler loop** (default cycle: every 5 minutes, configurable via `LOOP_INTERVAL_SECONDS`)
- **Market-hours awareness**: uses Alpaca's `get_clock()` API; skips cycles when market is closed
- **Graceful shutdown**: handles `SIGTERM` / `SIGINT` — completes the current cycle before stopping
- **EOD trigger**: automatically runs end-of-day reflection once per day after 21:00 UTC (≈ 4 PM ET)
- **Day-start reset**: portfolio risk controller resets its baseline at the start of each new calendar day
- **Error isolation**: unhandled errors in a single cycle are caught and logged; the loop continues

### Phase 3 — Portfolio Risk Controller
- `risk/controller.py` — `PortfolioRiskController` class with two hard limits:
  1. **Daily drawdown halt**: if portfolio drops ≥ `MAX_DAILY_DRAWDOWN_PCT` (default 5%) from day-open → all new trades blocked
  2. **Portfolio heat limit**: if total open market value ≥ `MAX_PORTFOLIO_HEAT_PCT` (default 20%) of equity → new trades blocked
- **Risk snapshots** written to `risk_snapshots` table every cycle for full audit trail
- Integrated into `main.py` before each ticker loop
- Dashboard shows current risk status, halt alerts, and snapshot history

### Phase 4 — Post-Trade Reflection
- `reflection/engine.py` — three reflection modes:
  1. **`reflect_on_stop_loss()`**: fires immediately when a stop-loss triggers; LLM identifies which signals were wrong and generates an actionable lesson
  2. **`reflect_on_trade()`**: fires after each executed trade; LLM commits to a 24-hour price prediction
  3. **`run_end_of_day_reflection()`**: fires once after market close; LLM reviews all of today's trades, identifies what worked/failed, and generates 3 rules for tomorrow
- **Lesson injection**: `pre_trade_analysis()` now fetches the most recent stop-loss and EOD lessons from the DB and injects them into the LLM prompt so the bot learns from its mistakes
- Lessons stored in `reflections` table (type, ticker, outcome, PnL, lesson text, full LLM analysis)
- Dashboard shows stop-loss reflections, EOD reviews, and full lesson history in separate tabs

### Asset Universe Update
- Default tickers updated to **ETFs-first** (macro hedge vehicles) + core equities:
  - Broad ETFs: `SPY`, `QQQ`, `IWM`, `EFA`, `EEM`
  - Global index proxies (benchmark + tradeable): `EWU` (FTSE), `EWJ` (Nikkei), `EWQ` (CAC40), `EWG` (DAX)
  - Sector ETFs: `XLF`, `XLK`, `XLE`, `XLV`, `GLD`, `TLT`
  - Core equities: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `NVDA`, `JPM`, `BAC`, `JNJ`, `XOM`, `WMT`
- Benchmark tickers defined: `["SPY", "EWU", "EWJ", "EWQ", "EWG"]`

---

## 🏗️ Architecture

```
trading-bot/
├── main.py                    # Continuous loop orchestrator (Phase 1)
├── config.py                  # All env vars, clients, ticker universe
├── dashboard.py               # Streamlit UI
├── db/
│   ├── schema.py              # SQLite schema (trades, settings, reflections, risk_snapshots)
│   └── queries.py             # Shared read/write helpers
├── signals/
│   ├── sentiment.py           # NewsAPI + LLM sentiment
│   ├── technical.py           # RSI, MACD, BBands, Volume spike, Momentum score (Phase 2)
│   ├── earnings.py            # Earnings proximity detection via NewsAPI (Phase 2)
│   └── macro.py               # Geopolitics, Fed rate, VIX/fear
├── trading/
│   ├── analysis.py            # Pre-trade LLM analysis (with lesson injection)
│   ├── execution.py           # Order submission + PnL tracking
│   ├── monitor.py             # Stop-loss / take-profit position monitor
│   └── sizing.py              # Ring-fence position sizing
├── reflection/
│   └── engine.py              # Post-trade reflection engine (Phase 4)
├── risk/
│   └── controller.py          # Portfolio risk controller (Phase 3)
├── pnl/
│   └── calculator.py          # Realized PnL calculator
├── Dockerfile                 # Python 3.11-slim + TA-Lib
└── docker-compose.yml         # bot + dashboard services
```

### Database Tables

| Table | Purpose |
|-------|---------|
| `trades` | Full trade log with signals, analysis, PnL |
| `settings` | Runtime-overridable risk params (stop %, take %) |
| `reflections` | LLM lessons from stop-losses & EOD reviews |
| `risk_snapshots` | Per-cycle portfolio risk audit trail |

---

## ⚙️ Configuration

```bash
# API Credentials (REQUIRED)
ALPACA_API_KEY=<alpaca-paper-key>
ALPACA_SECRET=<alpaca-secret>
NEWS_API_KEY=<newsapi-org-key>         # Optional — sentiment degrades to NEUTRAL without it

# Trading Parameters
DAILY_MAX_TRADES=1000                  # Hard cap per 24 hours
MAX_POSITION_PCT=0.03                  # Ring fence: 3% max per trade
STOP_LOSS_PCT=0.03                     # Default 3% stop loss
TAKE_PROFIT_PCT=0.05                   # Default 5% take profit
TICKERS="SPY,QQQ,AAPL"                # CSV override (defaults to curated universe)

# Continuous Loop
LOOP_INTERVAL_SECONDS=300              # Cycle interval (default 5 min)

# Portfolio Risk Controller
MAX_DAILY_DRAWDOWN_PCT=0.05            # Halt trading if down 5% from day-open
MAX_PORTFOLIO_HEAT_PCT=0.20            # Halt new trades if 20%+ of equity deployed

# Dashboard
TRADING_DB_PATH=trading_bot.db
```

---

## 🚀 Quick Start (Docker)

```bash
# 1. Create .env file
cat > .env << EOF
ALPACA_API_KEY=xxxxx
ALPACA_SECRET=yyyyy
NEWS_API_KEY=zzzzz
EOF

# 2. Start local Ollama with llama3.2:3b
docker run -d --name ollama -p 11434:11434 ollama/ollama
docker exec ollama ollama pull llama3.2:3b

# 3. Start bot + dashboard
docker-compose up -d

# 4. Dashboard → http://localhost:8501
# 5. Logs      → docker logs trading_bot -f
```

---

## 🔮 What Needs to Be Done

### Phase 6 — Performance Attribution *(Medium — high priority next)*
- Track which signals contributed to winning/losing trades
- Sharpe ratio, max drawdown, win rate calculation
- Benchmark comparison (SPY, EWU, EWJ, EWQ, EWG)
- Signal accuracy scoreboard (was BULLISH RSI/MACD followed by actual price up?)
- Per-signal P&L contribution breakdown

### Phase 5 — Multi-Strategy *(High complexity)*
- Implement distinct strategy classes: momentum, mean-reversion, pairs trading
- Strategy selection based on current market regime (trending vs. ranging)
- Per-strategy PnL tracking

### Phase 6 — Performance Attribution *(Medium)*
- Track which signals contributed to winning/losing trades
- Sharpe ratio, max drawdown, win rate calculation
- Benchmark comparison (SPY, EWU, EWJ, EWQ, EWG)
- Signal accuracy scoreboard (was BULLISH sentiment actually followed by price up?)

### Phase 7 — Dashboard Enhancements *(Low–Medium)*
- Live benchmark overlay on PnL chart
- Signal accuracy heat-map
- Open position P&L table with live prices
- Trade execution interface (manual override from UI)

### Phase 8 — Hardening *(Medium)*
- Unit tests for all modules (especially PnL calculator, risk controller)
- CI/CD pipeline (GitHub Actions)
- Slack/email alerts on stop-loss events and daily summary
- Proper secrets management (no `.env` file in production)
- Prometheus metrics export

---

## 📈 Benchmarks

The bot's performance is measured against these global index proxies (all US-listed ETFs):

| Benchmark | ETF Proxy | Index |
|-----------|-----------|-------|
| S&P 500 | SPY | US large-cap |
| FTSE 100 | EWU | UK large-cap |
| Nikkei 225 | EWJ | Japan large-cap |
| CAC 40 | EWQ | France large-cap |
| DAX | EWG | Germany large-cap |

---

## 🔄 Trading Mode

The bot runs in **Alpaca paper-trading mode** for training. Once target accuracy is achieved through paper trading, it can be switched to live mode by changing `paper=True` → `paper=False` in `config.py`.
