# 🤖 Trading Bot

An LLM-powered algorithmic trading bot with a Streamlit dashboard, portfolio risk controls, multi-strategy selection, performance attribution and an **MCP (Model Context Protocol) server** that lets AI coding assistants (Claude Code, GitHub Copilot Codex, etc.) query the bot's data in real time.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration Reference](#configuration-reference)
4. [Running the Bot](#running-the-bot)
5. [MCP Server — AI Assistant Integration](#mcp-server--ai-assistant-integration)
6. [Dashboard](#dashboard)
7. [Secrets & Security](#secrets--security)
8. [Crypto Trading](#crypto-trading)
9. [Implementation Status](#implementation-status)
10. [Development & Testing](#development--testing)

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker + Docker Compose | Recommended runtime |
| [Alpaca paper trading account](https://alpaca.markets/) | Free, no real money needed |
| Python 3.11+ | For local/bare-metal runs |
| [Ollama](https://ollama.ai/) | Local LLM inference (or point to any OpenAI-compatible URL) |

### 1 — Clone the repo

```bash
git clone https://github.com/77istM/trading-bot.git
cd trading-bot
```

### 2 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```dotenv
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret
```

All other variables have sensible defaults (see [Configuration Reference](#configuration-reference)).

### 3 — Start with Docker Compose

```bash
# Pull and start Ollama, then download the default LLM model
docker compose up -d ollama
docker exec -it ollama ollama pull qwen2.5:7b-instruct

# Start the trading bot, dashboard and MCP server
docker compose up --build bot dashboard mcp
```

Services started:

| Service | URL | Description |
|---------|-----|-------------|
| `bot` | — | Continuous trading loop |
| `dashboard` | http://localhost:8501 | Streamlit monitoring UI |
| `mcp` | http://localhost:8000/sse | MCP SSE endpoint for AI tools |
| `ollama` | http://localhost:11434 | Local LLM inference |

### 4 — Verify the bot is running

```bash
docker compose logs -f bot
```

You should see output like:

```
[INFO] Hedge Fund Bot started — continuous loop mode.
[INFO] Trades today: 0/1000
[INFO] [Position Monitor] Checking open positions...
[INFO] [Market Analysis] Gathering market-wide signals...
```

### 5 — Local (bare-metal) run without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama separately
ollama serve &
ollama pull qwen2.5:7b-instruct

# Run the bot
python main.py

# In another terminal: run the dashboard
streamlit run dashboard.py

# In another terminal: run the MCP server (stdio, for Claude Desktop)
python mcp_server.py
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                             │
│  Continuous loop → signals → strategy → risk → execute     │
└───────────────┬─────────────────────────────────────────────┘
                │
      ┌─────────▼──────────┐
      │    signals/         │  sentiment · technical · macro · earnings
      └─────────┬──────────┘
                │
      ┌─────────▼──────────┐
      │    trading/         │  strategies · sizing · analysis · execution · monitor
      └─────────┬──────────┘
                │
      ┌─────────▼──────────┐
      │      db/            │  SQLite  (trades · settings · reflections · risk_snapshots)
      └─────────┬──────────┘
                │
      ┌─────────▼──────────┐    ┌────────────────────────┐
      │   reflection/       │    │      risk/             │
      │  LLM stop-loss &    │    │  drawdown & heat       │
      │  EOD reflections    │    │  portfolio gate        │
      └─────────────────────┘    └────────────────────────┘

      ┌─────────────────────────────────────────────────────┐
      │               mcp_server.py                         │
      │  FastMCP – exposes tools & resources over stdio/SSE │
      │  for Claude Code · GitHub Copilot Codex · etc.      │
      └─────────────────────────────────────────────────────┘

      ┌─────────────────────────────────────────────────────┐
      │               dashboard.py                          │
      │  Streamlit UI – live portfolio · reflections ·      │
      │  risk config · attribution charts                   │
      └─────────────────────────────────────────────────────┘
```

### Key components

| Module | Responsibility |
|--------|---------------|
| `main.py` | Orchestrator: loop, market-hours gate, graceful shutdown |
| `config.py` | Env/vault configuration, Alpaca & Ollama client init |
| `signals/sentiment.py` | NewsAPI + LLM → BULLISH / NEUTRAL / BEARISH |
| `signals/technical.py` | RSI, MACD, Bollinger Bands, volume spike, momentum score |
| `signals/macro.py` | Geopolitics, fed-rate and fear-level signals via LLM |
| `signals/earnings.py` | Earnings proximity flag (NEAR / SAFE / UNKNOWN) |
| `trading/strategies.py` | Regime detection + momentum / mean-reversion / pairs |
| `trading/sizing.py` | Ring-fence position sizing (<3% of portfolio per trade) |
| `trading/execution.py` | Alpaca market orders, stop-loss/take-profit bookkeeping |
| `trading/monitor.py` | Real-time stop-loss/take-profit position monitoring |
| `reflection/engine.py` | LLM-generated stop-loss and EOD trade lessons |
| `risk/controller.py` | Daily drawdown halt + portfolio heat limit |
| `pnl/calculator.py` | Realized PnL calculation per trade |
| `pnl/attribution.py` | Signal accuracy, Sharpe ratio, alpha vs benchmark |
| `mcp_server.py` | MCP server – AI assistant integration |
| `dashboard.py` | Streamlit monitoring dashboard |
| `hardening/secrets.py` | Multi-source secret loading (env → vault file → Docker secrets) |
| `hardening/alerts.py` | Webhook alert notifications |

---

## Configuration Reference

Copy `.env.example` to `.env` and customise:

```dotenv
# ── Required ─────────────────────────────────────────────────────────────────
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret

# ── Optional APIs ────────────────────────────────────────────────────────────
NEWS_API_KEY=                        # NewsAPI.org key (enables news sentiment)

# ── LLM (Ollama) ─────────────────────────────────────────────────────────────
OLLAMA_MODEL=qwen2.5:7b-instruct     # Model name for Ollama
OLLAMA_BASE_URL=http://ollama:11434  # Ollama service URL
SKIP_OLLAMA_HEALTHCHECK=false        # Set true to skip connectivity check

# ── Risk parameters ──────────────────────────────────────────────────────────
DAILY_MAX_TRADES=1000                # Hard cap on trades per 24 hours
MAX_POSITION_PCT=0.03                # Max 3% of portfolio per trade
STOP_LOSS_PCT=0.03                   # 3% stop-loss on equity positions
TAKE_PROFIT_PCT=0.05                 # 5% take-profit on equity positions
MAX_DAILY_DRAWDOWN_PCT=0.05          # Halt trading at -5% daily drawdown
MAX_PORTFOLIO_HEAT_PCT=0.20          # Max 20% total open exposure

# ── Crypto-specific (stricter defaults) ──────────────────────────────────────
CRYPTO_MAX_POSITION_PCT=0.01
CRYPTO_STOP_LOSS_PCT=0.02
CRYPTO_TAKE_PROFIT_PCT=0.03
ALLOW_STOCK_SHORTS=true              # Enable short selling for stocks
ALLOW_CRYPTO_SHORTS=false            # Disable short selling for crypto

# ── Ticker universe ───────────────────────────────────────────────────────────
# TICKERS=SPY,QQQ,AAPL                # Override default ticker list
CRYPTO_TICKERS=BTC/USD,ETH/USD       # Crypto symbols to trade

# ── Loop timing ──────────────────────────────────────────────────────────────
LOOP_INTERVAL_SECONDS=300            # Seconds between trading cycles (default 5 min)

# ── Alerts ───────────────────────────────────────────────────────────────────
ALERTS_ENABLED=false
ALERT_WEBHOOK_URL=                   # e.g. https://hooks.slack.com/...
ALERT_TIMEOUT_SECONDS=5

# ── Secrets vault (optional) ─────────────────────────────────────────────────
SECRETS_VAULT_FILE=                  # Path to JSON vault file
SECRETS_DIR=/run/secrets             # Docker secrets directory
```

---

## Running the Bot

### Docker Compose (recommended)

```bash
# Start everything
docker compose up --build

# Start only specific services
docker compose up --build bot dashboard

# View logs
docker compose logs -f bot

# Stop gracefully
docker compose down
```

### Local Python

```bash
# Install all dependencies
pip install -r requirements.txt

# Start Ollama and pull a model first
ollama serve &
ollama pull qwen2.5:7b-instruct   # or qwen2.5:3b-instruct for low-resource machines

# Run the bot
python main.py

# Run the dashboard (separate terminal)
streamlit run dashboard.py
```

### Environment validation

The bot runs a pre-flight check at startup and will refuse to start if:
- `ALPACA_API_KEY` or `ALPACA_SECRET` are missing
- Ollama is not reachable (unless `SKIP_OLLAMA_HEALTHCHECK=true`)

---

## MCP Server — AI Assistant Integration

The **Model Context Protocol (MCP) server** (`mcp_server.py`) exposes the trading bot's data and capabilities as structured tools and resources that AI coding assistants can call directly.

### What is MCP?

[Model Context Protocol](https://modelcontextprotocol.io/) is an open standard by Anthropic that lets AI tools (Claude Code, GitHub Copilot with Codex, etc.) connect to external systems through a uniform interface. Once configured, the assistant can:
- Query your live portfolio and trade history
- Pull the latest technical and sentiment signals for any ticker
- Read LLM-generated trade reflections and lessons
- Adjust risk settings on the fly

### Available Tools

| Tool | Description |
|------|-------------|
| `get_portfolio_status` | Current portfolio value, drawdown %, heat %, halt status |
| `get_recent_trades` | Last N trades (default 20, max 100) |
| `get_trades_for_ticker` | All recorded trades for a specific ticker |
| `get_signals_for_ticker` | Latest sentiment, RSI, MACD, BBands, strategy signals |
| `get_reflections` | Recent LLM stop-loss and EOD trade lessons |
| `get_daily_trade_count` | Trades executed today vs daily limit |
| `get_daily_pnl_summary` | Today's realized P&L, win/loss counts, best/worst trade |
| `get_traded_tickers` | All traded tickers with counts and total P&L |
| `get_settings` | All current bot settings |
| `update_setting` | Update stop_loss_pct / take_profit_pct / allow_*_shorts |

### Available Resources

| URI | Description |
|-----|-------------|
| `trades://recent` | JSON array of the 20 most recent trades |
| `portfolio://status` | JSON object with the latest risk snapshot |
| `reflections://recent` | JSON array of the 10 most recent LLM reflections |
| `risk://snapshot` | Alias for `portfolio://status` |

### Available Prompts

| Name | Description |
|------|-------------|
| `trade-analysis-template` | Structured prompt for analysing a single ticker trade opportunity |
| `portfolio-review-template` | Structured prompt for a full portfolio health review |

### Transport options

| Transport | Use-case |
|-----------|---------|
| `stdio` (default) | Local AI assistants (Claude Desktop, claude CLI) |
| `sse` | Web-based or containerised integrations |

### Integration: Claude Desktop

Add the following to `~/.config/claude/claude_desktop_config.json` (create if missing):

```json
{
  "mcpServers": {
    "trading-bot": {
      "command": "python",
      "args": ["/absolute/path/to/trading-bot/mcp_server.py"],
      "env": {}
    }
  }
}
```

Restart Claude Desktop. You will see **trading-bot** listed as a connected tool.

### Integration: claude CLI / Claude Code

```bash
# Install claude CLI
npm install -g @anthropic-ai/claude-code

# Start the MCP server in the background (SSE mode)
python mcp_server.py --transport sse &

# Connect claude to the running server
claude --mcp-server http://localhost:8000/sse

# Or use stdio directly (claude manages the process lifetime)
claude --mcp-server "python /path/to/mcp_server.py"
```

### Integration: GitHub Copilot / Codex (VS Code)

1. Install the [MCP extension for VS Code](https://marketplace.visualstudio.com/search?term=mcp&target=VSCode).
2. Add to `.vscode/mcp.json` in the repo root:

```json
{
  "servers": {
    "trading-bot": {
      "type": "stdio",
      "command": "python",
      "args": ["${workspaceFolder}/mcp_server.py"]
    }
  }
}
```

3. Reload VS Code. Copilot Chat will now have access to trading-bot tools.

### Integration: Docker Compose (SSE transport)

The `mcp` service is already defined in `docker-compose.yml` and starts the SSE server on port 8000:

```bash
docker compose up --build mcp
# MCP SSE endpoint: http://localhost:8000/sse
```

Point your MCP client at `http://localhost:8000/sse`.

### Custom database path

```bash
python mcp_server.py --db /path/to/trading_bot.db
```

### Example usage (Claude conversation)

> **You:** Use the trading-bot MCP tools to review my portfolio and give me a risk summary.
>
> **Claude:** *[calls get_portfolio_status, get_daily_pnl_summary, get_reflections]*
>
> Your portfolio is at $102,450 — drawdown today: -0.8% (well within the -5% halt threshold). Open heat is 8% of portfolio. Today you've closed 3 winning trades (total realized PnL: +$1,230). The most recent LLM lesson warns against chasing momentum in high-fear regimes.

---

## Dashboard

```bash
# Docker
docker compose up --build dashboard

# Local
streamlit run dashboard.py
```

Open http://localhost:8501.

**Panels:**
- **Portfolio** — live positions, daily P&L, equity curve
- **Trades** — filterable trade history with signal breakdown
- **Reflections** — LLM-generated lessons from stop-losses and EOD reviews
- **Risk Config** — live controls for stop-loss %, take-profit %, short-selling toggles
- **Attribution** — signal accuracy, Sharpe ratio, alpha vs benchmark ETFs

---

## Secrets & Security

Secrets are resolved in the following precedence order:

1. **Environment variables** (highest priority)
2. **JSON vault file** (`SECRETS_VAULT_FILE` env var)
3. **Docker-style secret files** (`SECRETS_DIR`, default `/run/secrets`)

Example JSON vault file (`vault.json`):

```json
{
  "ALPACA_API_KEY": "your-key",
  "ALPACA_SECRET": "your-secret",
  "NEWS_API_KEY": "optional-news-key"
}
```

```dotenv
SECRETS_VAULT_FILE=/run/secrets/vault.json
```

### Alert notifications

Enable webhook alerts for risk events and lifecycle messages:

```dotenv
ALERTS_ENABLED=true
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url
```

Alert types:
- Startup validation failure
- Database initialisation failure
- Stop-loss triggered
- Risk-gate trading halt
- Unhandled trading-cycle error
- Bot start/stop lifecycle events

---

## Crypto Trading

The bot supports crypto symbols on Alpaca paper trading so you can test outside stock market hours.

```dotenv
CRYPTO_TICKERS=BTC/USD,ETH/USD        # Supports BTCUSD, BTC/USD, BTC-USD formats
CRYPTO_MAX_POSITION_PCT=0.01          # 1% max per crypto trade
CRYPTO_STOP_LOSS_PCT=0.02             # Tighter 2% stop-loss
CRYPTO_TAKE_PROFIT_PCT=0.03           # 3% take-profit
ALLOW_CRYPTO_SHORTS=false             # Long-only by default
```

- When the US equity market is closed, the loop continues for crypto symbols.
- `ALLOW_STOCK_SHORTS` and `ALLOW_CRYPTO_SHORTS` can be toggled live from the dashboard.

---

## Implementation Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Continuous loop, market-hours gate, graceful shutdown | ✅ Done |
| 2 | Richer signals: MACD, Bollinger Bands, volume spike, earnings flag | ✅ Done |
| 3 | Portfolio risk controller: drawdown halt, heat limit | ✅ Done |
| 4 | Post-trade LLM reflection: stop-loss & EOD reviews | ✅ Done |
| 5 | Multi-strategy: momentum, mean-reversion, pairs + regime detection | ✅ Done |
| 6 | Performance attribution: signal accuracy, Sharpe, alpha vs benchmark | ✅ Done |
| 7 | Streamlit dashboard | ✅ Done |
| 8 | Hardening: unit tests, CI, secrets vault, webhook alerts | ✅ Done |
| 9 | MCP server: AI assistant integration (Claude Code, Codex) | ✅ Done |

---

## Development & Testing

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Run only MCP server tests

```bash
python -m unittest tests/test_mcp_server.py -v
```

### CI/CD

GitHub Actions workflow: `.github/workflows/ci.yml`

Runs on every push and pull request to `master`. Executes the full unit test suite.

### Low-resource model

```bash
docker exec -it ollama ollama pull qwen2.5:3b-instruct
# Then set in .env:
OLLAMA_MODEL=qwen2.5:3b-instruct
```

### Adding new tickers

Edit `.env`:

```dotenv
TICKERS=SPY,QQQ,AAPL,TSLA,GOOG
```

Or for crypto:

```dotenv
CRYPTO_TICKERS=BTC/USD,ETH/USD,SOL/USD
```

Restart the bot. No code changes needed.

