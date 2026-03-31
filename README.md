

## 📊 Implementation Status

## Ollama + Qwen Runtime (Codespaces)

The bot is configured to use Ollama with Qwen by default:

- `OLLAMA_MODEL=qwen2.5:7b-instruct`
- `OLLAMA_BASE_URL=http://ollama:11434`
- `SKIP_OLLAMA_HEALTHCHECK=false`

`docker-compose.yml` includes an `ollama` service and wires `bot`/`dashboard`
to wait for it.

First-time model pull:

```bash
docker compose up -d ollama
docker exec -it ollama ollama pull qwen2.5:7b-instruct
```

Run the app stack:

```bash
docker compose up --build bot dashboard
```

Optional fast fallback model for lower CPU/RAM environments:

```bash
docker exec -it ollama ollama pull qwen2.5:3b-instruct
# then set OLLAMA_MODEL=qwen2.5:3b-instruct in .env
```

## Crypto Trading Support

The bot can now trade crypto symbols on Alpaca paper trading so you can test outside stock market hours.

- Configure crypto symbols with `CRYPTO_TICKERS` (examples: `BTC/USD,ETH/USD` or `BTCUSD,ETHUSD`).
- Crypto risk defaults are stricter than equities:
	- `CRYPTO_MAX_POSITION_PCT=0.01`
	- `CRYPTO_STOP_LOSS_PCT=0.02`
	- `CRYPTO_TAKE_PROFIT_PCT=0.03`
	- `ALLOW_STOCK_SHORTS=true` (stocks can be long+short by default)
	- `ALLOW_CRYPTO_SHORTS=false` (default long-only for safety)
- Both `ALLOW_STOCK_SHORTS` and `ALLOW_CRYPTO_SHORTS` can be toggled live from the dashboard
  Risk Configuration panel and are persisted in the database settings table.
- When equities are closed, the loop still runs for crypto symbols.

You can keep your stock universe in `TICKERS`; the bot handles both universes together.

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| **Phase 1** | Continuous Loop | ✅ **Done** | Infinite scheduler loop, market-hours gate, graceful shutdown |
| **Phase 2** | Richer Signals | ✅ **Done** | MACD, Bollinger Bands, volume spike, earnings flag, momentum score |
| **Phase 3** | Portfolio Risk Controller | ✅ **Done** | Daily drawdown halt, portfolio heat limit, risk snapshots |
| **Phase 4** | Post-Trade Reflection | ✅ **Done** | Stop-loss real-time reflection, EOD review, lesson injection |
| **Phase 5** | Multi-Strategy | ✅ **Done** | Regime-based strategy selector with momentum, mean-reversion, and pairs trading |
| **Phase 6** | Performance Attribution | ✅ **Done** | Signal accuracy tracking, Sharpe, drawdown, alpha vs benchmark |
| **Phase 7** | Dashboard | ✅ **Done** | Reflections viewer, risk status, and attribution charts |
| **Phase 8** | Hardening | ✅ **Done** | Unit tests, GitHub Actions CI, secrets vault loading, webhook alerts |


## Phase 8 Hardening

### 1) Tests

- Added hardening test coverage in `tests/test_hardening.py`.
- Run all tests locally:

```bash
python -m unittest discover -s tests -v
```

### 2) CI/CD

- GitHub Actions workflow: `.github/workflows/ci.yml`
- CI dependency lock for test/runtime imports: `requirements-ci.txt`

The CI workflow runs on pushes and PRs to `master` and executes the full unit test suite.

### 3) Secrets Vault

Secrets are now resolved in this precedence order:

1. Environment variables
2. JSON vault file (`SECRETS_VAULT_FILE`)
3. Docker-style secret files (`SECRETS_DIR`, default `/run/secrets`)

Required runtime secrets:

- `ALPACA_API_KEY`
- `ALPACA_SECRET`

Optional:

- `NEWS_API_KEY`
- `OLLAMA_MODEL`
- `OLLAMA_BASE_URL`
- `SKIP_OLLAMA_HEALTHCHECK`

Example JSON vault file:

```json
{
	"ALPACA_API_KEY": "your-key",
	"ALPACA_SECRET": "your-secret",
	"NEWS_API_KEY": "optional-news-key"
}
```

### 4) Alert Notifications

Webhook alerts are now supported for startup failures and runtime risk/error events.

Enable alerts with:

- `ALERTS_ENABLED=true`
- `ALERT_WEBHOOK_URL=https://your-endpoint`
- `ALERT_TIMEOUT_SECONDS=5` (optional)

Alert categories emitted by the bot:

- Startup validation failure
- Database initialization failure
- Stop-loss triggered
- Risk-gate trading halt
- Unhandled trading-cycle error
- Bot start/stop lifecycle events


