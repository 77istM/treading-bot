

## Trading Bot Quick Start

This repository contains an automated paper-trading bot with:

- Alpaca execution
- Multi-signal analysis (technical, sentiment, macro, earnings)
- Portfolio risk gates
- Post-trade reflection logging
- Streamlit dashboard
- Docker stack with Ollama integration

Use the sections below as copy-and-paste runbooks.

## 1) Prerequisites

Install:

- Python 3.11+
- Docker + Docker Compose (recommended path)
- Alpaca paper trading API credentials

Optional but recommended:

- News API key for sentiment/news quality

## 2) Fastest Start (Docker, Recommended)

Run everything (Ollama + bot + dashboard) with the commands below.

```bash
cd /workspaces/trading-bot

# Create local env file
cp .env.example .env 2>/dev/null || true

# If .env.example does not exist, create .env manually:
cat > .env << 'EOF'
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret

# Optional but recommended
NEWS_API_KEY=

# Ollama
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
SKIP_OLLAMA_HEALTHCHECK=false

# Runtime
LOOP_INTERVAL_SECONDS=300
DAILY_MAX_TRADES=1000

# Universes
TICKERS=SPY,QQQ,AAPL,MSFT,NVDA
CRYPTO_TICKERS=BTC/USD,ETH/USD

# Risk
MAX_POSITION_PCT=0.03
STOP_LOSS_PCT=0.03
TAKE_PROFIT_PCT=0.05
CRYPTO_MAX_POSITION_PCT=0.01
CRYPTO_STOP_LOSS_PCT=0.02
CRYPTO_TAKE_PROFIT_PCT=0.03
ALLOW_STOCK_SHORTS=true
ALLOW_CRYPTO_SHORTS=false

# Optional alerts
ALERTS_ENABLED=false
ALERT_WEBHOOK_URL=
ALERT_TIMEOUT_SECONDS=5
EOF

# Start Ollama first
docker compose up -d ollama

# Pull a model once (choose one)
docker exec -it ollama ollama pull llama3.2:3b
# docker exec -it ollama ollama pull qwen2.5:7b-instruct

# Start bot + dashboard
docker compose up --build bot dashboard
```

Open dashboard at:

- http://localhost:8501

Stop stack:

```bash
docker compose down
```

## 3) Local Start (Without Docker)

Use this path if you want to run directly on your host.

```bash
cd /workspaces/trading-bot

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Create .env in project root and fill credentials (same template as above)
cat > .env << 'EOF'
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret
NEWS_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
SKIP_OLLAMA_HEALTHCHECK=false
EOF

# Ensure Ollama is running locally and model exists
ollama serve
# In another terminal:
ollama pull llama3.2:3b

# Run bot
python main.py
```

In a second terminal, run dashboard:

```bash
cd /workspaces/trading-bot
source .venv/bin/activate
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

## 4) Tests

Run the full unit suite:

```bash
cd /workspaces/trading-bot
python -m unittest discover -s tests -v
```

## 5) Day-2 Operations

Useful commands:

```bash
# Follow container logs
docker compose logs -f bot
docker compose logs -f dashboard
docker compose logs -f ollama

# Check running containers
docker compose ps

# Rebuild after dependency/code changes
docker compose up --build bot dashboard

# Reset local DB (destructive)
rm -f trading_bot.db
```

## 6) Environment Variables Reference

Required:

- ALPACA_API_KEY
- ALPACA_SECRET

Common optional:

- NEWS_API_KEY
- OLLAMA_MODEL
- OLLAMA_BASE_URL
- SKIP_OLLAMA_HEALTHCHECK
- TICKERS
- CRYPTO_TICKERS
- LOOP_INTERVAL_SECONDS
- DAILY_MAX_TRADES

Risk controls:

- MAX_POSITION_PCT
- STOP_LOSS_PCT
- TAKE_PROFIT_PCT
- CRYPTO_MAX_POSITION_PCT
- CRYPTO_STOP_LOSS_PCT
- CRYPTO_TAKE_PROFIT_PCT
- ALLOW_STOCK_SHORTS
- ALLOW_CRYPTO_SHORTS

Secrets loading precedence:

1. Environment variables
2. JSON vault file via SECRETS_VAULT_FILE
3. Docker secret files in SECRETS_DIR (default /run/secrets)

## 7) Troubleshooting

Ollama health check fails:

```bash
docker compose logs ollama
docker exec -it ollama ollama list
```

Missing Alpaca credentials error:

- Ensure ALPACA_API_KEY and ALPACA_SECRET are set in .env
- Restart containers after updating .env

TA-Lib install issues on local host:

- Prefer Docker path (TA-Lib is already handled in Docker image)

Dashboard has no data:

- Let bot complete at least one cycle
- Confirm database file exists and bot logs show trade/analysis activity

## 8) Safety Notes

- This project is configured for paper trading and should be validated extensively before any live deployment.
- Start with long-only settings and conservative position sizing.
- Keep ALLOW_CRYPTO_SHORTS=false unless you intentionally want short exposure.


