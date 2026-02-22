<div align="center">

# LyTrade Scanner

### Open-Source Crypto Trading Signal Platform

**617 Markets | 13+ Strategies | Real-Time Binance WebSocket | Self-Hosted**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

[Features](#features) | [Quick Start](#quick-start) | [Screenshots](#screenshots) | [Strategies](#trading-strategies) | [Contributing](#contributing)

</div>

---

LyTrade Scanner is a self-hosted cryptocurrency trading signal platform that monitors **617 Binance USDT Perpetual Futures** markets in real-time. It runs 13+ independent trading strategies and delivers actionable signals with entry/exit levels, stop-loss, and take-profit targets through an interactive dashboard.

**No API keys required for basic use** -- Binance public endpoints are free. Just clone, install, and start scanning.

## Features

- **617 Market Coverage** -- Monitors all Binance USDT Perpetual Futures simultaneously via WebSocket
- **13+ Trading Strategies** -- Conservative Buy, Breakout-Retest, Momentum, RSI Divergence, Volume Surge, and more
- **Real-Time Signals** -- Sub-100ms latency with direct Binance WebSocket feed
- **Multi-Timeframe Analysis** -- Processes signals across multiple timeframes for higher confidence
- **Interactive Dashboard** -- ApexCharts candlestick charts + Recharts for portfolio visualization
- **AI Market Commentary** -- Works with any OpenAI-compatible API (Groq free tier, OpenAI, local Ollama)
- **Whale Tracking** -- Large transaction monitoring via on-chain data
- **News Aggregation** -- Crypto news from multiple sources with sentiment analysis
- **Traditional Markets** -- Forex, commodities, and indices alongside crypto
- **Push Notifications** -- Telegram bot + browser push notifications for signals
- **PWA Support** -- Install as a desktop/mobile app
- **Dark Theme** -- Professional trading interface with dark mode

## Screenshots

<div align="center">
  <img src="docs/screenshots/crypto-trading-grid.png" alt="LyTrade Scanner Dashboard" width="800">
  <br><em>Real-time trading grid with 24+ cryptocurrencies, live prices, volume, and trend indicators</em>
</div>

## Quick Start

### Prerequisites

- **Node.js** 20+ and **pnpm** 8+
- **PostgreSQL** (optional -- works without it using file-based storage)
- **Redis** (optional -- for caching in production)

### 1. Clone and Install

```bash
git clone https://github.com/AiLydian/lytrade-scanner.git
cd lytrade-scanner

pnpm install
```

### 2. Configure Environment

```bash
cp .env.example .env.local
```

Edit `.env.local` with your settings. **Minimum required:**

```env
# App URL (keep as-is for local development)
NEXT_PUBLIC_APP_URL=http://localhost:3000

# AI Provider (optional -- for market commentary feature)
# Option A: Groq (free) - https://console.groq.com
AI_API_URL=https://api.groq.com/openai/v1/chat/completions
AI_API_KEY=your_groq_key
AI_MODEL=llama-3.3-70b-versatile

# Option B: Local Ollama (free, private)
# AI_API_URL=http://localhost:11434/v1/chat/completions
# AI_API_KEY=ollama
# AI_MODEL=llama3.2
```

> **Note:** Binance market data works without any API key. AI features are optional.

### 3. Run

```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) and start scanning.

### Optional: Database Setup

For signal history and user accounts:

```bash
# Set DATABASE_URL in .env.local
# Example: postgresql://user:password@localhost:5432/lytrade

npx prisma generate
npx prisma db push
```

## Trading Strategies

| # | Strategy | Signal Type | Market Condition |
|---|----------|------------|-----------------|
| 1 | Conservative Buy | Long | Oversold with strong support |
| 2 | Breakout-Retest | Long | Resistance broken and retested |
| 3 | Momentum | Long/Short | Strong directional movement |
| 4 | Downtrend Reversal | Long | Exhausted sellers, accumulation zone |
| 5 | Volume Surge | Long/Short | Abnormal volume spike detection |
| 6 | RSI Divergence | Long/Short | Price/momentum divergence |
| 7 | Moving Average Cross | Long/Short | MA crossover systems (EMA/SMA) |
| 8 | Bollinger Squeeze | Long/Short | Volatility contraction breakout |
| 9 | MACD Signal | Long/Short | Histogram crossovers |
| 10 | Fibonacci Retracement | Long | Golden ratio support levels |
| 11 | Order Book Imbalance | Long/Short | Liquidity analysis |
| 12 | Funding Rate Extreme | Contrarian | Extreme sentiment reversal |
| 13 | Open Interest Spike | Long/Short | Unusual futures positioning |

## Architecture

```mermaid
graph TD
    subgraph "Data Ingestion"
        A[Binance WebSocket API] --> B[617 USDT Perpetual Markets]
        B --> C[Price Stream]
        B --> D[Volume Stream]
        B --> E[Order Book]
    end
    subgraph "Analysis Engine"
        C --> F[13+ Strategy Processor]
        D --> F
        E --> F
        F --> G[Signal Generator]
        G --> H[Risk Calculator]
        H --> I[Entry/Exit/SL/TP Levels]
    end
    subgraph "Presentation Layer"
        I --> J[Trading Dashboard]
        J --> K[ApexCharts Candlestick]
        J --> L[Signal Alert Cards]
        J --> M[Portfolio Tracker]
        J --> N[Top Gainers/Volume]
    end
    subgraph "Persistence"
        I --> O[PostgreSQL - Signal History]
        I --> P[Redis - Real-time Cache]
    end
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | [Next.js 16](https://nextjs.org) + [TypeScript 5](https://www.typescriptlang.org) |
| Styling | [Tailwind CSS](https://tailwindcss.com) |
| Charting | [ApexCharts](https://apexcharts.com) + [Recharts](https://recharts.org) |
| Market Data | [Binance WebSocket API](https://binance-docs.github.io/apidocs/) (real-time) |
| Database | [PostgreSQL](https://www.postgresql.org) + [Prisma ORM](https://www.prisma.io) |
| Cache | [Redis](https://redis.io) / [Upstash](https://upstash.com) |
| State | [Zustand](https://zustand-demo.pmnd.rs) |
| Auth | [NextAuth.js](https://next-auth.js.org) |
| Icons | [Lucide React](https://lucide.dev) |
| Validation | [Zod](https://zod.dev) |

## Project Structure

```
src/
  app/                  # Next.js App Router pages and API routes
    (auth)/             # Login, register, forgot password
    api/                # REST API endpoints
    market-scanner/     # Scanner dashboard
    market-commentary/  # AI-powered market analysis
    traditional-markets/# Forex, commodities, indices
    haberler/           # Crypto news aggregator
    pricing/            # Subscription plans
  components/           # React components
    settings/           # User settings panels
  lib/                  # Utility libraries
    strategies/         # Trading strategy implementations
    adapters/           # Data source adapters
    email/              # Email notification service
    push/               # Push notification service
    stripe/             # Payment integration
  types/                # TypeScript type definitions
prisma/                 # Database schema
public/                 # Static assets
```

## Environment Variables

See [`.env.example`](.env.example) for all available configuration options. Key categories:

| Category | Required | Description |
|----------|----------|-------------|
| Core | Yes | `NEXT_PUBLIC_APP_URL` |
| AI Services | No | Any OpenAI-compatible API (Groq, OpenAI, Ollama) |
| Market Data | No | CoinGlass, Alpha Vantage, MetalpriceAPI |
| Notifications | No | Telegram bot, VAPID keys for push |
| Database | No | PostgreSQL connection string |
| Cache | No | Redis/Upstash for production caching |
| Monitoring | No | Sentry DSN for error tracking |

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/lytrade-scanner.git
cd lytrade-scanner
pnpm install
pnpm dev

# Make changes, then submit a PR
```

## Security

If you discover a security vulnerability, please report it responsibly via [GitHub Security Advisories](https://github.com/AiLydian/lytrade-scanner/security/advisories/new). See [SECURITY.md](SECURITY.md) for details.

## Disclaimer

This platform is for **informational and educational purposes only**. Cryptocurrency trading carries significant financial risk. Never invest more than you can afford to lose. Past performance of trading signals does not guarantee future results. **This is not financial advice.**

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

If this project helps your trading workflow, consider giving it a star!

</div>
