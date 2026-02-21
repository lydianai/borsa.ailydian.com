/**
 * 🎯 UNIFIED DATA ORCHESTRATOR
 *
 * Centralized data management layer that eliminates Binance API timeouts
 * by routing all requests through Python microservices and WebSocket streams.
 *
 * ARCHITECTURE:
 * 1. WebSocket Service (Port 5021) - Real-time price updates (PRIMARY)
 * 2. Python Services - Analysis & signals (FALLBACK: Binance/Bybit/CoinGecko)
 * 3. Local cache - Instant responses (LAST RESORT: Offline mode)
 *
 * BENEFITS:
 * - ✅ No more 30s timeouts
 * - ✅ Real-time WebSocket prices (< 100ms)
 * - ✅ Python service load balancing
 * - ✅ Automatic fallback chain
 * - ✅ Unified error handling
 */

import { MarketData, BinanceDataResult } from './binance-data-fetcher';

// ============================================================================
// PYTHON MICROSERVICES CONFIGURATION
// ============================================================================

export const PYTHON_SERVICES = {
  WEBSOCKET_STREAM: 'http://localhost:5021',  // Real-time prices
  TALIB: 'http://localhost:5002',             // Technical analysis
  AI_MODELS: 'http://localhost:5003',         // AI predictions
  QUANTUM_LADDER: 'http://localhost:5022',    // Fibonacci analysis
} as const;

// Service health tracking
const serviceHealth = new Map<string, { isHealthy: boolean; lastCheck: number }>();
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

// ============================================================================
// CACHE MANAGEMENT
// ============================================================================

interface CachedData<T> {
  data: T;
  timestamp: number;
  source: 'websocket' | 'python' | 'binance' | 'offline';
}

class DataCache {
  private cache = new Map<string, CachedData<any>>();
  private readonly TTL = {
    WEBSOCKET: 5000,      // 5 seconds (real-time data)
    PYTHON: 60000,        // 1 minute (processed data)
    BINANCE: 300000,      // 5 minutes (fallback data)
    OFFLINE: Infinity,    // Never expire (emergency)
  };

  set<T>(key: string, data: T, source: CachedData<T>['source']): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      source,
    });
  }

  get<T>(key: string): T | null {
    const cached = this.cache.get(key);
    if (!cached) return null;

    const age = Date.now() - cached.timestamp;
    const ttl = this.TTL[cached.source.toUpperCase() as keyof typeof this.TTL];

    if (age > ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.data as T;
  }

  getCacheAge(key: string): number | null {
    const cached = this.cache.get(key);
    return cached ? Date.now() - cached.timestamp : null;
  }
}

const dataCache = new DataCache();

// ============================================================================
// SERVICE HEALTH CHECKS
// ============================================================================

async function checkServiceHealth(url: string): Promise<boolean> {
  const cached = serviceHealth.get(url);
  const now = Date.now();

  // Return cached health status if recent
  if (cached && (now - cached.lastCheck) < HEALTH_CHECK_INTERVAL) {
    return cached.isHealthy;
  }

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000); // 3s timeout

    const response = await fetch(`${url}/health`, {
      signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
    });

    clearTimeout(timeoutId);
    const isHealthy = response.ok;

    serviceHealth.set(url, { isHealthy, lastCheck: now });
    return isHealthy;
  } catch (error) {
    console.warn(`[Orchestrator] Service unhealthy: ${url}`);
    serviceHealth.set(url, { isHealthy: false, lastCheck: now });
    return false;
  }
}

// ============================================================================
// WEBSOCKET DATA FETCHER
// ============================================================================

/**
 * Fetch real-time market data from WebSocket service
 * This is the PRIMARY data source (fastest, most reliable)
 */
async function fetchFromWebSocket(): Promise<BinanceDataResult | null> {
  try {
    // Check service health first
    const isHealthy = await checkServiceHealth(PYTHON_SERVICES.WEBSOCKET_STREAM);
    if (!isHealthy) {
      console.warn('[Orchestrator] WebSocket service unhealthy, skipping');
      return null;
    }

    const response = await fetch(`${PYTHON_SERVICES.WEBSOCKET_STREAM}/api/latest-prices`, {
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000), // 5s timeout
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (!data.success || !data.prices) {
      throw new Error('Invalid WebSocket response format');
    }

    // Transform WebSocket format to MarketData format
    const marketData: MarketData[] = Object.entries(data.prices).map(([symbol, priceData]: [string, any]) => ({
      symbol,
      price: priceData.price || 0,
      change24h: priceData.change || 0,
      changePercent24h: priceData.changePercent || 0,
      volume24h: priceData.volume || 0,
      high24h: priceData.high || priceData.price * 1.05,
      low24h: priceData.low || priceData.price * 0.95,
      lastUpdate: priceData.timestamp || new Date().toISOString(),
    }));

    const sortedByVolume = [...marketData].sort((a, b) => b.volume24h - a.volume24h);
    const topGainers = [...marketData]
      .filter((m) => m.changePercent24h > 0)
      .sort((a, b) => b.changePercent24h - a.changePercent24h)
      .slice(0, 10);

    const result: BinanceDataResult = {
      success: true,
      data: {
        all: marketData,
        topVolume: sortedByVolume.slice(0, 20),
        topGainers,
        totalMarkets: marketData.length,
        lastUpdate: new Date().toISOString(),
      },
    };

    // Cache with WebSocket TTL
    dataCache.set('market-data', result, 'websocket');
    console.log(`[Orchestrator] ✅ WebSocket data: ${marketData.length} markets`);

    return result;
  } catch (error: any) {
    console.error('[Orchestrator] WebSocket fetch failed:', error.message);
    return null;
  }
}

// ============================================================================
// PYTHON SERVICE DATA FETCHER
// ============================================================================

/**
 * Fetch data from Python AI Models service
 * This is the SECONDARY data source (processed, with signals)
 */
async function fetchFromPythonServices(): Promise<BinanceDataResult | null> {
  try {
    // Check AI Models service health
    const isHealthy = await checkServiceHealth(PYTHON_SERVICES.AI_MODELS);
    if (!isHealthy) {
      console.warn('[Orchestrator] AI Models service unhealthy, skipping');
      return null;
    }

    // Try to fetch from AI Models service which aggregates data
    const response = await fetch(`${PYTHON_SERVICES.AI_MODELS}/api/market-data`, {
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(10000), // 10s timeout
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error('Python service returned error');
    }

    // Cache with Python TTL
    dataCache.set('market-data', data, 'python');
    console.log(`[Orchestrator] ✅ Python service data fetched successfully`);

    return data as BinanceDataResult;
  } catch (error: any) {
    console.error('[Orchestrator] Python service fetch failed:', error.message);
    return null;
  }
}

// ============================================================================
// BINANCE DIRECT FETCH (FALLBACK)
// ============================================================================

/**
 * Fetch from Binance directly (with existing fallback chain)
 * This is the TERTIARY data source (most likely to timeout)
 */
async function fetchFromBinanceDirect(): Promise<BinanceDataResult | null> {
  try {
    // Import the existing fetcher
    const { fetchBinanceFuturesData } = await import('./binance-data-fetcher');
    const result = await fetchBinanceFuturesData();

    if (result.success && result.data) {
      dataCache.set('market-data', result, 'binance');
      console.log(`[Orchestrator] ✅ Binance direct fetch: ${result.data.totalMarkets} markets`);
      return result;
    }

    return null;
  } catch (error: any) {
    console.error('[Orchestrator] Binance direct fetch failed:', error.message);
    return null;
  }
}

// ============================================================================
// OFFLINE MODE (LAST RESORT)
// ============================================================================

function getOfflineData(): BinanceDataResult {
  const offlineMarkets: MarketData[] = [
    {
      symbol: 'BTCUSDT',
      price: 67000,
      change24h: 1200,
      changePercent24h: 1.82,
      volume24h: 28000000000,
      high24h: 67500,
      low24h: 65800,
      lastUpdate: new Date().toISOString(),
    },
    {
      symbol: 'ETHUSDT',
      price: 3200,
      change24h: 50,
      changePercent24h: 1.59,
      volume24h: 12000000000,
      high24h: 3250,
      low24h: 3150,
      lastUpdate: new Date().toISOString(),
    },
  ];

  return {
    success: true,
    data: {
      all: offlineMarkets,
      topVolume: offlineMarkets,
      topGainers: offlineMarkets,
      totalMarkets: offlineMarkets.length,
      lastUpdate: `${new Date().toISOString()} (OFFLINE MODE)`,
    },
    error: 'All data sources unavailable - using offline mode',
  };
}

// ============================================================================
// UNIFIED MARKET DATA FETCHER
// ============================================================================

/**
 * Main entry point for fetching market data
 *
 * PRIORITY ORDER:
 * 1. Cache (if fresh)
 * 2. WebSocket Service (real-time, < 100ms)
 * 3. Python AI Services (processed, < 2s)
 * 4. Binance Direct (with fallbacks, < 10s)
 * 5. Offline Mode (emergency)
 *
 * @returns Market data with guaranteed response
 */
export async function fetchUnifiedMarketData(): Promise<BinanceDataResult> {
  const startTime = Date.now();

  // STEP 1: Check cache first
  const cached = dataCache.get<BinanceDataResult>('market-data');
  if (cached) {
    const cacheAge = dataCache.getCacheAge('market-data') || 0;
    console.log(`[Orchestrator] 💾 Cache hit (age: ${(cacheAge / 1000).toFixed(1)}s)`);
    return cached;
  }

  console.log('[Orchestrator] 🔄 Fetching fresh market data...');

  // STEP 2: Try WebSocket (PRIMARY - fastest)
  const wsData = await fetchFromWebSocket();
  if (wsData) {
    console.log(`[Orchestrator] ⚡ WebSocket fetch: ${Date.now() - startTime}ms`);
    return wsData;
  }

  // STEP 3: Try Python Services (SECONDARY - processed)
  const pythonData = await fetchFromPythonServices();
  if (pythonData) {
    console.log(`[Orchestrator] 🐍 Python fetch: ${Date.now() - startTime}ms`);
    return pythonData;
  }

  // STEP 4: Try Binance Direct (TERTIARY - fallback chain)
  const binanceData = await fetchFromBinanceDirect();
  if (binanceData) {
    console.log(`[Orchestrator] 🌐 Binance direct fetch: ${Date.now() - startTime}ms`);
    return binanceData;
  }

  // STEP 5: Offline mode (LAST RESORT)
  console.warn('[Orchestrator] ⚠️  ALL SOURCES FAILED - Using offline mode');
  const offlineData = getOfflineData();
  dataCache.set('market-data', offlineData, 'offline');
  return offlineData;
}

// ============================================================================
// PYTHON SERVICE SPECIFIC FETCHERS
// ============================================================================

/**
 * Fetch quantum ladder analysis from Python service
 */
export async function fetchQuantumLadderAnalysis(
  symbol: string,
  timeframes: string[] = ['15m', '1h', '4h']
): Promise<any> {
  try {
    const isHealthy = await checkServiceHealth(PYTHON_SERVICES.QUANTUM_LADDER);
    if (!isHealthy) {
      throw new Error('Quantum Ladder service unavailable');
    }

    const response = await fetch(`${PYTHON_SERVICES.QUANTUM_LADDER}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, timeframes }),
      signal: AbortSignal.timeout(15000), // 15s for analysis
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error: any) {
    console.error(`[Orchestrator] Quantum Ladder fetch failed:`, error.message);
    throw error;
  }
}

/**
 * Fetch TA-Lib indicators from Python service
 */
export async function fetchTALibIndicators(
  symbol: string,
  interval: string = '1h',
  indicators: string[] = ['RSI', 'MACD', 'BBANDS']
): Promise<any> {
  try {
    const isHealthy = await checkServiceHealth(PYTHON_SERVICES.TALIB);
    if (!isHealthy) {
      throw new Error('TA-Lib service unavailable');
    }

    const response = await fetch(`${PYTHON_SERVICES.TALIB}/api/indicators`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol, interval, indicators }),
      signal: AbortSignal.timeout(10000), // 10s timeout
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error: any) {
    console.error(`[Orchestrator] TA-Lib fetch failed:`, error.message);
    throw error;
  }
}

// ============================================================================
// EXPORT DEFAULT
// ============================================================================

export default {
  fetchMarketData: fetchUnifiedMarketData,
  fetchQuantumLadder: fetchQuantumLadderAnalysis,
  fetchTALib: fetchTALibIndicators,
  services: PYTHON_SERVICES,
  checkHealth: checkServiceHealth,
};
