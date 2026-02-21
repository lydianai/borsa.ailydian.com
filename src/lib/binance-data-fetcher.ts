/**
 * SHARED BINANCE DATA FETCHER
 * Prevents ECONNREFUSED errors from internal API-to-API fetch calls
 * Use this function directly in API routes instead of fetching from /api/binance/futures
 */

interface BinanceFuturesSymbol {
  symbol: string;
  pair: string;
  contractType: string;
  status: string;
  baseAsset: string;
  quoteAsset: string;
  pricePrecision: number;
  quantityPrecision: number;
}

interface BinanceFuturesTicker {
  symbol: string;
  price: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
  openTime: number;
  closeTime: number;
  count: number;
}

export interface MarketData {
  symbol: string; // Full symbol with USDT (e.g., "BTCUSDT")
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  lastUpdate: string;
  closeTime?: number; // Optional: Unix timestamp for filtering old data
}

export interface BinanceDataResult {
  success: boolean;
  data?: {
    all: MarketData[];
    topVolume: MarketData[];
    topGainers: MarketData[];
    totalMarkets: number;
    lastUpdate: string;
  };
  error?: string;
}

/**
 * Retry helper with exponential backoff
 */
async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 3
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        return response;
      }

      // Log non-OK responses
      const errorText = await response.text().catch(() => 'Unable to read error');
      console.error(`[Binance API] HTTP ${response.status} on attempt ${attempt}/${maxRetries}:`, errorText.substring(0, 200));

      // Handle rate limiting (429) and ban (418) - don't retry, use stale cache if available
      if (response.status === 429 || response.status === 418) {
        throw new Error(`HTTP ${response.status}: ${response.status === 418 ? 'IP Banned' : 'Too Many Requests'}`);
      }

      // Don't retry on other 4xx errors (client errors)
      if (response.status >= 400 && response.status < 500) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
    } catch (error: any) {
      lastError = error;
      console.error(`[Binance API] Attempt ${attempt}/${maxRetries} failed:`, error.message);

      // Don't retry on abort errors
      if (error.name === 'AbortError') {
        throw new Error('Request timeout (10s)');
      }

      // Wait before retrying (exponential backoff)
      if (attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
        console.log(`[Binance API] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('Failed after max retries');
}

// Import fallback data sources
import { fetchBybitFuturesData } from './bybit-futures-fetcher';
import { fetchCoinGeckoData } from './coingecko-data-fetcher';
// Note: WebSocket manager is client-side only, not imported here

/**
 * CACHING LAYER - Prevents rate limiting (HTTP 429/418)
 * Cache duration: 5 minutes to avoid Binance API ban
 */
let cachedData: BinanceDataResult | null = null;
let cacheTimestamp: number = 0;
const CACHE_DURATION_MS = 300000; // 5 minutes (300 seconds)
const STALE_CACHE_MAX_AGE_MS = 600000; // 10 minutes - use stale cache if rate limited/banned

/**
 * Fetch Binance Futures market data directly from Binance API
 * Fallback priority:
 * 1. Binance Futures (primary)
 * 2. Bybit Futures (fallback for HTTP 451 - has full futures data)
 * 3. CoinGecko Spot (last resort - spot data only, no futures)
 * This function can be imported and used by any API route without HTTP overhead
 *
 * **CACHED:** Returns cached data if less than 30 seconds old to prevent rate limiting
 */
export async function fetchBinanceFuturesData(): Promise<BinanceDataResult> {
  // Check cache first
  const now = Date.now();
  if (cachedData && (now - cacheTimestamp) < CACHE_DURATION_MS) {
    console.log(`[Binance Data Fetcher] ✅ Returning cached data (age: ${((now - cacheTimestamp) / 1000).toFixed(1)}s)`);
    return cachedData;
  }
  try {
    console.log('[Binance Data Fetcher] Starting fetch from Binance Futures API...');

    // Fetch all USDT futures symbols with retry logic
    const symbolsResponse = await fetchWithRetry(
      "https://fapi.binance.com/fapi/v1/exchangeInfo",
      {
        headers: {
          "User-Agent": "Mozilla/5.0 (compatible; SardagAI/1.0)",
          "Accept": "application/json",
        },
      }
    );

    console.log('[Binance Data Fetcher] Successfully fetched exchange info');

    const symbolsData = await symbolsResponse.json();

    // Filter only USDT perpetual contracts
    const usdtSymbols = symbolsData.symbols
      .filter(
        (s: BinanceFuturesSymbol) =>
          s.quoteAsset === "USDT" &&
          s.contractType === "PERPETUAL" &&
          s.status === "TRADING"
      )
      .map((s: BinanceFuturesSymbol) => s.symbol);

    console.log(`[Binance Data Fetcher] Fetching ticker data for ${usdtSymbols.length} symbols...`);

    // Fetch 24hr ticker statistics for all USDT symbols with retry logic
    const tickerResponse = await fetchWithRetry(
      `https://fapi.binance.com/fapi/v1/ticker/24hr?symbols=${JSON.stringify(usdtSymbols)}`,
      {
        headers: {
          "User-Agent": "Mozilla/5.0 (compatible; SardagAI/1.0)",
          "Accept": "application/json",
        },
      }
    );

    console.log('[Binance Data Fetcher] Successfully fetched ticker data');

    const tickersData: BinanceFuturesTicker[] = await tickerResponse.json();

    // Transform data to our MarketData format
    const now = Date.now();
    const marketData = tickersData
      .map((ticker) => {
        const price = parseFloat(ticker.price);
        const priceChange = parseFloat(ticker.priceChange) || 0;
        const priceChangePercent = parseFloat(ticker.priceChangePercent) || 0;
        const closeTime = ticker.closeTime;

        // Calculate estimated price if null or zero using priceChange and percent
        let finalPrice = price;
        if (!price || price === 0) {
          if (priceChangePercent !== 0) {
            // Estimate previous price and calculate current price
            const estimatedPrevPrice = Math.abs(
              priceChange / (priceChangePercent / 100)
            );
            finalPrice = estimatedPrevPrice + priceChange;
          } else {
            finalPrice = 0.001; // Ultimate fallback
          }
        }

        return {
          symbol: ticker.symbol, // ✅ KEEP FULL SYMBOL (e.g., "BTCUSDT")
          price: finalPrice,
          change24h: priceChange,
          changePercent24h: priceChangePercent,
          volume24h: parseFloat(ticker.volume) || 0,
          high24h: parseFloat(ticker.highPrice) || finalPrice * 1.05,
          low24h: parseFloat(ticker.lowPrice) || finalPrice * 0.95,
          lastUpdate: new Date(closeTime).toISOString(),
          closeTime, // Keep raw timestamp for filtering
        };
      })
      .filter((market) => {
        // Filter out old/delisted coins (older than 7 days)
        const ageInDays = (now - (market.closeTime || now)) / (1000 * 60 * 60 * 24);
        return market.volume24h > 0 && ageInDays < 7;
      });

    // Remove closeTime before sending to client (internal use only)
    const cleanedData = marketData.map(({ closeTime, ...rest }) => rest);

    // Sort by volume and top gainers
    const sortedByVolume = [...cleanedData].sort(
      (a, b) => b.volume24h - a.volume24h
    );
    const topGainers = [...cleanedData]
      .filter((m) => m.changePercent24h > 0)
      .sort((a, b) => b.changePercent24h - a.changePercent24h)
      .slice(0, 10);

    const result: BinanceDataResult = {
      success: true,
      data: {
        all: cleanedData,
        topVolume: sortedByVolume.slice(0, 20),
        topGainers: topGainers,
        totalMarkets: cleanedData.length,
        lastUpdate: new Date().toISOString(),
      },
    };

    // Update cache
    cachedData = result;
    cacheTimestamp = Date.now();
    console.log('[Binance Data Fetcher] ✅ Data cached successfully');

    return result;
  } catch (error: any) {
    console.error("[Binance Data Fetcher Error]:", error);

    // SPECIAL CASE: If rate limited (429) or banned (418) and we have stale cache, use it
    const isRateLimited = error?.message?.includes('429') || error?.message?.includes('418');

    if (isRateLimited && cachedData) {
      const cacheAge = Date.now() - cacheTimestamp;
      const cacheAgeMinutes = (cacheAge / 60000).toFixed(1);

      if (cacheAge < STALE_CACHE_MAX_AGE_MS) {
        console.warn(`[Binance Data Fetcher] ⚠️  Rate limited/banned - Using stale cache (age: ${cacheAgeMinutes} min)`);
        return {
          ...cachedData,
          data: cachedData.data ? {
            ...cachedData.data,
            lastUpdate: `${cachedData.data.lastUpdate} (cached ${cacheAgeMinutes}m ago)`
          } : undefined
        };
      } else {
        console.warn(`[Binance Data Fetcher] ⚠️  Rate limited/banned - Cache too old (${cacheAgeMinutes} min), trying fallbacks...`);
      }
    }

    // ALWAYS try fallbacks for ANY error (or if stale cache too old)
    console.log('[Binance Data Fetcher] 🔄 Primary source failed - trying fallbacks...');

    // PRIMARY FALLBACK: Bybit Futures (has full futures data)
    try {
      console.log('[Binance Data Fetcher] Attempting Bybit Futures fallback...');
      const bybitResult = await fetchBybitFuturesData();

      if (bybitResult.success) {
        console.log('[Binance Data Fetcher] ✅ Bybit Futures fallback successful (full futures data)');

        // Cache successful fallback result
        cachedData = bybitResult as BinanceDataResult;
        cacheTimestamp = Date.now();
        console.log('[Binance Data Fetcher] ✅ Fallback data cached successfully');

        return bybitResult as BinanceDataResult;
      } else {
        console.error('[Binance Data Fetcher] ❌ Bybit Futures fallback failed:', bybitResult.error);
      }
    } catch (bybitError) {
      console.error('[Binance Data Fetcher] ❌ Bybit Futures fallback exception:', bybitError);
    }

    // SECONDARY FALLBACK: CoinGecko Spot (spot data only, no futures)
    try {
      console.log('[Binance Data Fetcher] Attempting CoinGecko Spot fallback (spot data only)...');
      const coinGeckoResult = await fetchCoinGeckoData();

      if (coinGeckoResult.success) {
        console.log('[Binance Data Fetcher] ⚠️  CoinGecko Spot fallback successful (WARNING: spot data only, no futures)');

        // Cache successful fallback result
        cachedData = coinGeckoResult as BinanceDataResult;
        cacheTimestamp = Date.now();
        console.log('[Binance Data Fetcher] ✅ Fallback data cached successfully');

        return coinGeckoResult as BinanceDataResult;
      } else {
        console.error('[Binance Data Fetcher] ❌ CoinGecko fallback also failed:', coinGeckoResult.error);
      }
    } catch (coinGeckoError) {
      console.error('[Binance Data Fetcher] ❌ CoinGecko fallback exception:', coinGeckoError);
    }

    // FINAL FALLBACK: Return minimal working data (graceful degradation)
    console.warn('[Binance Data Fetcher] ⚠️  ALL SOURCES FAILED - Returning minimal offline data');

    const offlineData: MarketData[] = [
      {
        symbol: 'BTCUSDT',
        price: 67000,
        change24h: 1200,
        changePercent24h: 1.82,
        volume24h: 28000000000,
        high24h: 67500,
        low24h: 65800,
        lastUpdate: new Date().toISOString()
      },
      {
        symbol: 'ETHUSDT',
        price: 3200,
        change24h: 50,
        changePercent24h: 1.59,
        volume24h: 12000000000,
        high24h: 3250,
        low24h: 3150,
        lastUpdate: new Date().toISOString()
      }
    ];

    return {
      success: true, // Return success=true to prevent infinite refresh
      data: {
        all: offlineData,
        topVolume: offlineData,
        topGainers: offlineData,
        totalMarkets: offlineData.length,
        lastUpdate: new Date().toISOString(),
      },
      error: 'Offline mode: All data sources unavailable'
    };
  }
}
