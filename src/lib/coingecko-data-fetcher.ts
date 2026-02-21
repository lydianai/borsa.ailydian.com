/**
 * COINGECKO DATA FETCHER
 * Fallback data source when Binance is blocked
 * CoinGecko is more permissive and works from cloud providers
 */

interface CoinGeckoMarket {
  id: string;
  symbol: string;
  name: string;
  current_price: number;
  price_change_24h: number;
  price_change_percentage_24h: number;
  total_volume: number;
  high_24h: number;
  low_24h: number;
  last_updated: string;
}

export interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  lastUpdate: string;
}

export interface CoinGeckoDataResult {
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
  maxRetries = 2 // ✅ OPTIMIZED: Reduce from 3 to 2 for faster production experience
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000); // ✅ OPTIMIZED: 8s timeout (was 10s)

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
      console.error(`[CoinGecko API] HTTP ${response.status} on attempt ${attempt}/${maxRetries}:`, errorText.substring(0, 200));

      // Don't retry on 4xx errors (client errors)
      if (response.status >= 400 && response.status < 500) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
    } catch (error: any) {
      lastError = error;
      console.error(`[CoinGecko API] Attempt ${attempt}/${maxRetries} failed:`, error.message);

      // Don't retry on abort errors
      if (error.name === 'AbortError') {
        throw new Error('Request timeout (10s)');
      }

      // Wait before retrying (exponential backoff)
      if (attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
        console.log(`[CoinGecko API] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('Failed after max retries');
}

/**
 * Fetch crypto market data from CoinGecko API (free, no API key required)
 * This is a fallback when Binance API is blocked
 */
export async function fetchCoinGeckoData(): Promise<CoinGeckoDataResult> {
  try {
    console.log('[CoinGecko Data Fetcher] Starting fetch from CoinGecko API...');

    // Fetch top 250 coins by market cap (CoinGecko free tier)
    // ✅ OPTIMIZED: Faster timeout for better UX on production (10s → 8s)
    const response = await fetchWithRetry(
      "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1&sparkline=false",
      {
        headers: {
          "Accept": "application/json",
        },
      },
      2 // Reduce retries from 3 to 2 for faster fallback
    );

    console.log('[CoinGecko Data Fetcher] Successfully fetched market data');

    const marketsData: CoinGeckoMarket[] = await response.json();

    // Transform to our MarketData format
    // ✅ CRITICAL FIX: Add USDT suffix to match Binance format (BTCUSDT instead of BTC)
    const marketData: MarketData[] = marketsData
      .filter(coin => coin.current_price && coin.total_volume > 0)
      .map((coin) => ({
        symbol: `${coin.symbol.toUpperCase()}USDT`, // ✅ Add USDT suffix for compatibility
        price: coin.current_price || 0,
        change24h: coin.price_change_24h || 0,
        changePercent24h: coin.price_change_percentage_24h || 0,
        volume24h: coin.total_volume || 0,
        high24h: coin.high_24h || coin.current_price * 1.05,
        low24h: coin.low_24h || coin.current_price * 0.95,
        lastUpdate: coin.last_updated || new Date().toISOString(),
      }));

    // Sort by volume and top gainers
    const sortedByVolume = [...marketData].sort(
      (a, b) => b.volume24h - a.volume24h
    );
    const topGainers = [...marketData]
      .filter((m) => m.changePercent24h > 0)
      .sort((a, b) => b.changePercent24h - a.changePercent24h)
      .slice(0, 10);

    console.log(`[CoinGecko Data Fetcher] Processed ${marketData.length} markets`);

    return {
      success: true,
      data: {
        all: marketData,
        topVolume: sortedByVolume.slice(0, 50), // Top 50 by volume
        topGainers: topGainers,
        totalMarkets: marketData.length,
        lastUpdate: new Date().toISOString(),
      },
    };
  } catch (error) {
    console.error('[CoinGecko Data Fetcher Error]:', error);
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Failed to fetch CoinGecko data",
    };
  }
}
