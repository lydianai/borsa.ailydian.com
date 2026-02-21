/**
 * BYBIT FUTURES DATA FETCHER
 * Primary fallback for Binance Futures when geo-blocked (HTTP 451)
 * Bybit provides full USDT-M perpetual futures data
 */

interface BybitTicker {
  symbol: string;
  lastPrice: string;
  indexPrice: string;
  markPrice: string;
  prevPrice24h: string;
  price24hPcnt: string;
  highPrice24h: string;
  lowPrice24h: string;
  volume24h: string;
  turnover24h: string;
  fundingRate: string;
  nextFundingTime: string;
  openInterest: string;
  openInterestValue: string;
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
  // Futures-specific fields
  fundingRate?: number;
  openInterest?: number;
  markPrice?: number;
}

export interface BybitDataResult {
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
      console.error(`[Bybit API] HTTP ${response.status} on attempt ${attempt}/${maxRetries}:`, errorText.substring(0, 200));

      // Don't retry on 4xx errors (client errors)
      if (response.status >= 400 && response.status < 500) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      lastError = new Error(`HTTP ${response.status}: ${response.statusText}`);
    } catch (error: any) {
      lastError = error;
      console.error(`[Bybit API] Attempt ${attempt}/${maxRetries} failed:`, error.message);

      // Don't retry on abort errors
      if (error.name === 'AbortError') {
        throw new Error('Request timeout (10s)');
      }

      // Wait before retrying (exponential backoff)
      if (attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
        console.log(`[Bybit API] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError || new Error('Failed after max retries');
}

/**
 * Fetch Bybit USDT-M perpetual futures market data
 * This is a proper futures data source (unlike CoinGecko which only has spot)
 */
export async function fetchBybitFuturesData(): Promise<BybitDataResult> {
  try {
    console.log('[Bybit Futures Fetcher] Starting fetch from Bybit Futures API...');

    // Fetch all USDT perpetual futures (linear contracts)
    const response = await fetchWithRetry(
      "https://api.bybit.com/v5/market/tickers?category=linear",
      {
        headers: {
          "Accept": "application/json",
        },
      }
    );

    console.log('[Bybit Futures Fetcher] Successfully fetched futures data');

    const responseData = await response.json();

    if (responseData.retCode !== 0) {
      throw new Error(`Bybit API error: ${responseData.retMsg}`);
    }

    const tickersData: BybitTicker[] = responseData.result.list;

    // Filter only USDT pairs and transform to our MarketData format
    const marketData: MarketData[] = tickersData
      .filter(ticker => ticker.symbol.endsWith('USDT'))
      .map((ticker) => {
        const currentPrice = parseFloat(ticker.lastPrice) || 0;
        const prevPrice = parseFloat(ticker.prevPrice24h) || currentPrice;
        const priceChange = currentPrice - prevPrice;
        const priceChangePercent = parseFloat(ticker.price24hPcnt) * 100 || 0;

        return {
          symbol: ticker.symbol, // ✅ Keep full symbol format (BTCUSDT) to match Binance format
          price: currentPrice,
          change24h: priceChange,
          changePercent24h: priceChangePercent,
          volume24h: parseFloat(ticker.volume24h) || 0,
          high24h: parseFloat(ticker.highPrice24h) || currentPrice * 1.05,
          low24h: parseFloat(ticker.lowPrice24h) || currentPrice * 0.95,
          lastUpdate: new Date().toISOString(),
          // Futures-specific data
          fundingRate: parseFloat(ticker.fundingRate) || 0,
          openInterest: parseFloat(ticker.openInterest) || 0,
          markPrice: parseFloat(ticker.markPrice) || currentPrice,
        };
      })
      .filter((market) => market.volume24h > 0); // Filter by volume

    // Sort by volume and top gainers
    const sortedByVolume = [...marketData].sort(
      (a, b) => b.volume24h - a.volume24h
    );
    const topGainers = [...marketData]
      .filter((m) => m.changePercent24h > 0)
      .sort((a, b) => b.changePercent24h - a.changePercent24h)
      .slice(0, 10);

    console.log(`[Bybit Futures Fetcher] Processed ${marketData.length} futures markets`);

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
    console.error('[Bybit Futures Fetcher Error]:', error);
    return {
      success: false,
      error:
        error instanceof Error
          ? error.message
          : "Failed to fetch Bybit futures data",
    };
  }
}
