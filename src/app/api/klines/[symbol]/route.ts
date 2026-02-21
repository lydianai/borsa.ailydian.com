import { NextRequest, NextResponse } from 'next/server';

/**
 * Binance Kline/Candlestick Data API
 * Fetches historical price data for charting
 *
 * Features:
 * - In-memory caching (30 seconds TTL)
 * - Retry mechanism with exponential backoff
 * - Rate limit handling (HTTP 418)
 * - Fallback to cached data on errors
 */

interface KlineData {
  time: number; // Unix timestamp in seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface CacheEntry {
  data: KlineData[];
  timestamp: number;
}

// In-memory cache (30 seconds TTL)
const cache = new Map<string, CacheEntry>();
const KLINES_CACHE_TTL = 30 * 1000; // 30 seconds

/**
 * Retry fetch with exponential backoff
 */
async function fetchWithRetry(url: string, maxRetries = 3): Promise<Response> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });

      // If rate limited (418), wait longer before retry
      if (response.status === 418 && i < maxRetries - 1) {
        const waitTime = Math.pow(2, i + 1) * 1000; // 2s, 4s, 8s
        console.log(`[Klines API] Rate limited, waiting ${waitTime}ms before retry ${i + 1}/${maxRetries}`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }

      return response;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      const waitTime = Math.pow(2, i) * 1000; // 1s, 2s, 4s
      console.log(`[Klines API] Retry ${i + 1}/${maxRetries} after ${waitTime}ms`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
  }
  throw new Error('Max retries exceeded');
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;
    const { searchParams } = new URL(request.url);

    const interval = searchParams.get('interval') || '1h';
    const limit = parseInt(searchParams.get('limit') || '500');

    const cacheKey = `${symbol}-${interval}-${limit}`;

    // Check cache first
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < KLINES_CACHE_TTL) {
      console.log(`[Klines API] Cache HIT for ${cacheKey}`);
      return NextResponse.json({
        success: true,
        data: {
          symbol,
          interval,
          candles: cached.data,
          count: cached.data.length,
          cached: true
        }
      });
    }

    console.log(`[Klines API] Fetching ${symbol} data for ${interval} interval, limit: ${limit}`);

    // Try Binance first
    const binanceUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    let response: Response | null = null;
    let source = 'binance';

    try {
      response = await fetchWithRetry(binanceUrl);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Klines API] Binance API failed: ${response.status} - ${errorText}`);

        // Fallback to Bybit on rate limit or error
        console.log(`[Klines API] Falling back to Bybit...`);
        const bybitUrl = `https://api.bybit.com/v5/market/kline?category=linear&symbol=${symbol}&interval=${interval}&limit=${limit}`;

        try {
          response = await fetchWithRetry(bybitUrl);
          source = 'bybit';

          if (!response.ok) {
            const bybitError = await response.text();
            console.error(`[Klines API] Bybit API also failed: ${response.status} - ${bybitError}`);

            // If we have cached data, return it even if expired
            if (cached) {
              console.log(`[Klines API] Returning stale cache due to both APIs failing`);
              return NextResponse.json({
                success: true,
                data: {
                  symbol,
                  interval,
                  candles: cached.data,
                  count: cached.data.length,
                  cached: true,
                  warning: 'Using cached data due to API unavailability'
                }
              });
            }

            throw new Error(`Both Binance and Bybit APIs failed`);
          }
        } catch (bybitError) {
          console.error(`[Klines API] Bybit fallback error:`, bybitError);

          // If we have cached data, return it even if expired
          if (cached) {
            console.log(`[Klines API] Returning stale cache due to fallback error`);
            return NextResponse.json({
              success: true,
              data: {
                symbol,
                interval,
                candles: cached.data,
                count: cached.data.length,
                cached: true,
                warning: 'Using cached data due to temporary API issues'
              }
            });
          }

          throw bybitError;
        }
      }
    } catch (binanceError) {
      console.error(`[Klines API] Binance fetch error:`, binanceError);

      // Try Bybit fallback
      console.log(`[Klines API] Falling back to Bybit after Binance error...`);
      const bybitUrl = `https://api.bybit.com/v5/market/kline?category=linear&symbol=${symbol}&interval=${interval}&limit=${limit}`;

      try {
        response = await fetchWithRetry(bybitUrl);
        source = 'bybit';

        if (!response.ok) {
          const bybitError = await response.text();
          console.error(`[Klines API] Bybit API also failed: ${response.status} - ${bybitError}`);

          // If we have cached data, return it even if expired
          if (cached) {
            console.log(`[Klines API] Returning stale cache due to both APIs failing`);
            return NextResponse.json({
              success: true,
              data: {
                symbol,
                interval,
                candles: cached.data,
                count: cached.data.length,
                cached: true,
                warning: 'Using cached data due to API unavailability'
              }
            });
          }

          throw new Error(`Both Binance and Bybit APIs failed`);
        }
      } catch (bybitError) {
        console.error(`[Klines API] Bybit fallback also failed:`, bybitError);

        // Final fallback: return stale cache if available
        if (cached) {
          console.log(`[Klines API] Returning stale cache as final fallback`);
          return NextResponse.json({
            success: true,
            data: {
              symbol,
              interval,
              candles: cached.data,
              count: cached.data.length,
              cached: true,
              warning: 'Using cached data due to temporary API issues'
            }
          });
        }

        throw bybitError;
      }
    }

    if (!response) {
      throw new Error('No response received from any exchange');
    }

    try {
      const rawData = await response.json();

      // Transform to TradingView format
      let formattedData: KlineData[];

      if (source === 'bybit') {
        // Bybit format: {retCode: 0, result: {list: [[timestamp, open, high, low, close, volume, turnover]]}}
        if (!rawData.result || !rawData.result.list) {
          throw new Error('Invalid Bybit response format');
        }

        formattedData = rawData.result.list.map((candle: any[]) => ({
          time: Math.floor(parseInt(candle[0]) / 1000), // Bybit timestamp is in milliseconds
          open: parseFloat(candle[1]),
          high: parseFloat(candle[2]),
          low: parseFloat(candle[3]),
          close: parseFloat(candle[4]),
          volume: parseFloat(candle[5])
        })).reverse(); // Bybit returns newest first, reverse to oldest first
      } else {
        // Binance format: [[timestamp, open, high, low, close, volume, ...]]
        formattedData = rawData.map((candle: any[]) => ({
          time: Math.floor(candle[0] / 1000),
          open: parseFloat(candle[1]),
          high: parseFloat(candle[2]),
          low: parseFloat(candle[3]),
          close: parseFloat(candle[4]),
          volume: parseFloat(candle[5])
        }));
      }

      // Store in cache
      cache.set(cacheKey, {
        data: formattedData,
        timestamp: Date.now()
      });

      console.log(`[Klines API] Successfully fetched ${formattedData.length} candles for ${symbol} from ${source.toUpperCase()}`);

      return NextResponse.json({
        success: true,
        data: {
          symbol,
          interval,
          candles: formattedData,
          count: formattedData.length,
          cached: false,
          source: source // 'binance' or 'bybit'
        }
      });

    } catch (fetchError) {
      // Final fallback: return stale cache if available
      if (cached) {
        console.log(`[Klines API] Returning stale cache due to fetch error`);
        return NextResponse.json({
          success: true,
          data: {
            symbol,
            interval,
            candles: cached.data,
            count: cached.data.length,
            cached: true,
            warning: 'Using cached data due to temporary API issues'
          }
        });
      }
      throw fetchError;
    }
  } catch (error) {
    console.error('[Klines API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to fetch kline data',
        suggestion: 'Please try again in a few moments. API rate limits may apply.'
      },
      { status: 500 }
    );
  }
}

// Cleanup old cache entries every 5 minutes
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    for (const [key, entry] of cache.entries()) {
      if (now - entry.timestamp > KLINES_CACHE_TTL * 10) { // Remove entries older than 5 minutes
        cache.delete(key);
      }
    }
  }, 5 * 60 * 1000);
}
