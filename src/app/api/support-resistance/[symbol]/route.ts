import { NextRequest, NextResponse } from 'next/server';

/**
 * Support & Resistance Calculator API
 * Uses Pivot Points method to calculate support/resistance levels
 *
 * Features:
 * - In-memory caching (30 seconds TTL)
 * - Retry mechanism with exponential backoff
 * - Rate limit handling (HTTP 418)
 * - Fallback to cached data on errors
 */

interface PivotLevels {
  pivot: number;
  resistance: {
    r1: number;
    r2: number;
    r3: number;
  };
  support: {
    s1: number;
    s2: number;
    s3: number;
  };
}

interface CacheEntry {
  levels: PivotLevels;
  period: any;
  timestamp: number;
}

// In-memory cache (30 seconds TTL)
const cache = new Map<string, CacheEntry>();
const SR_CACHE_TTL = 30 * 1000; // 30 seconds

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
        console.log(`[Support/Resistance] Rate limited, waiting ${waitTime}ms before retry ${i + 1}/${maxRetries}`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }

      return response;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      const waitTime = Math.pow(2, i) * 1000; // 1s, 2s, 4s
      console.log(`[Support/Resistance] Retry ${i + 1}/${maxRetries} after ${waitTime}ms`);
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
    const cacheKey = `${symbol}-${interval}`;

    // Check cache first
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < SR_CACHE_TTL) {
      console.log(`[Support/Resistance] Cache HIT for ${cacheKey}`);
      return NextResponse.json({
        success: true,
        data: {
          symbol,
          interval,
          period: cached.period,
          levels: cached.levels,
          cached: true
        }
      });
    }

    console.log(`[Support/Resistance] Calculating for ${symbol} with interval ${interval}`);

    // Determine how many candles to fetch based on interval
    let limit = 1;
    if (interval === '1m' || interval === '15m' || interval === '1h') {
      limit = 24; // Use last 24 hours for short timeframes
    } else if (interval === '4h') {
      limit = 18; // Last 3 days
    } else if (interval === '1d') {
      limit = 7; // Last week
    } else if (interval === '1w') {
      limit = 4; // Last month
    }

    const binanceUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;

    try {
      const response = await fetchWithRetry(binanceUrl);

      if (!response.ok) {
        console.error(`[Support/Resistance] Binance API failed: ${response.status}`);

        // If we have cached data, return it even if expired
        if (cached) {
          console.log(`[Support/Resistance] Returning stale cache due to API error`);
          return NextResponse.json({
            success: true,
            data: {
              symbol,
              interval,
              period: cached.period,
              levels: cached.levels,
              cached: true,
              warning: 'Using cached data due to API unavailability'
            }
          });
        }

        throw new Error(`Binance API error: ${response.status}`);
      }

      const rawData = await response.json();

      // Calculate high, low, close from the period
      const candles = rawData.map((candle: any[]) => ({
        high: parseFloat(candle[2]),
        low: parseFloat(candle[3]),
        close: parseFloat(candle[4])
      }));

      const high = Math.max(...candles.map((c: any) => c.high));
      const low = Math.min(...candles.map((c: any) => c.low));
      const close = candles[candles.length - 1].close;

      // Pivot Points Formula
      const pivot = (high + low + close) / 3;

      // Resistance levels
      const r1 = (2 * pivot) - low;
      const r2 = pivot + (high - low);
      const r3 = pivot + 2 * (high - low);

      // Support levels
      const s1 = (2 * pivot) - high;
      const s2 = pivot - (high - low);
      const s3 = pivot - 2 * (high - low);

      const levels: PivotLevels = {
        pivot: parseFloat(pivot.toFixed(2)),
        resistance: {
          r1: parseFloat(r1.toFixed(2)),
          r2: parseFloat(r2.toFixed(2)),
          r3: parseFloat(r3.toFixed(2))
        },
        support: {
          s1: parseFloat(s1.toFixed(2)),
          s2: parseFloat(s2.toFixed(2)),
          s3: parseFloat(s3.toFixed(2))
        }
      };

      const period = {
        high,
        low,
        close,
        candlesUsed: candles.length
      };

      // Store in cache
      cache.set(cacheKey, {
        levels,
        period,
        timestamp: Date.now()
      });

      console.log(`[Support/Resistance] Calculated levels for ${symbol}:`, levels);

      return NextResponse.json({
        success: true,
        data: {
          symbol,
          interval,
          period,
          levels,
          cached: false
        }
      });

    } catch (fetchError) {
      // Final fallback: return stale cache if available
      if (cached) {
        console.log(`[Support/Resistance] Returning stale cache due to fetch error`);
        return NextResponse.json({
          success: true,
          data: {
            symbol,
            interval,
            period: cached.period,
            levels: cached.levels,
            cached: true,
            warning: 'Using cached data due to temporary API issues'
          }
        });
      }
      throw fetchError;
    }

  } catch (error) {
    console.error('[Support/Resistance] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to calculate support/resistance',
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
      if (now - entry.timestamp > SR_CACHE_TTL * 10) { // Remove entries older than 5 minutes
        cache.delete(key);
      }
    }
  }, 5 * 60 * 1000);
}
