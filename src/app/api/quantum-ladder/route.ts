import { NextRequest, NextResponse } from 'next/server';

// Use explicit IPv4 address to avoid IPv6 resolution issues
const PYTHON_SERVICE = 'http://127.0.0.1:5022';
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1 second

/**
 * Quantum Ladder Strategy API
 *
 * Bu endpoint Python Quantum Ladder servisine proxy yapar.
 * ZigZag + Fibonacci + MA 7-25-99 Bottom Hunter algoritması ile
 * destek/direnç seviyelerini "merdiven" gibi otomatik çizer.
 *
 * Features:
 * - ZigZag swing high/low detection
 * - Fibonacci retracement levels (ladder rungs)
 * - MA 7-25-99 bottom detection with scoring
 * - Multi-timeframe confluence zones
 */

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with timeout using AbortController
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Proxy request to Python service with retry logic
 */
async function proxyRequest(
  targetUrl: string,
  method: string,
  body?: string
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      // Apply retry delay (except for first attempt)
      if (attempt > 0) {
        await sleep(RETRY_DELAY * attempt);
      }

      // Prepare request options
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      };

      // Add body for POST requests
      if (body && method === 'POST') {
        options.body = body;
      }

      // Make request with timeout
      const response = await fetchWithTimeout(targetUrl, options, REQUEST_TIMEOUT);

      // Return response (even if status >= 400, let caller handle it)
      return response;

    } catch (error) {
      lastError = error as Error;

      // Don't retry on abort (timeout)
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${REQUEST_TIMEOUT}ms`);
      }

      // Log retry attempt
      if (attempt < MAX_RETRIES) {
        console.warn(
          `[Quantum Ladder] Retry ${attempt + 1}/${MAX_RETRIES} for ${targetUrl}:`,
          error instanceof Error ? error.message : 'Unknown error'
        );
      }
    }
  }

  // All retries failed
  throw new Error(
    `Failed after ${MAX_RETRIES} retries: ${lastError?.message || 'Unknown error'}`
  );
}

/**
 * Generate Binance-based fallback analysis when Python service is unavailable
 * Uses real Binance klines data to calculate approximate ZigZag, Fibonacci, and MA levels
 */
async function generateBinanceFallback(
  symbol: string,
  timeframes: string[],
  limit: number
): Promise<QuantumLadderResponse['data']> {
  console.log(`[Quantum Ladder] Generating Binance-based fallback for ${symbol}`);

  const analysis: any = {};

  // Process each timeframe
  for (const timeframe of timeframes) {
    try {
      // Fetch Binance klines
      const klinesUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${timeframe}&limit=${Math.min(limit, 1000)}`;
      const klinesResponse = await fetch(klinesUrl);

      if (!klinesResponse.ok) {
        throw new Error(`Binance API error: ${klinesResponse.status}`);
      }

      const klines = await klinesResponse.json();

      // Extract OHLC data
      const highs = klines.map((k: any) => parseFloat(k[2]));
      const lows = klines.map((k: any) => parseFloat(k[3]));
      const closes = klines.map((k: any) => parseFloat(k[4]));

      // Find swing highs and lows (simple approximation)
      const swings: Array<{ type: 'high' | 'low'; price: number; index: number }> = [];
      const _swingThreshold = 0.02; // 2% threshold for swing detection

      for (let i = 5; i < closes.length - 5; i++) {
        const isSwingHigh = highs.slice(i - 5, i).every((h: number) => h < highs[i]) &&
                            highs.slice(i + 1, i + 6).every((h: number) => h < highs[i]);
        const isSwingLow = lows.slice(i - 5, i).every((l: number) => l > lows[i]) &&
                           lows.slice(i + 1, i + 6).every((l: number) => l > lows[i]);

        if (isSwingHigh) {
          swings.push({ type: 'high', price: highs[i], index: i });
        } else if (isSwingLow) {
          swings.push({ type: 'low', price: lows[i], index: i });
        }
      }

      const lastSwingHigh = [...swings].filter(s => s.type === 'high').pop()?.price || Math.max(...highs);
      const lastSwingLow = [...swings].filter(s => s.type === 'low').pop()?.price || Math.min(...lows);

      // Calculate Fibonacci levels
      const range = lastSwingHigh - lastSwingLow;
      const fibLevels = {
        '0.0': lastSwingLow,
        '0.236': lastSwingLow + range * 0.236,
        '0.382': lastSwingLow + range * 0.382,
        '0.5': lastSwingLow + range * 0.5,
        '0.618': lastSwingLow + range * 0.618,
        '0.786': lastSwingLow + range * 0.786,
        '1.0': lastSwingHigh,
      };

      // Calculate Moving Averages
      const ma7 = closes.slice(-7).reduce((a: number, b: number) => a + b, 0) / 7;
      const ma25 = closes.slice(-25).reduce((a: number, b: number) => a + b, 0) / 25;
      const ma99 = closes.slice(-99).reduce((a: number, b: number) => a + b, 0) / 99;

      const currentPrice = closes[closes.length - 1];

      // Determine bottom MA and signal
      let bottomMA = 'MA99';
      let score = 5;
      let signal = 'NEUTRAL';
      let confidence = 50;
      let crossoverImminent = false;

      if (currentPrice > ma7 && currentPrice > ma25 && currentPrice > ma99) {
        bottomMA = 'Above All MAs';
        score = 9;
        signal = 'STRONG_BUY';
        confidence = 85;
      } else if (currentPrice < ma7 && currentPrice < ma25 && currentPrice < ma99) {
        bottomMA = 'Below All MAs';
        score = 2;
        signal = 'STRONG_SELL';
        confidence = 85;
      } else if (currentPrice > ma25 && currentPrice > ma99) {
        bottomMA = 'MA25';
        score = 7;
        signal = 'BUY';
        confidence = 70;
      } else if (currentPrice > ma99) {
        bottomMA = 'MA99';
        score = 6;
        signal = 'WEAK_BUY';
        confidence = 60;
      } else if (currentPrice < ma25) {
        bottomMA = 'MA7';
        score = 3;
        signal = 'SELL';
        confidence = 70;
      }

      // Check for imminent crossover
      const _ma7_prev = closes.slice(-8, -1).reduce((a: number, b: number) => a + b, 0) / 7;
      const _ma25_prev = closes.slice(-26, -1).reduce((a: number, b: number) => a + b, 0) / 25;
      crossoverImminent = Math.abs(ma7 - ma25) < (range * 0.005); // Within 0.5% of range

      analysis[timeframe] = {
        zigzag: {
          swings: swings.slice(-10), // Last 10 swings
          last_swing_high: lastSwingHigh,
          last_swing_low: lastSwingLow,
        },
        fibonacci: {
          levels: fibLevels,
          direction: currentPrice > fibLevels['0.5'] ? 'bullish' : 'bearish',
          range: range,
        },
        ma_analysis: {
          ma7: parseFloat(ma7.toFixed(2)),
          ma25: parseFloat(ma25.toFixed(2)),
          ma99: parseFloat(ma99.toFixed(2)),
          bottom_ma: bottomMA,
          score,
          signal,
          confidence,
          crossover_imminent: crossoverImminent,
          description: `Price at ${currentPrice.toFixed(2)}, ${bottomMA} support. ${signal} signal with ${confidence}% confidence.`,
        },
      };

    } catch (error) {
      console.error(`[Quantum Ladder] Error analyzing ${timeframe}:`, error);
      // Add minimal fallback for this timeframe
      analysis[timeframe] = {
        error: 'Timeframe analysis failed',
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // Calculate confluence zones (price levels appearing in multiple timeframes)
  const confluenceZones: Array<{
    price: number;
    timeframes: string[];
    levels: string[];
    power_score: number;
    description: string;
  }> = [];

  const allLevels: Map<number, { timeframes: Set<string>; levels: Set<string> }> = new Map();

  // Collect all Fibonacci levels from all timeframes
  for (const [tf, tfAnalysis] of Object.entries(analysis)) {
    if ((tfAnalysis as any).fibonacci && (tfAnalysis as any).fibonacci.levels) {
      for (const [levelName, price] of Object.entries((tfAnalysis as any).fibonacci.levels)) {
        const roundedPrice = Math.round((price as number) * 100) / 100;

        if (!allLevels.has(roundedPrice)) {
          allLevels.set(roundedPrice, { timeframes: new Set(), levels: new Set() });
        }

        const entry = allLevels.get(roundedPrice)!;
        entry.timeframes.add(tf);
        entry.levels.add(`Fib ${levelName}`);
      }
    }
  }

  // Find confluence zones (levels appearing in 2+ timeframes)
  for (const [price, data] of allLevels.entries()) {
    if (data.timeframes.size >= 2) {
      confluenceZones.push({
        price,
        timeframes: Array.from(data.timeframes),
        levels: Array.from(data.levels),
        power_score: data.timeframes.size * 10,
        description: `Strong confluence zone at ${price} (${data.timeframes.size} timeframes)`,
      });
    }
  }

  // Sort by power score
  confluenceZones.sort((a, b) => b.power_score - a.power_score);

  // Calculate overall signal
  const signals = Object.values(analysis)
    .filter((a: any) => a.ma_analysis && a.ma_analysis.signal)
    .map((a: any) => a.ma_analysis.signal);

  const buySignals = signals.filter(s => s.includes('BUY')).length;
  const sellSignals = signals.filter(s => s.includes('SELL')).length;

  let overallDirection = 'NEUTRAL';
  let overallConfidence = 50;
  let overallStrength = 'MEDIUM';

  if (buySignals > sellSignals * 1.5) {
    overallDirection = 'BULLISH';
    overallConfidence = Math.min(85, 60 + (buySignals * 10));
    overallStrength = buySignals >= 2 ? 'STRONG' : 'MEDIUM';
  } else if (sellSignals > buySignals * 1.5) {
    overallDirection = 'BEARISH';
    overallConfidence = Math.min(85, 60 + (sellSignals * 10));
    overallStrength = sellSignals >= 2 ? 'STRONG' : 'MEDIUM';
  }

  return {
    symbol,
    analysis,
    confluence_zones: confluenceZones.slice(0, 10), // Top 10
    overall_signal: {
      direction: overallDirection,
      confidence: overallConfidence,
      strength: overallStrength,
      description: `${overallDirection} trend detected with ${overallConfidence}% confidence across ${timeframes.length} timeframes (${buySignals} BUY, ${sellSignals} SELL signals). Using Binance-based fallback analysis.`,
    },
    timestamp: new Date().toISOString(),
  };
}

interface QuantumLadderRequest {
  symbol?: string;
  timeframes?: string[];
  limit?: number;
}

interface QuantumLadderResponse {
  success: boolean;
  data?: {
    symbol: string;
    analysis: {
      [timeframe: string]: {
        zigzag: {
          swings: Array<{
            type: 'high' | 'low';
            price: number;
            index: number;
          }>;
          last_swing_high: number;
          last_swing_low: number;
        };
        fibonacci: {
          levels: {
            [key: string]: number;
          };
          direction: string;
          range: number;
        };
        ma_analysis: {
          ma7: number;
          ma25: number;
          ma99: number;
          bottom_ma: string;
          score: number;
          signal: string;
          confidence: number;
          crossover_imminent: boolean;
          description: string;
        };
      };
    };
    confluence_zones?: Array<{
      price: number;
      timeframes: string[];
      levels: string[];
      power_score: number;
      description: string;
    }>;
    overall_signal: {
      direction: string;
      confidence: number;
      strength: string;
      description: string;
    };
    timestamp: string;
  };
  error?: string;
}

/**
 * POST /api/quantum-ladder
 *
 * Request body:
 * {
 *   "symbol": "BTCUSDT",
 *   "timeframes": ["15m", "1h", "4h"],
 *   "limit": 1000
 * }
 *
 * Response:
 * {
 *   "success": true,
 *   "data": {
 *     "symbol": "BTCUSDT",
 *     "analysis": { ... },
 *     "confluence_zones": [ ... ],
 *     "overall_signal": { ... }
 *   }
 * }
 */
export async function POST(request: NextRequest): Promise<NextResponse<QuantumLadderResponse>> {
  const startTime = Date.now();

  try {
    const body: QuantumLadderRequest = await request.json();

    const symbol = body.symbol || 'BTCUSDT';
    const timeframes = body.timeframes || ['15m', '1h', '4h'];
    const limit = body.limit || 1000;

    console.log(`[Quantum Ladder API] POST Analyzing ${symbol} on timeframes: ${timeframes.join(', ')}`);

    // Build request body
    const requestBody = JSON.stringify({
      symbol,
      timeframes,
      limit
    });

    // Try to proxy request to Python Quantum Ladder service with graceful fallback
    try {
      const response = await proxyRequest(
        `${PYTHON_SERVICE}/analyze`,
        'POST',
        requestBody
      );

      const data = await response.json();

      if (!response.ok) {
        // Pass through the actual error message from Python service
        const errorMessage = data.error || `Python service returned ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }

      const duration = Date.now() - startTime;

      console.log(`[Quantum Ladder API] POST Success - ${response.status} (${duration}ms)`);

      // Return the analysis data
      return NextResponse.json({
        success: true,
        data: data.data || data
      });

    } catch (proxyError: any) {
      // Graceful fallback if Python service unavailable - use Binance-based analysis
      const duration = Date.now() - startTime;
      console.warn(`[Quantum Ladder API] Python service unavailable (${duration}ms), using Binance fallback:`, proxyError.message);

      try {
        const fallbackData = await generateBinanceFallback(symbol, timeframes, limit);

        return NextResponse.json({
          success: true,
          fallback: true,
          data: fallbackData
        });
      } catch (fallbackError: any) {
        console.error(`[Quantum Ladder API] Fallback also failed:`, fallbackError);
        return NextResponse.json({
          success: false,
          error: 'Quantum Ladder analizi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.',
          fallback_error: fallbackError.message
        }, { status: 503 });
      }
    }

  } catch (error: any) {
    // Unexpected error (JSON parsing, etc.)
    const duration = Date.now() - startTime;
    console.error(`[Quantum Ladder API] POST Error (${duration}ms):`, error);

    return NextResponse.json({
      success: false,
      error: error.message || 'Quantum Ladder analizi başarısız oldu',
      data: undefined
    }, { status: 500 });
  }
}

/**
 * GET /api/quantum-ladder?symbol=BTCUSDT&timeframes=15m,1h,4h
 *
 * Query parameters:
 * - symbol: Trading pair (default: BTCUSDT)
 * - timeframes: Comma-separated timeframes (default: 15m,1h,4h)
 * - limit: Number of candles to analyze (default: 1000)
 */
export async function GET(request: NextRequest): Promise<NextResponse<QuantumLadderResponse>> {
  const startTime = Date.now();

  try {
    const searchParams = request.nextUrl.searchParams;

    const symbol = searchParams.get('symbol') || 'BTCUSDT';
    const timeframesParam = searchParams.get('timeframes') || '15m,1h,4h';
    const timeframes = timeframesParam.split(',').map(tf => tf.trim());
    const limit = parseInt(searchParams.get('limit') || '1000', 10);

    console.log(`[Quantum Ladder API] GET request for ${symbol} on ${timeframes.join(', ')}`);

    // Build request body
    const requestBody = JSON.stringify({
      symbol,
      timeframes,
      limit
    });

    // Try to proxy request to Python Quantum Ladder service with graceful fallback
    try {
      const response = await proxyRequest(
        `${PYTHON_SERVICE}/analyze`,
        'POST',
        requestBody
      );

      const data = await response.json();

      if (!response.ok) {
        // Pass through the actual error message from Python service
        const errorMessage = data.error || `Python service returned ${response.status}: ${response.statusText}`;
        throw new Error(errorMessage);
      }

      const duration = Date.now() - startTime;

      console.log(`[Quantum Ladder API] GET Success - ${response.status} (${duration}ms)`);

      return NextResponse.json({
        success: true,
        data: data.data || data
      });

    } catch (proxyError: any) {
      // Graceful fallback if Python service unavailable - use Binance-based analysis
      const duration = Date.now() - startTime;
      console.warn(`[Quantum Ladder API] Python service unavailable (${duration}ms), using Binance fallback:`, proxyError.message);

      try {
        const fallbackData = await generateBinanceFallback(symbol, timeframes, limit);

        return NextResponse.json({
          success: true,
          fallback: true,
          data: fallbackData
        });
      } catch (fallbackError: any) {
        console.error(`[Quantum Ladder API] Fallback also failed:`, fallbackError);
        return NextResponse.json({
          success: false,
          error: 'Quantum Ladder analizi şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.',
          fallback_error: fallbackError.message
        }, { status: 503 });
      }
    }

  } catch (error: any) {
    // Unexpected error (parameter parsing, etc.)
    const duration = Date.now() - startTime;
    console.error(`[Quantum Ladder API] GET Error (${duration}ms):`, error);

    return NextResponse.json({
      success: false,
      error: error.message || 'Quantum Ladder analizi başarısız oldu'
    }, { status: 500 });
  }
}
