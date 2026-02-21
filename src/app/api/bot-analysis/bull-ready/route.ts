/**
 * 🐂 BULL READY MOMENTUM DETECTOR
 *
 * Detects coins that have completed bullish momentum and are ready for upward movement
 *
 * TECHNICAL CRITERIA:
 * ✅ RSI > 50 (bullish momentum)
 * ✅ MACD bullish crossover (MACD line > Signal line)
 * ✅ Volume increase (current volume > average volume)
 * ✅ Price forming higher lows (bullish structure)
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Educational and research purposes only
 * - No trading execution or financial advice
 * - Transparent algorithmic criteria
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// TYPES
// ============================================================================

interface BullReadySignal {
  symbol: string;
  price: number;
  isBullReady: boolean;
  signals: {
    rsi: number;
    rsiBullish: boolean;
    macdBullish: boolean;
    volumeBullish: boolean;
    priceStructureBullish: boolean;
  };
  confidence: number;
  timestamp: number;
}

interface KlineData {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteAssetVolume: string;
  numberOfTrades: number;
  takerBuyBaseAssetVolume: string;
  takerBuyQuoteAssetVolume: string;
  ignore: string;
}

// ============================================================================
// TECHNICAL INDICATORS CALCULATIONS
// ============================================================================

/**
 * Calculate RSI (Relative Strength Index)
 * RSI > 50 indicates bullish momentum
 * RSI < 50 indicates bearish momentum
 */
function calculateRSI(closes: number[], period: number = 14): number {
  if (closes.length < period + 1) return 50; // Default neutral

  const changes = closes.slice(1).map((close, i) => close - closes[i]);
  const gains = changes.map(change => change > 0 ? change : 0);
  const losses = changes.map(change => change < 0 ? Math.abs(change) : 0);

  const avgGain = gains.slice(-period).reduce((a, b) => a + b, 0) / period;
  const avgLoss = losses.slice(-period).reduce((a, b) => a + b, 0) / period;

  if (avgLoss === 0) return 100;

  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  return rsi;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 * Returns true if MACD line is above signal line (bullish crossover)
 */
function calculateMACD(closes: number[]): boolean {
  if (closes.length < 26) return false;

  const ema12 = calculateEMA(closes, 12);
  const ema26 = calculateEMA(closes, 26);

  const macdLine = ema12 - ema26;

  // Calculate signal line (9-period EMA of MACD)
  const macdValues: number[] = [];
  for (let i = 26; i <= closes.length; i++) {
    const slice = closes.slice(0, i);
    const e12 = calculateEMA(slice, 12);
    const e26 = calculateEMA(slice, 26);
    macdValues.push(e12 - e26);
  }

  const signalLine = calculateEMA(macdValues, 9);

  return macdLine > signalLine;
}

/**
 * Calculate EMA (Exponential Moving Average)
 */
function calculateEMA(data: number[], period: number): number {
  if (data.length < period) return data[data.length - 1] || 0;

  const multiplier = 2 / (period + 1);
  let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;

  for (let i = period; i < data.length; i++) {
    ema = (data[i] - ema) * multiplier + ema;
  }

  return ema;
}

/**
 * Check if volume is increasing (bullish signal)
 */
function isVolumeIncreasing(volumes: number[]): boolean {
  if (volumes.length < 10) return false;

  const recentVolume = volumes[volumes.length - 1];
  const avgVolume = volumes.slice(-20).reduce((a, b) => a + b, 0) / 20;

  return recentVolume > avgVolume * 1.2; // 20% above average
}

/**
 * Check if price is forming higher lows (bullish price structure)
 */
function isFormingHigherLows(lows: number[]): boolean {
  if (lows.length < 3) return false;

  const recent = lows.slice(-3);
  return recent[2] > recent[1] && recent[1] > recent[0];
}

// ============================================================================
// DATA FETCHING
// ============================================================================

/**
 * Fetch kline/candlestick data from Binance
 */
async function fetchKlines(symbol: string, interval: string = '1h', limit: number = 100): Promise<KlineData[]> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6],
      quoteAssetVolume: k[7],
      numberOfTrades: k[8],
      takerBuyBaseAssetVolume: k[9],
      takerBuyQuoteAssetVolume: k[10],
      ignore: k[11]
    }));
  } catch (error) {
    console.error(`[Bull Ready] Failed to fetch klines for ${symbol}:`, error);
    return [];
  }
}

/**
 * Fetch all USDT-M Futures symbols
 */
async function fetchAllSymbols(): Promise<string[]> {
  try {
    const response = await fetch('https://fapi.binance.com/fapi/v1/exchangeInfo', {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return data.symbols
      .filter((s: any) => s.status === 'TRADING' && s.contractType === 'PERPETUAL' && s.quoteAsset === 'USDT')
      .map((s: any) => s.symbol);
  } catch (error) {
    console.error('[Bull Ready] Failed to fetch symbols:', error);
    return [];
  }
}

/**
 * Analyze single symbol for bull readiness
 */
async function analyzeBullMomentum(symbol: string): Promise<BullReadySignal | null> {
  try {
    const klines = await fetchKlines(symbol, '1h', 100);

    if (klines.length < 50) {
      return null; // Not enough data
    }

    const closes = klines.map(k => parseFloat(k.close));
    const lows = klines.map(k => parseFloat(k.low));
    const volumes = klines.map(k => parseFloat(k.volume));
    const currentPrice = closes[closes.length - 1];

    // Calculate indicators
    const rsi = calculateRSI(closes, 14);
    const rsiBullish = rsi > 50;
    const macdBullish = calculateMACD(closes);
    const volumeBullish = isVolumeIncreasing(volumes);
    const priceStructureBullish = isFormingHigherLows(lows);

    // Calculate confidence score
    let confidence = 0;
    if (rsiBullish) confidence += 25;
    if (macdBullish) confidence += 35;
    if (volumeBullish) confidence += 20;
    if (priceStructureBullish) confidence += 20;

    const isBullReady = confidence >= 70; // At least 70% confidence

    return {
      symbol,
      price: currentPrice,
      isBullReady,
      signals: {
        rsi,
        rsiBullish,
        macdBullish,
        volumeBullish,
        priceStructureBullish
      },
      confidence,
      timestamp: Date.now()
    };
  } catch (error) {
    console.error(`[Bull Ready] Error analyzing ${symbol}:`, error);
    return null;
  }
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    console.log('[Bull Ready] Starting bull momentum analysis...');

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const symbols = searchParams.get('symbols')?.split(',');
    const limit = parseInt(searchParams.get('limit') || '50');

    let symbolsToAnalyze: string[];

    if (symbols && symbols.length > 0) {
      // Analyze specific symbols
      symbolsToAnalyze = symbols;
    } else {
      // Fetch all symbols
      const allSymbols = await fetchAllSymbols();
      symbolsToAnalyze = allSymbols.slice(0, limit); // Limit to prevent timeout
    }

    console.log(`[Bull Ready] Analyzing ${symbolsToAnalyze.length} symbols...`);

    // Analyze symbols in batches to prevent rate limiting
    const batchSize = 10;
    const results: BullReadySignal[] = [];

    for (let i = 0; i < symbolsToAnalyze.length; i += batchSize) {
      const batch = symbolsToAnalyze.slice(i, i + batchSize);
      const batchPromises = batch.map(symbol => analyzeBullMomentum(symbol));
      const batchResults = await Promise.all(batchPromises);

      results.push(...batchResults.filter(r => r !== null) as BullReadySignal[]);

      // Small delay between batches
      if (i + batchSize < symbolsToAnalyze.length) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // Filter for bull ready signals
    const bullReadySignals = results.filter(r => r.isBullReady);

    // Sort by confidence
    bullReadySignals.sort((a, b) => b.confidence - a.confidence);

    const duration = Date.now() - startTime;

    console.log(
      `[Bull Ready] Found ${bullReadySignals.length} bull ready signals out of ${results.length} analyzed in ${duration}ms`
    );

    return NextResponse.json({
      success: true,
      data: {
        bullReadySignals,
        count: bullReadySignals.length,
        totalAnalyzed: results.length,
        timestamp: Date.now(),
        lastUpdate: new Date().toISOString()
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Bull Ready] Error:', error);

    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to analyze bull momentum',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    }, { status: 500 });
  }
}
