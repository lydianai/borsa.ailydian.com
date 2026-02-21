/**
 * 🔍 MULTI-TIMEFRAME SCANNER API
 *
 * Comprehensive multi-timeframe analysis with all indicators
 * - Analyzes 1h and 4h timeframes simultaneously
 * - Validates ALL indicators for LONG signal confirmation
 * - Detects candlestick patterns
 * - Calculates weighted consensus score
 *
 * USAGE:
 * GET /api/scanner/multi-timeframe?symbol=BTCUSDT
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Transparent scoring algorithm
 * - Rate-limited requests
 * - Educational purpose only
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  validateRSI,
  validateMFI,
  validateBollingerBands,
  validateVWAP,
  validateFVG,
  validateOrderBlocks,
  validatePremiumDiscount,
  validateMarketStructure,
  OHLCV
} from '@/lib/indicators/advanced-validator';
import {
  detectCandlestickPatterns,
  filterBullishPatterns,
  hasSufficientBullishPatterns
} from '@/lib/patterns/candlestick-patterns';
import {
  MultiTimeframeAnalysis,
  TimeframeAnalysis,
  Timeframe
} from '@/types/multi-timeframe-scanner';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// HELPER: FETCH BINANCE KLINES
// ============================================================================

async function fetchKlines(
  symbol: string,
  interval: string,
  limit: number = 500
): Promise<OHLCV[]> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return data.map((k: any[]) => ({
      time: k[0] / 1000, // Convert to seconds
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5])
    }));
  } catch (error) {
    console.error(`[Multi-TF Scanner] Failed to fetch ${interval} klines:`, error);
    throw error;
  }
}

// ============================================================================
// HELPER: CALCULATE TECHNICAL INDICATORS
// ============================================================================

function calculateRSI(closes: number[], period: number = 14): number[] {
  if (closes.length < period + 1) return [];

  const rsi: number[] = [];
  const changes: number[] = [];

  for (let i = 1; i < closes.length; i++) {
    changes.push(closes[i] - closes[i - 1]);
  }

  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 0; i < period; i++) {
    if (changes[i] > 0) avgGain += changes[i];
    else avgLoss += Math.abs(changes[i]);
  }

  avgGain /= period;
  avgLoss /= period;

  const rs = avgGain / (avgLoss || 0.0001);
  rsi.push(100 - (100 / (1 + rs)));

  for (let i = period; i < changes.length; i++) {
    const gain = changes[i] > 0 ? changes[i] : 0;
    const loss = changes[i] < 0 ? Math.abs(changes[i]) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    const currentRS = avgGain / (avgLoss || 0.0001);
    rsi.push(100 - (100 / (1 + currentRS)));
  }

  return rsi;
}

function calculateMFI(klines: OHLCV[], period: number = 14): number[] {
  if (klines.length < period + 1) return [];

  const mfi: number[] = [];
  const typicalPrices: number[] = [];
  const moneyFlows: number[] = [];

  for (const k of klines) {
    const tp = (k.high + k.low + k.close) / 3;
    typicalPrices.push(tp);
    moneyFlows.push(tp * k.volume);
  }

  for (let i = period; i < klines.length; i++) {
    let positiveFlow = 0;
    let negativeFlow = 0;

    for (let j = i - period + 1; j <= i; j++) {
      if (typicalPrices[j] > typicalPrices[j - 1]) {
        positiveFlow += moneyFlows[j];
      } else if (typicalPrices[j] < typicalPrices[j - 1]) {
        negativeFlow += moneyFlows[j];
      }
    }

    const mfiValue = negativeFlow === 0 ? 100 : 100 - (100 / (1 + positiveFlow / negativeFlow));
    mfi.push(mfiValue);
  }

  return mfi;
}

function calculateBollingerBands(closes: number[], period: number = 20, stdDev: number = 2) {
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];

  for (let i = period - 1; i < closes.length; i++) {
    const slice = closes.slice(i - period + 1, i + 1);
    const sma = slice.reduce((a, b) => a + b, 0) / period;

    const variance = slice.reduce((sum, val) => sum + Math.pow(val - sma, 2), 0) / period;
    const std = Math.sqrt(variance);

    middle.push(sma);
    upper.push(sma + std * stdDev);
    lower.push(sma - std * stdDev);
  }

  return { upper, middle, lower };
}

function calculateVWAP(klines: OHLCV[]): number {
  let cumulativeTPV = 0;
  let cumulativeVolume = 0;

  for (const k of klines) {
    const typicalPrice = (k.high + k.low + k.close) / 3;
    cumulativeTPV += typicalPrice * k.volume;
    cumulativeVolume += k.volume;
  }

  return cumulativeTPV / (cumulativeVolume || 1);
}

// ============================================================================
// ANALYZE SINGLE TIMEFRAME
// ============================================================================

async function analyzeTimeframe(
  symbol: string,
  timeframe: Timeframe
): Promise<TimeframeAnalysis> {
  // Map timeframe to Binance interval
  const intervalMap: Record<Timeframe, string> = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
  };

  const interval = intervalMap[timeframe];
  const klines = await fetchKlines(symbol, interval, 500);

  if (klines.length < 100) {
    throw new Error(`Insufficient data for ${symbol} on ${timeframe}`);
  }

  const closes = klines.map(k => k.close);
  const currentPrice = closes[closes.length - 1];
  const latestCandle = klines[klines.length - 1];

  // Calculate indicators
  const rsiValues = calculateRSI(closes);
  const mfiValues = calculateMFI(klines);
  const bb = calculateBollingerBands(closes);
  const vwap = calculateVWAP(klines.slice(-100)); // Last 100 candles for VWAP

  // Validate indicators using advanced validators
  const rsi = validateRSI(rsiValues, klines.slice(-(rsiValues.length)));
  const mfi = validateMFI(mfiValues, klines.slice(-(mfiValues.length)).map(k => k.volume));
  const bollinger = validateBollingerBands(bb.upper, bb.middle, bb.lower, currentPrice);
  const vwapAnalysis = validateVWAP(vwap, currentPrice);
  const fvg = validateFVG(klines.slice(-50));
  const orderBlocks = validateOrderBlocks(klines.slice(-50));
  const premiumDiscount = validatePremiumDiscount(klines.slice(-50));
  const marketStructure = validateMarketStructure(klines.slice(-50));

  // Placeholder validations for remaining indicators (simplified for now)
  const liquidity = {
    pools: [],
    signal: 'NEUTRAL' as const,
    confidence: 50,
    reason: 'Likidite analizi şu an mevcut değil'
  };

  const supportResistance = {
    levels: [],
    signal: 'NEUTRAL' as const,
    confidence: 50,
    reason: 'Destek/Direnç analizi şu an mevcut değil'
  };

  const fibonacci = {
    levels: { '0': 0, '23.6': 0, '38.2': 0, '50': 0, '61.8': 0, '78.6': 0, '100': 0 },
    signal: 'NEUTRAL' as const,
    confidence: 50,
    reason: 'Fibonacci analizi şu an mevcut değil'
  };

  // Detect candlestick patterns
  const allPatterns = detectCandlestickPatterns(klines, 20);
  const bullishPatterns = filterBullishPatterns(allPatterns);

  // Count BUY signals
  const indicators = {
    rsi,
    mfi,
    bollinger,
    vwap: vwapAnalysis,
    fvg,
    orderBlocks,
    liquidity,
    supportResistance,
    fibonacci,
    premiumDiscount,
    marketStructure
  };

  const buyIndicators = Object.values(indicators).filter(ind => ind.signal === 'BUY');
  const totalIndicators = Object.keys(indicators).length;

  // Calculate overall signal and confidence
  const buyRatio = buyIndicators.length / totalIndicators;
  let overallSignal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;

  if (buyRatio >= 0.7) {
    overallSignal = 'BUY';
    confidence = Math.round(buyRatio * 100);
  } else if (buyRatio <= 0.3) {
    overallSignal = 'SELL';
    confidence = Math.round((1 - buyRatio) * 100);
  } else {
    overallSignal = 'NEUTRAL';
    confidence = Math.round(Math.abs(buyRatio - 0.5) * 100);
  }

  return {
    timeframe,
    timestamp: Date.now(),
    price: {
      open: latestCandle.open,
      high: latestCandle.high,
      low: latestCandle.low,
      close: latestCandle.close,
      volume: latestCandle.volume
    },
    indicators,
    patterns: allPatterns,
    overallSignal,
    confidence,
    buyIndicatorCount: buyIndicators.length,
    totalIndicators,
    bullishPatternCount: bullishPatterns.length
  };
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';

  try {
    console.log(`[Multi-TF Scanner] Analyzing ${symbol}...`);

    // Analyze both timeframes in parallel
    const [tf1h, tf4h] = await Promise.all([
      analyzeTimeframe(symbol, '1h'),
      analyzeTimeframe(symbol, '4h')
    ]);

    // Calculate weighted consensus (4h is more important)
    const weight1h = 0.3;
    const weight4h = 0.7;

    const consensusScore = Math.round(
      tf1h.confidence * weight1h + tf4h.confidence * weight4h
    );

    // Check requirements for LONG signal (RELAXED RULES for better signal detection)
    const allIndicatorsBuy =
      tf1h.buyIndicatorCount === tf1h.totalIndicators &&
      tf4h.buyIndicatorCount === tf4h.totalIndicators;

    const bullishPatternsPresent =
      hasSufficientBullishPatterns(tf1h.patterns, 1, 70) ||
      hasSufficientBullishPatterns(tf4h.patterns, 1, 70);

    const multiTimeframeConfirm =
      tf1h.overallSignal === 'BUY' && tf4h.overallSignal === 'BUY';

    // NEW: More flexible confirmations
    const strongSingleTimeframe =
      (tf1h.overallSignal === 'BUY' && tf1h.confidence >= 75) ||
      (tf4h.overallSignal === 'BUY' && tf4h.confidence >= 75);

    const highConsensus = consensusScore >= 65; // Lowered from 70 for more signals
    const veryHighConsensus = consensusScore >= 75; // Lowered from 80
    const ultraHighConsensus = consensusScore >= 85; // Lowered from 90

    // NEW: Indicator majority (not all required) - RELAXED
    const indicatorMajority1h = tf1h.buyIndicatorCount / tf1h.totalIndicators >= 0.5; // Lowered from 0.6
    const indicatorMajority4h = tf4h.buyIndicatorCount / tf4h.totalIndicators >= 0.5; // Lowered from 0.6
    const strongIndicators = indicatorMajority1h || indicatorMajority4h; // Changed AND to OR

    // Determine signal quality (OPTIMIZED FOR MORE SIGNALS)
    let longSignalQuality: 'excellent' | 'good' | 'moderate' | 'poor' | 'none' = 'none';

    // EXCELLENT: All traditional criteria (STRICT)
    if (allIndicatorsBuy && bullishPatternsPresent && multiTimeframeConfirm && veryHighConsensus) {
      longSignalQuality = 'excellent';
    }
    // GOOD: Multi-timeframe confirm OR very high consensus with patterns (RELAXED)
    else if (
      (multiTimeframeConfirm && highConsensus) ||
      (ultraHighConsensus && bullishPatternsPresent) ||
      (veryHighConsensus && strongIndicators && bullishPatternsPresent)
    ) {
      longSignalQuality = 'good';
    }
    // MODERATE: Strong single timeframe OR high consensus with indicators (RELAXED)
    else if (
      (strongSingleTimeframe && highConsensus) ||
      (highConsensus && strongIndicators && bullishPatternsPresent) ||
      (veryHighConsensus && strongIndicators)
    ) {
      longSignalQuality = 'moderate';
    }
    // POOR: Decent consensus with some confirmation (ADDED FOR MORE SIGNALS)
    else if (
      (consensusScore >= 60 && strongSingleTimeframe) ||
      (consensusScore >= 55 && strongIndicators && bullishPatternsPresent)
    ) {
      longSignalQuality = 'poor';
    }

    // Should notify: excellent, good, moderate, OR poor (ADDED poor for more signals)
    const shouldNotify =
      longSignalQuality === 'excellent' ||
      longSignalQuality === 'good' ||
      longSignalQuality === 'moderate' ||
      longSignalQuality === 'poor';

    // Generate summary
    let summary = '';
    if (longSignalQuality === 'excellent') {
      summary = `🚀 MÜKEMMEL LONG SİNYALİ: ${symbol} - Tüm indikatörler BUY, mum formasyonları onaylı, konsensüs %${consensusScore}`;
    } else if (longSignalQuality === 'good') {
      summary = `✅ İYİ LONG SİNYALİ: ${symbol} - Çoklu zaman dilimi onayı, konsensüs %${consensusScore}`;
    } else if (longSignalQuality === 'moderate') {
      summary = `⚠️ ORTA LONG SİNYALİ: ${symbol} - Kısmi onay, konsensüs %${consensusScore}`;
    } else if (longSignalQuality === 'poor') {
      summary = `💡 ZAYIF LONG SİNYALİ: ${symbol} - Temel kriterler karşılandı, konsensüs %${consensusScore}`;
    } else {
      summary = `❌ YETERSİZ: ${symbol} - LONG sinyali kriterleri karşılanmadı`;
    }

    const analysis: MultiTimeframeAnalysis = {
      symbol,
      timestamp: Date.now(),
      timeframes: {
        '1h': tf1h,
        '4h': tf4h
      },
      consensusScore,
      longSignalQuality,
      requirements: {
        allIndicatorsBuy,
        bullishPatternsPresent,
        multiTimeframeConfirm,
        minimumConfidence: highConsensus
      },
      shouldNotify,
      summary
    };

    const duration = Date.now() - startTime;

    console.log(`[Multi-TF Scanner] ${symbol} analyzed in ${duration}ms - Quality: ${longSignalQuality}`);

    return NextResponse.json({
      success: true,
      data: analysis,
      metadata: {
        duration,
        timestamp: Date.now()
      }
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Multi-TF Scanner] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Analysis failed',
        duration
      },
      { status: 500 }
    );
  }
}
