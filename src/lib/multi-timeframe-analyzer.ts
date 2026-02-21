/**
 * ⏰ MULTI-TIMEFRAME ANALYSIS LIBRARY
 * Farklı zaman dilimlerinde (1h, 4h, 1d, 1w) teknik analiz
 *
 * ✅ GERÇEK VERİ: Binance kline data kullanır
 * ❌ Demo/Mock veri YOK
 *
 * White-Hat Compliance:
 * - Eğitim amaçlıdır
 * - Finansal tavsiye değildir
 * - Trend confirmation için kullanılır
 */

import {
  fetchBinanceKlines,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  type Candlestick,
  type RSI,
  type MACD,
  type BollingerBands,
} from './technical-indicators';

// ============================================================================
// INTERFACES
// ============================================================================

export type Timeframe = '1h' | '4h' | '1d' | '1w';

export interface TimeframeAnalysis {
  timeframe: Timeframe;
  rsi: RSI;
  macd: MACD;
  bollingerBands: BollingerBands;

  // Genel sinyal
  overallSignal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  signalStrength: number; // 0-100

  // Fiyat bilgisi
  currentPrice: number;
  priceChange24h?: number;

  // Timestamp
  timestamp: string;
}

export interface MultiTimeframeAnalysis {
  symbol: string;
  timeframes: {
    '1h': TimeframeAnalysis;
    '4h': TimeframeAnalysis;
    '1d': TimeframeAnalysis;
    '1w': TimeframeAnalysis;
  };

  // Consensus (Uyum)
  consensus: {
    signal: 'STRONG_BULLISH' | 'BULLISH' | 'NEUTRAL' | 'BEARISH' | 'STRONG_BEARISH';
    strength: number; // 0-100
    alignment: number; // Kaç timeframe aynı yönde (0-4)
    interpretation: string; // Türkçe açıklama
  };

  // Higher Timeframe Bias (Büyük zaman dilimi yönlülüğü)
  higherTimeframeBias: 'BULLISH' | 'BEARISH' | 'NEUTRAL';

  timestamp: string;
}

// ============================================================================
// TIMEFRAME CONFIGURATION
// ============================================================================

export const TIMEFRAME_CONFIG = {
  '1h': {
    label: '1 Saat',
    interval: '1h',
    limit: 100,
    weight: 1, // En düşük ağırlık
  },
  '4h': {
    label: '4 Saat',
    interval: '4h',
    limit: 100,
    weight: 2,
  },
  '1d': {
    label: '1 Gün',
    interval: '1d',
    limit: 100,
    weight: 3,
  },
  '1w': {
    label: '1 Hafta',
    interval: '1w',
    limit: 52, // 1 yıllık data (52 hafta)
    weight: 4, // En yüksek ağırlık
  },
} as const;

// ============================================================================
// SINGLE TIMEFRAME ANALYSIS
// ============================================================================

/**
 * Tek bir timeframe için kapsamlı analiz
 */
async function analyzeTimeframe(
  symbol: string,
  timeframe: Timeframe
): Promise<TimeframeAnalysis> {
  const config = TIMEFRAME_CONFIG[timeframe];

  // 1. Fetch klines data
  const klines = await fetchBinanceKlines(symbol, config.interval, config.limit);

  if (klines.length === 0) {
    throw new Error(`No kline data for ${symbol} on ${timeframe}`);
  }

  // 2. Extract close prices
  const closePrices = klines.map(k => k.close);
  const currentPrice = closePrices[closePrices.length - 1];

  // 3. Calculate technical indicators
  const rsi = calculateRSI(closePrices, 14);
  const macd = calculateMACD(closePrices, 12, 26, 9);
  const bollingerBands = calculateBollingerBands(closePrices, 20, 2);

  // 4. Calculate 24h price change (if available)
  let priceChange24h: number | undefined;
  if (timeframe === '1h' && klines.length >= 24) {
    const price24hAgo = klines[klines.length - 24].close;
    priceChange24h = ((currentPrice - price24hAgo) / price24hAgo) * 100;
  } else if (timeframe === '4h' && klines.length >= 6) {
    const price24hAgo = klines[klines.length - 6].close;
    priceChange24h = ((currentPrice - price24hAgo) / price24hAgo) * 100;
  }

  // 5. Determine overall signal
  const { overallSignal, signalStrength } = determineOverallSignal(rsi, macd, bollingerBands);

  return {
    timeframe,
    rsi,
    macd,
    bollingerBands,
    overallSignal,
    signalStrength,
    currentPrice,
    priceChange24h,
    timestamp: new Date().toISOString(),
  };
}

/**
 * RSI, MACD, Bollinger Bands'den genel sinyal belirle
 */
function determineOverallSignal(
  rsi: RSI,
  macd: MACD,
  bb: BollingerBands
): { overallSignal: 'BULLISH' | 'BEARISH' | 'NEUTRAL'; signalStrength: number } {
  let bullishScore = 0;
  let bearishScore = 0;

  // RSI scoring
  if (rsi.signal === 'OVERSOLD') {
    bullishScore += 30;
  } else if (rsi.signal === 'OVERBOUGHT') {
    bearishScore += 30;
  } else {
    // Neutral RSI - use value
    if (rsi.value > 50) {
      bullishScore += (rsi.value - 50) * 0.6; // Max +30
    } else {
      bearishScore += (50 - rsi.value) * 0.6; // Max +30
    }
  }

  // MACD scoring
  if (macd.signal === 'BULLISH') {
    bullishScore += 40;
  } else if (macd.signal === 'BEARISH') {
    bearishScore += 40;
  } else {
    bullishScore += 10;
    bearishScore += 10;
  }

  // Bollinger Bands scoring
  if (bb.signal === 'OVERSOLD') {
    bullishScore += 30;
  } else if (bb.signal === 'OVERBOUGHT') {
    bearishScore += 30;
  } else {
    bullishScore += 10;
    bearishScore += 10;
  }

  // Determine overall signal
  const totalScore = bullishScore + bearishScore;
  const bullishPercentage = (bullishScore / totalScore) * 100;
  const bearishPercentage = (bearishScore / totalScore) * 100;

  let overallSignal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  let signalStrength: number;

  if (bullishPercentage >= 60) {
    overallSignal = 'BULLISH';
    signalStrength = bullishPercentage;
  } else if (bearishPercentage >= 60) {
    overallSignal = 'BEARISH';
    signalStrength = bearishPercentage;
  } else {
    overallSignal = 'NEUTRAL';
    signalStrength = Math.max(bullishPercentage, bearishPercentage);
  }

  return { overallSignal, signalStrength };
}

// ============================================================================
// MULTI-TIMEFRAME CONSENSUS ALGORITHM
// ============================================================================

/**
 * Multi-timeframe consensus hesapla
 */
function calculateConsensus(
  tf1h: TimeframeAnalysis,
  tf4h: TimeframeAnalysis,
  tf1d: TimeframeAnalysis,
  tf1w: TimeframeAnalysis
): MultiTimeframeAnalysis['consensus'] {
  const timeframes = [tf1h, tf4h, tf1d, tf1w];
  const weights = [1, 2, 3, 4]; // Higher timeframes have more weight

  // Count signals
  let bullishCount = 0;
  let bearishCount = 0;
  let neutralCount = 0;
  let weightedBullishScore = 0;
  let weightedBearishScore = 0;

  timeframes.forEach((tf, index) => {
    const weight = weights[index];

    if (tf.overallSignal === 'BULLISH') {
      bullishCount++;
      weightedBullishScore += tf.signalStrength * weight;
    } else if (tf.overallSignal === 'BEARISH') {
      bearishCount++;
      weightedBearishScore += tf.signalStrength * weight;
    } else {
      neutralCount++;
    }
  });

  // Calculate alignment (how many timeframes agree)
  const alignment = Math.max(bullishCount, bearishCount, neutralCount);

  // Weighted average
  const totalWeight = weights.reduce((a, b) => a + b, 0);
  const avgBullishScore = weightedBullishScore / totalWeight;
  const avgBearishScore = weightedBearishScore / totalWeight;

  // Determine consensus signal
  let signal: MultiTimeframeAnalysis['consensus']['signal'];
  let strength: number;
  let interpretation: string;

  if (bullishCount >= 3 && avgBullishScore >= 70) {
    signal = 'STRONG_BULLISH';
    strength = Math.min(avgBullishScore, 100);
    interpretation = `Güçlü Yükseliş Sinyali! ${bullishCount}/4 timeframe yükselişte ve güç %${strength.toFixed(0)}. Tüm zaman dilimleri uyumlu.`;
  } else if (bullishCount >= 2 && avgBullishScore >= 60) {
    signal = 'BULLISH';
    strength = avgBullishScore;
    interpretation = `Yükseliş Sinyali. ${bullishCount}/4 timeframe yükselişte. ${alignment === 4 ? 'Tam uyum var!' : `Alignment: ${alignment}/4`}`;
  } else if (bearishCount >= 3 && avgBearishScore >= 70) {
    signal = 'STRONG_BEARISH';
    strength = Math.min(avgBearishScore, 100);
    interpretation = `Güçlü Düşüş Sinyali! ${bearishCount}/4 timeframe düşüşte ve güç %${strength.toFixed(0)}. Tüm zaman dilimleri uyumlu.`;
  } else if (bearishCount >= 2 && avgBearishScore >= 60) {
    signal = 'BEARISH';
    strength = avgBearishScore;
    interpretation = `Düşüş Sinyali. ${bearishCount}/4 timeframe düşüşte. ${alignment === 4 ? 'Tam uyum var!' : `Alignment: ${alignment}/4`}`;
  } else {
    signal = 'NEUTRAL';
    strength = 50;
    interpretation = `Karışık Sinyaller. Bullish: ${bullishCount}, Bearish: ${bearishCount}, Neutral: ${neutralCount}. Kesin trend yok.`;
  }

  return {
    signal,
    strength,
    alignment,
    interpretation,
  };
}

/**
 * Higher timeframe bias (1d ve 1w'den yönlülük belirle)
 */
function calculateHigherTimeframeBias(
  tf1d: TimeframeAnalysis,
  tf1w: TimeframeAnalysis
): 'BULLISH' | 'BEARISH' | 'NEUTRAL' {
  // 1w has 2x weight of 1d
  const weeklyWeight = 2;
  const dailyWeight = 1;

  let bullishScore = 0;
  let bearishScore = 0;

  if (tf1w.overallSignal === 'BULLISH') {
    bullishScore += weeklyWeight;
  } else if (tf1w.overallSignal === 'BEARISH') {
    bearishScore += weeklyWeight;
  }

  if (tf1d.overallSignal === 'BULLISH') {
    bullishScore += dailyWeight;
  } else if (tf1d.overallSignal === 'BEARISH') {
    bearishScore += dailyWeight;
  }

  if (bullishScore > bearishScore) {
    return 'BULLISH';
  } else if (bearishScore > bullishScore) {
    return 'BEARISH';
  } else {
    return 'NEUTRAL';
  }
}

// ============================================================================
// MAIN MULTI-TIMEFRAME ANALYZER
// ============================================================================

/**
 * Multi-timeframe analiz yap (1h, 4h, 1d, 1w)
 * ✅ GERÇEK VERİ: Binance kline data kullanır
 */
export async function analyzeMultiTimeframe(
  symbol: string
): Promise<MultiTimeframeAnalysis> {
  try {
    console.log(`[Multi-TF] Analyzing ${symbol} across 4 timeframes...`);

    // 1. Analyze all timeframes in parallel
    const [tf1h, tf4h, tf1d, tf1w] = await Promise.all([
      analyzeTimeframe(symbol, '1h'),
      analyzeTimeframe(symbol, '4h'),
      analyzeTimeframe(symbol, '1d'),
      analyzeTimeframe(symbol, '1w'),
    ]);

    console.log(`[Multi-TF] ${symbol} analyzed:`, {
      '1h': tf1h.overallSignal,
      '4h': tf4h.overallSignal,
      '1d': tf1d.overallSignal,
      '1w': tf1w.overallSignal,
    });

    // 2. Calculate consensus
    const consensus = calculateConsensus(tf1h, tf4h, tf1d, tf1w);

    // 3. Calculate higher timeframe bias
    const higherTimeframeBias = calculateHigherTimeframeBias(tf1d, tf1w);

    return {
      symbol,
      timeframes: {
        '1h': tf1h,
        '4h': tf4h,
        '1d': tf1d,
        '1w': tf1w,
      },
      consensus,
      higherTimeframeBias,
      timestamp: new Date().toISOString(),
    };

  } catch (error: any) {
    console.error(`[Multi-TF] Error analyzing ${symbol}:`, error.message);
    throw error;
  }
}

/**
 * Batch multi-timeframe analysis (birden fazla coin)
 */
export async function batchAnalyzeMultiTimeframe(
  symbols: string[],
  concurrency: number = 3
): Promise<Map<string, MultiTimeframeAnalysis>> {
  const results = new Map<string, MultiTimeframeAnalysis>();

  // Process in batches to avoid overwhelming Binance API
  for (let i = 0; i < symbols.length; i += concurrency) {
    const batch = symbols.slice(i, i + concurrency);

    const batchResults = await Promise.allSettled(
      batch.map(symbol => analyzeMultiTimeframe(symbol))
    );

    batchResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        results.set(batch[index], result.value);
      } else {
        console.warn(`[Multi-TF] Failed to analyze ${batch[index]}:`, result.reason);
      }
    });
  }

  return results;
}

// ============================================================================
// WHITE-HAT COMPLIANCE
// ============================================================================

export const MULTI_TIMEFRAME_CONFIG = {
  TIMEFRAMES: ['1h', '4h', '1d', '1w'] as Timeframe[],
  DEFAULT_SYMBOL: 'BTCUSDT',
  CONCURRENCY: 3, // Max 3 parallel requests
  CONSENSUS_THRESHOLD: {
    STRONG: 75,
    MODERATE: 60,
    WEAK: 50,
  },
};

console.log('✅ Multi-Timeframe Analysis Library initialized with White-Hat compliance');
console.log('⚠️ DISCLAIMER: For educational purposes only. Not financial advice.');
