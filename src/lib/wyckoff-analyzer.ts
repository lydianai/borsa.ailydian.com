/**
 * WYCKOFF METHOD ANALYZER
 * Real Wyckoff market cycle detection with volume profile and smart money analysis
 *
 * Wyckoff Market Cycle Phases:
 * 1. ACCUMULATION - Smart money accumulating, range-bound, low volatility
 * 2. MARKUP - Uptrend, strong buying, increasing volume
 * 3. DISTRIBUTION - Smart money distributing, range-bound at top, high volatility
 * 4. MARKDOWN - Downtrend, strong selling, panic volume
 *
 * Key Wyckoff Principles:
 * - Volume precedes price (volume analysis is critical)
 * - Price ranges indicate accumulation/distribution
 * - Test and confirmation (springs, upthrusts)
 * - Effort vs Result (volume vs price movement)
 */

interface OHLCVData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface VolumeProfile {
  averageVolume: number;
  currentVolume: number;
  volumeTrend: 'INCREASING' | 'DECREASING' | 'STABLE';
  volumeRatio: number;
  climaxVolume: boolean; // Buying/Selling climax detection
  dryUp: boolean; // Volume dry-up (low volume after high volume)
}

interface WyckoffPhase {
  phase: 'ACCUMULATION' | 'MARKUP' | 'DISTRIBUTION' | 'MARKDOWN' | 'UNKNOWN';
  confidence: number; // 0-100
  subPhase?: string; // PS (Preliminary Support), SC (Selling Climax), AR (Automatic Rally), etc.
  description: string;
  smartMoneyActivity: 'BUYING' | 'SELLING' | 'NEUTRAL';
  priceAction: string;
}

interface WyckoffAnalysis {
  wyckoffPhase: WyckoffPhase;
  volumeProfile: VolumeProfile;
  priceRange: {
    support: number;
    resistance: number;
    rangePercent: number;
    inRange: boolean;
  };
  trendStrength: number; // 0-100 (0 = strong down, 50 = sideways, 100 = strong up)
  effortVsResult: {
    effort: number; // Volume effort
    result: number; // Price result
    divergence: boolean; // High effort, low result = potential reversal
  };
  signal: 'BUY' | 'SELL' | 'WAIT';
  signalConfidence: number;
  recommendation: string;
}

/**
 * Fetch OHLCV data from Binance
 */
async function fetchOHLCV(symbol: string, interval: string = '1h', limit: number = 100): Promise<OHLCVData[]> {
  try {
    const response = await fetch(
      `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`,
      {
        headers: { 'User-Agent': 'Mozilla/5.0 (compatible; LyTradeAI/2.0)' },
        next: { revalidate: 300 }, // 5 min cache
      }
    );

    if (!response.ok) {
      console.error(`[Wyckoff Analyzer] Binance API error for ${symbol}: ${response.status}`);
      return [];
    }

    const klines: any[] = await response.json();
    if (!Array.isArray(klines) || klines.length === 0) {
      return [];
    }

    return klines.map(k => ({
      timestamp: k[0],
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5]),
    }));
  } catch (error: any) {
    console.error(`[Wyckoff Analyzer] Error fetching ${symbol}:`, error.message);
    return [];
  }
}

/**
 * Calculate Volume Profile
 */
function calculateVolumeProfile(ohlcv: OHLCVData[]): VolumeProfile {
  const volumes = ohlcv.map(d => d.volume);
  const averageVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;
  const currentVolume = volumes[volumes.length - 1];
  const volumeRatio = currentVolume / averageVolume;

  // Volume trend (last 10 vs previous 10)
  const recentVolume = volumes.slice(-10).reduce((a, b) => a + b, 0) / 10;
  const previousVolume = volumes.slice(-20, -10).reduce((a, b) => a + b, 0) / 10;
  const volumeChange = (recentVolume - previousVolume) / previousVolume;

  let volumeTrend: 'INCREASING' | 'DECREASING' | 'STABLE' = 'STABLE';
  if (volumeChange > 0.2) volumeTrend = 'INCREASING';
  else if (volumeChange < -0.2) volumeTrend = 'DECREASING';

  // Climax volume detection (extremely high volume)
  const maxVolume = Math.max(...volumes);
  const climaxVolume = currentVolume > maxVolume * 0.8 || volumeRatio > 3;

  // Volume dry-up (low volume after high volume period)
  const hadHighVolume = volumes.slice(-20, -5).some(v => v > averageVolume * 2);
  const recentLowVolume = volumes.slice(-5).every(v => v < averageVolume * 0.7);
  const dryUp = hadHighVolume && recentLowVolume;

  return {
    averageVolume,
    currentVolume,
    volumeTrend,
    volumeRatio,
    climaxVolume,
    dryUp,
  };
}

/**
 * Calculate Price Range (Support and Resistance)
 */
function calculatePriceRange(ohlcv: OHLCVData[], lookback: number = 50) {
  const recentData = ohlcv.slice(-lookback);
  const highs = recentData.map(d => d.high);
  const lows = recentData.map(d => d.low);
  const _currentPrice = ohlcv[ohlcv.length - 1].close;

  const resistance = Math.max(...highs);
  const support = Math.min(...lows);
  const rangePercent = ((resistance - support) / support) * 100;
  const inRange = rangePercent < 10; // Less than 10% range = consolidation

  return {
    support,
    resistance,
    rangePercent,
    inRange,
  };
}

/**
 * Calculate Trend Strength (0-100)
 * 0 = strong downtrend, 50 = sideways, 100 = strong uptrend
 */
function calculateTrendStrength(ohlcv: OHLCVData[]): number {
  const closes = ohlcv.map(d => d.close);
  const lookback = 20;
  const recentCloses = closes.slice(-lookback);

  // Calculate slope (linear regression)
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (let i = 0; i < recentCloses.length; i++) {
    sumX += i;
    sumY += recentCloses[i];
    sumXY += i * recentCloses[i];
    sumXX += i * i;
  }

  const n = recentCloses.length;
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const avgPrice = sumY / n;

  // Normalize slope to 0-100 scale
  const slopePercent = (slope / avgPrice) * 100 * 20; // Amplify for sensitivity
  const trendStrength = Math.max(0, Math.min(100, 50 + slopePercent));

  return trendStrength;
}

/**
 * Calculate Effort vs Result (Wyckoff principle)
 * High volume (effort) with small price change (result) = potential reversal
 */
function calculateEffortVsResult(ohlcv: OHLCVData[]): {
  effort: number;
  result: number;
  divergence: boolean;
} {
  const recent = ohlcv.slice(-5);
  const totalVolume = recent.reduce((sum, d) => sum + d.volume, 0);
  const avgVolume = ohlcv.slice(-20).reduce((sum, d) => sum + d.volume, 0) / 20;

  const effort = totalVolume / (avgVolume * 5); // Normalized effort

  const priceChange = Math.abs(recent[recent.length - 1].close - recent[0].close);
  const avgPrice = recent.reduce((sum, d) => sum + d.close, 0) / recent.length;
  const result = (priceChange / avgPrice) * 100; // Price change as percentage

  // Divergence: High effort (>1.5x avg) but low result (<2% move)
  const divergence = effort > 1.5 && result < 2;

  return {
    effort,
    result,
    divergence,
  };
}

/**
 * Detect Wyckoff Phase
 */
function detectWyckoffPhase(
  ohlcv: OHLCVData[],
  volumeProfile: VolumeProfile,
  priceRange: ReturnType<typeof calculatePriceRange>,
  trendStrength: number,
  effortVsResult: ReturnType<typeof calculateEffortVsResult>
): WyckoffPhase {
  const currentPrice = ohlcv[ohlcv.length - 1].close;
  const priceChangePercent = ((currentPrice - ohlcv[ohlcv.length - 2].close) / ohlcv[ohlcv.length - 2].close) * 100;

  // ACCUMULATION Phase Detection
  if (
    priceRange.inRange &&
    trendStrength < 55 &&
    trendStrength > 45 &&
    volumeProfile.dryUp &&
    effortVsResult.divergence
  ) {
    return {
      phase: 'ACCUMULATION',
      confidence: 80,
      subPhase: 'Spring Test',
      description: 'Akıllı para birikim yapıyor. Düşük hacimli konsolidasyon. Spring testi bekleniyor.',
      smartMoneyActivity: 'BUYING',
      priceAction: `Fiyat ${priceRange.support.toFixed(2)}-${priceRange.resistance.toFixed(2)} aralığında. Hacim azaldı (dry-up).`,
    };
  }

  // ACCUMULATION with Selling Climax
  if (
    priceRange.inRange &&
    volumeProfile.climaxVolume &&
    priceChangePercent < -2 &&
    trendStrength < 40
  ) {
    return {
      phase: 'ACCUMULATION',
      confidence: 85,
      subPhase: 'Selling Climax (SC)',
      description: 'Satış klimaksı! Panik satışlar, akıllı para alıma başladı. Reversal yakın.',
      smartMoneyActivity: 'BUYING',
      priceAction: `Yüksek hacimli düşüş (${priceChangePercent.toFixed(1)}%). Climax tespit edildi.`,
    };
  }

  // MARKUP Phase Detection
  if (
    trendStrength > 60 &&
    volumeProfile.volumeTrend === 'INCREASING' &&
    priceChangePercent > 1 &&
    !priceRange.inRange
  ) {
    return {
      phase: 'MARKUP',
      confidence: 85,
      description: 'Güçlü yükseliş trendi! Hacim artıyor, momentum devam ediyor.',
      smartMoneyActivity: 'BUYING',
      priceAction: `Trend gücü: ${trendStrength.toFixed(1)}/100. Hacim ${(volumeProfile.volumeRatio * 100).toFixed(0)}% ortalamanın üstünde.`,
    };
  }

  // DISTRIBUTION Phase Detection
  if (
    priceRange.inRange &&
    trendStrength > 55 &&
    trendStrength < 65 &&
    volumeProfile.volumeTrend === 'INCREASING' &&
    currentPrice > priceRange.support * 1.05 // Near top of range
  ) {
    return {
      phase: 'DISTRIBUTION',
      confidence: 75,
      subPhase: 'Preliminary Supply (PSY)',
      description: 'Akıllı para dağıtım yapıyor. Tepe bölgesinde yüksek hacim. DİKKAT!',
      smartMoneyActivity: 'SELLING',
      priceAction: `Fiyat direnç bölgesinde (${priceRange.resistance.toFixed(2)}). Hacim yüksek ama fiyat hareket etmiyor.`,
    };
  }

  // DISTRIBUTION with Buying Climax
  if (
    volumeProfile.climaxVolume &&
    priceChangePercent > 2 &&
    effortVsResult.divergence &&
    trendStrength > 70
  ) {
    return {
      phase: 'DISTRIBUTION',
      confidence: 90,
      subPhase: 'Buying Climax (BC)',
      description: 'Alım klimaksı! Euphoria zirvede, akıllı para satıyor. Reversal TEHLİKESİ!',
      smartMoneyActivity: 'SELLING',
      priceAction: `Yüksek hacim (${(volumeProfile.volumeRatio).toFixed(1)}x), düşük fiyat hareketi. Tehlikeli bölge!`,
    };
  }

  // MARKDOWN Phase Detection
  if (
    trendStrength < 40 &&
    volumeProfile.volumeTrend === 'INCREASING' &&
    priceChangePercent < -1 &&
    !priceRange.inRange
  ) {
    return {
      phase: 'MARKDOWN',
      confidence: 85,
      description: 'Güçlü düşüş trendi! Hacim artıyor, satış baskısı devam ediyor.',
      smartMoneyActivity: 'SELLING',
      priceAction: `Trend gücü: ${trendStrength.toFixed(1)}/100. Düşüş ${priceChangePercent.toFixed(1)}%.`,
    };
  }

  // UNKNOWN Phase (not clear)
  return {
    phase: 'UNKNOWN',
    confidence: 30,
    description: 'Wyckoff fazı belirsiz. Daha fazla fiyat aksiyonu bekleniyor.',
    smartMoneyActivity: 'NEUTRAL',
    priceAction: 'Net bir pattern tespit edilemedi.',
  };
}

/**
 * Main Wyckoff Analysis Function
 */
export async function analyzeWyckoff(symbol: string): Promise<WyckoffAnalysis> {
  try {
    console.log(`[Wyckoff Analyzer] Analyzing ${symbol}...`);

    // Fetch 100 hours of data (sufficient for Wyckoff analysis)
    const ohlcv = await fetchOHLCV(symbol, '1h', 100);

    if (ohlcv.length < 50) {
      throw new Error('Insufficient data for Wyckoff analysis');
    }

    // 1. Volume Profile
    const volumeProfile = calculateVolumeProfile(ohlcv);

    // 2. Price Range (Support/Resistance)
    const priceRange = calculatePriceRange(ohlcv, 50);

    // 3. Trend Strength
    const trendStrength = calculateTrendStrength(ohlcv);

    // 4. Effort vs Result
    const effortVsResult = calculateEffortVsResult(ohlcv);

    // 5. Wyckoff Phase Detection
    const wyckoffPhase = detectWyckoffPhase(ohlcv, volumeProfile, priceRange, trendStrength, effortVsResult);

    // 6. Generate Trading Signal
    let signal: 'BUY' | 'SELL' | 'WAIT' = 'WAIT';
    let signalConfidence = 50;
    let recommendation = '';

    if (wyckoffPhase.phase === 'ACCUMULATION' && wyckoffPhase.confidence > 75) {
      signal = 'BUY';
      signalConfidence = wyckoffPhase.confidence;
      recommendation = `🟢 ACCUMULATION PHASE: ${wyckoffPhase.description} Destek ${priceRange.support.toFixed(2)} yakınında alım fırsatı.`;
    } else if (wyckoffPhase.phase === 'MARKUP' && wyckoffPhase.confidence > 80 && trendStrength > 65) {
      signal = 'BUY';
      signalConfidence = Math.min(90, wyckoffPhase.confidence);
      recommendation = `🚀 MARKUP PHASE: ${wyckoffPhase.description} Trend güçlü, momentum devam ediyor.`;
    } else if (wyckoffPhase.phase === 'DISTRIBUTION' && wyckoffPhase.confidence > 70) {
      signal = 'SELL';
      signalConfidence = wyckoffPhase.confidence;
      recommendation = `🔴 DISTRIBUTION PHASE: ${wyckoffPhase.description} Direnç ${priceRange.resistance.toFixed(2)} yakınında çıkış zamanı.`;
    } else if (wyckoffPhase.phase === 'MARKDOWN' && wyckoffPhase.confidence > 80 && trendStrength < 35) {
      signal = 'SELL';
      signalConfidence = Math.min(95, wyckoffPhase.confidence);
      recommendation = `⬇️ MARKDOWN PHASE: ${wyckoffPhase.description} Güçlü düşüş, pozisyonları kapat.`;
    } else if (wyckoffPhase.phase === 'ACCUMULATION' && volumeProfile.dryUp) {
      signal = 'WAIT';
      signalConfidence = 60;
      recommendation = `⏸️ ACCUMULATION (Dry-Up): Hacim azaldı. Spring testi bekle, sonra alım yap.`;
    } else {
      signal = 'WAIT';
      signalConfidence = 40;
      recommendation = `⏸️ ${wyckoffPhase.phase} PHASE: ${wyckoffPhase.description} Net sinyal yok, bekle.`;
    }

    return {
      wyckoffPhase,
      volumeProfile,
      priceRange,
      trendStrength,
      effortVsResult,
      signal,
      signalConfidence,
      recommendation,
    };

  } catch (error: any) {
    console.error(`[Wyckoff Analyzer] Error analyzing ${symbol}:`, error);

    // Return safe default
    return {
      wyckoffPhase: {
        phase: 'UNKNOWN',
        confidence: 0,
        description: 'Analiz başarısız',
        smartMoneyActivity: 'NEUTRAL',
        priceAction: 'Veri yetersiz',
      },
      volumeProfile: {
        averageVolume: 0,
        currentVolume: 0,
        volumeTrend: 'STABLE',
        volumeRatio: 1,
        climaxVolume: false,
        dryUp: false,
      },
      priceRange: {
        support: 0,
        resistance: 0,
        rangePercent: 0,
        inRange: false,
      },
      trendStrength: 50,
      effortVsResult: {
        effort: 0,
        result: 0,
        divergence: false,
      },
      signal: 'WAIT',
      signalConfidence: 0,
      recommendation: 'Analiz başarısız, işlem yapma.',
    };
  }
}

/**
 * Batch Wyckoff Analysis for multiple symbols
 */
export async function batchAnalyzeWyckoff(
  symbols: string[],
  maxConcurrent: number = 10
): Promise<Map<string, WyckoffAnalysis>> {
  const results = new Map<string, WyckoffAnalysis>();

  // Process in batches
  for (let i = 0; i < symbols.length; i += maxConcurrent) {
    const batch = symbols.slice(i, i + maxConcurrent);
    const batchPromises = batch.map(symbol => analyzeWyckoff(symbol));
    const batchResults = await Promise.all(batchPromises);

    batch.forEach((symbol, index) => {
      results.set(symbol, batchResults[index]);
    });
  }

  return results;
}
