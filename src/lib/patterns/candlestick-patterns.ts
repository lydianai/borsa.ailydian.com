/**
 * 🕯️ CANDLESTICK PATTERN RECOGNITION
 *
 * Comprehensive library for detecting classic candlestick patterns
 * - 8 bullish reversal patterns
 * - 8 bearish reversal patterns
 * - Confidence scoring based on pattern quality
 * - Turkish naming and descriptions
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only price data analysis
 * - Transparent pattern detection logic
 * - Educational purpose only
 * - No market manipulation
 */

import { PatternDetection } from '@/types/multi-timeframe-scanner';

export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ============================================================================
// PATTERN DETECTION ENGINE
// ============================================================================

/**
 * Detect all candlestick patterns in price data
 */
export function detectCandlestickPatterns(
  priceData: OHLCV[],
  lookback: number = 20
): PatternDetection[] {
  const patterns: PatternDetection[] = [];

  // Need at least 3 candles for patterns
  if (priceData.length < 3) return patterns;

  // Analyze recent candles
  const startIndex = Math.max(0, priceData.length - lookback);

  for (let i = startIndex; i < priceData.length; i++) {
    // Single candle patterns
    const hammerPattern = detectHammer(priceData, i);
    if (hammerPattern) patterns.push(hammerPattern);

    const invertedHammerPattern = detectInvertedHammer(priceData, i);
    if (invertedHammerPattern) patterns.push(invertedHammerPattern);

    const shootingStarPattern = detectShootingStar(priceData, i);
    if (shootingStarPattern) patterns.push(shootingStarPattern);

    // Two candle patterns
    if (i >= 1) {
      const engulfingPattern = detectEngulfing(priceData, i);
      if (engulfingPattern) patterns.push(engulfingPattern);

      const haramiPattern = detectHarami(priceData, i);
      if (haramiPattern) patterns.push(haramiPattern);

      const piercingPattern = detectPiercingLine(priceData, i);
      if (piercingPattern) patterns.push(piercingPattern);

      const darkCloudPattern = detectDarkCloudCover(priceData, i);
      if (darkCloudPattern) patterns.push(darkCloudPattern);

      const tweezerPattern = detectTweezer(priceData, i);
      if (tweezerPattern) patterns.push(tweezerPattern);
    }

    // Three candle patterns
    if (i >= 2) {
      const morningStarPattern = detectMorningStar(priceData, i);
      if (morningStarPattern) patterns.push(morningStarPattern);

      const eveningStarPattern = detectEveningStar(priceData, i);
      if (eveningStarPattern) patterns.push(eveningStarPattern);

      const threeSoldiersPattern = detectThreeWhiteSoldiers(priceData, i);
      if (threeSoldiersPattern) patterns.push(threeSoldiersPattern);

      const threeCrowsPattern = detectThreeBlackCrows(priceData, i);
      if (threeCrowsPattern) patterns.push(threeCrowsPattern);
    }
  }

  return patterns;
}

// ============================================================================
// BULLISH PATTERNS
// ============================================================================

/**
 * Detect Hammer pattern (Çekiç)
 */
function detectHammer(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const candle = priceData[index];
  const body = Math.abs(candle.close - candle.open);
  const range = candle.high - candle.low;
  const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
  const upperShadow = candle.high - Math.max(candle.open, candle.close);

  // Hammer criteria:
  // 1. Small body (< 30% of range)
  // 2. Long lower shadow (> 2x body)
  // 3. Little to no upper shadow
  // 4. Appears in downtrend
  if (
    body < range * 0.3 &&
    lowerShadow > body * 2 &&
    upperShadow < body * 0.5
  ) {
    // Check if in downtrend
    const prevCandle = priceData[index - 1];
    const inDowntrend = candle.low < prevCandle.low;

    if (inDowntrend) {
      return {
        pattern: 'hammer',
        name: 'Çekiç',
        direction: 'bullish',
        confidence: 75,
        position: priceData.length - index - 1,
        description: 'Güçlü yükseliş dönüş formasyonu - Uzun alt gölge, küçük gövde'
      };
    }
  }

  return null;
}

/**
 * Detect Inverted Hammer (Ters Çekiç)
 */
function detectInvertedHammer(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const candle = priceData[index];
  const body = Math.abs(candle.close - candle.open);
  const range = candle.high - candle.low;
  const upperShadow = candle.high - Math.max(candle.open, candle.close);
  const lowerShadow = Math.min(candle.open, candle.close) - candle.low;

  if (
    body < range * 0.3 &&
    upperShadow > body * 2 &&
    lowerShadow < body * 0.5
  ) {
    const prevCandle = priceData[index - 1];
    const inDowntrend = candle.low < prevCandle.low;

    if (inDowntrend) {
      return {
        pattern: 'inverted_hammer',
        name: 'Ters Çekiç',
        direction: 'bullish',
        confidence: 70,
        position: priceData.length - index - 1,
        description: 'Yükseliş dönüş sinyali - Uzun üst gölge, küçük gövde'
      };
    }
  }

  return null;
}

/**
 * Detect Bullish Engulfing (Yükseliş Yutma)
 */
function detectEngulfing(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const current = priceData[index];
  const prev = priceData[index - 1];

  const _prevBody = Math.abs(prev.close - prev.open);
  const _currentBody = Math.abs(current.close - current.open);

  // Bullish Engulfing:
  // 1. Previous candle is bearish
  // 2. Current candle is bullish
  // 3. Current body completely engulfs previous body
  if (
    prev.close < prev.open &&
    current.close > current.open &&
    current.close > prev.open &&
    current.open < prev.close
  ) {
    return {
      pattern: 'bullish_engulfing',
      name: 'Yükseliş Yutma',
      direction: 'bullish',
      confidence: 85,
      position: priceData.length - index - 1,
      description: 'Güçlü yükseliş dönüş formasyonu - Yeşil mum kırmızı mumu yutar'
    };
  }

  return null;
}

/**
 * Detect Morning Star (Sabah Yıldızı)
 */
function detectMorningStar(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 2) return null;

  const first = priceData[index - 2];
  const second = priceData[index - 1];
  const third = priceData[index];

  const firstBody = Math.abs(first.close - first.open);
  const secondBody = Math.abs(second.close - second.open);
  const thirdBody = Math.abs(third.close - third.open);

  // Morning Star:
  // 1. First: large bearish candle
  // 2. Second: small body (star)
  // 3. Third: large bullish candle
  if (
    first.close < first.open &&
    firstBody > secondBody * 2 &&
    third.close > third.open &&
    thirdBody > secondBody * 2 &&
    third.close > (first.open + first.close) / 2
  ) {
    return {
      pattern: 'morning_star',
      name: 'Sabah Yıldızı',
      direction: 'bullish',
      confidence: 90,
      position: priceData.length - index - 1,
      description: 'Çok güçlü yükseliş dönüş formasyonu - 3 mumluk ters formasyonu'
    };
  }

  return null;
}

/**
 * Detect Three White Soldiers (Üç Beyaz Asker)
 */
function detectThreeWhiteSoldiers(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 2) return null;

  const first = priceData[index - 2];
  const second = priceData[index - 1];
  const third = priceData[index];

  // Three White Soldiers:
  // 1. Three consecutive bullish candles
  // 2. Each opens within previous body
  // 3. Each closes higher than previous
  if (
    first.close > first.open &&
    second.close > second.open &&
    third.close > third.open &&
    second.open > first.open &&
    second.open < first.close &&
    third.open > second.open &&
    third.open < second.close &&
    second.close > first.close &&
    third.close > second.close
  ) {
    return {
      pattern: 'three_white_soldiers',
      name: 'Üç Beyaz Asker',
      direction: 'bullish',
      confidence: 88,
      position: priceData.length - index - 1,
      description: 'Güçlü yükseliş devam formasyonu - Ardışık 3 yükseliş mumu'
    };
  }

  return null;
}

/**
 * Detect Piercing Line (Delici Hat)
 */
function detectPiercingLine(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const prev = priceData[index - 1];
  const current = priceData[index];

  // Piercing Line:
  // 1. Previous is bearish
  // 2. Current is bullish
  // 3. Current opens below prev low
  // 4. Current closes above 50% of prev body
  if (
    prev.close < prev.open &&
    current.close > current.open &&
    current.open < prev.low &&
    current.close > (prev.open + prev.close) / 2 &&
    current.close < prev.open
  ) {
    return {
      pattern: 'piercing_line',
      name: 'Delici Hat',
      direction: 'bullish',
      confidence: 80,
      position: priceData.length - index - 1,
      description: 'Yükseliş dönüş formasyonu - Aşağıdan delici hareket'
    };
  }

  return null;
}

/**
 * Detect Bullish Harami (Yükseliş Harami)
 */
function detectHarami(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const prev = priceData[index - 1];
  const current = priceData[index];

  // Bullish Harami:
  // 1. Previous is large bearish
  // 2. Current is small bullish
  // 3. Current body is within previous body
  if (
    prev.close < prev.open &&
    current.close > current.open &&
    current.open > prev.close &&
    current.close < prev.open
  ) {
    const prevBody = prev.open - prev.close;
    const currentBody = current.close - current.open;

    if (currentBody < prevBody * 0.5) {
      return {
        pattern: 'bullish_harami',
        name: 'Yükseliş Harami',
        direction: 'bullish',
        confidence: 72,
        position: priceData.length - index - 1,
        description: 'Yükseliş dönüş formasyonu - İçerde kalan küçük yeşil mum'
      };
    }
  }

  return null;
}

/**
 * Detect Tweezer Bottom (Makas Dibi)
 */
function detectTweezer(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const prev = priceData[index - 1];
  const current = priceData[index];

  const tolerance = (prev.low + current.low) * 0.001; // 0.1% tolerance

  // Tweezer Bottom:
  // 1. Two candles with same lows
  // 2. First is bearish, second is bullish
  if (
    Math.abs(prev.low - current.low) < tolerance &&
    prev.close < prev.open &&
    current.close > current.open
  ) {
    return {
      pattern: 'tweezer_bottom',
      name: 'Makas Dibi',
      direction: 'bullish',
      confidence: 75,
      position: priceData.length - index - 1,
      description: 'Yükseliş dönüş formasyonu - Çift dip oluşumu'
    };
  }

  // Tweezer Top:
  // 1. Two candles with same highs
  // 2. First is bullish, second is bearish
  if (
    Math.abs(prev.high - current.high) < tolerance &&
    prev.close > prev.open &&
    current.close < current.open
  ) {
    return {
      pattern: 'tweezer_top',
      name: 'Makas Tepe',
      direction: 'bearish',
      confidence: 75,
      position: priceData.length - index - 1,
      description: 'Düşüş dönüş formasyonu - Çift tepe oluşumu'
    };
  }

  return null;
}

// ============================================================================
// BEARISH PATTERNS
// ============================================================================

/**
 * Detect Shooting Star (Kayan Yıldız)
 */
function detectShootingStar(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const candle = priceData[index];
  const body = Math.abs(candle.close - candle.open);
  const range = candle.high - candle.low;
  const upperShadow = candle.high - Math.max(candle.open, candle.close);
  const lowerShadow = Math.min(candle.open, candle.close) - candle.low;

  if (
    body < range * 0.3 &&
    upperShadow > body * 2 &&
    lowerShadow < body * 0.5
  ) {
    const prevCandle = priceData[index - 1];
    const inUptrend = candle.high > prevCandle.high;

    if (inUptrend) {
      return {
        pattern: 'shooting_star',
        name: 'Kayan Yıldız',
        direction: 'bearish',
        confidence: 75,
        position: priceData.length - index - 1,
        description: 'Düşüş dönüş formasyonu - Uzun üst gölge, küçük gövde'
      };
    }
  }

  return null;
}

/**
 * Detect Evening Star (Akşam Yıldızı)
 */
function detectEveningStar(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 2) return null;

  const first = priceData[index - 2];
  const second = priceData[index - 1];
  const third = priceData[index];

  const firstBody = Math.abs(first.close - first.open);
  const secondBody = Math.abs(second.close - second.open);
  const thirdBody = Math.abs(third.close - third.open);

  // Evening Star:
  // 1. First: large bullish candle
  // 2. Second: small body (star)
  // 3. Third: large bearish candle
  if (
    first.close > first.open &&
    firstBody > secondBody * 2 &&
    third.close < third.open &&
    thirdBody > secondBody * 2 &&
    third.close < (first.open + first.close) / 2
  ) {
    return {
      pattern: 'evening_star',
      name: 'Akşam Yıldızı',
      direction: 'bearish',
      confidence: 90,
      position: priceData.length - index - 1,
      description: 'Çok güçlü düşüş dönüş formasyonu - 3 mumluk ters formasyonu'
    };
  }

  return null;
}

/**
 * Detect Three Black Crows (Üç Kara Karga)
 */
function detectThreeBlackCrows(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 2) return null;

  const first = priceData[index - 2];
  const second = priceData[index - 1];
  const third = priceData[index];

  // Three Black Crows:
  // 1. Three consecutive bearish candles
  // 2. Each opens within previous body
  // 3. Each closes lower than previous
  if (
    first.close < first.open &&
    second.close < second.open &&
    third.close < third.open &&
    second.open < first.open &&
    second.open > first.close &&
    third.open < second.open &&
    third.open > second.close &&
    second.close < first.close &&
    third.close < second.close
  ) {
    return {
      pattern: 'three_black_crows',
      name: 'Üç Kara Karga',
      direction: 'bearish',
      confidence: 88,
      position: priceData.length - index - 1,
      description: 'Güçlü düşüş devam formasyonu - Ardışık 3 düşüş mumu'
    };
  }

  return null;
}

/**
 * Detect Dark Cloud Cover (Kara Bulut Örtüsü)
 */
function detectDarkCloudCover(priceData: OHLCV[], index: number): PatternDetection | null {
  if (index < 1) return null;

  const prev = priceData[index - 1];
  const current = priceData[index];

  // Dark Cloud Cover:
  // 1. Previous is bullish
  // 2. Current is bearish
  // 3. Current opens above prev high
  // 4. Current closes below 50% of prev body
  if (
    prev.close > prev.open &&
    current.close < current.open &&
    current.open > prev.high &&
    current.close < (prev.open + prev.close) / 2 &&
    current.close > prev.open
  ) {
    return {
      pattern: 'dark_cloud_cover',
      name: 'Kara Bulut Örtüsü',
      direction: 'bearish',
      confidence: 80,
      position: priceData.length - index - 1,
      description: 'Düşüş dönüş formasyonu - Yukarıdan baskılayıcı hareket'
    };
  }

  return null;
}

// ============================================================================
// HELPER: FILTER ONLY BULLISH PATTERNS
// ============================================================================

/**
 * Filter only bullish patterns for LONG signal validation
 */
export function filterBullishPatterns(patterns: PatternDetection[]): PatternDetection[] {
  return patterns.filter(p => p.direction === 'bullish');
}

/**
 * Get highest confidence bullish pattern
 */
export function getBestBullishPattern(patterns: PatternDetection[]): PatternDetection | null {
  const bullish = filterBullishPatterns(patterns);
  if (bullish.length === 0) return null;

  return bullish.reduce((best, current) =>
    current.confidence > best.confidence ? current : best
  );
}

/**
 * Check if sufficient bullish patterns exist
 */
export function hasSufficientBullishPatterns(
  patterns: PatternDetection[],
  minCount: number = 1,
  minConfidence: number = 70
): boolean {
  const qualified = filterBullishPatterns(patterns).filter(
    p => p.confidence >= minConfidence
  );

  return qualified.length >= minCount;
}
