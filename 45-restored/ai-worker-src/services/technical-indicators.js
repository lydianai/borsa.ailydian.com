/**
 * Technical Indicators
 *
 * RSI, MACD, Bollinger Bands, EMA, Volume Analysis
 * White-hat compliant: Standard financial indicators, ethical usage
 */

/**
 * Calculate RSI (Relative Strength Index)
 * @param {Array} prices - Close prices
 * @param {number} period - RSI period (default 14)
 * @returns {Array} - RSI values
 */
export function calculateRSI(prices, period = 14) {
  if (prices.length < period + 1) {
    throw new Error(`Insufficient data for RSI. Need at least ${period + 1} prices`);
  }

  const rsi = [];
  let gains = 0;
  let losses = 0;

  // First RSI calculation
  for (let i = 1; i <= period; i++) {
    const change = prices[i] - prices[i - 1];
    if (change >= 0) {
      gains += change;
    } else {
      losses -= change;
    }
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  // First RSI value
  let rs = avgGain / avgLoss;
  rsi.push(100 - (100 / (1 + rs)));

  // Subsequent RSI values (smoothed)
  for (let i = period + 1; i < prices.length; i++) {
    const change = prices[i] - prices[i - 1];
    const gain = change >= 0 ? change : 0;
    const loss = change < 0 ? -change : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    rs = avgGain / avgLoss;
    rsi.push(100 - (100 / (1 + rs)));
  }

  return rsi;
}

/**
 * Calculate EMA (Exponential Moving Average)
 * @param {Array} prices - Close prices
 * @param {number} period - EMA period
 * @returns {Array} - EMA values
 */
export function calculateEMA(prices, period) {
  if (prices.length < period) {
    throw new Error(`Insufficient data for EMA. Need at least ${period} prices`);
  }

  const k = 2 / (period + 1);
  const ema = [];

  // First EMA = SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += prices[i];
  }
  ema.push(sum / period);

  // Subsequent EMA values
  for (let i = period; i < prices.length; i++) {
    ema.push(prices[i] * k + ema[ema.length - 1] * (1 - k));
  }

  return ema;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 * @param {Array} prices - Close prices
 * @param {number} fastPeriod - Fast EMA period (default 12)
 * @param {number} slowPeriod - Slow EMA period (default 26)
 * @param {number} signalPeriod - Signal line period (default 9)
 * @returns {Object} - MACD line, signal line, histogram
 */
export function calculateMACD(prices, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  if (prices.length < slowPeriod) {
    throw new Error(`Insufficient data for MACD. Need at least ${slowPeriod} prices`);
  }

  const fastEMA = calculateEMA(prices, fastPeriod);
  const slowEMA = calculateEMA(prices, slowPeriod);

  // MACD line = fast EMA - slow EMA
  const macdLine = [];
  const offset = slowPeriod - fastPeriod;

  for (let i = 0; i < slowEMA.length; i++) {
    macdLine.push(fastEMA[i + offset] - slowEMA[i]);
  }

  // Signal line = EMA of MACD line
  const signalLine = calculateEMA(macdLine, signalPeriod);

  // Histogram = MACD line - signal line
  const histogram = [];
  const signalOffset = signalPeriod - 1;

  for (let i = 0; i < signalLine.length; i++) {
    histogram.push(macdLine[i + signalOffset] - signalLine[i]);
  }

  return {
    macd: macdLine,
    signal: signalLine,
    histogram: histogram
  };
}

/**
 * Calculate Bollinger Bands
 * @param {Array} prices - Close prices
 * @param {number} period - Period (default 20)
 * @param {number} stdDev - Standard deviations (default 2)
 * @returns {Object} - Upper, middle, lower bands
 */
export function calculateBollingerBands(prices, period = 20, stdDev = 2) {
  if (prices.length < period) {
    throw new Error(`Insufficient data for Bollinger Bands. Need at least ${period} prices`);
  }

  const middle = []; // SMA
  const upper = [];
  const lower = [];

  for (let i = period - 1; i < prices.length; i++) {
    // Calculate SMA (middle band)
    let sum = 0;
    for (let j = 0; j < period; j++) {
      sum += prices[i - j];
    }
    const sma = sum / period;
    middle.push(sma);

    // Calculate standard deviation
    let variance = 0;
    for (let j = 0; j < period; j++) {
      variance += Math.pow(prices[i - j] - sma, 2);
    }
    const std = Math.sqrt(variance / period);

    // Upper and lower bands
    upper.push(sma + stdDev * std);
    lower.push(sma - stdDev * std);
  }

  return {
    upper,
    middle,
    lower
  };
}

/**
 * Calculate volume profile
 * @param {Array} ohlcv - OHLCV data
 * @returns {Object} - Volume analysis
 */
export function calculateVolumeProfile(ohlcv) {
  if (ohlcv.length === 0) {
    throw new Error('Empty OHLCV data');
  }

  const volumes = ohlcv.map(c => c.volume);
  const avgVolume = volumes.reduce((a, b) => a + b, 0) / volumes.length;

  const currentVolume = volumes[volumes.length - 1];
  const volumeRatio = currentVolume / avgVolume;

  // Volume trend (increasing/decreasing)
  const recentVolumes = volumes.slice(-10);
  const volumeTrend = recentVolumes[recentVolumes.length - 1] > recentVolumes[0]
    ? 'increasing'
    : 'decreasing';

  return {
    currentVolume,
    avgVolume,
    volumeRatio,
    volumeTrend,
    isHighVolume: volumeRatio > 1.5,
    isLowVolume: volumeRatio < 0.5
  };
}

/**
 * Detect trend (simple)
 * @param {Array} prices - Close prices
 * @param {number} shortPeriod - Short EMA period
 * @param {number} longPeriod - Long EMA period
 * @returns {string} - 'bullish', 'bearish', or 'neutral'
 */
export function detectTrend(prices, shortPeriod = 9, longPeriod = 21) {
  if (prices.length < longPeriod) {
    return 'neutral';
  }

  const shortEMA = calculateEMA(prices, shortPeriod);
  const longEMA = calculateEMA(prices, longPeriod);

  const shortLatest = shortEMA[shortEMA.length - 1];
  const longLatest = longEMA[longEMA.length - 1];

  if (shortLatest > longLatest * 1.01) {
    return 'bullish';
  } else if (shortLatest < longLatest * 0.99) {
    return 'bearish';
  } else {
    return 'neutral';
  }
}

/**
 * Calculate all indicators for a symbol
 * @param {Array} ohlcv - OHLCV data
 * @returns {Object} - All technical indicators
 */
export function calculateAllIndicators(ohlcv) {
  if (!ohlcv || ohlcv.length < 50) {
    throw new Error('Insufficient data for indicators (need at least 50 candles)');
  }

  const closes = ohlcv.map(c => c.close);

  try {
    const rsi = calculateRSI(closes, 14);
    const macd = calculateMACD(closes, 12, 26, 9);
    const bollinger = calculateBollingerBands(closes, 20, 2);
    const ema9 = calculateEMA(closes, 9);
    const ema21 = calculateEMA(closes, 21);
    const ema50 = calculateEMA(closes, 50);
    const volume = calculateVolumeProfile(ohlcv);
    const trend = detectTrend(closes, 9, 21);

    return {
      rsi: {
        current: rsi[rsi.length - 1],
        values: rsi,
        signal: rsi[rsi.length - 1] > 70 ? 'overbought' :
                rsi[rsi.length - 1] < 30 ? 'oversold' : 'neutral'
      },
      macd: {
        current: macd.macd[macd.macd.length - 1],
        signal: macd.signal[macd.signal.length - 1],
        histogram: macd.histogram[macd.histogram.length - 1],
        crossover: macd.histogram[macd.histogram.length - 1] > 0 ? 'bullish' : 'bearish'
      },
      bollinger: {
        upper: bollinger.upper[bollinger.upper.length - 1],
        middle: bollinger.middle[bollinger.middle.length - 1],
        lower: bollinger.lower[bollinger.lower.length - 1],
        position: closes[closes.length - 1] > bollinger.upper[bollinger.upper.length - 1] ? 'above' :
                  closes[closes.length - 1] < bollinger.lower[bollinger.lower.length - 1] ? 'below' : 'inside'
      },
      ema: {
        ema9: ema9[ema9.length - 1],
        ema21: ema21[ema21.length - 1],
        ema50: ema50[ema50.length - 1]
      },
      volume,
      trend,
      timestamp: Date.now()
    };

  } catch (error) {
    console.error('Indicator calculation error:', error);
    throw error;
  }
}
