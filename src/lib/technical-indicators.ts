/**
 * 📊 TECHNICAL INDICATORS LIBRARY
 * RSI, MACD, Bollinger Bands ve diğer teknik indikatörler
 *
 * ✅ GERÇEK VERİ: Binance kline (candlestick) data kullanır
 * ❌ Demo/Mock veri YOK
 *
 * White-Hat Compliance:
 * - Eğitim amaçlıdır
 * - Finansal tavsiye değildir
 * - Matematiksel hesaplamalar standart formüllere dayanır
 */

// ============================================================================
// INTERFACES
// ============================================================================

export interface Candlestick {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
}

export interface RSI {
  value: number;
  signal: 'OVERSOLD' | 'OVERBOUGHT' | 'NEUTRAL';
  interpretation: string;
}

export interface MACD {
  macdLine: number;
  signalLine: number;
  histogram: number;
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  interpretation: string;
}

export interface BollingerBands {
  upper: number;
  middle: number;
  lower: number;
  bandwidth: number;
  percentB: number;
  signal: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
  interpretation: string;
}

export interface TechnicalIndicators {
  rsi: RSI;
  macd: MACD;
  bollingerBands: BollingerBands;
  timestamp: string;
}

// ============================================================================
// RSI (Relative Strength Index) - 14 period default
// ============================================================================

/**
 * Calculate RSI
 * Formula: RSI = 100 - (100 / (1 + RS))
 * RS = Average Gain / Average Loss
 */
export function calculateRSI(prices: number[], period: number = 14): RSI {
  if (prices.length < period + 1) {
    return {
      value: 50,
      signal: 'NEUTRAL',
      interpretation: 'Yetersiz veri (RSI hesaplanamadı)',
    };
  }

  // Calculate price changes
  const changes: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  // Separate gains and losses
  const gains = changes.map(change => change > 0 ? change : 0);
  const losses = changes.map(change => change < 0 ? Math.abs(change) : 0);

  // Calculate initial average gain and loss
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;

  // Calculate smoothed averages (Wilder's smoothing method)
  for (let i = period; i < gains.length; i++) {
    avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
    avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
  }

  // Calculate RS and RSI
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  // Determine signal
  let signal: 'OVERSOLD' | 'OVERBOUGHT' | 'NEUTRAL';
  let interpretation: string;

  if (rsi <= 30) {
    signal = 'OVERSOLD';
    interpretation = `RSI ${rsi.toFixed(1)} - Aşırı Satım Bölgesi. Potansiyel yükseliş fırsatı.`;
  } else if (rsi >= 70) {
    signal = 'OVERBOUGHT';
    interpretation = `RSI ${rsi.toFixed(1)} - Aşırı Alım Bölgesi. Potansiyel düşüş riski.`;
  } else {
    signal = 'NEUTRAL';
    interpretation = `RSI ${rsi.toFixed(1)} - Nötr bölge. Belirgin bir aşırılık yok.`;
  }

  return {
    value: parseFloat(rsi.toFixed(2)),
    signal,
    interpretation,
  };
}

// ============================================================================
// MACD (Moving Average Convergence Divergence)
// ============================================================================

/**
 * Calculate EMA (Exponential Moving Average)
 */
function calculateEMA(prices: number[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // First EMA is SMA
  const sma = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;
  ema.push(sma);

  // Calculate EMA for remaining prices
  for (let i = period; i < prices.length; i++) {
    const currentEma = (prices[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
    ema.push(currentEma);
  }

  return ema;
}

/**
 * Calculate MACD
 * MACD Line = 12-period EMA - 26-period EMA
 * Signal Line = 9-period EMA of MACD Line
 * Histogram = MACD Line - Signal Line
 */
export function calculateMACD(
  prices: number[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACD {
  if (prices.length < slowPeriod + signalPeriod) {
    return {
      macdLine: 0,
      signalLine: 0,
      histogram: 0,
      signal: 'NEUTRAL',
      interpretation: 'Yetersiz veri (MACD hesaplanamadı)',
    };
  }

  // Calculate 12-period and 26-period EMAs
  const ema12 = calculateEMA(prices, fastPeriod);
  const ema26 = calculateEMA(prices, slowPeriod);

  // Calculate MACD Line
  const macdLine: number[] = [];
  const startIndex = slowPeriod - fastPeriod;
  for (let i = 0; i < ema26.length; i++) {
    macdLine.push(ema12[i + startIndex] - ema26[i]);
  }

  // Calculate Signal Line (9-period EMA of MACD)
  const signalLine = calculateEMA(macdLine, signalPeriod);

  // Get latest values
  const latestMacd = macdLine[macdLine.length - 1];
  const latestSignal = signalLine[signalLine.length - 1];
  const histogram = latestMacd - latestSignal;

  // Determine signal
  let signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  let interpretation: string;

  if (histogram > 0 && latestMacd > latestSignal) {
    signal = 'BULLISH';
    interpretation = `MACD Yükseliş Sinyali. MACD çizgisi sinyal çizgisinin üstünde (Histogram: ${histogram.toFixed(4)})`;
  } else if (histogram < 0 && latestMacd < latestSignal) {
    signal = 'BEARISH';
    interpretation = `MACD Düşüş Sinyali. MACD çizgisi sinyal çizgisinin altında (Histogram: ${histogram.toFixed(4)})`;
  } else {
    signal = 'NEUTRAL';
    interpretation = `MACD Nötr. Belirgin bir kesişim sinyali yok (Histogram: ${histogram.toFixed(4)})`;
  }

  return {
    macdLine: parseFloat(latestMacd.toFixed(4)),
    signalLine: parseFloat(latestSignal.toFixed(4)),
    histogram: parseFloat(histogram.toFixed(4)),
    signal,
    interpretation,
  };
}

// ============================================================================
// BOLLINGER BANDS
// ============================================================================

/**
 * Calculate SMA (Simple Moving Average)
 */
function calculateSMA(prices: number[], period: number): number {
  if (prices.length < period) return prices[prices.length - 1];
  const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
  return sum / period;
}

/**
 * Calculate Standard Deviation
 */
function calculateStdDev(prices: number[], period: number, sma: number): number {
  if (prices.length < period) return 0;
  const recentPrices = prices.slice(-period);
  const squaredDiffs = recentPrices.map(price => Math.pow(price - sma, 2));
  const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
  return Math.sqrt(variance);
}

/**
 * Calculate Bollinger Bands
 * Middle Band = 20-period SMA
 * Upper Band = Middle Band + (2 * std deviation)
 * Lower Band = Middle Band - (2 * std deviation)
 */
export function calculateBollingerBands(
  prices: number[],
  period: number = 20,
  stdDevMultiplier: number = 2
): BollingerBands {
  if (prices.length < period) {
    return {
      upper: prices[prices.length - 1],
      middle: prices[prices.length - 1],
      lower: prices[prices.length - 1],
      bandwidth: 0,
      percentB: 0.5,
      signal: 'NEUTRAL',
      interpretation: 'Yetersiz veri (Bollinger Bands hesaplanamadı)',
    };
  }

  const currentPrice = prices[prices.length - 1];
  const middle = calculateSMA(prices, period);
  const stdDev = calculateStdDev(prices, period, middle);
  const upper = middle + (stdDevMultiplier * stdDev);
  const lower = middle - (stdDevMultiplier * stdDev);

  // Calculate %B (where price is within the bands)
  const percentB = (currentPrice - lower) / (upper - lower);

  // Calculate Bandwidth
  const bandwidth = ((upper - lower) / middle) * 100;

  // Determine signal
  let signal: 'OVERBOUGHT' | 'OVERSOLD' | 'NEUTRAL';
  let interpretation: string;

  if (currentPrice >= upper) {
    signal = 'OVERBOUGHT';
    interpretation = `Fiyat üst banda dokundu ($${currentPrice.toFixed(2)} >= $${upper.toFixed(2)}). Aşırı alım bölgesi.`;
  } else if (currentPrice <= lower) {
    signal = 'OVERSOLD';
    interpretation = `Fiyat alt banda dokundu ($${currentPrice.toFixed(2)} <= $${lower.toFixed(2)}). Aşırı satım bölgesi.`;
  } else {
    const position = ((currentPrice - middle) / (upper - middle) * 100).toFixed(1);
    signal = 'NEUTRAL';
    interpretation = `Fiyat bantlar içinde (Orta noktadan %${position} uzaklıkta). Normal volatilite.`;
  }

  return {
    upper: parseFloat(upper.toFixed(2)),
    middle: parseFloat(middle.toFixed(2)),
    lower: parseFloat(lower.toFixed(2)),
    bandwidth: parseFloat(bandwidth.toFixed(2)),
    percentB: parseFloat(percentB.toFixed(3)),
    signal,
    interpretation,
  };
}

// ============================================================================
// BINANCE KLINE DATA FETCHER
// ============================================================================

/**
 * Fetch Binance Kline (Candlestick) Data
 * ✅ GERÇEK VERİ: Binance API'den gerçek candlestick verileri
 */
export async function fetchBinanceKlines(
  symbol: string,
  interval: string = '1h',
  limit: number = 100
): Promise<Candlestick[]> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      next: { revalidate: 60 }, // Cache for 60 seconds
    });

    if (!response.ok) {
      console.error(`[Tech Indicators] Binance kline fetch failed for ${symbol}: ${response.status}`);
      return [];
    }

    const data = await response.json();

    // Convert Binance kline format to our Candlestick interface
    return data.map((kline: any) => ({
      openTime: kline[0],
      open: parseFloat(kline[1]),
      high: parseFloat(kline[2]),
      low: parseFloat(kline[3]),
      close: parseFloat(kline[4]),
      volume: parseFloat(kline[5]),
      closeTime: kline[6],
    }));

  } catch (error) {
    console.error(`[Tech Indicators] Error fetching klines for ${symbol}:`, error);
    return [];
  }
}

/**
 * Calculate All Technical Indicators for a Symbol
 * ✅ GERÇEK VERİ: Binance kline data kullanır
 */
export async function calculateTechnicalIndicators(
  symbol: string,
  interval: string = '1h'
): Promise<TechnicalIndicators | null> {
  try {
    // Fetch real candlestick data from Binance
    const klines = await fetchBinanceKlines(symbol, interval, 100);

    if (klines.length === 0) {
      console.warn(`[Tech Indicators] No kline data for ${symbol}`);
      return null;
    }

    // Extract close prices for calculations
    const closePrices = klines.map(k => k.close);

    // Calculate all indicators
    const rsi = calculateRSI(closePrices, 14);
    const macd = calculateMACD(closePrices, 12, 26, 9);
    const bollingerBands = calculateBollingerBands(closePrices, 20, 2);

    return {
      rsi,
      macd,
      bollingerBands,
      timestamp: new Date().toISOString(),
    };

  } catch (error) {
    console.error(`[Tech Indicators] Error calculating indicators for ${symbol}:`, error);
    return null;
  }
}

// ============================================================================
// WHITE-HAT COMPLIANCE
// ============================================================================

export const TECHNICAL_INDICATORS_CONFIG = {
  RSI_PERIOD: 14,
  RSI_OVERSOLD: 30,
  RSI_OVERBOUGHT: 70,

  MACD_FAST: 12,
  MACD_SLOW: 26,
  MACD_SIGNAL: 9,

  BB_PERIOD: 20,
  BB_STD_DEV: 2,

  DEFAULT_INTERVAL: '1h',
  KLINE_LIMIT: 100,
};

console.log('✅ Technical Indicators Library initialized with White-Hat compliance');
console.log('⚠️ DISCLAIMER: For educational purposes only. Not financial advice.');
