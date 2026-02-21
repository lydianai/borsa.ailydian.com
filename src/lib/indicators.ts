/**
 * Technical Indicators Library
 * RSI, MFI and other technical analysis calculations
 */

interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Calculate RSI (Relative Strength Index)
 * @param candles - Array of candle data
 * @param period - RSI period (default 14)
 * @returns Array of RSI values with timestamps
 */
export function calculateRSI(candles: Candle[], period: number = 14): { time: number; value: number }[] {
  if (candles.length < period + 1) return [];

  const rsiValues: { time: number; value: number }[] = [];
  const changes: number[] = [];

  // Calculate price changes
  for (let i = 1; i < candles.length; i++) {
    changes.push(candles[i].close - candles[i - 1].close);
  }

  // Calculate initial average gain and loss
  let avgGain = 0;
  let avgLoss = 0;

  for (let i = 0; i < period; i++) {
    const change = changes[i];
    if (change > 0) {
      avgGain += change;
    } else {
      avgLoss += Math.abs(change);
    }
  }

  avgGain /= period;
  avgLoss /= period;

  // Calculate RSI for the first period
  let rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
  let rsi = 100 - (100 / (1 + rs));
  rsiValues.push({ time: candles[period].time, value: rsi });

  // Calculate RSI for remaining periods using smoothed averages
  for (let i = period; i < changes.length; i++) {
    const change = changes[i];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? Math.abs(change) : 0;

    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;

    rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
    rsi = 100 - (100 / (1 + rs));
    rsiValues.push({ time: candles[i + 1].time, value: rsi });
  }

  return rsiValues;
}

/**
 * Calculate MFI (Money Flow Index)
 * @param candles - Array of candle data
 * @param period - MFI period (default 14)
 * @returns Array of MFI values with timestamps
 */
export function calculateMFI(candles: Candle[], period: number = 14): { time: number; value: number }[] {
  if (candles.length < period + 1) return [];

  const mfiValues: { time: number; value: number }[] = [];
  const typicalPrices: number[] = [];
  const rawMoneyFlows: number[] = [];

  // Calculate typical price and raw money flow
  for (let i = 0; i < candles.length; i++) {
    const tp = (candles[i].high + candles[i].low + candles[i].close) / 3;
    typicalPrices.push(tp);
    rawMoneyFlows.push(tp * candles[i].volume);
  }

  // Calculate MFI for each period
  for (let i = period; i < candles.length; i++) {
    let positiveFlow = 0;
    let negativeFlow = 0;

    for (let j = i - period + 1; j <= i; j++) {
      if (typicalPrices[j] > typicalPrices[j - 1]) {
        positiveFlow += rawMoneyFlows[j];
      } else if (typicalPrices[j] < typicalPrices[j - 1]) {
        negativeFlow += rawMoneyFlows[j];
      }
    }

    const mfi = negativeFlow === 0 ? 100 : 100 - (100 / (1 + (positiveFlow / negativeFlow)));
    mfiValues.push({ time: candles[i].time, value: mfi });
  }

  return mfiValues;
}

/**
 * Calculate EMA (Exponential Moving Average)
 * @param values - Array of values
 * @param period - EMA period
 * @returns Array of EMA values
 */
export function calculateEMA(values: number[], period: number): number[] {
  if (values.length < period) return [];

  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Calculate initial SMA
  let sum = 0;
  for (let i = 0; i < period; i++) {
    sum += values[i];
  }
  ema.push(sum / period);

  // Calculate EMA for remaining values
  for (let i = period; i < values.length; i++) {
    const currentEMA = (values[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
    ema.push(currentEMA);
  }

  return ema;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 * @param candles - Array of candle data
 * @param fastPeriod - Fast EMA period (default 12)
 * @param slowPeriod - Slow EMA period (default 26)
 * @param signalPeriod - Signal line period (default 9)
 * @returns Object with MACD line, signal line, and histogram
 */
export function calculateMACD(
  candles: Candle[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): {
  macd: { time: number; value: number }[];
  signal: { time: number; value: number }[];
  histogram: { time: number; value: number }[];
} {
  const closePrices = candles.map(c => c.close);
  const fastEMA = calculateEMA(closePrices, fastPeriod);
  const slowEMA = calculateEMA(closePrices, slowPeriod);

  const macdLine: number[] = [];
  for (let i = 0; i < Math.min(fastEMA.length, slowEMA.length); i++) {
    macdLine.push(fastEMA[i + (fastPeriod - slowPeriod)] - slowEMA[i]);
  }

  const signalLine = calculateEMA(macdLine, signalPeriod);

  const macd: { time: number; value: number }[] = [];
  const signal: { time: number; value: number }[] = [];
  const histogram: { time: number; value: number }[] = [];

  const startIndex = slowPeriod - 1;
  for (let i = 0; i < signalLine.length; i++) {
    const timeIndex = startIndex + i;
    if (timeIndex < candles.length) {
      macd.push({ time: candles[timeIndex].time, value: macdLine[i + (signalPeriod - 1)] });
      signal.push({ time: candles[timeIndex].time, value: signalLine[i] });
      histogram.push({ time: candles[timeIndex].time, value: macdLine[i + (signalPeriod - 1)] - signalLine[i] });
    }
  }

  return { macd, signal, histogram };
}

/**
 * Calculate Bollinger Bands
 * @param candles - Array of candle data
 * @param period - Period for SMA and standard deviation (default 20)
 * @param stdDev - Number of standard deviations (default 2)
 * @returns Array of Bollinger Band values
 */
export function calculateBollingerBands(
  candles: Candle[],
  period: number = 20,
  stdDev: number = 2
): { time: number; upper: number; middle: number; lower: number }[] {
  if (candles.length < period) return [];

  const bands: { time: number; upper: number; middle: number; lower: number }[] = [];
  const closePrices = candles.map(c => c.close);

  for (let i = period - 1; i < closePrices.length; i++) {
    // Calculate SMA (middle band)
    let sum = 0;
    for (let j = i - period + 1; j <= i; j++) {
      sum += closePrices[j];
    }
    const sma = sum / period;

    // Calculate standard deviation
    let variance = 0;
    for (let j = i - period + 1; j <= i; j++) {
      variance += Math.pow(closePrices[j] - sma, 2);
    }
    const std = Math.sqrt(variance / period);

    bands.push({
      time: candles[i].time,
      upper: sma + stdDev * std,
      middle: sma,
      lower: sma - stdDev * std,
    });
  }

  return bands;
}
