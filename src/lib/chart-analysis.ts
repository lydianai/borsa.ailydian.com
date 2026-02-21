/**
 * Advanced Chart Analysis Library
 * Support/Resistance, Fibonacci, Order Blocks, FVG, Volume Profile
 */

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SupportResistanceLevel {
  price: number;
  strength: number;
  touches: number;
  type: 'support' | 'resistance';
  startTime: number;
  endTime: number;
}

export interface FibonacciLevel {
  level: number;
  price: number;
  label: string;
}

export interface OrderBlock {
  high: number;
  low: number;
  time: number;
  type: 'bullish' | 'bearish';
  strength: number;
}

export interface FairValueGap {
  high: number;
  low: number;
  startTime: number;
  endTime: number;
  type: 'bullish' | 'bearish';
  filled: boolean;
}

export interface VolumeProfile {
  price: number;
  volume: number;
  percentage: number;
}

/**
 * Detect Support and Resistance Levels
 */
export function detectSupportResistance(
  candles: Candle[],
  lookback: number = 20,
  touchTolerance: number = 0.002 // 0.2% tolerance
): SupportResistanceLevel[] {
  if (candles.length < lookback) return [];

  const levels: SupportResistanceLevel[] = [];
  const prices: number[] = [];

  // Collect all highs and lows as potential levels
  for (let i = 0; i < candles.length; i++) {
    const candle = candles[i];

    // Check if this is a local high
    let isLocalHigh = true;
    for (let j = Math.max(0, i - 5); j <= Math.min(candles.length - 1, i + 5); j++) {
      if (j !== i && candles[j].high > candle.high) {
        isLocalHigh = false;
        break;
      }
    }
    if (isLocalHigh) prices.push(candle.high);

    // Check if this is a local low
    let isLocalLow = true;
    for (let j = Math.max(0, i - 5); j <= Math.min(candles.length - 1, i + 5); j++) {
      if (j !== i && candles[j].low < candle.low) {
        isLocalLow = false;
        break;
      }
    }
    if (isLocalLow) prices.push(candle.low);
  }

  // Cluster similar prices
  const clusters: { price: number; count: number; times: number[] }[] = [];

  prices.forEach(price => {
    let found = false;
    for (const cluster of clusters) {
      if (Math.abs(price - cluster.price) / cluster.price < touchTolerance) {
        cluster.count++;
        cluster.price = (cluster.price * (cluster.count - 1) + price) / cluster.count;
        found = true;
        break;
      }
    }
    if (!found) {
      clusters.push({ price, count: 1, times: [] });
    }
  });

  // Convert clusters to levels
  clusters.forEach(cluster => {
    if (cluster.count >= 2) {
      const currentPrice = candles[candles.length - 1].close;
      const type: 'support' | 'resistance' = cluster.price < currentPrice ? 'support' : 'resistance';

      levels.push({
        price: cluster.price,
        strength: cluster.count,
        touches: cluster.count,
        type,
        startTime: candles[0].time,
        endTime: candles[candles.length - 1].time,
      });
    }
  });

  // Sort by strength
  return levels.sort((a, b) => b.strength - a.strength).slice(0, 10);
}

/**
 * Calculate Fibonacci Retracement Levels
 */
export function calculateFibonacci(
  highPrice: number,
  lowPrice: number,
  direction: 'uptrend' | 'downtrend' = 'uptrend'
): FibonacciLevel[] {
  const diff = highPrice - lowPrice;

  const levels = [
    { level: 0, label: '0%' },
    { level: 0.236, label: '23.6%' },
    { level: 0.382, label: '38.2%' },
    { level: 0.5, label: '50%' },
    { level: 0.618, label: '61.8%' },
    { level: 0.786, label: '78.6%' },
    { level: 1, label: '100%' },
  ];

  if (direction === 'uptrend') {
    return levels.map(l => ({
      level: l.level,
      price: highPrice - (diff * l.level),
      label: l.label,
    }));
  } else {
    return levels.map(l => ({
      level: l.level,
      price: lowPrice + (diff * l.level),
      label: l.label,
    }));
  }
}

/**
 * Detect Order Blocks (SMC)
 */
export function detectOrderBlocks(candles: Candle[], _lookback: number = 50): OrderBlock[] {
  if (candles.length < 10) return [];

  const orderBlocks: OrderBlock[] = [];

  for (let i = 3; i < candles.length - 1; i++) {
    const current = candles[i];
    const prev = candles[i - 1];
    const _prev2 = candles[i - 2];
    const _next = candles[i + 1];

    // Bullish Order Block
    // Look for: down candle followed by strong up move
    if (
      prev.close < prev.open && // Previous candle is bearish
      current.close > current.open && // Current candle is bullish
      current.close > prev.high && // Breaks previous high
      (current.close - current.open) > (prev.open - prev.close) * 1.5 // Strong move
    ) {
      orderBlocks.push({
        high: prev.high,
        low: prev.low,
        time: prev.time,
        type: 'bullish',
        strength: (current.close - current.open) / current.open,
      });
    }

    // Bearish Order Block
    // Look for: up candle followed by strong down move
    if (
      prev.close > prev.open && // Previous candle is bullish
      current.close < current.open && // Current candle is bearish
      current.close < prev.low && // Breaks previous low
      (current.open - current.close) > (prev.close - prev.open) * 1.5 // Strong move
    ) {
      orderBlocks.push({
        high: prev.high,
        low: prev.low,
        time: prev.time,
        type: 'bearish',
        strength: (current.open - current.close) / current.close,
      });
    }
  }

  return orderBlocks.slice(-10); // Keep last 10
}

/**
 * Detect Fair Value Gaps (FVG/Imbalance)
 */
export function detectFairValueGaps(candles: Candle[]): FairValueGap[] {
  if (candles.length < 3) return [];

  const gaps: FairValueGap[] = [];

  for (let i = 1; i < candles.length - 1; i++) {
    const prev = candles[i - 1];
    const current = candles[i];
    const next = candles[i + 1];

    // Bullish FVG: gap between prev low and next high
    if (prev.low > next.high) {
      const gap: FairValueGap = {
        high: prev.low,
        low: next.high,
        startTime: current.time,
        endTime: candles[candles.length - 1].time,
        type: 'bullish',
        filled: false,
      };

      // Check if gap is filled
      for (let j = i + 2; j < candles.length; j++) {
        if (candles[j].low <= gap.low) {
          gap.filled = true;
          gap.endTime = candles[j].time;
          break;
        }
      }

      gaps.push(gap);
    }

    // Bearish FVG: gap between prev high and next low
    if (prev.high < next.low) {
      const gap: FairValueGap = {
        high: next.low,
        low: prev.high,
        startTime: current.time,
        endTime: candles[candles.length - 1].time,
        type: 'bearish',
        filled: false,
      };

      // Check if gap is filled
      for (let j = i + 2; j < candles.length; j++) {
        if (candles[j].high >= gap.high) {
          gap.filled = true;
          gap.endTime = candles[j].time;
          break;
        }
      }

      gaps.push(gap);
    }
  }

  return gaps.filter(g => !g.filled).slice(-15); // Keep last 15 unfilled gaps
}

/**
 * Calculate Volume Profile
 */
export function calculateVolumeProfile(
  candles: Candle[],
  bins: number = 24
): VolumeProfile[] {
  if (candles.length === 0) return [];

  // Find price range
  const allPrices = candles.flatMap(c => [c.high, c.low]);
  const minPrice = Math.min(...allPrices);
  const maxPrice = Math.max(...allPrices);
  const priceRange = maxPrice - minPrice;
  const binSize = priceRange / bins;

  // Initialize bins
  const volumeBins: { price: number; volume: number }[] = [];
  for (let i = 0; i < bins; i++) {
    volumeBins.push({
      price: minPrice + (i + 0.5) * binSize,
      volume: 0,
    });
  }

  // Distribute volume to bins
  candles.forEach(candle => {
    const _candleRange = candle.high - candle.low;
    const avgPrice = (candle.high + candle.low) / 2;
    const binIndex = Math.min(
      Math.floor((avgPrice - minPrice) / binSize),
      bins - 1
    );

    if (binIndex >= 0 && binIndex < bins) {
      volumeBins[binIndex].volume += candle.volume;
    }
  });

  // Calculate total volume
  const totalVolume = volumeBins.reduce((sum, bin) => sum + bin.volume, 0);

  // Convert to profile
  return volumeBins.map(bin => ({
    price: bin.price,
    volume: bin.volume,
    percentage: totalVolume > 0 ? (bin.volume / totalVolume) * 100 : 0,
  }));
}

/**
 * Find swing highs and lows
 */
export function findSwingPoints(
  candles: Candle[],
  leftBars: number = 5,
  rightBars: number = 5
): { highs: { time: number; price: number }[]; lows: { time: number; price: number }[] } {
  const highs: { time: number; price: number }[] = [];
  const lows: { time: number; price: number }[] = [];

  for (let i = leftBars; i < candles.length - rightBars; i++) {
    const current = candles[i];

    // Check for swing high
    let isSwingHigh = true;
    for (let j = i - leftBars; j <= i + rightBars; j++) {
      if (j !== i && candles[j].high >= current.high) {
        isSwingHigh = false;
        break;
      }
    }
    if (isSwingHigh) {
      highs.push({ time: current.time, price: current.high });
    }

    // Check for swing low
    let isSwingLow = true;
    for (let j = i - leftBars; j <= i + rightBars; j++) {
      if (j !== i && candles[j].low <= current.low) {
        isSwingLow = false;
        break;
      }
    }
    if (isSwingLow) {
      lows.push({ time: current.time, price: current.low });
    }
  }

  return { highs, lows };
}

/**
 * Detect trend direction
 */
export function detectTrend(candles: Candle[], period: number = 20): 'uptrend' | 'downtrend' | 'sideways' {
  if (candles.length < period) return 'sideways';

  const recent = candles.slice(-period);
  const { highs, lows } = findSwingPoints(recent, 3, 3);

  if (highs.length < 2 || lows.length < 2) return 'sideways';

  // Check if highs are increasing
  const highsIncreasing = highs[highs.length - 1].price > highs[0].price;
  const lowsIncreasing = lows[lows.length - 1].price > lows[0].price;

  if (highsIncreasing && lowsIncreasing) return 'uptrend';
  if (!highsIncreasing && !lowsIncreasing) return 'downtrend';
  return 'sideways';
}

/**
 * Calculate Point of Control (POC) - highest volume price level
 */
export function calculatePOC(volumeProfile: VolumeProfile[]): number {
  if (volumeProfile.length === 0) return 0;

  const maxVolumePoint = volumeProfile.reduce((max, current) =>
    current.volume > max.volume ? current : max
  );

  return maxVolumePoint.price;
}

/**
 * Calculate Value Area (70% of volume)
 */
export function calculateValueArea(
  volumeProfile: VolumeProfile[]
): { high: number; low: number; poc: number } {
  if (volumeProfile.length === 0) {
    return { high: 0, low: 0, poc: 0 };
  }

  const poc = calculatePOC(volumeProfile);
  const pocIndex = volumeProfile.findIndex(p => p.price === poc);

  const totalVolume = volumeProfile.reduce((sum, p) => sum + p.volume, 0);
  const targetVolume = totalVolume * 0.7;

  let accumulatedVolume = volumeProfile[pocIndex].volume;
  let upperIndex = pocIndex;
  let lowerIndex = pocIndex;

  while (accumulatedVolume < targetVolume) {
    const upperVolume = upperIndex < volumeProfile.length - 1 ? volumeProfile[upperIndex + 1].volume : 0;
    const lowerVolume = lowerIndex > 0 ? volumeProfile[lowerIndex - 1].volume : 0;

    if (upperVolume > lowerVolume && upperIndex < volumeProfile.length - 1) {
      upperIndex++;
      accumulatedVolume += upperVolume;
    } else if (lowerIndex > 0) {
      lowerIndex--;
      accumulatedVolume += lowerVolume;
    } else {
      break;
    }
  }

  return {
    high: volumeProfile[upperIndex].price,
    low: volumeProfile[lowerIndex].price,
    poc,
  };
}
