/**
 * 📊 Volume Profile & Market Profile Indicators
 * Professional volume analysis for institutional trading
 *
 * Features:
 * - Volume Profile (VP)
 * - Point of Control (POC)
 * - Value Area (VA) - 70% of volume
 * - High Volume Nodes (HVN)
 * - Low Volume Nodes (LVN)
 * - Volume Weighted Average Price (VWAP) with bands
 * - Anchored VWAP
 * - Session Volume Profile
 */

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface VolumePriceLevel {
  price: number;
  volume: number;
  percentage: number;
}

export interface VolumeProfile {
  levels: VolumePriceLevel[];
  poc: number; // Point of Control (highest volume price)
  valueAreaHigh: number; // Top of 70% volume area
  valueAreaLow: number; // Bottom of 70% volume area
  totalVolume: number;
  highVolumeNodes: number[]; // Prices with high volume
  lowVolumeNodes: number[]; // Prices with low volume (potential breakout areas)
}

export interface Session {
  name: 'asian' | 'london' | 'newyork';
  startHour: number; // UTC hours
  endHour: number;
  color: string;
}

export interface SessionData {
  session: Session;
  high: number;
  low: number;
  open: number;
  close: number;
  volume: number;
  startTime: number;
  endTime: number;
}

export interface VWAPData {
  time: number;
  vwap: number;
  upperBand1: number; // +1 std dev
  upperBand2: number; // +2 std dev
  lowerBand1: number; // -1 std dev
  lowerBand2: number; // -2 std dev
}

/**
 * Calculate Volume Profile
 * Distributes volume across price levels
 */
export function calculateVolumeProfile(
  candles: CandleData[],
  priceLevels: number = 50
): VolumeProfile {
  if (candles.length === 0) {
    return {
      levels: [],
      poc: 0,
      valueAreaHigh: 0,
      valueAreaLow: 0,
      totalVolume: 0,
      highVolumeNodes: [],
      lowVolumeNodes: []
    };
  }

  const high = Math.max(...candles.map(c => c.high));
  const low = Math.min(...candles.map(c => c.low));
  const priceStep = (high - low) / priceLevels;

  // Initialize price levels
  const levels: VolumePriceLevel[] = [];
  for (let i = 0; i < priceLevels; i++) {
    const price = low + (i * priceStep) + (priceStep / 2);
    levels.push({ price, volume: 0, percentage: 0 });
  }

  // Distribute volume across price levels
  let totalVolume = 0;
  for (const candle of candles) {
    const candleRange = candle.high - candle.low;
    if (candleRange === 0) continue;

    const volumePerPrice = candle.volume / candleRange;

    for (const level of levels) {
      // Check if this price level is within the candle's range
      const priceInRange = level.price >= candle.low && level.price <= candle.high;
      if (priceInRange) {
        level.volume += volumePerPrice * priceStep;
      }
    }

    totalVolume += candle.volume;
  }

  // Calculate percentages
  levels.forEach(level => {
    level.percentage = totalVolume > 0 ? (level.volume / totalVolume) * 100 : 0;
  });

  // Find Point of Control (POC) - highest volume price
  const poc = levels.reduce((max, level) =>
    level.volume > max.volume ? level : max
  , levels[0]).price;

  // Calculate Value Area (70% of volume)
  const valueArea = calculateValueArea(levels, totalVolume);

  // Identify High and Low Volume Nodes
  const avgVolume = totalVolume / levels.length;
  const highVolumeNodes = levels
    .filter(l => l.volume > avgVolume * 1.5)
    .map(l => l.price);

  const lowVolumeNodes = levels
    .filter(l => l.volume < avgVolume * 0.5)
    .map(l => l.price);

  return {
    levels,
    poc,
    valueAreaHigh: valueArea.high,
    valueAreaLow: valueArea.low,
    totalVolume,
    highVolumeNodes,
    lowVolumeNodes
  };
}

/**
 * Calculate Value Area (70% of volume around POC)
 */
function calculateValueArea(
  levels: VolumePriceLevel[],
  totalVolume: number
): { high: number; low: number } {
  // Sort by volume descending
  const sorted = [...levels].sort((a, b) => b.volume - a.volume);

  let accumulatedVolume = 0;
  const targetVolume = totalVolume * 0.7;
  const vaLevels: VolumePriceLevel[] = [];

  for (const level of sorted) {
    vaLevels.push(level);
    accumulatedVolume += level.volume;
    if (accumulatedVolume >= targetVolume) break;
  }

  const prices = vaLevels.map(l => l.price);
  return {
    high: Math.max(...prices),
    low: Math.min(...prices)
  };
}

/**
 * Calculate VWAP with Standard Deviation Bands
 */
export function calculateVWAPWithBands(candles: CandleData[]): VWAPData[] {
  const vwapData: VWAPData[] = [];

  let cumulativeTPV = 0; // Typical Price × Volume
  let cumulativeVolume = 0;
  let cumulativeTPVSquared = 0;

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    const tpv = typicalPrice * candle.volume;

    cumulativeTPV += tpv;
    cumulativeVolume += candle.volume;
    cumulativeTPVSquared += (typicalPrice * typicalPrice * candle.volume);

    const vwap = cumulativeVolume > 0 ? cumulativeTPV / cumulativeVolume : typicalPrice;

    // Calculate variance and standard deviation
    const variance = cumulativeVolume > 0
      ? (cumulativeTPVSquared / cumulativeVolume) - (vwap * vwap)
      : 0;
    const stdDev = Math.sqrt(Math.max(0, variance));

    vwapData.push({
      time: candle.time,
      vwap,
      upperBand1: vwap + stdDev,
      upperBand2: vwap + (2 * stdDev),
      lowerBand1: vwap - stdDev,
      lowerBand2: vwap - (2 * stdDev)
    });
  }

  return vwapData;
}

/**
 * Detect Trading Sessions (Asian, London, New York)
 */
export const TRADING_SESSIONS: Session[] = [
  {
    name: 'asian',
    startHour: 0,  // 00:00 UTC
    endHour: 8,    // 08:00 UTC
    color: '#FFD700' // Gold
  },
  {
    name: 'london',
    startHour: 7,  // 07:00 UTC
    endHour: 16,   // 16:00 UTC
    color: '#00D4FF' // Cyan
  },
  {
    name: 'newyork',
    startHour: 13, // 13:00 UTC
    endHour: 22,   // 22:00 UTC
    color: '#10B981' // Green
  }
];

/**
 * Calculate Session Highs/Lows
 */
export function calculateSessionData(candles: CandleData[]): SessionData[] {
  const sessionData: SessionData[] = [];

  for (const session of TRADING_SESSIONS) {
    const sessionCandles = candles.filter(candle => {
      const date = new Date(candle.time);
      const hour = date.getUTCHours();

      if (session.startHour < session.endHour) {
        return hour >= session.startHour && hour < session.endHour;
      } else {
        // For sessions that cross midnight
        return hour >= session.startHour || hour < session.endHour;
      }
    });

    if (sessionCandles.length > 0) {
      const high = Math.max(...sessionCandles.map(c => c.high));
      const low = Math.min(...sessionCandles.map(c => c.low));
      const open = sessionCandles[0].open;
      const close = sessionCandles[sessionCandles.length - 1].close;
      const volume = sessionCandles.reduce((sum, c) => sum + c.volume, 0);

      sessionData.push({
        session,
        high,
        low,
        open,
        close,
        volume,
        startTime: sessionCandles[0].time,
        endTime: sessionCandles[sessionCandles.length - 1].time
      });
    }
  }

  return sessionData;
}

/**
 * Calculate Anchored VWAP from a specific time
 */
export function calculateAnchoredVWAP(
  candles: CandleData[],
  anchorTime: number
): VWAPData[] {
  const anchorIndex = candles.findIndex(c => c.time >= anchorTime);
  if (anchorIndex === -1) return [];

  const anchoredCandles = candles.slice(anchorIndex);
  return calculateVWAPWithBands(anchoredCandles);
}

/**
 * Identify Volume Clusters (areas of high volume concentration)
 */
export function identifyVolumeClusters(
  volumeProfile: VolumeProfile,
  threshold: number = 1.5 // 1.5x average volume
): { startPrice: number; endPrice: number; volume: number }[] {
  const { levels, totalVolume } = volumeProfile;
  const avgVolumePerLevel = totalVolume / levels.length;
  const clusters: { startPrice: number; endPrice: number; volume: number }[] = [];

  let inCluster = false;
  let clusterStart = 0;
  let clusterVolume = 0;

  for (let i = 0; i < levels.length; i++) {
    const level = levels[i];

    if (level.volume > avgVolumePerLevel * threshold) {
      if (!inCluster) {
        inCluster = true;
        clusterStart = i;
        clusterVolume = 0;
      }
      clusterVolume += level.volume;
    } else if (inCluster) {
      // End of cluster
      clusters.push({
        startPrice: levels[clusterStart].price,
        endPrice: levels[i - 1].price,
        volume: clusterVolume
      });
      inCluster = false;
    }
  }

  // Handle cluster at the end
  if (inCluster) {
    clusters.push({
      startPrice: levels[clusterStart].price,
      endPrice: levels[levels.length - 1].price,
      volume: clusterVolume
    });
  }

  return clusters;
}

/**
 * Calculate Delta (Buy Volume - Sell Volume estimation)
 * Uses close vs open to estimate buying/selling pressure
 */
export function calculateDelta(candles: CandleData[]): number[] {
  return candles.map(candle => {
    const isBullish = candle.close > candle.open;
    const volume = candle.volume;
    const bodySize = Math.abs(candle.close - candle.open);
    const range = candle.high - candle.low;
    const bodyRatio = range > 0 ? bodySize / range : 0;

    // Estimate buying/selling pressure
    const buyVolume = isBullish ? volume * bodyRatio : volume * (1 - bodyRatio);
    const sellVolume = volume - buyVolume;

    return buyVolume - sellVolume;
  });
}

/**
 * Calculate Cumulative Delta
 */
export function calculateCumulativeDelta(candles: CandleData[]): number[] {
  const deltas = calculateDelta(candles);
  const cumulativeDelta: number[] = [];
  let cumulative = 0;

  for (const delta of deltas) {
    cumulative += delta;
    cumulativeDelta.push(cumulative);
  }

  return cumulativeDelta;
}
