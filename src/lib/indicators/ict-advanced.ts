/**
 * 🎯 ICT (Inner Circle Trader) Advanced Indicators
 * Professional institutional trading concepts
 *
 * Features:
 * - Order Blocks (Bullish/Bearish)
 * - Fair Value Gaps (FVG) - Imbalance detection
 * - Breaker Blocks
 * - Mitigation Blocks
 * - Liquidity Pools (Buy/Sell Side Liquidity)
 * - Market Structure Breaks (MSB/BOS)
 * - Premium/Discount Zones
 * - Optimal Trade Entry (OTE)
 */

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OrderBlock {
  type: 'bullish' | 'bearish';
  time: number;
  startTime: number;
  endTime: number;
  high: number;
  low: number;
  volume: number;
  strength: number; // 1-100
  mitigated: boolean;
}

export interface FairValueGap {
  type: 'bullish' | 'bearish';
  startTime: number;
  endTime: number;
  high: number;
  low: number;
  filled: boolean;
  fillPercentage: number;
}

export interface BreakerBlock {
  type: 'bullish' | 'bearish';
  time: number;
  high: number;
  low: number;
  originalOrderBlock: OrderBlock;
  broken: boolean;
}

export interface LiquidityPool {
  type: 'buy_side' | 'sell_side';
  price: number;
  time: number;
  strength: number;
  swept: boolean;
}

export interface MarketStructure {
  type: 'higher_high' | 'higher_low' | 'lower_high' | 'lower_low';
  time: number;
  price: number;
  broken: boolean;
}

export interface PremiumDiscountZone {
  type: 'premium' | 'equilibrium' | 'discount';
  high: number;
  low: number;
  equilibrium: number;
  startTime: number;
  endTime: number;
}

/**
 * Detect Institutional Order Blocks
 * Order blocks are the last up/down candle before a strong move
 */
export function detectOrderBlocks(
  candles: CandleData[],
  lookback: number = 20,
  minVolume: number = 0
): OrderBlock[] {
  const orderBlocks: OrderBlock[] = [];

  for (let i = lookback; i < candles.length - 3; i++) {
    const current = candles[i];
    const _prev = candles[i - 1];
    const next1 = candles[i + 1];
    const next2 = candles[i + 2];
    const next3 = candles[i + 3];

    // Bullish Order Block: Last down candle before strong up move
    const isBullishOB =
      current.close < current.open && // Down candle
      next1.close > next1.open && // Up candle
      next2.close > next2.open && // Up candle
      next3.close > next3.open && // Up candle
      next3.close > current.high && // Strong move up
      current.volume > minVolume;

    if (isBullishOB) {
      const strength = calculateOBStrength(candles, i, 'bullish');
      orderBlocks.push({
        type: 'bullish',
        time: current.time,
        startTime: current.time,
        endTime: candles[candles.length - 1].time,
        high: current.high,
        low: current.low,
        volume: current.volume,
        strength,
        mitigated: false
      });
    }

    // Bearish Order Block: Last up candle before strong down move
    const isBearishOB =
      current.close > current.open && // Up candle
      next1.close < next1.open && // Down candle
      next2.close < next2.open && // Down candle
      next3.close < next3.open && // Down candle
      next3.close < current.low && // Strong move down
      current.volume > minVolume;

    if (isBearishOB) {
      const strength = calculateOBStrength(candles, i, 'bearish');
      orderBlocks.push({
        type: 'bearish',
        time: current.time,
        startTime: current.time,
        endTime: candles[candles.length - 1].time,
        high: current.high,
        low: current.low,
        volume: current.volume,
        strength,
        mitigated: false
      });
    }
  }

  // Check for mitigation
  return checkOrderBlockMitigation(orderBlocks, candles);
}

/**
 * Calculate Order Block Strength (1-100)
 */
function calculateOBStrength(
  candles: CandleData[],
  index: number,
  type: 'bullish' | 'bearish'
): number {
  const current = candles[index];
  let strength = 50; // Base strength

  // Volume factor
  const avgVolume = candles.slice(index - 20, index).reduce((sum, c) => sum + c.volume, 0) / 20;
  if (current.volume > avgVolume * 1.5) strength += 20;
  else if (current.volume > avgVolume * 1.2) strength += 10;

  // Body size factor
  const bodySize = Math.abs(current.close - current.open);
  const candleRange = current.high - current.low;
  const bodyRatio = bodySize / candleRange;
  if (bodyRatio > 0.7) strength += 15;
  else if (bodyRatio > 0.5) strength += 10;

  // Rejection wicks
  if (type === 'bullish') {
    const lowerWick = current.open - current.low;
    if (lowerWick > bodySize * 0.5) strength += 15;
  } else {
    const upperWick = current.high - current.open;
    if (upperWick > bodySize * 0.5) strength += 15;
  }

  return Math.min(100, Math.max(0, strength));
}

/**
 * Check if Order Blocks are mitigated (price returned to OB zone)
 */
function checkOrderBlockMitigation(
  orderBlocks: OrderBlock[],
  candles: CandleData[]
): OrderBlock[] {
  return orderBlocks.map(ob => {
    for (const candle of candles) {
      if (candle.time <= ob.time) continue;

      if (ob.type === 'bullish') {
        // Bullish OB mitigated if price returns to the zone
        if (candle.low <= ob.high && candle.low >= ob.low) {
          ob.mitigated = true;
          break;
        }
      } else {
        // Bearish OB mitigated if price returns to the zone
        if (candle.high >= ob.low && candle.high <= ob.high) {
          ob.mitigated = true;
          break;
        }
      }
    }
    return ob;
  });
}

/**
 * Detect Fair Value Gaps (FVG) - Imbalance
 * FVG occurs when there's a gap between candle 1 high and candle 3 low (or vice versa)
 */
export function detectFairValueGaps(candles: CandleData[]): FairValueGap[] {
  const fvgs: FairValueGap[] = [];

  for (let i = 1; i < candles.length - 1; i++) {
    const prev = candles[i - 1];
    const _current = candles[i];
    const next = candles[i + 1];

    // Bullish FVG: Gap between prev.high and next.low
    const bullishGap = next.low - prev.high;
    if (bullishGap > 0) {
      const fvg: FairValueGap = {
        type: 'bullish',
        startTime: prev.time,
        endTime: candles[candles.length - 1].time,
        high: next.low,
        low: prev.high,
        filled: false,
        fillPercentage: 0
      };

      // Check if filled
      for (let j = i + 2; j < candles.length; j++) {
        const fillCandle = candles[j];
        if (fillCandle.low <= fvg.low) {
          fvg.filled = true;
          fvg.fillPercentage = 100;
          break;
        } else if (fillCandle.low < fvg.high) {
          fvg.fillPercentage = ((fvg.high - fillCandle.low) / (fvg.high - fvg.low)) * 100;
        }
      }

      fvgs.push(fvg);
    }

    // Bearish FVG: Gap between next.high and prev.low
    const bearishGap = prev.low - next.high;
    if (bearishGap > 0) {
      const fvg: FairValueGap = {
        type: 'bearish',
        startTime: prev.time,
        endTime: candles[candles.length - 1].time,
        high: prev.low,
        low: next.high,
        filled: false,
        fillPercentage: 0
      };

      // Check if filled
      for (let j = i + 2; j < candles.length; j++) {
        const fillCandle = candles[j];
        if (fillCandle.high >= fvg.high) {
          fvg.filled = true;
          fvg.fillPercentage = 100;
          break;
        } else if (fillCandle.high > fvg.low) {
          fvg.fillPercentage = ((fillCandle.high - fvg.low) / (fvg.high - fvg.low)) * 100;
        }
      }

      fvgs.push(fvg);
    }
  }

  return fvgs;
}

/**
 * Detect Liquidity Pools (areas where stop losses cluster)
 * Buy Side Liquidity: Above swing highs
 * Sell Side Liquidity: Below swing lows
 */
export function detectLiquidityPools(
  candles: CandleData[],
  swingStrength: number = 5
): LiquidityPool[] {
  const pools: LiquidityPool[] = [];

  for (let i = swingStrength; i < candles.length - swingStrength; i++) {
    const current = candles[i];

    // Check for swing high (buy side liquidity above it)
    let isSwingHigh = true;
    for (let j = i - swingStrength; j <= i + swingStrength; j++) {
      if (j !== i && candles[j].high >= current.high) {
        isSwingHigh = false;
        break;
      }
    }

    if (isSwingHigh) {
      const strength = calculateLiquidityStrength(candles, i, 'buy_side');
      pools.push({
        type: 'buy_side',
        price: current.high,
        time: current.time,
        strength,
        swept: false
      });
    }

    // Check for swing low (sell side liquidity below it)
    let isSwingLow = true;
    for (let j = i - swingStrength; j <= i + swingStrength; j++) {
      if (j !== i && candles[j].low <= current.low) {
        isSwingLow = false;
        break;
      }
    }

    if (isSwingLow) {
      const strength = calculateLiquidityStrength(candles, i, 'sell_side');
      pools.push({
        type: 'sell_side',
        price: current.low,
        time: current.time,
        strength,
        swept: false
      });
    }
  }

  return checkLiquiditySweep(pools, candles);
}

/**
 * Calculate Liquidity Pool Strength
 */
function calculateLiquidityStrength(
  candles: CandleData[],
  index: number,
  type: 'buy_side' | 'sell_side'
): number {
  const current = candles[index];
  let strength = 50;

  // Volume factor
  const avgVolume = candles.slice(Math.max(0, index - 20), index).reduce((sum, c) => sum + c.volume, 0) / 20;
  if (current.volume > avgVolume * 2) strength += 30;
  else if (current.volume > avgVolume * 1.5) strength += 20;

  // Wick size (rejection)
  const _body = Math.abs(current.close - current.open);
  const range = current.high - current.low;

  if (type === 'buy_side') {
    const upperWick = current.high - Math.max(current.open, current.close);
    const wickRatio = upperWick / range;
    if (wickRatio > 0.5) strength += 20;
  } else {
    const lowerWick = Math.min(current.open, current.close) - current.low;
    const wickRatio = lowerWick / range;
    if (wickRatio > 0.5) strength += 20;
  }

  return Math.min(100, strength);
}

/**
 * Check if liquidity was swept
 */
function checkLiquiditySweep(
  pools: LiquidityPool[],
  candles: CandleData[]
): LiquidityPool[] {
  return pools.map(pool => {
    for (const candle of candles) {
      if (candle.time <= pool.time) continue;

      if (pool.type === 'buy_side' && candle.high >= pool.price) {
        pool.swept = true;
        break;
      } else if (pool.type === 'sell_side' && candle.low <= pool.price) {
        pool.swept = true;
        break;
      }
    }
    return pool;
  });
}

/**
 * Calculate Premium/Discount Zones
 * Based on swing high and swing low
 */
export function calculatePremiumDiscountZones(
  candles: CandleData[],
  lookback: number = 100
): PremiumDiscountZone | null {
  if (candles.length < lookback) return null;

  const recentCandles = candles.slice(-lookback);
  const high = Math.max(...recentCandles.map(c => c.high));
  const low = Math.min(...recentCandles.map(c => c.low));
  const equilibrium = (high + low) / 2;

  // 0.618 and 0.382 Fibonacci levels for premium/discount
  const _premiumLevel = equilibrium + (high - equilibrium) * 0.618;
  const _discountLevel = equilibrium - (equilibrium - low) * 0.618;

  return {
    type: 'equilibrium', // Will be determined by current price
    high,
    low,
    equilibrium,
    startTime: recentCandles[0].time,
    endTime: recentCandles[recentCandles.length - 1].time
  };
}

/**
 * Detect Market Structure (Higher Highs, Higher Lows, etc.)
 */
export function detectMarketStructure(
  candles: CandleData[],
  swingStrength: number = 5
): MarketStructure[] {
  const structures: MarketStructure[] = [];
  const swingHighs: { time: number; price: number }[] = [];
  const swingLows: { time: number; price: number }[] = [];

  // Find swing highs and lows
  for (let i = swingStrength; i < candles.length - swingStrength; i++) {
    const current = candles[i];

    // Swing High
    let isSwingHigh = true;
    for (let j = i - swingStrength; j <= i + swingStrength; j++) {
      if (j !== i && candles[j].high >= current.high) {
        isSwingHigh = false;
        break;
      }
    }
    if (isSwingHigh) {
      swingHighs.push({ time: current.time, price: current.high });
    }

    // Swing Low
    let isSwingLow = true;
    for (let j = i - swingStrength; j <= i + swingStrength; j++) {
      if (j !== i && candles[j].low <= current.low) {
        isSwingLow = false;
        break;
      }
    }
    if (isSwingLow) {
      swingLows.push({ time: current.time, price: current.low });
    }
  }

  // Determine market structure
  for (let i = 1; i < swingHighs.length; i++) {
    const current = swingHighs[i];
    const prev = swingHighs[i - 1];

    if (current.price > prev.price) {
      structures.push({
        type: 'higher_high',
        time: current.time,
        price: current.price,
        broken: false
      });
    } else if (current.price < prev.price) {
      structures.push({
        type: 'lower_high',
        time: current.time,
        price: current.price,
        broken: false
      });
    }
  }

  for (let i = 1; i < swingLows.length; i++) {
    const current = swingLows[i];
    const prev = swingLows[i - 1];

    if (current.price > prev.price) {
      structures.push({
        type: 'higher_low',
        time: current.time,
        price: current.price,
        broken: false
      });
    } else if (current.price < prev.price) {
      structures.push({
        type: 'lower_low',
        time: current.time,
        price: current.price,
        broken: false
      });
    }
  }

  return structures.sort((a, b) => a.time - b.time);
}
