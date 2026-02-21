/**
 * 🎯 ADVANCED INDICATOR VALIDATOR
 *
 * Comprehensive validation library for all technical indicators
 * - Deep analysis with multiple confirmation layers
 * - Divergence detection for RSI/MFI
 * - Multi-condition validation for complex indicators
 * - Turkish explanations for all signals
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Transparent scoring algorithm
 * - Educational purpose only
 * - No market manipulation
 */

import {
  RSIValidation,
  MFIValidation,
  BollingerValidation,
  VWAPValidation,
  FVGValidation,
  OrderBlockValidation,
  LiquidityValidation,
  SupportResistanceValidation,
  FibonacciValidation,
  PremiumDiscountValidation,
  MarketStructureValidation,
  IndicatorAnalysis
} from '@/types/multi-timeframe-scanner';

// ============================================================================
// OHLCV DATA TYPE
// ============================================================================

export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ============================================================================
// RSI VALIDATOR
// ============================================================================

/**
 * Validate RSI with divergence detection
 */
export function validateRSI(
  rsiValues: number[],
  priceData: OHLCV[]
): RSIValidation {
  const currentRSI = rsiValues[rsiValues.length - 1];

  // Determine zone
  let zone: 'oversold' | 'neutral' | 'overbought';
  if (currentRSI < 30) zone = 'oversold';
  else if (currentRSI > 70) zone = 'overbought';
  else zone = 'neutral';

  // Detect divergence (last 10 candles)
  let divergence: 'bullish' | 'bearish' | undefined;
  if (rsiValues.length >= 10 && priceData.length >= 10) {
    const recentRSI = rsiValues.slice(-10);
    const recentPrices = priceData.slice(-10);

    // Bullish divergence: price lower lows, RSI higher lows
    const priceLowIndices = findLows(recentPrices.map(p => p.low));
    const rsiLowIndices = findLows(recentRSI);

    if (priceLowIndices.length >= 2 && rsiLowIndices.length >= 2) {
      const priceDecreasing = recentPrices[priceLowIndices[1]].low < recentPrices[priceLowIndices[0]].low;
      const rsiIncreasing = recentRSI[rsiLowIndices[1]] > recentRSI[rsiLowIndices[0]];

      if (priceDecreasing && rsiIncreasing) {
        divergence = 'bullish';
      }
    }

    // Bearish divergence: price higher highs, RSI lower highs
    const priceHighIndices = findHighs(recentPrices.map(p => p.high));
    const rsiHighIndices = findHighs(recentRSI);

    if (priceHighIndices.length >= 2 && rsiHighIndices.length >= 2) {
      const priceIncreasing = recentPrices[priceHighIndices[1]].high > recentPrices[priceHighIndices[0]].high;
      const rsiDecreasing = recentRSI[rsiHighIndices[1]] < recentRSI[rsiHighIndices[0]];

      if (priceIncreasing && rsiDecreasing && !divergence) {
        divergence = 'bearish';
      }
    }
  }

  // Generate signal
  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (zone === 'oversold') {
    signal = 'BUY';
    confidence = 70;
    reason = 'RSI aşırı satım bölgesinde (< 30)';

    if (divergence === 'bullish') {
      confidence = 90;
      reason = 'RSI aşırı satım bölgesinde + Yükseliş Diverjansı tespit edildi';
    }
  } else if (zone === 'overbought') {
    signal = 'SELL';
    confidence = 70;
    reason = 'RSI aşırı alım bölgesinde (> 70)';

    if (divergence === 'bearish') {
      confidence = 90;
      reason = 'RSI aşırı alım bölgesinde + Düşüş Diverjansı tespit edildi';
    }
  } else {
    // Neutral zone
    if (divergence === 'bullish') {
      signal = 'BUY';
      confidence = 60;
      reason = 'RSI nötr bölgede ancak Yükseliş Diverjansı mevcut';
    } else if (divergence === 'bearish') {
      signal = 'SELL';
      confidence = 60;
      reason = 'RSI nötr bölgede ancak Düşüş Diverjansı mevcut';
    } else {
      confidence = 40;
      reason = 'RSI nötr bölgede, net sinyal yok';
    }
  }

  return {
    value: currentRSI,
    zone,
    divergence,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// MFI VALIDATOR
// ============================================================================

/**
 * Validate MFI (Money Flow Index)
 */
export function validateMFI(
  mfiValues: number[],
  _volumeData: number[]
): MFIValidation {
  const currentMFI = mfiValues[mfiValues.length - 1];

  // Determine zone
  let zone: 'oversold' | 'neutral' | 'overbought';
  if (currentMFI < 20) zone = 'oversold';
  else if (currentMFI > 80) zone = 'overbought';
  else zone = 'neutral';

  // Calculate money flow trend (last 5 periods)
  let moneyFlowTrend: 'increasing' | 'decreasing' | 'stable' = 'stable';
  if (mfiValues.length >= 5) {
    const recent = mfiValues.slice(-5);
    const avgFirst = (recent[0] + recent[1]) / 2;
    const avgLast = (recent[3] + recent[4]) / 2;

    if (avgLast > avgFirst + 5) moneyFlowTrend = 'increasing';
    else if (avgLast < avgFirst - 5) moneyFlowTrend = 'decreasing';
  }

  // Generate signal
  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (zone === 'oversold') {
    signal = 'BUY';
    confidence = 75;
    reason = 'MFI aşırı satım bölgesinde (< 20), para akışı zayıf';

    if (moneyFlowTrend === 'increasing') {
      confidence = 85;
      reason = 'MFI aşırı satım bölgesinde + Para akışı artıyor';
    }
  } else if (zone === 'overbought') {
    signal = 'SELL';
    confidence = 75;
    reason = 'MFI aşırı alım bölgesinde (> 80), para akışı aşırı';

    if (moneyFlowTrend === 'decreasing') {
      confidence = 85;
      reason = 'MFI aşırı alım bölgesinde + Para akışı azalıyor';
    }
  } else {
    if (moneyFlowTrend === 'increasing') {
      signal = 'BUY';
      confidence = 55;
      reason = 'MFI nötr bölgede, para akışı artıyor';
    } else if (moneyFlowTrend === 'decreasing') {
      signal = 'SELL';
      confidence = 55;
      reason = 'MFI nötr bölgede, para akışı azalıyor';
    } else {
      confidence = 40;
      reason = 'MFI nötr bölgede, net sinyal yok';
    }
  }

  return {
    value: currentMFI,
    zone,
    moneyFlowTrend,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// BOLLINGER BANDS VALIDATOR
// ============================================================================

/**
 * Validate Bollinger Bands with squeeze detection
 */
export function validateBollingerBands(
  upper: number[],
  middle: number[],
  lower: number[],
  currentPrice: number
): BollingerValidation {
  const currentUpper = upper[upper.length - 1];
  const currentMiddle = middle[middle.length - 1];
  const currentLower = lower[lower.length - 1];

  // Determine position
  let position: 'above_upper' | 'near_upper' | 'middle' | 'near_lower' | 'below_lower';
  const upperThreshold = currentUpper - (currentUpper - currentMiddle) * 0.2;
  const lowerThreshold = currentLower + (currentMiddle - currentLower) * 0.2;

  if (currentPrice > currentUpper) position = 'above_upper';
  else if (currentPrice > upperThreshold) position = 'near_upper';
  else if (currentPrice < currentLower) position = 'below_lower';
  else if (currentPrice < lowerThreshold) position = 'near_lower';
  else position = 'middle';

  // Calculate bandwidth (volatility)
  const bandwidth = ((currentUpper - currentLower) / currentMiddle) * 100;

  // Detect squeeze (narrow bands = low volatility, potential breakout)
  const squeeze = bandwidth < 10; // Less than 10% = squeeze

  // Generate signal
  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (position === 'below_lower') {
    signal = 'BUY';
    confidence = 80;
    reason = 'Fiyat alt bandın altında, aşırı satım';

    if (squeeze) {
      confidence = 85;
      reason = 'Fiyat alt bandın altında + Bollinger Daralması (olası yükseliş hareketi)';
    }
  } else if (position === 'near_lower') {
    signal = 'BUY';
    confidence = 65;
    reason = 'Fiyat alt banda yakın, potansiyel destek';
  } else if (position === 'above_upper') {
    signal = 'SELL';
    confidence = 80;
    reason = 'Fiyat üst bandın üstünde, aşırı alım';
  } else if (position === 'near_upper') {
    signal = 'SELL';
    confidence = 65;
    reason = 'Fiyat üst banda yakın, potansiyel direnç';
  } else {
    if (squeeze) {
      signal = 'NEUTRAL';
      confidence = 45;
      reason = 'Fiyat orta bölgede + Bollinger Daralması (kırılım bekleniyor)';
    } else {
      confidence = 40;
      reason = 'Fiyat orta bölgede, net sinyal yok';
    }
  }

  return {
    upper: currentUpper,
    middle: currentMiddle,
    lower: currentLower,
    currentPrice,
    position,
    bandwidth,
    squeeze,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// VWAP VALIDATOR
// ============================================================================

/**
 * Validate VWAP (Volume Weighted Average Price)
 */
export function validateVWAP(
  vwapValue: number,
  currentPrice: number
): VWAPValidation {
  const deviation = ((currentPrice - vwapValue) / vwapValue) * 100;

  let position: 'above' | 'at' | 'below';
  if (Math.abs(deviation) < 0.5) position = 'at';
  else if (currentPrice > vwapValue) position = 'above';
  else position = 'below';

  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (position === 'below') {
    signal = 'BUY';
    confidence = 70;
    reason = `Fiyat VWAP'ın %${Math.abs(deviation).toFixed(2)} altında, potansiyel alım fırsatı`;

    if (deviation < -2) {
      confidence = 85;
      reason = `Fiyat VWAP'ın %${Math.abs(deviation).toFixed(2)} altında, güçlü alım fırsatı`;
    }
  } else if (position === 'above') {
    signal = 'SELL';
    confidence = 70;
    reason = `Fiyat VWAP'ın %${deviation.toFixed(2)} üstünde, potansiyel satış bölgesi`;

    if (deviation > 2) {
      confidence = 85;
      reason = `Fiyat VWAP'ın %${deviation.toFixed(2)} üstünde, güçlü satış bölgesi`;
    }
  } else {
    confidence = 45;
    reason = 'Fiyat VWAP seviyesinde, net sinyal yok';
  }

  return {
    value: vwapValue,
    currentPrice,
    position,
    deviation,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// FVG (FAIR VALUE GAP) VALIDATOR
// ============================================================================

/**
 * Validate FVG (Imbalances)
 */
export function validateFVG(
  priceData: OHLCV[]
): FVGValidation {
  const gaps: Array<{
    high: number;
    low: number;
    type: 'bullish' | 'bearish';
    filled: boolean;
    age: number;
  }> = [];

  // Detect FVG in last 20 candles
  for (let i = priceData.length - 20; i < priceData.length - 2; i++) {
    if (i < 0) continue;

    const prev = priceData[i];
    const current = priceData[i + 1];
    const next = priceData[i + 2];

    // Bullish FVG: gap between prev.high and next.low (current candle creates gap)
    if (next.low > prev.high && current.close > current.open) {
      const gap = {
        high: next.low,
        low: prev.high,
        type: 'bullish' as const,
        filled: false,
        age: priceData.length - (i + 2)
      };

      // Check if filled by subsequent price action
      for (let j = i + 3; j < priceData.length; j++) {
        if (priceData[j].low <= gap.low) {
          gap.filled = true;
          break;
        }
      }

      gaps.push(gap);
    }

    // Bearish FVG: gap between prev.low and next.high
    if (next.high < prev.low && current.close < current.open) {
      const gap = {
        high: prev.low,
        low: next.high,
        type: 'bearish' as const,
        filled: false,
        age: priceData.length - (i + 2)
      };

      for (let j = i + 3; j < priceData.length; j++) {
        if (priceData[j].high >= gap.high) {
          gap.filled = true;
          break;
        }
      }

      gaps.push(gap);
    }
  }

  const currentPrice = priceData[priceData.length - 1].close;

  // Find nearest unfilled gap
  const unfilledGaps = gaps.filter(g => !g.filled);
  let nearestGap: { distance: number; type: 'bullish' | 'bearish' } | undefined;

  if (unfilledGaps.length > 0) {
    const sorted = unfilledGaps
      .map(g => ({
        gap: g,
        distance: Math.abs(((g.low + g.high) / 2 - currentPrice) / currentPrice) * 100
      }))
      .sort((a, b) => a.distance - b.distance);

    nearestGap = {
      distance: sorted[0].distance,
      type: sorted[0].gap.type
    };
  }

  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (nearestGap) {
    if (nearestGap.type === 'bullish' && nearestGap.distance < 2) {
      signal = 'BUY';
      confidence = 75;
      reason = `Yükseliş FVG yakınında (%${nearestGap.distance.toFixed(2)}), boşluk dolum potansiyeli`;
    } else if (nearestGap.type === 'bearish' && nearestGap.distance < 2) {
      signal = 'SELL';
      confidence = 75;
      reason = `Düşüş FVG yakınında (%${nearestGap.distance.toFixed(2)}), boşluk dolum potansiyeli`;
    } else {
      confidence = 45;
      reason = 'FVG mevcut ancak uzakta, net sinyal yok';
    }
  } else {
    confidence = 40;
    reason = 'Yakında FVG tespit edilmedi';
  }

  return {
    gaps,
    nearestGap,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// ORDER BLOCK VALIDATOR
// ============================================================================

/**
 * Validate Order Blocks
 */
export function validateOrderBlocks(
  priceData: OHLCV[]
): OrderBlockValidation {
  const blocks: Array<{
    high: number;
    low: number;
    type: 'bullish' | 'bearish';
    tested: boolean;
    strength: number;
    age: number;
  }> = [];

  // Detect order blocks in last 30 candles
  for (let i = priceData.length - 30; i < priceData.length - 3; i++) {
    if (i < 1) continue;

    const candle = priceData[i];
    const nextCandles = priceData.slice(i + 1, i + 4);

    // Bullish OB: strong down candle followed by strong up move
    if (candle.close < candle.open) {
      const bodySize = candle.open - candle.close;
      const avgBody = (candle.high - candle.low) * 0.7;

      if (bodySize > avgBody) {
        // Check if followed by bullish move
        const upMove = nextCandles.filter(c => c.close > c.open).length >= 2;
        if (upMove) {
          const block = {
            high: candle.high,
            low: candle.low,
            type: 'bullish' as const,
            tested: false,
            strength: 0,
            age: priceData.length - (i + 1)
          };

          // Check if tested later
          let testCount = 0;
          for (let j = i + 4; j < priceData.length; j++) {
            if (priceData[j].low <= block.high && priceData[j].high >= block.low) {
              block.tested = true;
              testCount++;
            }
          }

          block.strength = Math.min(100, testCount * 25 + 50);
          blocks.push(block);
        }
      }
    }

    // Bearish OB: strong up candle followed by strong down move
    if (candle.close > candle.open) {
      const bodySize = candle.close - candle.open;
      const avgBody = (candle.high - candle.low) * 0.7;

      if (bodySize > avgBody) {
        const downMove = nextCandles.filter(c => c.close < c.open).length >= 2;
        if (downMove) {
          const block = {
            high: candle.high,
            low: candle.low,
            type: 'bearish' as const,
            tested: false,
            strength: 0,
            age: priceData.length - (i + 1)
          };

          let testCount = 0;
          for (let j = i + 4; j < priceData.length; j++) {
            if (priceData[j].low <= block.high && priceData[j].high >= block.low) {
              block.tested = true;
              testCount++;
            }
          }

          block.strength = Math.min(100, testCount * 25 + 50);
          blocks.push(block);
        }
      }
    }
  }

  const currentPrice = priceData[priceData.length - 1].close;

  // Find nearest block
  let nearestBlock: { distance: number; type: 'bullish' | 'bearish'; strength: number } | undefined;

  if (blocks.length > 0) {
    const sorted = blocks
      .map(b => ({
        block: b,
        distance: Math.abs(((b.low + b.high) / 2 - currentPrice) / currentPrice) * 100
      }))
      .sort((a, b) => a.distance - b.distance);

    nearestBlock = {
      distance: sorted[0].distance,
      type: sorted[0].block.type,
      strength: sorted[0].block.strength
    };
  }

  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (nearestBlock) {
    if (nearestBlock.type === 'bullish' && nearestBlock.distance < 1.5) {
      signal = 'BUY';
      confidence = 70 + (nearestBlock.strength * 0.2);
      reason = `Yükseliş Order Block yakınında (güç: ${nearestBlock.strength}/100), destek potansiyeli`;
    } else if (nearestBlock.type === 'bearish' && nearestBlock.distance < 1.5) {
      signal = 'SELL';
      confidence = 70 + (nearestBlock.strength * 0.2);
      reason = `Düşüş Order Block yakınında (güç: ${nearestBlock.strength}/100), direnç potansiyeli`;
    } else {
      confidence = 45;
      reason = 'Order Block mevcut ancak uzakta';
    }
  } else {
    confidence = 40;
    reason = 'Yakında Order Block tespit edilmedi';
  }

  return {
    blocks,
    nearestBlock,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// PREMIUM/DISCOUNT ZONE VALIDATOR
// ============================================================================

/**
 * Validate Premium/Discount zones
 */
export function validatePremiumDiscount(
  priceData: OHLCV[]
): PremiumDiscountValidation {
  // Find high and low of recent range (last 50 candles)
  const recentData = priceData.slice(-50);
  const high = Math.max(...recentData.map(c => c.high));
  const low = Math.min(...recentData.map(c => c.low));
  const equilibrium = (high + low) / 2;
  const currentPrice = priceData[priceData.length - 1].close;

  // Calculate percentage within range
  const percentage = ((currentPrice - low) / (high - low)) * 100;

  let zone: 'premium' | 'equilibrium' | 'discount';
  if (percentage > 55) zone = 'premium';
  else if (percentage < 45) zone = 'discount';
  else zone = 'equilibrium';

  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (zone === 'discount') {
    signal = 'BUY';
    confidence = 75;
    reason = `Fiyat İndirim Bölgesinde (%${percentage.toFixed(1)}), değerli alım fırsatı`;

    if (percentage < 30) {
      confidence = 85;
      reason = `Fiyat Derin İndirim Bölgesinde (%${percentage.toFixed(1)}), güçlü alım fırsatı`;
    }
  } else if (zone === 'premium') {
    signal = 'SELL';
    confidence = 75;
    reason = `Fiyat Premium Bölgesinde (%${percentage.toFixed(1)}), satış baskısı olası`;

    if (percentage > 70) {
      confidence = 85;
      reason = `Fiyat Yüksek Premium Bölgesinde (%${percentage.toFixed(1)}), güçlü satış bölgesi`;
    }
  } else {
    confidence = 45;
    reason = `Fiyat Denge Bölgesinde (%${percentage.toFixed(1)}), net sinyal yok`;
  }

  return {
    equilibrium,
    currentPrice,
    zone,
    percentage,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// MARKET STRUCTURE VALIDATOR
// ============================================================================

/**
 * Validate Market Structure (SMC - Smart Money Concepts)
 */
export function validateMarketStructure(
  priceData: OHLCV[]
): MarketStructureValidation {
  const swings = identifySwingPoints(priceData);

  let higherHighs = 0;
  let higherLows = 0;
  let lowerHighs = 0;
  let lowerLows = 0;

  // Analyze swing relationships
  for (let i = 1; i < swings.length; i++) {
    const prev = swings[i - 1];
    const current = swings[i];

    if (current.type === 'high' && prev.type === 'high') {
      if (current.price > prev.price) higherHighs++;
      else lowerHighs++;
    } else if (current.type === 'low' && prev.type === 'low') {
      if (current.price > prev.price) higherLows++;
      else lowerLows++;
    }
  }

  // Determine trend
  let trend: 'bullish' | 'bearish' | 'ranging';
  if (higherHighs >= 2 && higherLows >= 2) trend = 'bullish';
  else if (lowerHighs >= 2 && lowerLows >= 2) trend = 'bearish';
  else trend = 'ranging';

  // Detect ChoCh (Change of Character) - trend weakening
  let choch: { detected: boolean; type: 'bullish' | 'bearish'; position: number } | undefined;

  if (trend === 'bullish' && lowerLows > 0) {
    choch = { detected: true, type: 'bearish', position: swings.length - 1 };
  } else if (trend === 'bearish' && higherHighs > 0) {
    choch = { detected: true, type: 'bullish', position: swings.length - 1 };
  }

  // Detect BOS (Break of Structure) - strong trend continuation signal
  let bos: { detected: boolean; type: 'bullish' | 'bearish'; position: number } | undefined;

  if (trend === 'bullish' && higherHighs > 0) {
    bos = { detected: true, type: 'bullish', position: swings.length - 1 };
  } else if (trend === 'bearish' && lowerLows > 0) {
    bos = { detected: true, type: 'bearish', position: swings.length - 1 };
  }

  let signal: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;
  let reason = '';

  if (trend === 'bullish') {
    signal = 'BUY';
    confidence = 70;
    reason = `Yükseliş trendi (HH: ${higherHighs}, HL: ${higherLows})`;

    if (bos && bos.type === 'bullish') {
      confidence = 85;
      reason = `Güçlü yükseliş trendi + BOS tespit edildi`;
    }
  } else if (trend === 'bearish') {
    signal = 'SELL';
    confidence = 70;
    reason = `Düşüş trendi (LH: ${lowerHighs}, LL: ${lowerLows})`;

    if (bos && bos.type === 'bearish') {
      confidence = 85;
      reason = `Güçlü düşüş trendi + BOS tespit edildi`;
    }
  } else {
    if (choch) {
      signal = choch.type === 'bullish' ? 'BUY' : 'SELL';
      confidence = 60;
      reason = `Yatay piyasa + ${choch.type === 'bullish' ? 'Yükseliş' : 'Düşüş'} ChoCh tespit edildi`;
    } else {
      confidence = 40;
      reason = 'Yatay piyasa, net trend yok';
    }
  }

  return {
    trend,
    higherHighs,
    higherLows,
    lowerHighs,
    lowerLows,
    choch,
    bos,
    signal,
    confidence,
    reason
  };
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Find local lows in a data series
 */
function findLows(data: number[]): number[] {
  const lows: number[] = [];
  for (let i = 1; i < data.length - 1; i++) {
    if (data[i] < data[i - 1] && data[i] < data[i + 1]) {
      lows.push(i);
    }
  }
  return lows;
}

/**
 * Find local highs in a data series
 */
function findHighs(data: number[]): number[] {
  const highs: number[] = [];
  for (let i = 1; i < data.length - 1; i++) {
    if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
      highs.push(i);
    }
  }
  return highs;
}

/**
 * Identify swing high/low points
 */
function identifySwingPoints(priceData: OHLCV[]): Array<{ type: 'high' | 'low'; price: number; index: number }> {
  const swings: Array<{ type: 'high' | 'low'; price: number; index: number }> = [];

  for (let i = 2; i < priceData.length - 2; i++) {
    // Swing high
    if (
      priceData[i].high > priceData[i - 1].high &&
      priceData[i].high > priceData[i - 2].high &&
      priceData[i].high > priceData[i + 1].high &&
      priceData[i].high > priceData[i + 2].high
    ) {
      swings.push({ type: 'high', price: priceData[i].high, index: i });
    }

    // Swing low
    if (
      priceData[i].low < priceData[i - 1].low &&
      priceData[i].low < priceData[i - 2].low &&
      priceData[i].low < priceData[i + 1].low &&
      priceData[i].low < priceData[i + 2].low
    ) {
      swings.push({ type: 'low', price: priceData[i].low, index: i });
    }
  }

  return swings.sort((a, b) => a.index - b.index);
}
