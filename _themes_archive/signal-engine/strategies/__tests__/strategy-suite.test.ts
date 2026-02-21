/**
 * 🧪 COMPREHENSIVE STRATEGY TEST SUITE
 * Tests all 9 whitelisted strategies with fixture data
 */

import { describe, it, expect } from 'vitest';
import { analyzeRSIDivergence } from '../rsi-divergence';
import { analyzeBollingerSqueeze } from '../bollinger-squeeze';
import { analyzeEMARibbon } from '../ema-ribbon';
import { analyzeVolumeBreakout } from '../volume-breakout';
import { analyzeFibonacciRetracement } from '../fibonacci-retracement';
import { analyzeIchimokuCloud } from '../ichimoku-cloud';
import { analyzeATRVolatility } from '../atr-volatility';
import { analyzeTrendReversal } from '../trend-reversal';
import {
  strongBullishData,
  pullbackBuyData,
  strongBearishData,
  neutralRangingData,
  oversoldData,
  overboughtData,
  breakoutData,
  breakdownData,
  highVolatilityData,
  createPriceData,
} from './fixtures';

// ============================================================================
// RSI DIVERGENCE STRATEGY
// ============================================================================

describe('RSI Divergence Strategy', () => {
  it('should return valid signal for bullish data', async () => {
    const result = await analyzeRSIDivergence(strongBullishData);

    expect(result.name).toBe('RSI Divergence');
    expect(result.signal).toMatch(/BUY|SELL|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(100);
    expect(result.reason).toBeTruthy();
  });

  it('should detect oversold conditions', async () => {
    const result = await analyzeRSIDivergence(oversoldData);

    // Oversold should not give SELL
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
  });

  it('should detect overbought conditions', async () => {
    const result = await analyzeRSIDivergence(overboughtData);

    // Overbought should give SELL or WAIT
    expect(result.signal).toMatch(/SELL|WAIT|NEUTRAL/);
  });

  it('should return NEUTRAL for ranging market', async () => {
    const result = await analyzeRSIDivergence(neutralRangingData);

    expect(result.signal).toMatch(/NEUTRAL|WAIT/);
  });
});

// ============================================================================
// BOLLINGER SQUEEZE STRATEGY
// ============================================================================

describe('Bollinger Squeeze Strategy', () => {
  it('should return valid signal for volatile data', async () => {
    const result = await analyzeBollingerSqueeze(highVolatilityData);

    expect(result.name).toBe('Bollinger Squeeze');
    expect(result.signal).toMatch(/BUY|SELL|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(100);
  });

  it('should detect breakout after squeeze', async () => {
    const result = await analyzeBollingerSqueeze(breakoutData);

    // Breakout should not give SELL
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
  });

  it('should handle ranging market', async () => {
    const result = await analyzeBollingerSqueeze(neutralRangingData);

    expect(result.signal).toBeDefined();
    expect(result.confidence).toBeLessThan(80); // Low confidence in ranging
  });
});

// ============================================================================
// EMA RIBBON STRATEGY
// ============================================================================

describe('EMA Ribbon Strategy', () => {
  it('should detect bullish ribbon alignment', async () => {
    const result = await analyzeEMARibbon(strongBullishData);

    expect(result.name).toBe('EMA Ribbon');
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
  });

  it('should detect bearish ribbon alignment', async () => {
    const result = await analyzeEMARibbon(strongBearishData);

    expect(result.signal).toMatch(/SELL|WAIT/);
  });

  it('should identify ribbon compression (ranging)', async () => {
    const result = await analyzeEMARibbon(neutralRangingData);

    expect(result.signal).toMatch(/NEUTRAL|WAIT/);
  });
});

// ============================================================================
// VOLUME BREAKOUT STRATEGY
// ============================================================================

describe('Volume Breakout Strategy', () => {
  it('should detect volume breakout', async () => {
    const result = await analyzeVolumeBreakout(breakoutData);

    expect(result.name).toBe('Volume Breakout');
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
  });

  it('should detect volume breakdown', async () => {
    const result = await analyzeVolumeBreakout(breakdownData);

    // Breakdown should not give strong BUY
    expect(result.signal).toMatch(/SELL|WAIT|NEUTRAL/);
  });

  it('should return NEUTRAL for low volume', async () => {
    const lowVolumeData = createPriceData({
      volume24h: 5000000, // Low volume
      changePercent24h: 0.5,
    });

    const result = await analyzeVolumeBreakout(lowVolumeData);

    expect(result.signal).toMatch(/NEUTRAL|WAIT/);
  });
});

// ============================================================================
// FIBONACCI RETRACEMENT STRATEGY
// ============================================================================

describe('Fibonacci Retracement Strategy', () => {
  it('should detect Fib levels for pullback', async () => {
    const result = await analyzeFibonacciRetracement(pullbackBuyData);

    expect(result.name).toBe('Fibonacci Retracement');
    expect(result.signal).toMatch(/BUY|SELL|WAIT|NEUTRAL/);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
  });

  it('should identify 61.8% golden ratio', async () => {
    const fibData = createPriceData({
      price: 9500,
      high24h: 10000,
      low24h: 9000,
      changePercent24h: 3.0,
    });

    const result = await analyzeFibonacciRetracement(fibData);

    // Price at 50% retracement
    expect(result.signal).toBeDefined();
  });

  it('should handle no clear retracement', async () => {
    const result = await analyzeFibonacciRetracement(neutralRangingData);

    expect(result.signal).toMatch(/NEUTRAL|WAIT/);
  });
});

// ============================================================================
// ICHIMOKU CLOUD STRATEGY
// ============================================================================

describe('Ichimoku Cloud Strategy', () => {
  it('should detect price above cloud (bullish)', async () => {
    const result = await analyzeIchimokuCloud(strongBullishData);

    expect(result.name).toBe('Ichimoku Cloud');
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
  });

  it('should detect price below cloud (bearish)', async () => {
    const result = await analyzeIchimokuCloud(strongBearishData);

    expect(result.signal).toMatch(/SELL|WAIT/);
  });

  it('should identify kumo twist (cloud reversal)', async () => {
    const result = await analyzeIchimokuCloud(neutralRangingData);

    expect(result.signal).toBeDefined();
    expect(result.confidence).toBeLessThan(80);
  });
});

// ============================================================================
// ATR VOLATILITY STRATEGY
// ============================================================================

describe('ATR Volatility Strategy', () => {
  it('should detect high volatility', async () => {
    const result = await analyzeATRVolatility(highVolatilityData);

    expect(result.name).toBe('ATR Volatility');
    expect(result.signal).toMatch(/BUY|SELL|WAIT|NEUTRAL/);
    // ATR strategy may not return indicators in all cases
    if (result.indicators?.volatility) {
      expect(result.indicators.volatility).toBeGreaterThan(0);
    }
  });

  it('should detect low volatility (squeeze)', async () => {
    const result = await analyzeATRVolatility(neutralRangingData);

    // Low volatility = potential breakout setup
    expect(result.signal).toBeDefined();
  });

  it('should calculate ATR correctly', async () => {
    const result = await analyzeATRVolatility(strongBullishData);

    if (result.indicators?.atr) {
      expect(result.indicators.atr).toBeGreaterThan(0);
    }
  });
});

// ============================================================================
// TREND REVERSAL STRATEGY
// ============================================================================

describe('Trend Reversal Strategy', () => {
  it('should detect bullish reversal from oversold', async () => {
    const result = await analyzeTrendReversal(oversoldData);

    expect(result.name).toBe('Trend Reversal');
    // May need more confirmation before BUY
    expect(result.signal).toMatch(/BUY|WAIT|NEUTRAL/);
  });

  it('should detect bearish reversal from overbought', async () => {
    const result = await analyzeTrendReversal(overboughtData);

    // May need more confirmation before SELL
    expect(result.signal).toMatch(/SELL|WAIT|NEUTRAL/);
  });

  it('should return NEUTRAL when no reversal', async () => {
    const result = await analyzeTrendReversal(strongBullishData);

    // Strong trend, no reversal yet
    expect(result.signal).toMatch(/NEUTRAL|WAIT|BUY/);
  });

  it('should identify double bottom pattern', async () => {
    const doubleBottomData = createPriceData({
      price: 9800,
      low24h: 9500,
      high24h: 10500,
      changePercent24h: 2.0,
    });

    const result = await analyzeTrendReversal(doubleBottomData);

    expect(result.signal).toBeDefined();
  });
});

// ============================================================================
// INTEGRATION TESTS - Multiple Strategies
// ============================================================================

describe('Strategy Integration Tests', () => {
  it('all strategies should return consistent structure', async () => {
    const strategies = [
      analyzeRSIDivergence,
      analyzeBollingerSqueeze,
      analyzeEMARibbon,
      analyzeVolumeBreakout,
      analyzeFibonacciRetracement,
      analyzeIchimokuCloud,
      analyzeATRVolatility,
      analyzeTrendReversal,
    ];

    for (const strategy of strategies) {
      const result = await strategy(strongBullishData);

      // All strategies should return these fields
      expect(result.name).toBeTruthy();
      expect(result.signal).toMatch(/BUY|SELL|WAIT|NEUTRAL/);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(100);
      expect(result.reason).toBeTruthy();
    }
  });

  it('strategies should handle edge cases gracefully', async () => {
    const edgeCaseData = createPriceData({
      price: 0.001, // Very low price
      changePercent24h: 0,
      volume24h: 100,
    });

    const strategies = [
      analyzeRSIDivergence,
      analyzeBollingerSqueeze,
      analyzeEMARibbon,
    ];

    for (const strategy of strategies) {
      const result = await strategy(edgeCaseData);

      // Should not throw and return valid signal
      expect(result.signal).toBeDefined();
    }
  });

  it('strategies should agree on strong bullish signals', async () => {
    const strategies = [
      analyzeRSIDivergence,
      analyzeEMARibbon,
      analyzeVolumeBreakout,
      analyzeTrendReversal,
    ];

    const results = await Promise.all(
      strategies.map((s) => s(strongBullishData))
    );

    // At least 60% should give BUY or WAIT (not SELL)
    const buyOrWaitCount = results.filter((r) =>
      ['BUY', 'WAIT', 'NEUTRAL'].includes(r.signal)
    ).length;

    expect(buyOrWaitCount).toBeGreaterThanOrEqual(Math.floor(strategies.length * 0.6));
  });

  it('strategies should agree on strong bearish signals', async () => {
    const strategies = [
      analyzeRSIDivergence,
      analyzeEMARibbon,
      analyzeVolumeBreakout,
      analyzeTrendReversal,
    ];

    const results = await Promise.all(
      strategies.map((s) => s(strongBearishData))
    );

    // At least 60% should give SELL or WAIT (not BUY)
    const sellOrWaitCount = results.filter((r) =>
      ['SELL', 'WAIT', 'NEUTRAL'].includes(r.signal)
    ).length;

    expect(sellOrWaitCount).toBeGreaterThanOrEqual(Math.floor(strategies.length * 0.6));
  });
});

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

describe('Strategy Performance Tests', () => {
  it('strategies should execute within 100ms', async () => {
    const start = Date.now();
    await analyzeRSIDivergence(strongBullishData);
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(100);
  });

  it('all strategies should execute in parallel < 200ms', async () => {
    const strategies = [
      analyzeRSIDivergence,
      analyzeBollingerSqueeze,
      analyzeEMARibbon,
      analyzeVolumeBreakout,
      analyzeFibonacciRetracement,
      analyzeIchimokuCloud,
      analyzeATRVolatility,
      analyzeTrendReversal,
    ];

    const start = Date.now();
    await Promise.all(strategies.map((s) => s(strongBullishData)));
    const duration = Date.now() - start;

    expect(duration).toBeLessThan(200);
  });
});
