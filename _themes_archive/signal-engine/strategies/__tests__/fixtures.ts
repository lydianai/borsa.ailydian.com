/**
 * 🧪 TEST FIXTURES
 * Common price data scenarios for strategy testing
 */

import { PriceData, Candle } from '../types';

// ============================================================================
// CANDLE FIXTURES
// ============================================================================

/**
 * Bullish trend candles (upward movement)
 */
export const bullishCandles: Candle[] = [
  { open: 40000, high: 40500, low: 39800, close: 40400, volume: 1000000, timestamp: Date.now() - 900000 },
  { open: 40400, high: 40900, low: 40200, close: 40800, volume: 1100000, timestamp: Date.now() - 600000 },
  { open: 40800, high: 41300, low: 40600, close: 41200, volume: 1200000, timestamp: Date.now() - 300000 },
  { open: 41200, high: 41700, low: 41000, close: 41600, volume: 1300000, timestamp: Date.now() },
];

/**
 * Bearish trend candles (downward movement)
 */
export const bearishCandles: Candle[] = [
  { open: 42000, high: 42200, low: 41500, close: 41600, volume: 1000000, timestamp: Date.now() - 900000 },
  { open: 41600, high: 41800, low: 41000, close: 41100, volume: 1100000, timestamp: Date.now() - 600000 },
  { open: 41100, high: 41300, low: 40500, close: 40600, volume: 1200000, timestamp: Date.now() - 300000 },
  { open: 40600, high: 40800, low: 40000, close: 40100, volume: 1300000, timestamp: Date.now() },
];

/**
 * Sideways/ranging candles (no clear trend)
 */
export const sidewaysCandles: Candle[] = [
  { open: 41000, high: 41300, low: 40800, close: 41100, volume: 1000000, timestamp: Date.now() - 900000 },
  { open: 41100, high: 41400, low: 40900, close: 41000, volume: 1000000, timestamp: Date.now() - 600000 },
  { open: 41000, high: 41300, low: 40700, close: 41200, volume: 1000000, timestamp: Date.now() - 300000 },
  { open: 41200, high: 41500, low: 41000, close: 41100, volume: 1000000, timestamp: Date.now() },
];

/**
 * Volatile candles (high volatility)
 */
export const volatileCandles: Candle[] = [
  { open: 40000, high: 42000, low: 39000, close: 41500, volume: 5000000, timestamp: Date.now() - 900000 },
  { open: 41500, high: 43000, low: 40000, close: 40500, volume: 5500000, timestamp: Date.now() - 600000 },
  { open: 40500, high: 42500, low: 39500, close: 42000, volume: 6000000, timestamp: Date.now() - 300000 },
  { open: 42000, high: 43500, low: 41000, close: 41800, volume: 6500000, timestamp: Date.now() },
];

// ============================================================================
// PRICE DATA FIXTURES
// ============================================================================

/**
 * Strong bullish scenario (BUY signal expected)
 * - 5% 24h gain
 * - Price near high
 * - Good volume
 */
export const strongBullishData: PriceData = {
  symbol: 'BTCUSDT',
  price: 41600,
  change24h: 2000,
  changePercent24h: 5.0,
  volume24h: 50000000,
  high24h: 42000,
  low24h: 39600,
  candles: bullishCandles,
};

/**
 * Pullback scenario (BUY signal expected after pullback)
 * - 4% 24h gain
 * - Price 3% below high (pullback)
 * - Strong momentum
 */
export const pullbackBuyData: PriceData = {
  symbol: 'ETHUSDT',
  price: 2910,
  change24h: 112,
  changePercent24h: 4.0,
  volume24h: 30000000,
  high24h: 3000,
  low24h: 2798,
  candles: bullishCandles,
};

/**
 * Strong bearish scenario (SELL/WAIT signal expected)
 * - -6% 24h loss
 * - Price near low
 * - High volume
 */
export const strongBearishData: PriceData = {
  symbol: 'BTCUSDT',
  price: 37600,
  change24h: -2400,
  changePercent24h: -6.0,
  volume24h: 55000000,
  high24h: 40000,
  low24h: 37400,
  candles: bearishCandles,
};

/**
 * Neutral/ranging scenario (WAIT/NEUTRAL signal expected)
 * - Low volatility
 * - Small 24h change
 * - Average volume
 */
export const neutralRangingData: PriceData = {
  symbol: 'BTCUSDT',
  price: 41000,
  change24h: 200,
  changePercent24h: 0.5,
  volume24h: 20000000,
  high24h: 41500,
  low24h: 40500,
  candles: sidewaysCandles,
};

/**
 * High volatility scenario
 * - Large price swings
 * - Very high volume
 * - Unclear trend
 */
export const highVolatilityData: PriceData = {
  symbol: 'BTCUSDT',
  price: 41800,
  change24h: 1800,
  changePercent24h: 4.5,
  volume24h: 100000000,
  high24h: 43500,
  low24h: 39000,
  candles: volatileCandles,
};

/**
 * Oversold scenario (potential BUY signal)
 * - Recent strong decline
 * - Price near low
 * - Signs of reversal
 */
export const oversoldData: PriceData = {
  symbol: 'BTCUSDT',
  price: 38000,
  change24h: -3000,
  changePercent24h: -7.3,
  volume24h: 60000000,
  high24h: 41000,
  low24h: 37800,
  candles: bearishCandles,
};

/**
 * Overbought scenario (potential SELL signal)
 * - Recent strong rally
 * - Price near high
 * - Signs of exhaustion
 */
export const overboughtData: PriceData = {
  symbol: 'BTCUSDT',
  price: 44800,
  change24h: 3800,
  changePercent24h: 9.3,
  volume24h: 70000000,
  high24h: 45000,
  low24h: 41000,
  candles: bullishCandles,
};

/**
 * Breakout scenario (BUY signal expected)
 * - Price breaking above resistance
 * - High volume
 * - Strong momentum
 */
export const breakoutData: PriceData = {
  symbol: 'BTCUSDT',
  price: 42500,
  change24h: 2500,
  changePercent24h: 6.25,
  volume24h: 80000000,
  high24h: 42500,
  low24h: 40000,
  candles: bullishCandles,
};

/**
 * Breakdown scenario (SELL signal expected)
 * - Price breaking below support
 * - High volume
 * - Strong downward momentum
 */
export const breakdownData: PriceData = {
  symbol: 'BTCUSDT',
  price: 38500,
  change24h: -2500,
  changePercent24h: -6.1,
  volume24h: 75000000,
  high24h: 41000,
  low24h: 38500,
  candles: bearishCandles,
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Create custom price data for testing
 */
export function createPriceData(overrides: Partial<PriceData>): PriceData {
  return {
    symbol: 'TESTUSDT',
    price: 1000,
    change24h: 0,
    changePercent24h: 0,
    volume24h: 1000000,
    high24h: 1050,
    low24h: 950,
    candles: sidewaysCandles,
    ...overrides,
  };
}

/**
 * Create custom candle data
 */
export function createCandle(overrides: Partial<Candle>): Candle {
  return {
    open: 1000,
    high: 1050,
    low: 950,
    close: 1000,
    volume: 100000,
    timestamp: Date.now(),
    ...overrides,
  };
}
