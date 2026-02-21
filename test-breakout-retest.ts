/**
 * Test Breakout-Retest Strategy with simulated perfect conditions
 */

import { analyzeBreakoutRetest } from './apps/signal-engine/strategies/breakout-retest';
import { PriceData } from './apps/signal-engine/strategies/types';

async function testBreakoutRetest() {
  console.log('🧪 Testing Breakout-Retest Strategy\n');

  // Test 1: Perfect LONG setup
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('TEST 1: Perfect LONG Breakout-Retest');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const perfectLong: PriceData = {
    symbol: 'BTCUSDT',
    price: 50500, // At retest level
    change24h: 1000,
    changePercent24h: 2.5, // Moderate bullish momentum
    volume24h: 250_000_000, // High volume
    high24h: 50500, // Breakout level (consolidated before)
    low24h: 48000, // 5% range from consolidation low
  };

  const signal1 = await analyzeBreakoutRetest(perfectLong);
  console.log('Signal:', signal1.signal);
  console.log('Confidence:', signal1.confidence);
  console.log('Reason:\n', signal1.reason);
  console.log('\nTargets:', signal1.targets);
  console.log('Stop Loss:', signal1.stopLoss);
  console.log('\n');

  // Test 2: No consolidation (ranging too wide)
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('TEST 2: Invalid - Range Too Wide');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const tooWide: PriceData = {
    symbol: 'ETHUSDT',
    price: 3000,
    change24h: 50,
    changePercent24h: 1.7,
    volume24h: 180_000_000,
    high24h: 3300, // 10% range (too wide, need 2-8%)
    low24h: 3000,
  };

  const signal2 = await analyzeBreakoutRetest(tooWide);
  console.log('Signal:', signal2.signal);
  console.log('Confidence:', signal2.confidence);
  console.log('Reason:\n', signal2.reason);
  console.log('\n');

  // Test 3: No breakout yet
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('TEST 3: Valid Consolidation, No Breakout');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const noBreakout: PriceData = {
    symbol: 'SOLUSDT',
    price: 95, // In middle of range
    change24h: 1,
    changePercent24h: 1.0,
    volume24h: 150_000_000,
    high24h: 100, // 5.26% range (perfect)
    low24h: 95,
  };

  const signal3 = await analyzeBreakoutRetest(noBreakout);
  console.log('Signal:', signal3.signal);
  console.log('Confidence:', signal3.confidence);
  console.log('Reason:\n', signal3.reason);
  console.log('\n');

  // Test 4: Real market data example
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('TEST 4: Real Market Data - BTC');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const realBTC: PriceData = {
    symbol: 'BTCUSDT',
    price: 110967.80,
    change24h: 992.00,
    changePercent24h: 0.902,
    volume24h: 111879.393 * 110967.80, // Convert to USDT
    high24h: 112086.00,
    low24h: 109650.00,
  };

  const signal4 = await analyzeBreakoutRetest(realBTC);
  console.log('Signal:', signal4.signal);
  console.log('Confidence:', signal4.confidence);
  console.log('Reason:\n', signal4.reason.substring(0, 500) + '...');
  console.log('\n');

  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('✅ Test Complete');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
}

testBreakoutRetest().catch(console.error);
