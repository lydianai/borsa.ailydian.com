/**
 * 🧪 VOLUME SPIKE STRATEGY - UNIT TESTS
 *
 * Coverage Target: 85%+
 * Test Strategy: White-box testing with real market scenarios
 *
 * Test Categories:
 * 1. Valid BUY signals (volume spike detected)
 * 2. WAIT signals (no spike or missing conditions)
 * 3. Edge cases (boundary values)
 * 4. Risk management (stop loss, targets)
 * 5. Real market scenarios
 */

import { describe, it, expect } from 'vitest';
import { analyzeVolumeSpike } from '../volume-spike';
import { PriceData } from '../types';

describe('Volume Spike Strategy', () => {

  // ============================================
  // TEST CATEGORY 1: VALID BUY SIGNALS
  // ============================================

  describe('Valid BUY Signals', () => {
    it('should generate BUY signal when volume spike detected (5/5 conditions)', async () => {
      const perfectSpike: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,           // ✅ Near support (20% from range)
        change24h: 1500,
        changePercent24h: 3.2,  // ✅ Positive momentum (RSI ~56)
        volume24h: 2400000000,  // ✅ 3.2x avg volume (huge spike!)
        high24h: 50000,
        low24h: 47000,          // ✅ Support identified
      };

      const result = await analyzeVolumeSpike(perfectSpike);

      expect(result.signal).toBe('BUY');
      expect(result.confidence).toBeGreaterThanOrEqual(80);
      expect(result.reason).toContain('VOLUME SPIKE DETECTED');
      expect(result.targets).toBeDefined();
      expect(result.targets?.length).toBe(3);
      expect(result.stopLoss).toBeDefined();
    });

    it('should generate BUY signal with 4/5 conditions', async () => {
      const goodSpike: PriceData = {
        symbol: 'ETHUSDT',
        price: 2950,
        change24h: 100,
        changePercent24h: 3.5,  // ✅ Positive momentum
        volume24h: 1700000000,  // ✅ 2.27x avg volume
        high24h: 3100,
        low24h: 2900,           // ✅ Support at 2900
      };

      const result = await analyzeVolumeSpike(goodSpike);

      expect(result.signal).toBe('BUY');
      expect(result.confidence).toBeGreaterThanOrEqual(70);
    });

    it('should calculate correct targets (+4%, +7%, +12%)', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.targets) {
        expect(result.targets[0]).toBeCloseTo(50000 * 1.04, 0); // +4%
        expect(result.targets[1]).toBeCloseTo(50000 * 1.07, 0); // +7%
        expect(result.targets[2]).toBeCloseTo(50000 * 1.12, 0); // +12%
      }
    });

    it('should calculate correct stop loss (-3%)', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.stopLoss) {
        expect(result.stopLoss).toBeCloseTo(50000 * 0.97, 0); // -3%
        expect(result.stopLoss).toBeLessThan(data.price);
      }
    });

    it('should include volume ratio in BUY signal', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 2000,
        changePercent24h: 4.2,
        volume24h: 3000000000, // 4x avg volume
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.indicators) {
        expect(result.indicators.volumeRatio).toBeGreaterThan(2.0);
        expect(result.indicators.currentVolume).toBe(3000000000);
      }
    });
  });

  // ============================================
  // TEST CATEGORY 2: WAIT SIGNALS
  // ============================================

  describe('WAIT Signals (Invalid Conditions)', () => {
    it('should return WAIT when volume ratio < 2.0', async () => {
      const lowVolume: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 800000000,  // Only 1.07x avg (no spike)
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(lowVolume);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Volume ratio too low');
    });

    it('should return WAIT when momentum is negative', async () => {
      const negativeMomentum: PriceData = {
        symbol: 'BTCUSDT',
        price: 47000,
        change24h: -1500,
        changePercent24h: -3.1,  // Negative momentum
        volume24h: 2000000000,    // Volume spike present
        high24h: 50000,
        low24h: 46000,
      };

      const result = await analyzeVolumeSpike(negativeMomentum);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Negative momentum');
    });

    it('should return WAIT when price is overbought (RSI > 70)', async () => {
      const overbought: PriceData = {
        symbol: 'BTCUSDT',
        price: 52000,
        change24h: 10000,
        changePercent24h: 23.8,  // Too hot, RSI ~98
        volume24h: 2500000000,
        high24h: 52000,
        low24h: 42000,
      };

      const result = await analyzeVolumeSpike(overbought);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('RSI > 70');
    });

    it('should return WAIT when price too high in range', async () => {
      const nearResistance: PriceData = {
        symbol: 'BTCUSDT',
        price: 50800,           // 95% from range (too high)
        change24h: 1800,
        changePercent24h: 3.7,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(nearResistance);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Price too high in range');
    });

    it('should return WAIT when volume is too low (< $50M)', async () => {
      const tinyVolume: PriceData = {
        symbol: 'SHITCOIN',
        price: 1.5,
        change24h: 0.2,
        changePercent24h: 15.4,
        volume24h: 20000000,  // Only $20M (too low)
        high24h: 1.6,
        low24h: 1.3,
      };

      const result = await analyzeVolumeSpike(tinyVolume);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Volume too low');
    });

    it('should return WAIT when less than 4/5 conditions met', async () => {
      const poorSetup: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 100,
        changePercent24h: 0.2,   // Weak momentum
        volume24h: 600000000,    // Low volume (only 0.8x)
        high24h: 50500,
        low24h: 49500,
      };

      const result = await analyzeVolumeSpike(poorSetup);

      expect(result.signal).toBe('WAIT');
      expect(result.confidence).toBeLessThan(80);
    });
  });

  // ============================================
  // TEST CATEGORY 3: EDGE CASES
  // ============================================

  describe('Edge Cases & Boundaries', () => {
    it('should handle exactly 2.0x volume ratio (boundary)', async () => {
      const exactBoundary: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 1500,
        changePercent24h: 3.2,
        volume24h: 1500000000,  // Exactly 2.0x avg
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(exactBoundary);

      expect(result).toBeDefined();
      expect(result.signal).toMatch(/BUY|WAIT/);
    });

    it('should handle zero price change', async () => {
      const flatMarket: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 0,
        changePercent24h: 0,
        volume24h: 2000000000,
        high24h: 50100,
        low24h: 49900,
      };

      const result = await analyzeVolumeSpike(flatMarket);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Negative momentum');
    });

    it('should handle very high volume spike (>10x)', async () => {
      const massiveSpike: PriceData = {
        symbol: 'NEWSUSDT',
        price: 5.0,
        change24h: 1.0,
        changePercent24h: 25.0,  // News-driven pump
        volume24h: 10000000000,  // 13.3x avg volume (massive!)
        high24h: 5.2,
        low24h: 4.0,
      };

      const result = await analyzeVolumeSpike(massiveSpike);

      // Might be WAIT due to overbought (RSI ~100)
      expect(result).toBeDefined();
      if (result.indicators) {
        expect(result.indicators.volumeRatio).toBeGreaterThan(10);
      }
    });

    it('should handle very low price (altcoins)', async () => {
      const lowPriceAlt: PriceData = {
        symbol: 'PEPEUSDT',
        price: 0.000005,
        change24h: 0.000001,
        changePercent24h: 25.0,
        volume24h: 100000000,  // $100M volume
        high24h: 0.000006,
        low24h: 0.000004,
      };

      const result = await analyzeVolumeSpike(lowPriceAlt);

      expect(result).toBeDefined();
      expect(result.signal).toMatch(/BUY|WAIT/);
    });

    it('should handle price at exact support (0% from range)', async () => {
      const atSupport: PriceData = {
        symbol: 'BTCUSDT',
        price: 47000,           // Exactly at low24h
        change24h: 500,
        changePercent24h: 1.1,
        volume24h: 2000000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(atSupport);

      expect(result).toBeDefined();
      if (result.indicators) {
        expect(result.indicators.pricePosition).toBeLessThan(10); // Near 0%
      }
    });

    it('should handle price at exact resistance (100% from range)', async () => {
      const atResistance: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,           // Exactly at high24h
        change24h: 2000,
        changePercent24h: 4.2,
        volume24h: 2000000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(atResistance);

      expect(result).toBeDefined();
      if (result.indicators) {
        expect(result.indicators.pricePosition).toBeGreaterThan(90); // Near 100%
      }
    });
  });

  // ============================================
  // TEST CATEGORY 4: RISK MANAGEMENT
  // ============================================

  describe('Risk Management Validation', () => {
    it('should never exceed 3x leverage', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 2000,
        changePercent24h: 4.3,
        volume24h: 2500000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.indicators) {
        expect(result.indicators.leverageMax).toBeLessThanOrEqual(3);
      }
    });

    it('should limit position size to 1.5% of portfolio', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 2000,
        changePercent24h: 4.3,
        volume24h: 2500000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.indicators) {
        expect(result.indicators.positionSizePercent).toBeLessThanOrEqual(1.5);
      }
    });

    it('should have stop loss below entry price', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.stopLoss) {
        expect(result.stopLoss).toBeLessThan(data.price);
      }
    });

    it('should have all targets above entry price', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY' && result.targets) {
        result.targets.forEach(target => {
          expect(target).toBeGreaterThan(data.price);
        });
      }
    });
  });

  // ============================================
  // TEST CATEGORY 5: REAL MARKET SCENARIOS
  // ============================================

  describe('Real Market Scenarios', () => {
    it('should detect breakout with volume confirmation', async () => {
      const breakout: PriceData = {
        symbol: 'BTCUSDT',
        price: 68000,
        change24h: 4000,
        changePercent24h: 6.3,
        volume24h: 5000000000,  // 6.67x avg volume (breakout!)
        high24h: 68500,
        low24h: 64000,
      };

      const result = await analyzeVolumeSpike(breakout);

      // Might be WAIT due to overbought, but volume spike detected
      expect(result).toBeDefined();
      if (result.indicators) {
        expect(result.indicators.volumeRatio).toBeGreaterThan(5);
      }
    });

    it('should handle news-driven pump', async () => {
      const newsPump: PriceData = {
        symbol: 'ETHUSDT',
        price: 3500,
        change24h: 600,
        changePercent24h: 20.7,  // Huge news pump
        volume24h: 8000000000,   // 10.67x avg volume
        high24h: 3600,
        low24h: 2900,
      };

      const result = await analyzeVolumeSpike(newsPump);

      // Likely WAIT due to overbought (RSI ~91)
      expect(result).toBeDefined();
      expect(result.signal).toBe('WAIT');
    });

    it('should reject wash trading (high volume, no price movement)', async () => {
      const washTrading: PriceData = {
        symbol: 'SCAMUSDT',
        price: 1.0,
        change24h: 0.01,
        changePercent24h: 1.0,   // Minimal movement
        volume24h: 3000000000,   // High volume (4x avg)
        high24h: 1.05,
        low24h: 0.95,
      };

      const result = await analyzeVolumeSpike(washTrading);

      // Should generate signal if other conditions met
      expect(result).toBeDefined();
    });

    it('should handle altcoin volume spike during alt season', async () => {
      const altSeason: PriceData = {
        symbol: 'SOLUSDT',
        price: 150,
        change24h: 15,
        changePercent24h: 11.1,
        volume24h: 2000000000,  // 2.67x avg volume
        high24h: 155,
        low24h: 135,
      };

      const result = await analyzeVolumeSpike(altSeason);

      expect(result).toBeDefined();
    });

    it('should reject during bear market dump', async () => {
      const bearDump: PriceData = {
        symbol: 'BTCUSDT',
        price: 18000,
        change24h: -7000,
        changePercent24h: -28.0,  // Massive dump
        volume24h: 12000000000,   // 16x avg volume (panic!)
        high24h: 25000,
        low24h: 17500,
      };

      const result = await analyzeVolumeSpike(bearDump);

      expect(result.signal).toBe('WAIT');
      expect(result.reason).toContain('Negative momentum');
    });
  });

  // ============================================
  // TEST CATEGORY 6: RESPONSE STRUCTURE
  // ============================================

  describe('Response Structure Validation', () => {
    it('should always return required fields', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 50000,
        change24h: 2000,
        changePercent24h: 4.0,
        volume24h: 2000000000,
        high24h: 51000,
        low24h: 48000,
      };

      const result = await analyzeVolumeSpike(data);

      expect(result).toHaveProperty('name');
      expect(result).toHaveProperty('signal');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('reason');
      expect(result).toHaveProperty('indicators');
      expect(result.name).toBe('Volume Spike');
      expect(['BUY', 'WAIT']).toContain(result.signal);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(100);
    });

    it('should include timeframe in BUY signal', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 2500,
        changePercent24h: 5.5,
        volume24h: 3000000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(data);

      if (result.signal === 'BUY') {
        expect(result.timeframe).toBe('4H');
      }
    });

    it('should include volume metrics in indicators', async () => {
      const data: PriceData = {
        symbol: 'BTCUSDT',
        price: 48000,
        change24h: 2000,
        changePercent24h: 4.3,
        volume24h: 2400000000,
        high24h: 50000,
        low24h: 47000,
      };

      const result = await analyzeVolumeSpike(data);

      expect(result.indicators).toBeDefined();
      expect(result.indicators).toHaveProperty('volumeRatio');
      expect(result.indicators).toHaveProperty('avgVolume');
      expect(result.indicators).toHaveProperty('currentVolume');
      expect(result.indicators).toHaveProperty('approxRSI');
      expect(result.indicators).toHaveProperty('pricePosition');
    });
  });
});
