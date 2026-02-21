/**
 * 🧪 MA CROSSOVER PULLBACK STRATEGY TESTS
 */

import { describe, it, expect } from 'vitest';
import { analyzeMACrossoverPullback } from '../ma-crossover-pullback';
import {
  strongBullishData,
  pullbackBuyData,
  strongBearishData,
  neutralRangingData,
  createPriceData,
} from './fixtures';

describe('MA Crossover Pullback Strategy', () => {
  it('should return BUY signal for pullback scenario', async () => {
    const result = await analyzeMACrossoverPullback(pullbackBuyData);

    expect(result.name).toBe('MA Crossover Pullback');
    expect(result.signal).toBe('BUY');
    expect(result.confidence).toBeGreaterThan(80);
    expect(result.reason).toContain('pullback');
    expect(result.targets).toBeDefined();
    expect(result.targets?.length).toBeGreaterThan(0);
    expect(result.stopLoss).toBeDefined();
    expect(result.indicators).toBeDefined();
  });

  it('should return BUY signal when price is 3% below high with strong momentum', async () => {
    const data = createPriceData({
      price: 9700,
      changePercent24h: 4.5,
      high24h: 10000,
      low24h: 9000,
    });

    const result = await analyzeMACrossoverPullback(data);

    expect(result.signal).toBe('BUY');
    expect(result.confidence).toBeGreaterThanOrEqual(85);
  });

  it('should return WAIT signal for strong bearish trend', async () => {
    const result = await analyzeMACrossoverPullback(strongBearishData);

    expect(result.name).toBe('MA Crossover Pullback');
    expect(result.signal).toBe('WAIT');
    expect(result.confidence).toBe(60);
    expect(result.reason).toContain('Düşüş');
  });

  it('should return NEUTRAL signal when pullback conditions not met', async () => {
    const result = await analyzeMACrossoverPullback(neutralRangingData);

    expect(result.name).toBe('MA Crossover Pullback');
    expect(result.signal).toBe('NEUTRAL');
    expect(result.confidence).toBe(50);
    expect(result.reason).toContain('Pullback koşulları');
  });

  it('should return NEUTRAL when momentum too weak', async () => {
    const data = createPriceData({
      price: 9700,
      changePercent24h: 1.5, // Too weak momentum
      high24h: 10000,
      low24h: 9500,
    });

    const result = await analyzeMACrossoverPullback(data);

    expect(result.signal).toBe('NEUTRAL');
  });

  it('should calculate correct targets and stop loss', async () => {
    const data = createPriceData({
      price: 10000,
      changePercent24h: 5.0,
      high24h: 10300,
      low24h: 9500,
    });

    const result = await analyzeMACrossoverPullback(data);

    if (result.signal === 'BUY') {
      expect(result.targets?.[0]).toBeCloseTo(10500, 0); // 5% target
      expect(result.targets?.[1]).toBeCloseTo(11000, 0); // 10% target
      expect(result.stopLoss).toBeLessThan(data.price);
    }
  });

  it('should have pullback indicators in response', async () => {
    const result = await analyzeMACrossoverPullback(pullbackBuyData);

    if (result.signal === 'BUY') {
      expect(result.indicators?.pullbackPercent).toBeDefined();
      expect(result.indicators?.momentum24h).toBeDefined();
      expect(result.indicators?.momentum24h).toBe(pullbackBuyData.changePercent24h);
    }
  });

  it('should handle edge case: price exactly at high', async () => {
    const data = createPriceData({
      price: 10000,
      changePercent24h: 5.0,
      high24h: 10000, // Price = High (no pullback)
      low24h: 9500,
    });

    const result = await analyzeMACrossoverPullback(data);

    // Should be NEUTRAL because pullback % is 0
    expect(result.signal).toBe('NEUTRAL');
  });

  it('should cap confidence at 95', async () => {
    const data = createPriceData({
      price: 9700,
      changePercent24h: 15.0, // Very high momentum
      high24h: 10000,
      low24h: 9000,
    });

    const result = await analyzeMACrossoverPullback(data);

    if (result.signal === 'BUY') {
      expect(result.confidence).toBeLessThanOrEqual(95);
    }
  });
});
