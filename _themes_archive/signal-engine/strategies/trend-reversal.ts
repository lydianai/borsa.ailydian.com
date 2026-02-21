/**
 * 🔄 TREND REVERSAL STRATEGY
 * Trend dönüşü tespiti (support/resistance kırılımları)
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeTrendReversal(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, high24h, low24h } = data;

  // Basit reversal logic: güçlü düşüşten güçlü yükselişe geçiş
  const strongReversal = changePercent24h > 10 && price < high24h * 0.95;

  if (strongReversal) {
    return {
      name: 'Trend Reversal',
      signal: 'BUY',
      confidence: 84,
      reason: 'Güçlü trend dönüşü! Downtrend kırıldı, uptrend başladı.',
      targets: [price * 1.10, price * 1.15],
      stopLoss: low24h * 1.02,
      indicators: {
        reversalStrength: changePercent24h,
      },
    };
  }

  if (changePercent24h < -10) {
    return {
      name: 'Trend Reversal',
      signal: 'WAIT',
      confidence: 70,
      reason: 'Downtrend devam ediyor. Dönüş sinyali bekle.',
      indicators: {
        trend: 'DOWN',
      },
    };
  }

  return {
    name: 'Trend Reversal',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Net trend dönüşü yok.',
  };
}
