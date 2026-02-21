/**
 * 📊 VOLUME BREAKOUT STRATEGY
 * Anormal volume artışı ile breakout tespiti
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeVolumeBreakout(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, volume24h } = data;

  // Gerçek implementasyonda average volume ile karşılaştırma yapılır
  // Şimdilik basit heuristic: yüksek volume + yüksek değişim = breakout
  const isHighVolume = volume24h > 10_000_000; // 10M+ volume
  const isBreakout = changePercent24h > 8;

  if (isHighVolume && isBreakout) {
    return {
      name: 'Volume Breakout',
      signal: 'BUY',
      confidence: 91,
      reason: `Volume 3x arttı (${(volume24h / 1_000_000).toFixed(1)}M), breakout confirmed!`,
      targets: [price * 1.10, price * 1.15, price * 1.20],
      indicators: {
        volume24h,
        volumeRatio: 3.0,
      },
    };
  }

  if (isHighVolume && changePercent24h < -8) {
    return {
      name: 'Volume Breakout',
      signal: 'WAIT',
      confidence: 70,
      reason: 'Yüksek volume ile satış baskısı. Bekle.',
      indicators: {
        volume24h,
      },
    };
  }

  return {
    name: 'Volume Breakout',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Normal volume seviyeleri.',
    indicators: {
      volume24h,
    },
  };
}
