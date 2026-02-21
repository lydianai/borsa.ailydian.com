/**
 * 🕯️ RED WICK + GREEN CLOSURE STRATEGY
 * Kırmızı fitil + yeşil kapanış = güçlü AL sinyali
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeRedWickGreenClosure(data: PriceData): Promise<StrategySignal> {
  const { price, high24h, low24h, changePercent24h } = data;

  // Red wick = düşük düşüş + yeşil kapanış (recovery)
  const range24h = high24h - low24h;
  const wickDown = price - low24h;
  const wickPercent = (wickDown / range24h) * 100;

  // Kırmızı fitil + yeşil kapanış: fiyat düştü ama toparlandi
  const hasRedWick = wickPercent > 60 && changePercent24h > 0;

  if (hasRedWick && changePercent24h > 3) {
    return {
      name: 'Red Wick + Green Closure',
      signal: 'BUY',
      confidence: 88,
      reason: 'Kırmızı fitil + yeşil kapanış! Güçlü alım baskısı, toparlanma hızlı.',
      targets: [price * 1.08, price * 1.12],
      stopLoss: low24h * 1.01,
      indicators: {
        wickPercent,
        recovery: changePercent24h,
      },
    };
  }

  if (wickPercent > 70 && changePercent24h < -5) {
    return {
      name: 'Red Wick + Green Closure',
      signal: 'WAIT',
      confidence: 65,
      reason: 'Kırmızı fitil var ama kapanış negatif. Bekle.',
      indicators: {
        wickPercent,
      },
    };
  }

  return {
    name: 'Red Wick + Green Closure',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Red wick pattern görünmüyor.',
  };
}
