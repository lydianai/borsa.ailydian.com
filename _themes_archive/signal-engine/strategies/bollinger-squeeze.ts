/**
 * 🎯 BOLLINGER SQUEEZE STRATEGY
 * Bollinger Bands daralması (squeeze) sonrası breakout beklentisi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeBollingerSqueeze(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, high24h, low24h } = data;

  // Basit Bollinger Bands tahmini
  // 24h range dar ise (<%3) = squeeze, sonra breakout beklenir
  const range24h = ((high24h - low24h) / low24h) * 100;

  if (range24h < 3 && Math.abs(changePercent24h) < 2) {
    return {
      name: 'Bollinger Squeeze',
      signal: 'WAIT',
      confidence: 68,
      reason: `Bollinger Bands sıkışması (range: ${range24h.toFixed(1)}%). Breakout bekleniyor.`,
      indicators: {
        range24h,
        bandWidth: range24h,
      },
    };
  }

  if (range24h < 3 && changePercent24h > 5) {
    return {
      name: 'Bollinger Squeeze',
      signal: 'BUY',
      confidence: 86,
      reason: 'Squeeze sonrası yukarı breakout! Güçlü AL sinyali.',
      targets: [price * 1.08, price * 1.12],
      indicators: {
        range24h,
        breakoutDirection: 'UP',
      },
    };
  }

  return {
    name: 'Bollinger Squeeze',
    signal: 'NEUTRAL',
    confidence: 52,
    reason: 'Normal Bollinger Bands genişliği.',
    indicators: {
      range24h,
    },
  };
}
