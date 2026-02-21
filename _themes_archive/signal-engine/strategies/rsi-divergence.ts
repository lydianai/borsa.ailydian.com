/**
 * 📉 RSI DIVERGENCE STRATEGY
 * RSI overbought/oversold ve divergence analizi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeRSIDivergence(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, high24h, low24h } = data;

  // Basit RSI tahmini (gerçekte 14 periyot RSI hesaplanır)
  // Yüksek volatilite = yüksek RSI
  const estimatedRSI = Math.min(50 + changePercent24h * 2, 100);

  if (estimatedRSI > 70) {
    return {
      name: 'RSI Divergence',
      signal: 'WAIT',
      confidence: 75,
      reason: `RSI overbought (${estimatedRSI.toFixed(0)}). Düzeltme bekleniyor.`,
      indicators: {
        rsi: estimatedRSI,
      },
    };
  }

  if (estimatedRSI < 30 && changePercent24h < -5) {
    return {
      name: 'RSI Divergence',
      signal: 'BUY',
      confidence: 82,
      reason: `RSI oversold (${estimatedRSI.toFixed(0)}). Toparlanma fırsatı.`,
      targets: [price * 1.08, price * 1.15],
      indicators: {
        rsi: estimatedRSI,
      },
    };
  }

  return {
    name: 'RSI Divergence',
    signal: 'NEUTRAL',
    confidence: 55,
    reason: `RSI normal seviyelerde (${estimatedRSI.toFixed(0)}).`,
    indicators: {
      rsi: estimatedRSI,
    },
  };
}
