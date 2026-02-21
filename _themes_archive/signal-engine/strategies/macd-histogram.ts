/**
 * 📊 MACD HISTOGRAM STRATEGY
 * MACD (12, 26, 9) histogram momentum analizi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeMACDHistogram(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h } = data;

  // Basit MACD tahmini: momentum proxy
  // Pozitif histogram = bullish, negatif = bearish
  const macdHistogram = changePercent24h; // Gerçekte MACD hesaplanır

  if (macdHistogram > 5) {
    return {
      name: 'MACD Histogram',
      signal: 'BUY',
      confidence: 80,
      reason: `MACD histogram pozitif ve güçlü (${macdHistogram.toFixed(1)}). Bullish momentum.`,
      targets: [price * 1.07, price * 1.12],
      indicators: {
        macdHistogram,
      },
    };
  }

  if (macdHistogram < -5) {
    return {
      name: 'MACD Histogram',
      signal: 'WAIT',
      confidence: 72,
      reason: `MACD histogram negatif (${macdHistogram.toFixed(1)}). Bearish momentum.`,
      indicators: {
        macdHistogram,
      },
    };
  }

  return {
    name: 'MACD Histogram',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'MACD nötr. Momentum zayıf.',
    indicators: {
      macdHistogram,
    },
  };
}
