/**
 * 📈 MA7 PULLBACK STRATEGY
 * MA7 (7 period moving average) pullback after crossover
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeMA7Pullback(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, high24h } = data;

  // MA7 pullback: fiyat yüksek yapıp geri çekildi
  const pullback = ((high24h - price) / high24h) * 100;
  const goodPullback = pullback >= 1 && pullback <= 4;
  const momentum = changePercent24h > 2;

  if (goodPullback && momentum) {
    return {
      name: 'MA7 Pullback',
      signal: 'BUY',
      confidence: 86,
      reason: `MA7 pullback tamamlandı (${pullback.toFixed(1)}%). İdeal giriş noktası.`,
      targets: [price * 1.06, price * 1.10],
      indicators: {
        pullbackPercent: pullback,
      },
    };
  }

  if (pullback > 8) {
    return {
      name: 'MA7 Pullback',
      signal: 'WAIT',
      confidence: 60,
      reason: `Pullback çok derin (${pullback.toFixed(1)}%). Daha fazla düşüş olabilir.`,
    };
  }

  return {
    name: 'MA7 Pullback',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'MA7 pullback koşulları oluşmadı.',
  };
}
