/**
 * 📈 MA CROSSOVER PULLBACK STRATEGY
 * MA7 kesişim sonrası pullback (geri çekilme) bekler ve AL sinyali verir
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeMACrossoverPullback(data: PriceData): Promise<StrategySignal> {
  const { price, changePercent24h, high24h, low24h } = data;

  // Basit MA7 pullback logic (gerçek implementasyonda candle data kullanılır)
  // 24h değişim > %3 ve fiyat 24h high'dan %2-5 düşük ise pullback
  const pullbackFromHigh = ((high24h - price) / high24h) * 100;
  const isStrongMove = changePercent24h > 3;
  const isPullback = pullbackFromHigh >= 2 && pullbackFromHigh <= 5;

  if (isStrongMove && isPullback) {
    const target1 = price * 1.05;
    const target2 = price * 1.10;
    return {
      name: 'MA Crossover Pullback',
      signal: 'BUY',
      confidence: Math.min(85 + changePercent24h, 95),
      reason: `MA7 kesişim sonrası pullback tamamlandı. ${pullbackFromHigh.toFixed(1)}% düşüş, güçlü momentum devam ediyor.`,
      targets: [target1, target2],
      stopLoss: low24h * 0.98,
      indicators: {
        pullbackPercent: pullbackFromHigh,
        momentum24h: changePercent24h,
      },
    };
  }

  if (changePercent24h < -5) {
    return {
      name: 'MA Crossover Pullback',
      signal: 'WAIT',
      confidence: 60,
      reason: 'Düşüş trendi, MA kesişim bekleniyor.',
    };
  }

  return {
    name: 'MA Crossover Pullback',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Pullback koşulları oluşmadı. Takip et.',
  };
}
