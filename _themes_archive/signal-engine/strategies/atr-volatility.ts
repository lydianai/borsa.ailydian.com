/**
 * 📊 ATR VOLATILITY STRATEGY
 * Average True Range - Volatilite analizi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeATRVolatility(data: PriceData): Promise<StrategySignal> {
  const { price, high24h, low24h, changePercent24h } = data;

  // ATR tahmini: 24h range
  const atr24h = high24h - low24h;
  const atrPercent = (atr24h / price) * 100;

  // Yüksek volatilite = fırsat VEYA risk
  if (atrPercent > 10 && changePercent24h > 5) {
    return {
      name: 'ATR Volatility',
      signal: 'BUY',
      confidence: 73,
      reason: `Yüksek volatilite (ATR: ${atrPercent.toFixed(1)}%) + pozitif momentum. Trend güçlü.`,
      targets: [price * 1.08],
      stopLoss: price * 0.95,
      indicators: {
        atr: atr24h,
        atrPercent,
      },
    };
  }

  if (atrPercent > 15) {
    return {
      name: 'ATR Volatility',
      signal: 'WAIT',
      confidence: 68,
      reason: `Çok yüksek volatilite (ATR: ${atrPercent.toFixed(1)}%). Risk yüksek, bekle.`,
      indicators: {
        atr: atr24h,
        atrPercent,
      },
    };
  }

  return {
    name: 'ATR Volatility',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: `Normal volatilite (ATR: ${atrPercent.toFixed(1)}%).`,
    indicators: {
      atr: atr24h,
      atrPercent,
    },
  };
}
