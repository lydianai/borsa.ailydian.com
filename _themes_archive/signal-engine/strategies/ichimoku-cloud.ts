/**
 * ☁️ ICHIMOKU CLOUD STRATEGY
 * Ichimoku Kinko Hyo - Cloud (bulut) analizi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeIchimokuCloud(data: PriceData): Promise<StrategySignal> {
  const { price, high24h, low24h, changePercent24h } = data;

  // Basitleştirilmiş Ichimoku (gerçekte Tenkan, Kijun, Senkou A/B hesaplanır)
  // Fiyat > 24h mid = bullish cloud
  const mid24h = (high24h + low24h) / 2;
  const aboveCloud = price > mid24h;
  const cloudStrength = ((price - mid24h) / mid24h) * 100;

  if (aboveCloud && cloudStrength > 2) {
    return {
      name: 'Ichimoku Cloud',
      signal: 'BUY',
      confidence: 77,
      reason: `Fiyat bulutun üzerinde (+${cloudStrength.toFixed(1)}%). Güçlü uptrend.`,
      targets: [price * 1.05, price * 1.08],
      indicators: {
        cloudPosition: 'ABOVE',
        cloudStrength,
      },
    };
  }

  if (!aboveCloud && cloudStrength < -2) {
    return {
      name: 'Ichimoku Cloud',
      signal: 'WAIT',
      confidence: 65,
      reason: `Fiyat bulutun altında (${cloudStrength.toFixed(1)}%). Downtrend.`,
      indicators: {
        cloudPosition: 'BELOW',
        cloudStrength,
      },
    };
  }

  return {
    name: 'Ichimoku Cloud',
    signal: 'NEUTRAL',
    confidence: 52,
    reason: 'Fiyat bulut içinde. Net sinyal yok.',
    indicators: {
      cloudPosition: 'INSIDE',
    },
  };
}
