/**
 * 📏 SUPPORT/RESISTANCE STRATEGY
 * Destek ve direnç seviyeleri analizi
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeSupportResistance(data: PriceData): Promise<StrategySignal> {
  const { price, high24h, low24h, changePercent24h } = data;

  // Destek = 24h low, Direnç = 24h high
  const support = low24h;
  const resistance = high24h;
  const mid = (support + resistance) / 2;

  const nearSupport = Math.abs(price - support) / price < 0.03; // %3 içinde
  const nearResistance = Math.abs(price - resistance) / price < 0.03;

  if (nearSupport && changePercent24h > 0) {
    return {
      name: 'Support/Resistance',
      signal: 'BUY',
      confidence: 83,
      reason: `Fiyat destek seviyesinde ($${support.toFixed(6)}) ve yükseliyor. Güçlü AL fırsatı.`,
      targets: [mid, resistance],
      stopLoss: support * 0.98,
      indicators: {
        support,
        resistance,
      },
    };
  }

  if (nearResistance) {
    return {
      name: 'Support/Resistance',
      signal: 'WAIT',
      confidence: 70,
      reason: `Fiyat direnç seviyesinde ($${resistance.toFixed(6)}). Kırılım bekle.`,
      indicators: {
        resistance,
      },
    };
  }

  return {
    name: 'Support/Resistance',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Fiyat destek-direnç arasında.',
    indicators: {
      support,
      resistance,
    },
  };
}
