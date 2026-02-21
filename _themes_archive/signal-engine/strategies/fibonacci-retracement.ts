/**
 * 📐 FIBONACCI RETRACEMENT STRATEGY
 * Fibonacci seviyeleri (0.382, 0.5, 0.618) ile destek/direnç
 */

import { PriceData, StrategySignal } from './types';

export async function analyzeFibonacciRetracement(data: PriceData): Promise<StrategySignal> {
  const { price, high24h, low24h, changePercent24h } = data;

  // Fibonacci retracement levels
  const range = high24h - low24h;
  const fib382 = high24h - range * 0.382;
  const fib50 = high24h - range * 0.5;
  const fib618 = high24h - range * 0.618;

  // Fiyat Fibonacci desteğine yakın mı?
  const nearFib618 = Math.abs(price - fib618) / price < 0.02; // %2 içinde
  const nearFib50 = Math.abs(price - fib50) / price < 0.02;

  if (nearFib618 && changePercent24h < 0) {
    return {
      name: 'Fibonacci Retracement',
      signal: 'BUY',
      confidence: 81,
      reason: `Fiyat Fibonacci 0.618 desteğinde ($${fib618.toFixed(6)}). Güçlü destek seviyesi.`,
      targets: [fib50, fib382, high24h],
      stopLoss: fib618 * 0.98,
      indicators: {
        fib618,
        fib50,
        fib382,
      },
    };
  }

  if (nearFib50) {
    return {
      name: 'Fibonacci Retracement',
      signal: 'WAIT',
      confidence: 70,
      reason: `Fiyat Fibonacci 0.5 seviyesinde ($${fib50.toFixed(6)}). Kırılım bekle.`,
      indicators: {
        fib50,
      },
    };
  }

  return {
    name: 'Fibonacci Retracement',
    signal: 'NEUTRAL',
    confidence: 50,
    reason: 'Fibonacci seviyeleri arasında.',
  };
}
