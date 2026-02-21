/**
 * 🎯 STRATEGY AGGREGATOR
 * 15 stratejiyi birleştirip genel analiz sunar
 */

import { PriceData, StrategyAnalysis, StrategySignal } from './strategies/types';
import { analyzeMACrossoverPullback } from './strategies/ma-crossover-pullback';
import { analyzeMA7Pullback } from './strategies/ma7-pullback';
import { analyzeRSIDivergence } from './strategies/rsi-divergence';
import { analyzeVolumeBreakout } from './strategies/volume-breakout';
import { analyzeBollingerSqueeze } from './strategies/bollinger-squeeze';
import { analyzeEMARibbon } from './strategies/ema-ribbon';
import { analyzeFibonacciRetracement } from './strategies/fibonacci-retracement';
import { analyzeIchimokuCloud } from './strategies/ichimoku-cloud';
import { analyzeATRVolatility } from './strategies/atr-volatility';
import { analyzeTrendReversal } from './strategies/trend-reversal';
import { analyzeMACDHistogram } from './strategies/macd-histogram';
import { analyzeSupportResistance } from './strategies/support-resistance';
import { analyzeRedWickGreenClosure } from './strategies/red-wick-green-closure';
import { analyzeConservativeBuySignal } from './strategies/conservative-buy-signal';
import { analyzeOmnipotentFuturesMatrix } from './strategies/omnipotent-futures-matrix';

/**
 * Tüm 15 stratejiyi çalıştırıp birleştirilmiş analiz döner
 */
export async function analyzeAllStrategies(data: PriceData): Promise<StrategyAnalysis> {
  // Paralel olarak tüm stratejileri çalıştır
  const strategies = await Promise.all([
    analyzeMACrossoverPullback(data),
    analyzeMA7Pullback(data),
    analyzeRSIDivergence(data),
    analyzeVolumeBreakout(data),
    analyzeBollingerSqueeze(data),
    analyzeEMARibbon(data),
    analyzeFibonacciRetracement(data),
    analyzeIchimokuCloud(data),
    analyzeATRVolatility(data),
    analyzeTrendReversal(data),
    analyzeMACDHistogram(data),
    analyzeSupportResistance(data),
    analyzeRedWickGreenClosure(data),
    analyzeConservativeBuySignal(data),
    analyzeOmnipotentFuturesMatrix(data),
  ]);

  // Sinyal sayılarını hesapla
  const buyCount = strategies.filter((s) => s.signal === 'BUY').length;
  const waitCount = strategies.filter((s) => s.signal === 'WAIT').length;
  const sellCount = strategies.filter((s) => s.signal === 'SELL').length;
  const neutralCount = strategies.filter((s) => s.signal === 'NEUTRAL').length;

  // Genel skor hesapla (weighted average confidence)
  const totalConfidence = strategies.reduce((sum, s) => {
    if (s.signal === 'BUY') return sum + s.confidence * 1.5;
    if (s.signal === 'SELL') return sum + s.confidence * 0.5;
    if (s.signal === 'WAIT') return sum + s.confidence * 0.8;
    return sum + s.confidence * 0.5;
  }, 0);

  const overallScore = Math.min(Math.round(totalConfidence / strategies.length), 100);

  // Genel öneri
  let recommendation: StrategyAnalysis['recommendation'];

  if (overallScore >= 75 && buyCount >= 8) {
    recommendation = 'STRONG_BUY';
  } else if (overallScore >= 60 && buyCount >= 6) {
    recommendation = 'BUY';
  } else if (overallScore <= 40 && sellCount >= 6) {
    recommendation = 'SELL';
  } else if (waitCount >= buyCount) {
    recommendation = 'WAIT';
  } else {
    recommendation = 'NEUTRAL';
  }

  return {
    symbol: data.symbol,
    price: data.price,
    change24h: data.change24h,
    changePercent24h: data.changePercent24h,
    strategies,
    overallScore,
    recommendation,
    buyCount,
    waitCount,
    sellCount,
    neutralCount,
    timestamp: new Date().toISOString(),
  };
}
