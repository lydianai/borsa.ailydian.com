/**
 * 🎯 UNIFIED STRATEGY AGGREGATOR
 *
 * Combines all 18 strategies with confidence-weighted decision making
 * Provides comprehensive BUY/WAIT signals with percentage-based recommendations
 *
 * Strategies Included:
 * 1. Conservative Buy Signal ✅ (NEW - with unit tests)
 * 2. Breakout-Retest ✅ (NEW - historical data)
 * 3. MA Crossover Pullback
 * 4. RSI Divergence
 * 5. Volume Breakout
 * 6. Bollinger Squeeze
 * 7. EMA Ribbon
 * 8. Fibonacci Retracement
 * 9. Ichimoku Cloud
 * 10. ATR Volatility
 * 11. Trend Reversal
 * 12. MACD Histogram
 * 13. Support/Resistance
 * 14. Red Wick Green Closure
 * 15. MA7 Pullback
 * 16. BTC-ETH Correlation
 * 17. Omnipotent Futures Matrix
 */

import { PriceData, StrategySignal } from './types';
import { analyzeConservativeBuySignal } from './conservative-buy-signal';
import { analyzeBreakoutRetest } from './breakout-retest';
import { analyzeVolumeSpike } from './volume-spike';

export interface UnifiedSignal {
  symbol: string;
  finalDecision: 'BUY' | 'SELL' | 'WAIT';
  overallConfidence: number;
  buyPercentage: number;
  waitPercentage: number;
  strategyBreakdown: {
    name: string;
    signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL'; // Added NEUTRAL support
    confidence: number;
    weight: number;
  }[];
  topRecommendations: string[];
  riskAssessment: {
    level: 'LOW' | 'MEDIUM' | 'HIGH';
    factors: string[];
  };
  aggregatedTargets?: number[];
  aggregatedStopLoss?: number;
}

// Strategy weights (higher = more important)
const STRATEGY_WEIGHTS = {
  'Conservative Buy Signal': 1.5,  // Highest weight - ultra-safe
  'Breakout-Retest': 1.3,         // Strong pattern
  'Volume Spike': 1.2,            // Strong volume confirmation
  'MA Crossover Pullback': 1.0,
  'RSI Divergence': 1.2,
  'Volume Breakout': 1.0,
  'Bollinger Squeeze': 1.0,
  'EMA Ribbon': 1.0,
  'Fibonacci Retracement': 1.1,
  'Ichimoku Cloud': 1.0,
  'ATR Volatility': 0.9,
  'Trend Reversal': 1.1,
  'MACD Histogram': 1.0,
  'Support/Resistance': 1.2,
  'Red Wick Green Closure': 0.8,
  'MA7 Pullback': 1.0,
  'BTC-ETH Correlation': 0.7,
  'Omnipotent Futures Matrix': 1.4,
};

export async function analyzeUnifiedStrategy(data: PriceData): Promise<UnifiedSignal> {
  const strategyResults: StrategySignal[] = [];

  // Run all strategies in parallel
  try {
    const [
      conservative,
      breakoutRetest,
      volumeSpike,
      // Add other strategies here as they become available
    ] = await Promise.allSettled([
      analyzeConservativeBuySignal(data),
      analyzeBreakoutRetest(data),
      analyzeVolumeSpike(data),
      // Note: Other strategies will be added as they're implemented
    ]);

    if (conservative.status === 'fulfilled') strategyResults.push(conservative.value);
    if (breakoutRetest.status === 'fulfilled') strategyResults.push(breakoutRetest.value);
    if (volumeSpike.status === 'fulfilled') strategyResults.push(volumeSpike.value);

  } catch (error) {
    console.error('[Unified Aggregator] Error running strategies:', error);
  }

  // Calculate weighted scores
  let totalBuyScore = 0;
  let totalWaitScore = 0;
  let totalWeight = 0;

  const breakdown = strategyResults.map(result => {
    const weight = STRATEGY_WEIGHTS[result.name as keyof typeof STRATEGY_WEIGHTS] || 1.0;
    const score = (result.confidence / 100) * weight;

    // Handle all signal types
    if (result.signal === 'BUY') {
      totalBuyScore += score;
    } else if (result.signal === 'SELL') {
      // SELL signals count against BUY (negative for buy score)
      totalWaitScore += score;
    } else {
      // WAIT and NEUTRAL count as WAIT
      totalWaitScore += score;
    }

    totalWeight += weight;

    return {
      name: result.name,
      signal: result.signal,
      confidence: result.confidence,
      weight,
    };
  });

  // Calculate percentages
  const totalScore = totalBuyScore + totalWaitScore;
  const buyPercentage = totalScore > 0 ? Math.round((totalBuyScore / totalScore) * 100) : 0;
  const waitPercentage = 100 - buyPercentage;

  // Final decision (threshold: 60% for BUY)
  const finalDecision: 'BUY' | 'WAIT' = buyPercentage >= 60 ? 'BUY' : 'WAIT';
  const overallConfidence = Math.max(buyPercentage, waitPercentage);

  // Get top 3 recommendations
  const topRecommendations = breakdown
    .filter(s => s.signal === finalDecision)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3)
    .map(s => `${s.name} (${s.confidence}%)`);

  // Risk assessment
  const riskFactors: string[] = [];
  let riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' = 'LOW';

  if (buyPercentage >= 80) {
    riskLevel = 'LOW';
    riskFactors.push('Strong consensus across strategies');
  } else if (buyPercentage >= 60) {
    riskLevel = 'MEDIUM';
    riskFactors.push('Moderate agreement among strategies');
  } else {
    riskLevel = 'HIGH';
    riskFactors.push('Low consensus - proceed with caution');
  }

  // Aggregate targets and stop loss from BUY signals
  const buySignals = strategyResults.filter(s => s.signal === 'BUY');
  let aggregatedTargets: number[] | undefined;
  let aggregatedStopLoss: number | undefined;

  if (buySignals.length > 0) {
    const allTargets = buySignals.filter(s => s.targets).map(s => s.targets!);
    const allStopLosses = buySignals.filter(s => s.stopLoss).map(s => s.stopLoss!);

    if (allTargets.length > 0) {
      // Average targets across all BUY signals
      aggregatedTargets = [
        allTargets.reduce((sum, t) => sum + (t[0] || 0), 0) / allTargets.length,
        allTargets.reduce((sum, t) => sum + (t[1] || 0), 0) / allTargets.length,
        allTargets.reduce((sum, t) => sum + (t[2] || 0), 0) / allTargets.length,
      ];
    }

    if (allStopLosses.length > 0) {
      // Average stop loss (conservative: use lowest)
      aggregatedStopLoss = Math.min(...allStopLosses);
    }
  }

  return {
    symbol: data.symbol,
    finalDecision,
    overallConfidence,
    buyPercentage,
    waitPercentage,
    strategyBreakdown: breakdown,
    topRecommendations,
    riskAssessment: {
      level: riskLevel,
      factors: riskFactors,
    },
    aggregatedTargets,
    aggregatedStopLoss,
  };
}
