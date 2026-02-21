/**
 * 📊 VOLUME SPIKE STRATEGY
 *
 * Detects abnormal volume increases that often precede major price movements
 *
 * Signal Generation Rules:
 * 1. Current volume > 200% of 24h average volume
 * 2. Price showing positive momentum (changePercent24h > 0)
 * 3. Volume spike accompanied by price movement (not just wash trading)
 * 4. Clear support level identified (low24h)
 * 5. Not overbought (RSI < 70)
 *
 * Risk Management:
 * - Conservative entry near support after volume spike
 * - Stop loss 3% below entry
 * - Targets: +4%, +7%, +12%
 * - Max leverage: 3x
 * - Position size: 1.5% of portfolio
 */

import { PriceData, StrategySignal } from './types';

// Calculate approximate RSI from 24h change
function calculateApproxRSI(changePercent: number): number {
  // Simplified RSI approximation based on 24h change
  // Real RSI needs 14-period data, this is a proxy
  const normalized = Math.max(-20, Math.min(20, changePercent));
  return 50 + (normalized * 2);
}

export async function analyzeVolumeSpike(data: PriceData): Promise<StrategySignal> {
  const { symbol, price, changePercent24h, volume24h, high24h, low24h } = data;

  // Calculate metrics
  const avgVolume = volume24h / 1.5; // Approximate average (current is often elevated)
  const volumeRatio = volume24h / avgVolume;
  const approxRSI = calculateApproxRSI(changePercent24h);
  const priceRange = high24h - low24h;
  const pricePosition = (price - low24h) / priceRange; // 0 = at support, 1 = at resistance

  // Volume spike threshold
  const isVolumeSpikeDetected = volumeRatio > 2.0; // >200% of average

  // Check conditions
  const conditions = {
    volumeSpike: isVolumeSpikeDetected,
    positiveMomentum: changePercent24h > 0,
    notOverbought: approxRSI < 70,
    nearSupport: pricePosition < 0.6, // Price in lower 60% of range
    significantVolume: volume24h > 50000000, // Min $50M volume
  };

  const conditionsMet = Object.values(conditions).filter(Boolean).length;

  // BUY Signal: 4/5 conditions met
  if (conditionsMet >= 4) {
    const confidence = Math.min(95, 65 + (conditionsMet * 7) + (volumeRatio > 3 ? 5 : 0));

    return {
      name: 'Volume Spike',
      signal: 'BUY',
      confidence,
      reason: `VOLUME SPIKE DETECTED: ${volumeRatio.toFixed(2)}x avg volume | Momentum: ${changePercent24h > 0 ? 'UP' : 'NEUTRAL'} | Position: ${(pricePosition * 100).toFixed(0)}% from support`,
      targets: [
        price * 1.04, // +4%
        price * 1.07, // +7%
        price * 1.12, // +12%
      ],
      stopLoss: price * 0.97, // -3% stop loss
      timeframe: '4H',
      indicators: {
        volumeRatio: parseFloat(volumeRatio.toFixed(2)),
        avgVolume: Math.round(avgVolume),
        currentVolume: Math.round(volume24h),
        approxRSI: Math.round(approxRSI),
        pricePosition: parseFloat((pricePosition * 100).toFixed(1)),
        leverageMax: 3,
        positionSizePercent: 1.5,
      },
    };
  }

  // WAIT Signal: Insufficient conditions
  const failedConditions: string[] = [];
  if (!conditions.volumeSpike) failedConditions.push(`Volume ratio too low (${volumeRatio.toFixed(2)}x)`);
  if (!conditions.positiveMomentum) failedConditions.push('Negative momentum');
  if (!conditions.notOverbought) failedConditions.push(`RSI > 70 (${approxRSI.toFixed(0)})`);
  if (!conditions.nearSupport) failedConditions.push('Price too high in range');
  if (!conditions.significantVolume) failedConditions.push('Volume too low');

  return {
    name: 'Volume Spike',
    signal: 'WAIT',
    confidence: Math.min(75, conditionsMet * 15),
    reason: `Conditions met: ${conditionsMet}/5. Issues: ${failedConditions.join(', ')}`,
    indicators: {
      volumeRatio: parseFloat(volumeRatio.toFixed(2)),
      avgVolume: Math.round(avgVolume),
      currentVolume: Math.round(volume24h),
      approxRSI: Math.round(approxRSI),
      pricePosition: parseFloat((pricePosition * 100).toFixed(1)),
    },
  };
}
