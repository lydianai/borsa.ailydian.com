/**
 * 🌟 OMNIPOTENT FUTURES MATRIX v6.0 - Simplified & Optimized
 *
 * Simplified Binance Futures USDT-M analysis system
 * - 5-layer confirmation system (reduced from 12)
 * - Deterministic calculations (no randomness)
 * - Ultra-conservative risk management
 * - Clear, maintainable code
 *
 * Target: 85%+ accuracy, max 0.25% risk per trade
 */

import { PriceData, StrategySignal, SignalType } from './types';

interface SimplifiedAnalysis {
  // Core metrics
  marketPhase: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  volumeStrength: number;
  volatility: number;
  momentum: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';

  // Scores
  totalScore: number;
  confidence: number;
  signal: SignalType;
}

/**
 * Detect market phase based on price position and momentum
 */
function detectMarketPhase(data: PriceData): 'BULLISH' | 'BEARISH' | 'NEUTRAL' {
  const { price, high24h, low24h, changePercent24h } = data;
  const pricePosition = (price - low24h) / (high24h - low24h || 1);

  // Strong signals
  if (changePercent24h > 5 && pricePosition > 0.6) return 'BULLISH';
  if (changePercent24h < -5 && pricePosition < 0.4) return 'BEARISH';

  // Moderate signals
  if (changePercent24h > 2 && pricePosition > 0.5) return 'BULLISH';
  if (changePercent24h < -2 && pricePosition < 0.5) return 'BEARISH';

  return 'NEUTRAL';
}

/**
 * Calculate volume strength (0-100)
 */
function calculateVolumeStrength(data: PriceData): number {
  const { volume24h, high24h, changePercent24h } = data;

  // Normalized volume relative to price
  const normalizedVolume = volume24h / (high24h * 1000000 || 1);

  // Volume-price correlation strength
  const vpStrength = Math.abs(changePercent24h) * normalizedVolume * 10;

  return Math.min(100, Math.round(vpStrength));
}

/**
 * Calculate volatility percentage
 */
function calculateVolatility(data: PriceData): number {
  const { high24h, low24h, price } = data;
  const volatility = ((high24h - low24h) / (price || 1)) * 100;
  return Math.round(volatility * 10) / 10;
}

/**
 * Assess risk level
 */
function assessRiskLevel(volatility: number, volumeStrength: number): 'LOW' | 'MEDIUM' | 'HIGH' {
  // High volatility or low volume = high risk
  if (volatility > 8 || volumeStrength < 30) return 'HIGH';
  if (volatility > 5 || volumeStrength < 50) return 'MEDIUM';
  return 'LOW';
}

/**
 * Main analysis function - 5-layer simplified system
 */
export async function analyzeOmnipotentFuturesMatrix(data: PriceData): Promise<StrategySignal> {
  const { changePercent24h, price, symbol } = data;

  // LAYER 1: Market Phase Detection
  const marketPhase = detectMarketPhase(data);
  const phaseScore = marketPhase === 'BULLISH' ? 80 : marketPhase === 'BEARISH' ? 20 : 50;

  // LAYER 2: Volume Analysis
  const volumeStrength = calculateVolumeStrength(data);
  const volumeScore = volumeStrength;

  // LAYER 3: Volatility Check
  const volatility = calculateVolatility(data);
  const volatilityScore = Math.max(0, 100 - (volatility * 8)); // High volatility = lower score

  // LAYER 4: Momentum Assessment
  const momentum = Math.abs(changePercent24h);
  const momentumScore = Math.min(100, momentum * 8);

  // LAYER 5: Risk Assessment
  const riskLevel = assessRiskLevel(volatility, volumeStrength);
  const riskScore = riskLevel === 'LOW' ? 90 : riskLevel === 'MEDIUM' ? 60 : 30;

  // TOTAL SCORE (weighted average)
  const totalScore = Math.round(
    (phaseScore * 0.30) +      // 30% weight on market phase
    (volumeScore * 0.25) +     // 25% weight on volume
    (volatilityScore * 0.20) + // 20% weight on volatility
    (momentumScore * 0.15) +   // 15% weight on momentum
    (riskScore * 0.10)         // 10% weight on risk
  );

  // CONFIDENCE CALCULATION
  let confidence = totalScore;

  // Bonuses and penalties
  if (volumeStrength > 70) confidence += 5;
  if (riskLevel === 'HIGH') confidence -= 15;
  if (volatility > 10) confidence -= 10;

  confidence = Math.max(0, Math.min(100, confidence));

  // SIGNAL DETERMINATION
  let signal: SignalType;

  if (totalScore >= 75 && marketPhase === 'BULLISH' && riskLevel !== 'HIGH') {
    signal = 'BUY';
  } else if (totalScore <= 30 && marketPhase === 'BEARISH' && riskLevel !== 'HIGH') {
    signal = 'SELL';
  } else if (totalScore >= 50 && totalScore < 75) {
    signal = 'WAIT';
  } else {
    signal = 'NEUTRAL';
  }

  // REASON GENERATION
  const reason = [
    `Omnipotent Matrix v6.0: Score ${totalScore}/100`,
    `Phase: ${marketPhase}`,
    `Volume: ${volumeStrength}/100`,
    `Volatility: ${volatility}%`,
    `Momentum: ${changePercent24h > 0 ? '+' : ''}${changePercent24h.toFixed(2)}%`,
    `Risk: ${riskLevel}`,
  ].join(' | ');

  // TARGETS & STOP LOSS (ultra-conservative)
  let targets: number[] | undefined;
  let stopLoss: number | undefined;

  if (signal === 'BUY') {
    targets = [
      price * 1.01,  // TP1: +1%
      price * 1.02,  // TP2: +2%
      price * 1.03,  // TP3: +3%
    ];
    stopLoss = price * 0.9975; // SL: -0.25%
  } else if (signal === 'SELL') {
    targets = [
      price * 0.99,  // TP1: -1%
      price * 0.98,  // TP2: -2%
      price * 0.97,  // TP3: -3%
    ];
    stopLoss = price * 1.0025; // SL: +0.25%
  }

  return {
    name: 'Omnipotent Futures Matrix',
    signal,
    confidence,
    reason,
    targets,
    stopLoss,
    timeframe: '1D',
    indicators: {
      totalScore,
      marketPhase,
      volumeStrength,
      volatility,
      momentum: Math.round(momentum * 10) / 10,
      riskLevel,
      phaseScore,
      volumeScore,
      volatilityScore,
      momentumScore,
      riskScore,
    },
  };
}
