/**
 * TRADING SIGNAL CALCULATOR
 * Professional trading signal calculation with entry, exit, SL, TP, and leverage
 * Zero-error design for alfabetik pattern trading
 */

export interface TradingSignal {
  symbol: string;
  currentPrice: number;
  signal: "LONG" | "SHORT" | "HOLD";
  confidence: number;

  // Entry & Exit
  entryPrice: number;
  targetPrice: number;
  stopLoss: number;

  // Risk Management
  riskRewardRatio: number;
  recommendedLeverage: number;
  positionSize: number; // Percentage of portfolio

  // Performance metrics
  expectedProfit: number; // Percentage
  maxLoss: number; // Percentage

  // Additional info
  timeframe: string;
  volatility: number;
  trend: "BULLISH" | "BEARISH" | "NEUTRAL";
}

export interface CoinTradingDetails extends TradingSignal {
  // Market data
  volume24h: number;
  changePercent24h: number;
  changePercent7d: number;
  changePercent30d: number;

  // Technical indicators (simplified)
  rsi: number;
  momentum: string;

  // Pattern info
  patternLetter: string;
  patternConfidence: number;
}

/**
 * Calculate RSI (simplified version)
 */
function calculateRSI(
  _currentPrice: number,
  change24h: number,
  change7d: number
): number {
  // Simplified RSI based on recent performance
  const recentGain = Math.max(0, change24h);
  const recentLoss = Math.abs(Math.min(0, change24h));

  const avgGain = (recentGain + Math.max(0, change7d)) / 2;
  const avgLoss = (recentLoss + Math.abs(Math.min(0, change7d))) / 2;

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));

  return Math.max(0, Math.min(100, rsi));
}

/**
 * Calculate volatility based on performance
 */
function calculateVolatility(
  change24h: number,
  change7d: number,
  change30d: number
): number {
  const changes = [change24h, change7d / 7, change30d / 30];
  const avg = changes.reduce((a, b) => a + b, 0) / changes.length;
  const variance = changes.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / changes.length;
  return Math.sqrt(variance);
}

/**
 * Determine trend from performance metrics
 */
function determineTrend(
  change24h: number,
  change7d: number,
  change30d: number
): "BULLISH" | "BEARISH" | "NEUTRAL" {
  const shortTerm = change24h;
  const mediumTerm = change7d / 7;
  const longTerm = change30d / 30;

  const bullishScore =
    (shortTerm > 0 ? 1 : 0) +
    (mediumTerm > 0 ? 1 : 0) +
    (longTerm > 0 ? 1 : 0);

  if (bullishScore >= 2 && shortTerm > 2) return "BULLISH";
  if (bullishScore <= 1 && shortTerm < -2) return "BEARISH";
  return "NEUTRAL";
}

/**
 * Calculate recommended leverage based on confidence and volatility
 */
function calculateLeverage(
  confidence: number,
  volatility: number,
  signal: string
): number {
  // Base leverage on confidence
  let leverage = 1;

  if (signal === "HOLD") return 1;

  // High confidence = higher leverage potential
  if (confidence >= 80) leverage = 5;
  else if (confidence >= 70) leverage = 3;
  else if (confidence >= 60) leverage = 2;
  else leverage = 1;

  // Reduce leverage for high volatility
  if (volatility > 10) leverage = Math.max(1, leverage - 1);
  if (volatility > 20) leverage = 1;

  return Math.max(1, Math.min(10, leverage));
}

/**
 * Calculate entry price with slight buffer
 */
function calculateEntryPrice(
  currentPrice: number,
  signal: "LONG" | "SHORT" | "HOLD",
  volatility: number
): number {
  if (signal === "HOLD") return currentPrice;

  // Add small buffer based on volatility
  const buffer = volatility > 5 ? 0.005 : 0.002; // 0.5% or 0.2%

  if (signal === "LONG") {
    return currentPrice * (1 + buffer); // Slightly above current
  } else {
    return currentPrice * (1 - buffer); // Slightly below current
  }
}

/**
 * Calculate target price (take profit)
 */
function calculateTargetPrice(
  entryPrice: number,
  signal: "LONG" | "SHORT" | "HOLD",
  confidence: number,
  volatility: number,
  change7d: number
): number {
  if (signal === "HOLD") return entryPrice;

  // Base target on 7d performance and confidence
  let targetPercent = (Math.abs(change7d) / 7) * 3; // 3x daily average

  // Adjust by confidence
  targetPercent *= (confidence / 100);

  // Add volatility bonus
  targetPercent += volatility * 0.3;

  // Minimum 2%, maximum 25%
  targetPercent = Math.max(2, Math.min(25, targetPercent));

  if (signal === "LONG") {
    return entryPrice * (1 + targetPercent / 100);
  } else {
    return entryPrice * (1 - targetPercent / 100);
  }
}

/**
 * Calculate stop loss
 */
function calculateStopLoss(
  entryPrice: number,
  signal: "LONG" | "SHORT" | "HOLD",
  volatility: number,
  leverage: number
): number {
  if (signal === "HOLD") return entryPrice * 0.95;

  // Base stop loss on volatility and leverage
  let stopPercent = volatility * 0.5; // Half of volatility

  // Tighter stop loss for higher leverage
  if (leverage >= 5) stopPercent *= 0.6;
  else if (leverage >= 3) stopPercent *= 0.75;

  // Minimum 1%, maximum 8%
  stopPercent = Math.max(1, Math.min(8, stopPercent));

  if (signal === "LONG") {
    return entryPrice * (1 - stopPercent / 100);
  } else {
    return entryPrice * (1 + stopPercent / 100);
  }
}

/**
 * Calculate position size (% of portfolio to risk)
 */
function calculatePositionSize(
  confidence: number,
  volatility: number,
  leverage: number
): number {
  // Base on confidence
  let positionSize = confidence / 100 * 10; // Max 10% of portfolio

  // Reduce for high volatility
  if (volatility > 15) positionSize *= 0.5;
  else if (volatility > 10) positionSize *= 0.75;

  // Reduce for high leverage
  if (leverage >= 5) positionSize *= 0.6;
  else if (leverage >= 3) positionSize *= 0.8;

  // Minimum 1%, maximum 10%
  return Math.max(1, Math.min(10, positionSize));
}

/**
 * Main function: Generate trading signal for a coin
 */
export function generateTradingSignal(
  symbol: string,
  currentPrice: number,
  patternSignal: "STRONG_BUY" | "BUY" | "SELL" | "HOLD",
  patternConfidence: number,
  change24h: number,
  change7d: number,
  change30d: number,
  volume24h: number,
  momentum: "YUKARIDA" | "ASAGIDA" | "YATAY",
  patternLetter: string
): CoinTradingDetails {
  // Convert pattern signal to trading signal
  let signal: "LONG" | "SHORT" | "HOLD";
  if (patternSignal === "STRONG_BUY" || patternSignal === "BUY") {
    signal = "LONG";
  } else if (patternSignal === "SELL") {
    signal = "SHORT";
  } else {
    signal = "HOLD";
  }

  // Calculate technical indicators
  const rsi = calculateRSI(currentPrice, change24h, change7d);
  const volatility = calculateVolatility(change24h, change7d, change30d);
  const trend = determineTrend(change24h, change7d, change30d);

  // Adjust confidence based on technical factors
  let adjustedConfidence = patternConfidence;

  // RSI confirmation
  if (signal === "LONG" && rsi < 30) adjustedConfidence += 10; // Oversold = good for long
  if (signal === "LONG" && rsi > 70) adjustedConfidence -= 15; // Overbought = bad for long
  if (signal === "SHORT" && rsi > 70) adjustedConfidence += 10; // Overbought = good for short
  if (signal === "SHORT" && rsi < 30) adjustedConfidence -= 15; // Oversold = bad for short

  // Trend confirmation
  if (signal === "LONG" && trend === "BULLISH") adjustedConfidence += 10;
  if (signal === "LONG" && trend === "BEARISH") adjustedConfidence -= 20;
  if (signal === "SHORT" && trend === "BEARISH") adjustedConfidence += 10;
  if (signal === "SHORT" && trend === "BULLISH") adjustedConfidence -= 20;

  // Momentum confirmation
  if (signal === "LONG" && momentum === "YUKARIDA") adjustedConfidence += 5;
  if (signal === "LONG" && momentum === "ASAGIDA") adjustedConfidence -= 10;
  if (signal === "SHORT" && momentum === "ASAGIDA") adjustedConfidence += 5;
  if (signal === "SHORT" && momentum === "YUKARIDA") adjustedConfidence -= 10;

  // Clamp confidence between 0-100
  adjustedConfidence = Math.max(0, Math.min(100, adjustedConfidence));

  // Calculate entry, target, and stop loss
  const entryPrice = calculateEntryPrice(currentPrice, signal, volatility);
  const leverage = calculateLeverage(adjustedConfidence, volatility, signal);
  const targetPrice = calculateTargetPrice(entryPrice, signal, adjustedConfidence, volatility, change7d);
  const stopLoss = calculateStopLoss(entryPrice, signal, volatility, leverage);

  // Calculate risk/reward ratio
  const potentialProfit = Math.abs(targetPrice - entryPrice);
  const potentialLoss = Math.abs(entryPrice - stopLoss);
  const riskRewardRatio = potentialLoss > 0 ? potentialProfit / potentialLoss : 3;

  // Calculate expected profit and max loss percentages
  const expectedProfit = ((targetPrice - entryPrice) / entryPrice) * 100 * (signal === "SHORT" ? -1 : 1);
  const maxLoss = ((stopLoss - entryPrice) / entryPrice) * 100 * (signal === "SHORT" ? -1 : 1);

  // Calculate position size
  const positionSize = calculatePositionSize(adjustedConfidence, volatility, leverage);

  // Determine timeframe based on volatility and confidence
  let timeframe = "24h-72h";
  if (volatility > 15 || adjustedConfidence < 60) timeframe = "1h-24h";
  if (volatility < 5 && adjustedConfidence > 75) timeframe = "3d-7d";

  return {
    symbol,
    currentPrice,
    signal,
    confidence: Math.round(adjustedConfidence),
    entryPrice: Number(entryPrice.toFixed(6)),
    targetPrice: Number(targetPrice.toFixed(6)),
    stopLoss: Number(stopLoss.toFixed(6)),
    riskRewardRatio: Number(riskRewardRatio.toFixed(2)),
    recommendedLeverage: leverage,
    positionSize: Number(positionSize.toFixed(1)),
    expectedProfit: Number(expectedProfit.toFixed(2)),
    maxLoss: Number(maxLoss.toFixed(2)),
    timeframe,
    volatility: Number(volatility.toFixed(2)),
    trend,
    volume24h,
    changePercent24h: change24h,
    changePercent7d: change7d,
    changePercent30d: change30d,
    rsi: Math.round(rsi),
    momentum: momentum === "YUKARIDA" ? "Strong Up" : momentum === "ASAGIDA" ? "Strong Down" : "Neutral",
    patternLetter,
    patternConfidence
  };
}

/**
 * Batch generate trading signals for multiple coins
 */
export function generateBatchTradingSignals(
  coins: Array<{
    symbol: string;
    currentPrice: number;
    patternSignal: "STRONG_BUY" | "BUY" | "SELL" | "HOLD";
    patternConfidence: number;
    change24h: number;
    change7d: number;
    change30d: number;
    volume24h: number;
    momentum: "YUKARIDA" | "ASAGIDA" | "YATAY";
    patternLetter: string;
  }>
): CoinTradingDetails[] {
  return coins.map(coin => generateTradingSignal(
    coin.symbol,
    coin.currentPrice,
    coin.patternSignal,
    coin.patternConfidence,
    coin.change24h,
    coin.change7d,
    coin.change30d,
    coin.volume24h,
    coin.momentum,
    coin.patternLetter
  ));
}
