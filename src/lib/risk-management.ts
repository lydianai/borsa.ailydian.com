/**
 * 🎯 RISK MANAGEMENT CALCULATOR
 * Kelly Criterion, Position Sizing, Portfolio Risk Metrics
 *
 * White-Hat Compliance:
 * - Educational purposes only
 * - No financial advice
 * - Mathematical calculations only
 */

// ============================================================================
// INTERFACES
// ============================================================================

export interface TradeHistory {
  wins: number;
  losses: number;
  avgWin: number;
  avgLoss: number;
  totalTrades: number;
}

export interface KellyCriterion {
  kellyPercentage: number;  // Optimal position size
  fractionalKelly: number;  // Half Kelly (safer)
  quarterKelly: number;     // Quarter Kelly (very conservative)
  recommendation: string;
  riskLevel: 'AGGRESSIVE' | 'MODERATE' | 'CONSERVATIVE' | 'TOO_RISKY';
}

export interface PositionSizeRecommendation {
  accountSize: number;
  riskPerTrade: number;      // Dollars
  riskPercentage: number;    // Percentage of account
  positionSize: number;      // Dollars to allocate
  leverage: number;
  stopLossDistance: number;  // Percentage
  contractQuantity?: number; // For futures
}

export interface PortfolioRiskMetrics {
  totalValue: number;
  allocatedCapital: number;
  availableCapital: number;
  usedLeverage: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  profitFactor: number;
  expectancy: number;
  riskScore: number;        // 0-100
  riskRating: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
}

export interface RiskReward {
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  riskAmount: number;
  rewardAmount: number;
  riskRewardRatio: number;
  recommendation: string;
}

// ============================================================================
// KELLY CRITERION CALCULATOR
// ============================================================================

/**
 * Calculate Kelly Criterion
 * Formula: K% = W - (1-W)/R
 * Where:
 *   W = Win rate (probability of winning)
 *   R = Win/Loss ratio (average win / average loss)
 */
export function calculateKellyCriterion(tradeHistory: TradeHistory): KellyCriterion {
  const { wins, losses, avgWin, avgLoss, totalTrades } = tradeHistory;

  if (totalTrades === 0 || avgLoss === 0) {
    return {
      kellyPercentage: 0,
      fractionalKelly: 0,
      quarterKelly: 0,
      recommendation: 'Insufficient data for Kelly calculation',
      riskLevel: 'CONSERVATIVE',
    };
  }

  // Win rate
  const winRate = wins / totalTrades;

  // Win/Loss ratio
  const winLossRatio = avgWin / avgLoss;

  // Kelly percentage
  const kelly = winRate - (1 - winRate) / winLossRatio;

  // Clamp to 0-100%
  const kellyPercentage = Math.max(0, Math.min(100, kelly * 100));

  // Fractional Kelly (safer approach)
  const fractionalKelly = kellyPercentage * 0.5;  // Half Kelly
  const quarterKelly = kellyPercentage * 0.25;    // Quarter Kelly

  // Risk level assessment
  let riskLevel: 'AGGRESSIVE' | 'MODERATE' | 'CONSERVATIVE' | 'TOO_RISKY';
  let recommendation: string;

  if (kellyPercentage <= 0) {
    riskLevel = 'TOO_RISKY';
    recommendation = 'Negative Kelly - Do not trade this strategy. Edge is negative.';
  } else if (kellyPercentage < 5) {
    riskLevel = 'CONSERVATIVE';
    recommendation = `Kelly suggests ${kellyPercentage.toFixed(2)}% position size. Use ${fractionalKelly.toFixed(2)}% (Half Kelly) for safety.`;
  } else if (kellyPercentage < 15) {
    riskLevel = 'MODERATE';
    recommendation = `Kelly suggests ${kellyPercentage.toFixed(2)}% position size. Use ${fractionalKelly.toFixed(2)}% (Half Kelly) recommended.`;
  } else if (kellyPercentage < 30) {
    riskLevel = 'AGGRESSIVE';
    recommendation = `Kelly suggests ${kellyPercentage.toFixed(2)}% position size. Consider ${fractionalKelly.toFixed(2)}% (Half Kelly) to reduce risk.`;
  } else {
    riskLevel = 'TOO_RISKY';
    recommendation = `Kelly suggests ${kellyPercentage.toFixed(2)}% position size. This is too high! Use max ${quarterKelly.toFixed(2)}% (Quarter Kelly).`;
  }

  return {
    kellyPercentage,
    fractionalKelly,
    quarterKelly,
    recommendation,
    riskLevel,
  };
}

// ============================================================================
// POSITION SIZING CALCULATOR
// ============================================================================

/**
 * Calculate recommended position size
 * Uses percentage risk method
 */
export function calculatePositionSize(
  accountSize: number,
  riskPercentage: number,  // e.g., 1 = 1% risk per trade
  entryPrice: number,
  stopLossPrice: number,
  leverage: number = 1
): PositionSizeRecommendation {
  // Risk amount in dollars
  const riskAmount = accountSize * (riskPercentage / 100);

  // Stop loss distance (percentage)
  const stopLossDistance = Math.abs((entryPrice - stopLossPrice) / entryPrice) * 100;

  // Position size without leverage
  const basePositionSize = riskAmount / (stopLossDistance / 100);

  // Position size with leverage
  const positionSize = basePositionSize * leverage;

  // Contract quantity (for futures)
  const contractQuantity = Math.floor(positionSize / entryPrice);

  return {
    accountSize,
    riskPerTrade: riskAmount,
    riskPercentage,
    positionSize,
    leverage,
    stopLossDistance,
    contractQuantity,
  };
}

// ============================================================================
// RISK/REWARD CALCULATOR
// ============================================================================

/**
 * Calculate risk/reward ratio
 */
export function calculateRiskReward(
  entryPrice: number,
  stopLoss: number,
  takeProfit: number
): RiskReward {
  const riskAmount = Math.abs(entryPrice - stopLoss);
  const rewardAmount = Math.abs(takeProfit - entryPrice);
  const riskRewardRatio = rewardAmount / riskAmount;

  let recommendation: string;

  if (riskRewardRatio < 1) {
    recommendation = `Poor R:R (${riskRewardRatio.toFixed(2)}:1). Risk more than reward - NOT RECOMMENDED.`;
  } else if (riskRewardRatio < 1.5) {
    recommendation = `Low R:R (${riskRewardRatio.toFixed(2)}:1). Need high win rate (>60%) to be profitable.`;
  } else if (riskRewardRatio < 2) {
    recommendation = `Decent R:R (${riskRewardRatio.toFixed(2)}:1). Need 50%+ win rate to be profitable.`;
  } else if (riskRewardRatio < 3) {
    recommendation = `Good R:R (${riskRewardRatio.toFixed(2)}:1). Need 40%+ win rate to be profitable.`;
  } else {
    recommendation = `Excellent R:R (${riskRewardRatio.toFixed(2)}:1). Even 30% win rate can be profitable.`;
  }

  return {
    entryPrice,
    stopLoss,
    takeProfit,
    riskAmount,
    rewardAmount,
    riskRewardRatio,
    recommendation,
  };
}

// ============================================================================
// PORTFOLIO RISK METRICS
// ============================================================================

/**
 * Calculate Sharpe Ratio
 * Measures risk-adjusted returns
 */
export function calculateSharpeRatio(
  returns: number[],  // Array of period returns (%)
  riskFreeRate: number = 0  // Annual risk-free rate (e.g., 3% = 3)
): number {
  if (returns.length === 0) return 0;

  const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);

  if (stdDev === 0) return 0;

  return (avgReturn - riskFreeRate / 100) / stdDev;
}

/**
 * Calculate Max Drawdown
 * Maximum peak-to-trough decline
 */
export function calculateMaxDrawdown(equityCurve: number[]): number {
  if (equityCurve.length === 0) return 0;

  let maxDrawdown = 0;
  let peak = equityCurve[0];

  for (const value of equityCurve) {
    if (value > peak) {
      peak = value;
    }

    const drawdown = ((peak - value) / peak) * 100;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return maxDrawdown;
}

/**
 * Calculate Profit Factor
 * Gross profit / Gross loss
 */
export function calculateProfitFactor(
  grossProfit: number,
  grossLoss: number
): number {
  if (grossLoss === 0) return grossProfit > 0 ? Infinity : 0;
  return grossProfit / Math.abs(grossLoss);
}

/**
 * Calculate Expectancy
 * Average expected profit per trade
 */
export function calculateExpectancy(
  winRate: number,      // 0-1 (e.g., 0.6 = 60%)
  avgWin: number,
  avgLoss: number
): number {
  const lossRate = 1 - winRate;
  return (winRate * avgWin) - (lossRate * Math.abs(avgLoss));
}

/**
 * Calculate overall portfolio risk metrics
 */
export function calculatePortfolioRiskMetrics(
  accountValue: number,
  openPositions: Array<{
    size: number;
    leverage: number;
    entryPrice: number;
    currentPrice: number;
  }>,
  tradeHistory: TradeHistory,
  equityCurve: number[]
): PortfolioRiskMetrics {
  // Calculate allocated capital
  const allocatedCapital = openPositions.reduce((sum, pos) => sum + pos.size, 0);
  const availableCapital = accountValue - allocatedCapital;

  // Calculate used leverage
  const leveragedExposure = openPositions.reduce(
    (sum, pos) => sum + pos.size * pos.leverage,
    0
  );
  const usedLeverage = leveragedExposure / accountValue;

  // Calculate max drawdown
  const maxDrawdown = calculateMaxDrawdown(equityCurve);

  // Calculate Sharpe ratio (using last 30 returns)
  const returns = [];
  for (let i = 1; i < equityCurve.length && i <= 30; i++) {
    const returnPct = ((equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1]) * 100;
    returns.push(returnPct);
  }
  const sharpeRatio = calculateSharpeRatio(returns);

  // Calculate win rate
  const winRate = tradeHistory.totalTrades > 0
    ? (tradeHistory.wins / tradeHistory.totalTrades) * 100
    : 0;

  // Calculate profit factor
  const grossProfit = tradeHistory.wins * tradeHistory.avgWin;
  const grossLoss = tradeHistory.losses * tradeHistory.avgLoss;
  const profitFactor = calculateProfitFactor(grossProfit, grossLoss);

  // Calculate expectancy
  const expectancy = calculateExpectancy(
    winRate / 100,
    tradeHistory.avgWin,
    tradeHistory.avgLoss
  );

  // Calculate risk score (0-100)
  let riskScore = 0;
  riskScore += Math.min(maxDrawdown, 50); // Max drawdown contributes up to 50 points
  riskScore += Math.min(usedLeverage * 10, 30); // Leverage contributes up to 30 points
  riskScore += Math.min((allocatedCapital / accountValue) * 20, 20); // Capital usage up to 20 points

  // Risk rating
  let riskRating: 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
  if (riskScore < 25) {
    riskRating = 'LOW';
  } else if (riskScore < 50) {
    riskRating = 'MEDIUM';
  } else if (riskScore < 75) {
    riskRating = 'HIGH';
  } else {
    riskRating = 'EXTREME';
  }

  return {
    totalValue: accountValue,
    allocatedCapital,
    availableCapital,
    usedLeverage,
    maxDrawdown,
    sharpeRatio,
    winRate,
    profitFactor,
    expectancy,
    riskScore,
    riskRating,
  };
}

// ============================================================================
// RISK MANAGEMENT RECOMMENDATIONS
// ============================================================================

/**
 * Get risk management recommendations based on current state
 */
export function getRiskRecommendations(metrics: PortfolioRiskMetrics): string[] {
  const recommendations: string[] = [];

  // Leverage warnings
  if (metrics.usedLeverage > 10) {
    recommendations.push('⚠️ EXTREME LEVERAGE: Reduce leverage immediately. Risk of liquidation is very high.');
  } else if (metrics.usedLeverage > 5) {
    recommendations.push('⚠️ HIGH LEVERAGE: Consider reducing leverage to below 5x for safety.');
  } else if (metrics.usedLeverage > 3) {
    recommendations.push('⚡ Moderate leverage detected. Monitor positions closely.');
  }

  // Drawdown warnings
  if (metrics.maxDrawdown > 30) {
    recommendations.push('🚨 MAX DRAWDOWN EXCEEDED: Stop trading! Review strategy. Max drawdown is too high.');
  } else if (metrics.maxDrawdown > 20) {
    recommendations.push('⚠️ HIGH DRAWDOWN: Reduce position sizes. Current drawdown is concerning.');
  } else if (metrics.maxDrawdown > 10) {
    recommendations.push('⚡ Moderate drawdown detected. Consider tightening stop losses.');
  }

  // Capital allocation
  if (metrics.allocatedCapital / metrics.totalValue > 0.8) {
    recommendations.push('⚠️ OVER-ALLOCATED: 80%+ capital in use. Keep some cash for opportunities.');
  } else if (metrics.allocatedCapital / metrics.totalValue < 0.2) {
    recommendations.push('💡 UNDER-UTILIZED: Only 20% capital used. Consider increasing exposure if conditions are favorable.');
  }

  // Win rate
  if (metrics.winRate < 30) {
    recommendations.push('📉 LOW WIN RATE: Win rate below 30%. Need better R:R ratio or strategy adjustment.');
  } else if (metrics.winRate > 70) {
    recommendations.push('✅ EXCELLENT WIN RATE: Above 70% wins. Maintain discipline.');
  }

  // Profit factor
  if (metrics.profitFactor < 1) {
    recommendations.push('🚨 NEGATIVE PROFIT FACTOR: Losing money overall. Stop trading and review strategy.');
  } else if (metrics.profitFactor < 1.5) {
    recommendations.push('⚠️ LOW PROFIT FACTOR: Barely profitable. Need improvements.');
  } else if (metrics.profitFactor > 2) {
    recommendations.push('✅ STRONG PROFIT FACTOR: Above 2.0. Strategy is working well.');
  }

  // Sharpe ratio
  if (metrics.sharpeRatio < 0) {
    recommendations.push('📉 NEGATIVE SHARPE RATIO: Returns not compensating for risk.');
  } else if (metrics.sharpeRatio > 2) {
    recommendations.push('✅ EXCELLENT SHARPE RATIO: Risk-adjusted returns are strong.');
  }

  // Expectancy
  if (metrics.expectancy < 0) {
    recommendations.push('🚨 NEGATIVE EXPECTANCY: Expected to lose money per trade. Stop trading.');
  } else if (metrics.expectancy > 0) {
    recommendations.push(`✅ POSITIVE EXPECTANCY: Expected $${metrics.expectancy.toFixed(2)} profit per trade.`);
  }

  // Overall risk
  if (metrics.riskRating === 'EXTREME') {
    recommendations.push('🚨 EXTREME RISK: Reduce all positions immediately. Account is in danger.');
  } else if (metrics.riskRating === 'HIGH') {
    recommendations.push('⚠️ HIGH RISK: Consider reducing exposure and leverage.');
  } else if (metrics.riskRating === 'LOW') {
    recommendations.push('✅ LOW RISK: Portfolio risk is well-managed.');
  }

  return recommendations;
}

// ============================================================================
// WHITE-HAT COMPLIANCE
// ============================================================================

export const RISK_MANAGEMENT_CONFIG = {
  // Max recommended values
  MAX_LEVERAGE: 10,
  MAX_POSITION_SIZE_PCT: 20,  // 20% of account per position
  MAX_TOTAL_RISK_PCT: 10,     // 10% total account risk
  MAX_DRAWDOWN_PCT: 20,       // 20% max drawdown threshold

  // Recommended values
  RECOMMENDED_RISK_PER_TRADE: 1,  // 1% per trade
  RECOMMENDED_LEVERAGE: 3,         // 3x leverage max
  RECOMMENDED_WIN_RATE: 50,        // 50%+ win rate target

  // Kelly Criterion
  USE_FRACTIONAL_KELLY: true,  // Use half Kelly by default
  MAX_KELLY_PCT: 25,           // Never risk more than 25% even if Kelly says so
};

console.log('✅ Risk Management Calculator initialized with White-Hat compliance');
console.log('⚠️ DISCLAIMER: For educational purposes only. Not financial advice.');
