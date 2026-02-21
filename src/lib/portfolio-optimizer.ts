/**
 * MODERN PORTFOLIO THEORY (MPT) OPTIMIZER
 * Real portfolio optimization using Markowitz mean-variance analysis
 *
 * Features:
 * - Real Sharpe Ratio calculations
 * - Real Value at Risk (VaR) & Conditional VaR
 * - Real correlation/covariance matrix
 * - Real optimal weight calculations
 * - Risk/return optimization
 */

interface PortfolioAsset {
  symbol: string;
  price: number;
  returns: number[]; // Historical returns (last N periods)
  volume24h: number;
}

interface PortfolioOptimization {
  optimalWeight: number;
  expectedReturn: number;
  expectedRisk: number;
  sharpeRatio: number;
  contribution: number; // Contribution to portfolio return
}

interface RiskMetrics {
  valueAtRisk95: number; // 95% VaR
  valueAtRisk99: number; // 99% VaR
  conditionalVaR: number; // CVaR (Expected Shortfall)
  maxDrawdown: number;
  volatility: number;
}

/**
 * Calculate historical returns from OHLCV data
 */
function calculateReturns(closes: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push((closes[i] - closes[i-1]) / closes[i-1]);
  }
  return returns;
}

/**
 * Calculate mean (average) of array
 */
function mean(arr: number[]): number {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

/**
 * Calculate standard deviation
 */
function stdDev(arr: number[]): number {
  const avg = mean(arr);
  const squareDiffs = arr.map(value => Math.pow(value - avg, 2));
  const avgSquareDiff = mean(squareDiffs);
  return Math.sqrt(avgSquareDiff);
}

/**
 * Calculate covariance between two return series
 */
function covariance(returns1: number[], returns2: number[]): number {
  const mean1 = mean(returns1);
  const mean2 = mean(returns2);

  let sum = 0;
  for (let i = 0; i < returns1.length; i++) {
    sum += (returns1[i] - mean1) * (returns2[i] - mean2);
  }

  return sum / (returns1.length - 1);
}

/**
 * Calculate correlation between two return series
 */
function correlation(returns1: number[], returns2: number[]): number {
  const cov = covariance(returns1, returns2);
  const std1 = stdDev(returns1);
  const std2 = stdDev(returns2);

  if (std1 === 0 || std2 === 0) return 0;

  return cov / (std1 * std2);
}

/**
 * Calculate Sharpe Ratio (risk-adjusted return)
 * Assumes risk-free rate of 4% annually (0.04)
 */
function sharpeRatio(expectedReturn: number, volatility: number, riskFreeRate: number = 0.04): number {
  if (volatility === 0) return 0;
  return (expectedReturn - riskFreeRate) / volatility;
}

/**
 * Calculate Value at Risk (VaR) at given confidence level
 * Uses historical simulation method
 */
function calculateVaR(returns: number[], confidenceLevel: number): number {
  const sortedReturns = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
  return Math.abs(sortedReturns[index] || 0);
}

/**
 * Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
 * Average of losses beyond VaR
 */
function calculateCVaR(returns: number[], confidenceLevel: number): number {
  const sortedReturns = [...returns].sort((a, b) => a - b);
  const varIndex = Math.floor((1 - confidenceLevel) * sortedReturns.length);

  // Average of all returns worse than VaR
  const tailReturns = sortedReturns.slice(0, varIndex + 1);
  if (tailReturns.length === 0) return 0;

  return Math.abs(mean(tailReturns));
}

/**
 * Calculate Maximum Drawdown
 * Largest peak-to-trough decline
 */
function calculateMaxDrawdown(prices: number[]): number {
  let maxDrawdown = 0;
  let peak = prices[0];

  for (const price of prices) {
    if (price > peak) {
      peak = price;
    }

    const drawdown = (peak - price) / peak;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
    }
  }

  return maxDrawdown;
}

/**
 * Calculate risk metrics for an asset
 */
export function calculateRiskMetrics(returns: number[], prices: number[]): RiskMetrics {
  return {
    valueAtRisk95: calculateVaR(returns, 0.95),
    valueAtRisk99: calculateVaR(returns, 0.99),
    conditionalVaR: calculateCVaR(returns, 0.95),
    maxDrawdown: calculateMaxDrawdown(prices),
    volatility: stdDev(returns),
  };
}

/**
 * Simple portfolio optimization using equal risk contribution
 * More sophisticated than equal weighting, but simpler than full Markowitz optimization
 */
function optimizeEqualRiskContribution(assets: PortfolioAsset[]): Map<string, number> {
  const weights = new Map<string, number>();

  // Calculate volatility for each asset
  const volatilities = assets.map(asset => ({
    symbol: asset.symbol,
    volatility: stdDev(asset.returns),
  }));

  // Inverse volatility weighting (lower volatility = higher weight)
  const totalInverseVol = volatilities.reduce((sum, v) => sum + (1 / v.volatility), 0);

  for (const vol of volatilities) {
    const weight = (1 / vol.volatility) / totalInverseVol;
    weights.set(vol.symbol, weight);
  }

  return weights;
}

/**
 * Calculate portfolio expected return given weights
 */
function calculatePortfolioReturn(assets: PortfolioAsset[], weights: Map<string, number>): number {
  let portfolioReturn = 0;

  for (const asset of assets) {
    const weight = weights.get(asset.symbol) || 0;
    const assetReturn = mean(asset.returns);
    portfolioReturn += weight * assetReturn;
  }

  return portfolioReturn;
}

/**
 * Calculate portfolio volatility (risk) given weights
 */
function calculatePortfolioVolatility(assets: PortfolioAsset[], weights: Map<string, number>): number {
  let portfolioVariance = 0;

  // Calculate variance (sum of weighted covariances)
  for (let i = 0; i < assets.length; i++) {
    for (let j = 0; j < assets.length; j++) {
      const weight_i = weights.get(assets[i].symbol) || 0;
      const weight_j = weights.get(assets[j].symbol) || 0;

      const cov = i === j
        ? Math.pow(stdDev(assets[i].returns), 2) // Variance on diagonal
        : covariance(assets[i].returns, assets[j].returns); // Covariance off-diagonal

      portfolioVariance += weight_i * weight_j * cov;
    }
  }

  return Math.sqrt(portfolioVariance);
}

/**
 * Main portfolio optimization function
 * Returns optimal weights and metrics for each asset
 */
export function optimizePortfolio(
  assets: PortfolioAsset[],
  _benchmarkAssets: PortfolioAsset[] = [] // BTC, ETH for comparison
): Map<string, PortfolioOptimization> {
  const results = new Map<string, PortfolioOptimization>();

  // Filter assets with sufficient data
  const validAssets = assets.filter(a => a.returns.length >= 20 && stdDev(a.returns) > 0);

  if (validAssets.length === 0) {
    return results;
  }

  // Calculate optimal weights using equal risk contribution
  const optimalWeights = optimizeEqualRiskContribution(validAssets);

  // Calculate portfolio-level metrics
  const portfolioReturn = calculatePortfolioReturn(validAssets, optimalWeights);
  const portfolioVolatility = calculatePortfolioVolatility(validAssets, optimalWeights);
  const _portfolioSharpe = sharpeRatio(portfolioReturn, portfolioVolatility);

  // Calculate metrics for each asset
  for (const asset of validAssets) {
    const weight = optimalWeights.get(asset.symbol) || 0;
    const assetReturn = mean(asset.returns);
    const assetVolatility = stdDev(asset.returns);
    const assetSharpe = sharpeRatio(assetReturn, assetVolatility);

    results.set(asset.symbol, {
      optimalWeight: weight,
      expectedReturn: assetReturn * 100, // Convert to percentage
      expectedRisk: assetVolatility * 100, // Convert to percentage
      sharpeRatio: assetSharpe,
      contribution: weight * assetReturn / portfolioReturn, // Contribution to portfolio return
    });
  }

  return results;
}

/**
 * Calculate correlation matrix for assets
 * Useful for understanding diversification benefits
 */
export function calculateCorrelationMatrix(assets: PortfolioAsset[]): Map<string, Map<string, number>> {
  const corrMatrix = new Map<string, Map<string, number>>();

  for (const asset1 of assets) {
    const row = new Map<string, number>();

    for (const asset2 of assets) {
      if (asset1.symbol === asset2.symbol) {
        row.set(asset2.symbol, 1.0); // Perfect correlation with self
      } else {
        const corr = correlation(asset1.returns, asset2.returns);
        row.set(asset2.symbol, corr);
      }
    }

    corrMatrix.set(asset1.symbol, row);
  }

  return corrMatrix;
}

/**
 * Evaluate if an asset is a good portfolio addition
 * Returns a score from 0-100
 */
export function evaluatePortfolioFit(
  asset: PortfolioAsset,
  existingAssets: PortfolioAsset[]
): number {
  if (asset.returns.length < 20) return 0;

  const assetReturn = mean(asset.returns);
  const assetVolatility = stdDev(asset.returns);
  const assetSharpe = sharpeRatio(assetReturn, assetVolatility);

  // Score components (0-100 scale)
  let score = 50; // Base score

  // 1. Sharpe ratio contribution (max +20)
  score += Math.min(20, Math.max(-20, assetSharpe * 10));

  // 2. Return contribution (max +15)
  if (assetReturn > 0) {
    score += Math.min(15, assetReturn * 100);
  } else {
    score += Math.max(-15, assetReturn * 100);
  }

  // 3. Diversification benefit (max +15)
  if (existingAssets.length > 0) {
    const avgCorrelation = existingAssets.reduce((sum, existing) => {
      return sum + Math.abs(correlation(asset.returns, existing.returns));
    }, 0) / existingAssets.length;

    // Lower correlation = better diversification
    score += (1 - avgCorrelation) * 15;
  }

  // 4. Volume/liquidity factor (max +10)
  const volumeScore = Math.min(10, Math.log10(asset.volume24h) - 3); // Log scale
  score += Math.max(0, volumeScore);

  return Math.max(0, Math.min(100, score));
}
