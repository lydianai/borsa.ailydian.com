/**
 * ALFABETİK PATTERN BACKTEST ENGINE
 * Simulated backtest based on pattern performance metrics
 * Uses current 7d/30d data to simulate historical performance
 */

import type { AlfabetikPattern } from './alfabetik-pattern-analyzer';

export interface BacktestTrade {
  letter: string;
  entryDate: string;
  exitDate: string;
  entrySignal: string;
  profitPercent: number;
  isWin: boolean;
}

export interface LetterBacktestResult {
  letter: string;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  avgProfit: number;
  avgLoss: number;
  maxProfit: number;
  maxLoss: number;
  profitFactor: number;
  bestTrade: BacktestTrade | null;
  worstTrade: BacktestTrade | null;
}

export interface BacktestSummary {
  period: string;
  totalPatterns: number;
  totalTrades: number;
  totalWins: number;
  totalLosses: number;
  overallWinRate: number;
  avgProfit: number;
  totalProfitPercent: number;
  profitFactor: number;
  bestLetter: string;
  worstLetter: string;
  letterResults: LetterBacktestResult[];
}

/**
 * Generate simulated trades based on pattern performance
 * Uses 7d and 30d performance to estimate trade outcomes
 */
function generateSimulatedTrades(
  pattern: AlfabetikPattern,
  days: number = 30
): BacktestTrade[] {
  const trades: BacktestTrade[] = [];

  // Estimate number of trades based on coin count and signal strength
  const estimatedTradesPerMonth = Math.max(
    1,
    Math.floor(pattern.coinSayisi / 3) // Roughly 1 trade per 3 coins
  );

  // Use pattern performance to simulate trade outcomes
  const baseSuccessRate = pattern.güvenilirlik / 100;
  const momentumBonus = pattern.momentum === "YUKARIDA" ? 0.15 :
                        pattern.momentum === "ASAGIDA" ? -0.15 : 0;

  const adjustedSuccessRate = Math.max(0.3, Math.min(0.9, baseSuccessRate + momentumBonus));

  // Generate trades
  for (let i = 0; i < estimatedTradesPerMonth; i++) {
    const daysAgo = Math.floor(Math.random() * days);
    const holdingPeriod = Math.floor(Math.random() * 7) + 1; // 1-7 days

    const entryDate = new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000);
    const exitDate = new Date(entryDate.getTime() + holdingPeriod * 24 * 60 * 60 * 1000);

    // Determine if trade is a win based on adjusted success rate
    const isWin = Math.random() < adjustedSuccessRate;

    // Calculate profit based on pattern performance
    let profitPercent = 0;

    if (isWin) {
      // Use 7d performance as baseline for wins
      const baseProfit = Math.max(5, pattern.ortalamaPerformans7d * 1.5);
      profitPercent = baseProfit * (0.7 + Math.random() * 0.6); // 70-130% of base
    } else {
      // Losses are typically smaller than wins (risk management)
      const baseLoss = Math.min(-3, pattern.ortalamaPerformans7d * 0.5);
      profitPercent = baseLoss * (0.5 + Math.random() * 0.5); // 50-100% of base loss
    }

    trades.push({
      letter: pattern.harf,
      entryDate: entryDate.toISOString().split('T')[0],
      exitDate: exitDate.toISOString().split('T')[0],
      entrySignal: pattern.signal,
      profitPercent: Number(profitPercent.toFixed(2)),
      isWin
    });
  }

  return trades;
}

/**
 * Calculate backtest statistics for a single letter
 */
function calculateLetterStats(trades: BacktestTrade[]): LetterBacktestResult {
  if (trades.length === 0) {
    return {
      letter: '',
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      avgProfit: 0,
      avgLoss: 0,
      maxProfit: 0,
      maxLoss: 0,
      profitFactor: 0,
      bestTrade: null,
      worstTrade: null
    };
  }

  const letter = trades[0].letter;
  const wins = trades.filter(t => t.isWin);
  const losses = trades.filter(t => !t.isWin);

  const totalProfit = wins.reduce((sum, t) => sum + t.profitPercent, 0);
  const totalLoss = Math.abs(losses.reduce((sum, t) => sum + t.profitPercent, 0));

  const avgProfit = wins.length > 0 ? totalProfit / wins.length : 0;
  const avgLoss = losses.length > 0 ? totalLoss / losses.length : 0;

  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;

  const sortedByProfit = [...trades].sort((a, b) => b.profitPercent - a.profitPercent);

  return {
    letter,
    totalTrades: trades.length,
    winningTrades: wins.length,
    losingTrades: losses.length,
    winRate: Number(((wins.length / trades.length) * 100).toFixed(2)),
    avgProfit: Number(avgProfit.toFixed(2)),
    avgLoss: Number(avgLoss.toFixed(2)),
    maxProfit: sortedByProfit[0].profitPercent,
    maxLoss: sortedByProfit[sortedByProfit.length - 1].profitPercent,
    profitFactor: Number(profitFactor.toFixed(2)),
    bestTrade: sortedByProfit[0],
    worstTrade: sortedByProfit[sortedByProfit.length - 1]
  };
}

/**
 * Run backtest simulation on alfabetik patterns
 */
export function runBacktest(
  patterns: AlfabetikPattern[],
  days: number = 30
): BacktestSummary {
  console.log(`🔄 Running backtest simulation for ${patterns.length} patterns over ${days} days...`);

  // Generate trades for each pattern
  const allTrades: BacktestTrade[] = [];
  const letterResults: LetterBacktestResult[] = [];

  for (const pattern of patterns) {
    // Only backtest patterns with meaningful data
    if (pattern.coinSayisi < 2 || pattern.güvenilirlik < 40) {
      continue;
    }

    const trades = generateSimulatedTrades(pattern, days);
    allTrades.push(...trades);

    const stats = calculateLetterStats(trades);
    letterResults.push(stats);
  }

  // Sort by profit factor (best performers first)
  letterResults.sort((a, b) => b.profitFactor - a.profitFactor);

  // Calculate overall statistics
  const totalTrades = allTrades.length;
  const totalWins = allTrades.filter(t => t.isWin).length;
  const totalLosses = allTrades.filter(t => !t.isWin).length;

  const totalProfit = allTrades
    .filter(t => t.isWin)
    .reduce((sum, t) => sum + t.profitPercent, 0);

  const totalLoss = Math.abs(
    allTrades
      .filter(t => !t.isWin)
      .reduce((sum, t) => sum + t.profitPercent, 0)
  );

  const profitFactor = totalLoss > 0 ? totalProfit / totalLoss : totalProfit > 0 ? 999 : 0;

  const bestLetter = letterResults[0]?.letter || 'N/A';
  const worstLetter = letterResults[letterResults.length - 1]?.letter || 'N/A';

  console.log(`✅ Backtest complete: ${totalTrades} trades, ${totalWins} wins, ${totalLosses} losses`);

  return {
    period: `${days}d`,
    totalPatterns: letterResults.length,
    totalTrades,
    totalWins,
    totalLosses,
    overallWinRate: totalTrades > 0 ? Number(((totalWins / totalTrades) * 100).toFixed(2)) : 0,
    avgProfit: totalWins > 0 ? Number((totalProfit / totalWins).toFixed(2)) : 0,
    totalProfitPercent: Number((totalProfit - totalLoss).toFixed(2)),
    profitFactor: Number(profitFactor.toFixed(2)),
    bestLetter,
    worstLetter,
    letterResults
  };
}

/**
 * Get backtest results for a specific letter
 */
export function getLetterBacktest(
  pattern: AlfabetikPattern,
  days: number = 30
): LetterBacktestResult {
  const trades = generateSimulatedTrades(pattern, days);
  return calculateLetterStats(trades);
}
