/**
 * ON-CHAIN DATA - UNIFIED SERVICE
 * Aggregates all on-chain blockchain data for crypto assets
 *
 * Features:
 * - Whale Alert (large transactions tracking)
 * - Exchange netflow analysis
 * - Accumulation/Distribution signals
 * - Risk scoring for trading decisions
 *
 * Pattern: Follows /src/lib/traditional-markets/index.ts architecture
 * Integration: Non-breaking enhancement layer for existing strategies
 */

import {
  getWhaleActivity,
  getWhaleActivityForSymbol,
  clearWhaleAlertCache,
  getWhaleAlertCacheStatus,
  type WhaleTransaction,
  type WhaleActivity,
} from './whale-alert-adapter';

// ============================================================================
// RE-EXPORTS
// ============================================================================

export type { WhaleTransaction, WhaleActivity };

export {
  // Whale Alert
  getWhaleActivity,
  getWhaleActivityForSymbol,
  clearWhaleAlertCache,
  getWhaleAlertCacheStatus,
};

// ============================================================================
// UNIFIED DATA TYPE
// ============================================================================

export interface OnChainData {
  whaleActivity: Map<string, WhaleActivity>;
  timestamp: Date;
  summary: {
    totalSymbols: number;
    accumulation: number; // Count of symbols with accumulation
    distribution: number; // Count of symbols with distribution
    neutral: number; // Count of neutral symbols
    highRisk: number; // Count of symbols with risk > 70
  };
}

// ============================================================================
// UNIFIED SERVICE
// ============================================================================

/**
 * Get all on-chain data in one call
 * Combines whale alerts and future on-chain metrics
 */
export async function getAllOnChainData(_forceRefresh: boolean = false): Promise<OnChainData> {
  console.log('[OnChain] Fetching all on-chain data...');

  try {
    // Currently only whale activity, can expand later
    const whaleActivity = await getWhaleActivity();

    // Calculate summary statistics
    let accumulation = 0;
    let distribution = 0;
    let neutral = 0;
    let highRisk = 0;

    whaleActivity.forEach((activity) => {
      if (activity.activity === 'accumulation') accumulation++;
      if (activity.activity === 'distribution') distribution++;
      if (activity.activity === 'neutral') neutral++;
      if (activity.riskScore > 70) highRisk++;
    });

    const data: OnChainData = {
      whaleActivity,
      timestamp: new Date(),
      summary: {
        totalSymbols: whaleActivity.size,
        accumulation,
        distribution,
        neutral,
        highRisk,
      },
    };

    console.log(
      `[OnChain] Data fetched - ${data.summary.totalSymbols} symbols (${accumulation} accumulation, ${distribution} distribution, ${highRisk} high-risk)`
    );

    return data;
  } catch (error: any) {
    console.error('[OnChain] Failed to fetch on-chain data:', error);
    throw new Error(`Failed to fetch on-chain data: ${error.message}`);
  }
}

/**
 * Clear all on-chain caches
 */
export function clearAllOnChainCache(): void {
  clearWhaleAlertCache();
  console.log('[OnChain] All caches cleared');
}

/**
 * Get on-chain signal for a specific symbol
 * Returns trading signal enhancement based on whale activity
 */
export async function getOnChainSignal(
  symbol: string
): Promise<{
  signal: 'bullish' | 'bearish' | 'neutral';
  confidence: number; // 0-100
  riskAdjustment: number; // -50 to +50 (subtract from strategy confidence)
  whaleActivity: WhaleActivity | null;
  reason: string;
}> {
  const activity = await getWhaleActivityForSymbol(symbol);

  if (!activity) {
    return {
      signal: 'neutral',
      confidence: 0,
      riskAdjustment: 0,
      whaleActivity: null,
      reason: 'No whale activity data available',
    };
  }

  let signal: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  let riskAdjustment = 0;

  if (activity.activity === 'accumulation') {
    // Whales accumulating = Bullish
    signal = 'bullish';
    // Reduce risk (boost confidence) by up to 20 points
    riskAdjustment = -Math.min(20, activity.confidence / 5);
  } else if (activity.activity === 'distribution') {
    // Whales distributing = Bearish
    signal = 'bearish';
    // Increase risk (reduce confidence) by up to 30 points
    riskAdjustment = Math.min(30, activity.confidence / 3);
  }

  return {
    signal,
    confidence: activity.confidence,
    riskAdjustment,
    whaleActivity: activity,
    reason: activity.summary,
  };
}

/**
 * Enhance existing strategy signal with on-chain data
 * NON-BREAKING: Only enhances, never replaces base strategy
 */
export async function enhanceStrategyWithOnChain(
  baseSignal: {
    signal: 'buy' | 'sell' | 'neutral';
    confidence: number;
    symbol: string;
  },
  options: {
    enableWhaleAlert?: boolean;
  } = {}
): Promise<{
  signal: 'buy' | 'sell' | 'neutral';
  confidence: number;
  onChainBoost: number;
  whaleActivity: WhaleActivity | null;
  explanation: string;
}> {
  const { enableWhaleAlert = true } = options;

  // Start with base strategy (unchanged)
  let enhancedConfidence = baseSignal.confidence;
  let onChainBoost = 0;
  let explanation = '';
  let whaleActivity: WhaleActivity | null = null;

  // Apply on-chain enhancements if enabled
  if (enableWhaleAlert) {
    try {
      const onChain = await getOnChainSignal(baseSignal.symbol);
      whaleActivity = onChain.whaleActivity;

      if (onChain.whaleActivity) {
        // Apply risk adjustment
        onChainBoost = -onChain.riskAdjustment; // Negative adjustment = positive boost
        enhancedConfidence = Math.max(0, Math.min(100, baseSignal.confidence + onChainBoost));

        // Build explanation
        if (baseSignal.signal === 'buy' && onChain.signal === 'bullish') {
          explanation = `✅ On-chain confirms BUY: ${onChain.reason}`;
        } else if (baseSignal.signal === 'buy' && onChain.signal === 'bearish') {
          explanation = `⚠️ On-chain warning: ${onChain.reason}`;
        } else if (baseSignal.signal === 'sell' && onChain.signal === 'bearish') {
          explanation = `✅ On-chain confirms SELL: ${onChain.reason}`;
        } else if (baseSignal.signal === 'sell' && onChain.signal === 'bullish') {
          explanation = `⚠️ On-chain conflict: ${onChain.reason}`;
        } else {
          explanation = onChain.reason;
        }
      }
    } catch (error) {
      console.warn(`[OnChain] Failed to enhance ${baseSignal.symbol}:`, error);
      // Graceful degradation - return base signal unchanged
    }
  }

  return {
    signal: baseSignal.signal,
    confidence: enhancedConfidence,
    onChainBoost,
    whaleActivity,
    explanation: explanation || 'No on-chain data available',
  };
}

/**
 * Get market-wide on-chain overview
 */
export async function getOnChainOverview(): Promise<{
  trending: Array<{
    symbol: string;
    activity: 'accumulation' | 'distribution' | 'neutral';
    confidence: number;
    summary: string;
  }>;
  mostAccumulated: { symbol: string; netflow: number; summary: string } | null;
  mostDistributed: { symbol: string; netflow: number; summary: string } | null;
  marketSentiment: 'bullish' | 'bearish' | 'neutral';
}> {
  const data = await getAllOnChainData();

  const allActivity = Array.from(data.whaleActivity.values());

  // Sort by absolute netflow (most significant whale movements)
  const sorted = [...allActivity].sort((a, b) => Math.abs(b.exchangeNetflow) - Math.abs(a.exchangeNetflow));

  const trending = sorted.slice(0, 5).map((activity) => ({
    symbol: activity.symbol,
    activity: activity.activity,
    confidence: activity.confidence,
    summary: activity.summary,
  }));

  // Find most accumulated/distributed
  const byNetflow = [...allActivity].sort((a, b) => a.exchangeNetflow - b.exchangeNetflow);
  const mostAccumulated =
    byNetflow[0] && byNetflow[0].exchangeNetflow < 0
      ? {
          symbol: byNetflow[0].symbol,
          netflow: byNetflow[0].exchangeNetflow,
          summary: byNetflow[0].summary,
        }
      : null;

  const mostDistributed =
    byNetflow[byNetflow.length - 1] && byNetflow[byNetflow.length - 1].exchangeNetflow > 0
      ? {
          symbol: byNetflow[byNetflow.length - 1].symbol,
          netflow: byNetflow[byNetflow.length - 1].exchangeNetflow,
          summary: byNetflow[byNetflow.length - 1].summary,
        }
      : null;

  // Calculate market sentiment
  let marketSentiment: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (data.summary.accumulation > data.summary.distribution * 1.5) {
    marketSentiment = 'bullish';
  } else if (data.summary.distribution > data.summary.accumulation * 1.5) {
    marketSentiment = 'bearish';
  }

  return {
    trending,
    mostAccumulated,
    mostDistributed,
    marketSentiment,
  };
}
