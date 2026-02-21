/**
 * ON-CHAIN STRATEGY ENHANCER
 * Non-breaking enhancement layer for existing trading strategies
 *
 * How it works:
 * 1. Takes base strategy signal (unchanged)
 * 2. Fetches on-chain whale activity
 * 3. Adjusts confidence based on whale movements
 * 4. Returns enhanced signal with explanation
 *
 * KEY: This NEVER modifies existing strategies
 * It only wraps them with additional on-chain intelligence
 */

import {
  getOnChainSignal,
  enhanceStrategyWithOnChain,
  type WhaleActivity,
} from './index';

// ============================================================================
// TYPES
// ============================================================================

export interface BaseStrategySignal {
  signal: 'buy' | 'sell' | 'neutral';
  confidence: number; // 0-100
  symbol: string;
  strategy?: string; // Strategy name (optional)
  price?: number; // Current price (optional)
  reason?: string; // Base strategy reason (optional)
}

export interface EnhancedStrategySignal extends BaseStrategySignal {
  // Original fields preserved
  originalConfidence: number;

  // On-chain enhancements
  onChainAnalysis: {
    enabled: boolean;
    whaleActivity: WhaleActivity | null;
    signal: 'bullish' | 'bearish' | 'neutral';
    riskAdjustment: number; // Points added/subtracted from confidence
    explanation: string;
  };

  // Final decision
  finalDecision: {
    signal: 'buy' | 'sell' | 'neutral';
    confidence: number;
    recommendation: string;
    warnings: string[];
  };
}

// ============================================================================
// ENHANCEMENT RULES
// ============================================================================

const ENHANCEMENT_RULES = {
  // Maximum confidence boost from on-chain data
  MAX_BOOST: 20,

  // Maximum confidence penalty from on-chain data
  MAX_PENALTY: 30,

  // Minimum whale confidence to apply adjustment
  MIN_WHALE_CONFIDENCE: 30,

  // Risk score thresholds
  RISK: {
    LOW: 30,
    MEDIUM: 50,
    HIGH: 70,
    CRITICAL: 85,
  },

  // Confidence thresholds for trading decisions
  CONFIDENCE: {
    STRONG_BUY: 75,
    BUY: 60,
    WEAK_BUY: 50,
    NEUTRAL: 40,
    WEAK_SELL: 50,
    SELL: 60,
    STRONG_SELL: 75,
  },
};

// ============================================================================
// ENHANCEMENT FUNCTION
// ============================================================================

/**
 * Enhance any strategy signal with on-chain whale analysis
 *
 * @param baseSignal - Original strategy signal (unchanged)
 * @param options - Enhancement options
 * @returns Enhanced signal with on-chain intelligence
 */
export async function enhanceWithOnChain(
  baseSignal: BaseStrategySignal,
  options: {
    enableWhaleAlert?: boolean;
    aggressiveMode?: boolean; // Apply stronger adjustments
  } = {}
): Promise<EnhancedStrategySignal> {
  const { enableWhaleAlert = true, aggressiveMode = false } = options;

  // Store original values
  const originalConfidence = baseSignal.confidence;
  const originalSignal = baseSignal.signal;

  // Initialize enhanced signal
  let enhancedConfidence = baseSignal.confidence;
  let riskAdjustment = 0;
  let whaleActivity: WhaleActivity | null = null;
  let onChainSignal: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  let explanation = 'On-chain analysis disabled';
  const warnings: string[] = [];

  // Apply on-chain analysis if enabled
  if (enableWhaleAlert) {
    try {
      const onChain = await getOnChainSignal(baseSignal.symbol);
      whaleActivity = onChain.whaleActivity;
      onChainSignal = onChain.signal;
      explanation = onChain.reason;

      if (onChain.whaleActivity && onChain.confidence >= ENHANCEMENT_RULES.MIN_WHALE_CONFIDENCE) {
        // Calculate adjustment multiplier
        const multiplier = aggressiveMode ? 1.5 : 1.0;

        // Apply whale signal adjustment
        if (onChain.whaleActivity?.activity === 'accumulation') {
          // Whales buying = Boost buy signals, penalize sell signals
          if (baseSignal.signal === 'buy') {
            riskAdjustment = Math.min(
              ENHANCEMENT_RULES.MAX_BOOST,
              (onChain.confidence / 5) * multiplier
            );
            explanation = `✅ Whales accumulating - BUY signal strengthened`;
          } else if (baseSignal.signal === 'sell') {
            riskAdjustment = -Math.min(
              ENHANCEMENT_RULES.MAX_PENALTY,
              (onChain.confidence / 3) * multiplier
            );
            warnings.push('⚠️ Whale accumulation conflicts with SELL signal');
            explanation = `⚠️ Whales accumulating but strategy says SELL - proceed with caution`;
          }
        } else if (onChain.whaleActivity?.activity === 'distribution') {
          // Whales selling = Boost sell signals, penalize buy signals
          if (baseSignal.signal === 'sell') {
            riskAdjustment = Math.min(
              ENHANCEMENT_RULES.MAX_BOOST,
              (onChain.confidence / 5) * multiplier
            );
            explanation = `✅ Whales distributing - SELL signal strengthened`;
          } else if (baseSignal.signal === 'buy') {
            riskAdjustment = -Math.min(
              ENHANCEMENT_RULES.MAX_PENALTY,
              (onChain.confidence / 3) * multiplier
            );
            warnings.push('🚨 Whale distribution conflicts with BUY signal');
            explanation = `🚨 Whales selling but strategy says BUY - HIGH RISK`;
          }
        }

        // Apply risk-based adjustments
        if ((onChain.whaleActivity?.riskScore ?? 0) >= ENHANCEMENT_RULES.RISK.CRITICAL) {
          warnings.push('🔴 CRITICAL RISK: Extreme whale activity detected');
          if (baseSignal.signal === 'buy') {
            riskAdjustment -= 10; // Extra penalty for buying in critical risk
          }
        } else if ((onChain.whaleActivity?.riskScore ?? 0) >= ENHANCEMENT_RULES.RISK.HIGH) {
          warnings.push('⚠️ HIGH RISK: Significant whale activity');
        }

        // Calculate enhanced confidence
        enhancedConfidence = Math.max(0, Math.min(100, baseSignal.confidence + riskAdjustment));
      } else if (onChain.whaleActivity) {
        explanation = 'Whale activity detected but confidence too low';
      } else {
        explanation = 'No significant whale activity';
      }
    } catch (error) {
      console.warn(`[StrategyEnhancer] Failed to fetch on-chain data for ${baseSignal.symbol}:`, error);
      explanation = 'On-chain data unavailable';
      warnings.push('⚠️ On-chain analysis failed - using base strategy only');
    }
  }

  // Generate final recommendation
  let recommendation = '';
  let finalSignal = originalSignal;

  if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.STRONG_BUY && originalSignal === 'buy') {
    recommendation = `🟢 STRONG BUY (${enhancedConfidence}% confidence)`;
  } else if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.BUY && originalSignal === 'buy') {
    recommendation = `🟢 BUY (${enhancedConfidence}% confidence)`;
  } else if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.WEAK_BUY && originalSignal === 'buy') {
    recommendation = `🟡 WEAK BUY (${enhancedConfidence}% confidence)`;
  } else if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.STRONG_SELL && originalSignal === 'sell') {
    recommendation = `🔴 STRONG SELL (${enhancedConfidence}% confidence)`;
  } else if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.SELL && originalSignal === 'sell') {
    recommendation = `🔴 SELL (${enhancedConfidence}% confidence)`;
  } else if (enhancedConfidence >= ENHANCEMENT_RULES.CONFIDENCE.WEAK_SELL && originalSignal === 'sell') {
    recommendation = `🟡 WEAK SELL (${enhancedConfidence}% confidence)`;
  } else {
    recommendation = `⚪ WAIT / NEUTRAL (${enhancedConfidence}% confidence)`;
    finalSignal = 'neutral';
  }

  // Add whale activity summary to recommendation if available
  if (whaleActivity) {
    recommendation += ` | ${whaleActivity.summary}`;
  }

  return {
    ...baseSignal,
    originalConfidence,
    onChainAnalysis: {
      enabled: enableWhaleAlert,
      whaleActivity,
      signal: onChainSignal,
      riskAdjustment,
      explanation,
    },
    finalDecision: {
      signal: finalSignal,
      confidence: enhancedConfidence,
      recommendation,
      warnings,
    },
  };
}

/**
 * Batch enhance multiple signals
 * Useful for scanning multiple symbols
 */
export async function enhanceMultipleSignals(
  signals: BaseStrategySignal[],
  options?: {
    enableWhaleAlert?: boolean;
    aggressiveMode?: boolean;
  }
): Promise<EnhancedStrategySignal[]> {
  const enhanced = await Promise.all(
    signals.map((signal) => enhanceWithOnChain(signal, options))
  );

  return enhanced;
}

/**
 * Get on-chain summary for a symbol
 * Quick check without full enhancement
 */
export async function getOnChainSummary(symbol: string): Promise<{
  hasWhaleActivity: boolean;
  activity: 'accumulation' | 'distribution' | 'neutral';
  confidence: number;
  riskScore: number;
  summary: string;
}> {
  try {
    const onChain = await getOnChainSignal(symbol);

    if (!onChain.whaleActivity) {
      return {
        hasWhaleActivity: false,
        activity: 'neutral',
        confidence: 0,
        riskScore: 50,
        summary: 'No whale activity detected',
      };
    }

    return {
      hasWhaleActivity: true,
      activity: onChain.whaleActivity.activity,
      confidence: onChain.whaleActivity.confidence,
      riskScore: onChain.whaleActivity.riskScore,
      summary: onChain.whaleActivity.summary,
    };
  } catch (error) {
    return {
      hasWhaleActivity: false,
      activity: 'neutral',
      confidence: 0,
      riskScore: 50,
      summary: 'On-chain data unavailable',
    };
  }
}

/**
 * Check if a trade should be blocked based on on-chain risk
 */
export async function shouldBlockTrade(
  signal: BaseStrategySignal,
  riskTolerance: 'low' | 'medium' | 'high' = 'medium'
): Promise<{
  blocked: boolean;
  reason: string;
  riskScore: number;
}> {
  try {
    const onChain = await getOnChainSignal(signal.symbol);

    if (!onChain.whaleActivity) {
      return { blocked: false, reason: 'No on-chain risk detected', riskScore: 50 };
    }

    const riskThresholds = {
      low: ENHANCEMENT_RULES.RISK.MEDIUM,
      medium: ENHANCEMENT_RULES.RISK.HIGH,
      high: ENHANCEMENT_RULES.RISK.CRITICAL,
    };

    const threshold = riskThresholds[riskTolerance];

    if (onChain.whaleActivity.riskScore >= threshold) {
      // Additional check: Block BUY if whales are distributing heavily
      if (
        signal.signal === 'buy' &&
        onChain.whaleActivity.activity === 'distribution' &&
        onChain.whaleActivity.confidence >= 70
      ) {
        return {
          blocked: true,
          reason: `🚨 BLOCKED: Heavy whale distribution detected (${onChain.whaleActivity.confidence}% confidence)`,
          riskScore: onChain.whaleActivity.riskScore,
        };
      }

      return {
        blocked: true,
        reason: `⚠️ BLOCKED: Risk score ${onChain.whaleActivity.riskScore} exceeds threshold ${threshold}`,
        riskScore: onChain.whaleActivity.riskScore,
      };
    }

    return {
      blocked: false,
      reason: 'Risk within acceptable limits',
      riskScore: onChain.whaleActivity.riskScore,
    };
  } catch (error) {
    // On error, don't block (fail open for availability)
    return {
      blocked: false,
      reason: 'On-chain risk check failed - proceeding with caution',
      riskScore: 50,
    };
  }
}
