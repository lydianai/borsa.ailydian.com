/**
 * UNIFIED DECISION ENGINE v1.0
 *
 * Synchronizes 13+ trading strategies into a single unified decision
 *
 * Features:
 * - Weighted voting system
 * - Conflict resolution algorithm
 * - Dynamic weight adjustment based on market conditions
 * - Multi-layer confirmation system
 *
 * White-hat: All weights and logic are transparent and auditable
 */

// ============================================================================
// TYPE DEFINITIONS (temporarily duplicated until path aliases are fixed)
// ============================================================================

interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

export interface PriceData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  candles?: Candle[];
}

export type SignalType = 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';

export interface StrategySignal {
  name: string;
  signal: SignalType;
  confidence: number;
  reason: string;
  targets?: number[];
  stopLoss?: number;
  timeframe?: string;
  indicators?: Record<string, number | string>;
}

export interface StrategyAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  groqAnalysis?: string;
  strategies: StrategySignal[];
  overallScore: number;
  recommendation: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'WAIT' | 'SELL' | 'STRONG_SELL';
  buyCount: number;
  waitCount: number;
  sellCount: number;
  neutralCount: number;
  timestamp: string;
}

// ============================================================================
// UNIFIED DECISION ENGINE TYPES
// ============================================================================

export interface StrategyWeight {
  name: string;
  baseWeight: number; // 0-1 (base importance of strategy)
  volatilityMultiplier: number; // How weight changes in volatile markets
  trendMultiplier: number; // How weight changes in trending markets
}

export interface UnifiedDecision {
  signal: SignalType;
  confidence: number; // 0-100 (aggregated weighted confidence)
  recommendation: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'WAIT' | 'SELL' | 'STRONG_SELL';
  consensus: number; // 0-100 (how much strategies agree)
  weightedVotes: {
    BUY: number;
    SELL: number;
    WAIT: number;
    NEUTRAL: number;
  };
  contributingStrategies: {
    name: string;
    signal: SignalType;
    confidence: number;
    weight: number;
    contribution: number; // How much this strategy influenced the decision
  }[];
  conflictWarnings: string[];
  marketCondition: 'volatile' | 'trending' | 'ranging' | 'uncertain';
}

export interface MarketCondition {
  volatility: number; // 0-100
  trend: 'strong_up' | 'up' | 'neutral' | 'down' | 'strong_down';
  strength: number; // 0-100
}

// ============================================================================
// STRATEGY WEIGHTS CONFIGURATION
// ============================================================================

const STRATEGY_WEIGHTS: Record<string, StrategyWeight> = {
  'MA Crossover Pullback': {
    name: 'MA Crossover Pullback',
    baseWeight: 0.15,
    volatilityMultiplier: 0.8, // Less reliable in volatile markets
    trendMultiplier: 1.3, // More reliable in trending markets
  },
  'MA7 Pullback': {
    name: 'MA7 Pullback',
    baseWeight: 0.12,
    volatilityMultiplier: 0.85,
    trendMultiplier: 1.25,
  },
  'RSI Divergence': {
    name: 'RSI Divergence',
    baseWeight: 0.18,
    volatilityMultiplier: 1.2, // More reliable in volatile markets
    trendMultiplier: 0.9,
  },
  'Bollinger Squeeze': {
    name: 'Bollinger Squeeze',
    baseWeight: 0.16,
    volatilityMultiplier: 1.4, // Very reliable for volatility breakouts
    trendMultiplier: 0.8,
  },
  'EMA Ribbon': {
    name: 'EMA Ribbon',
    baseWeight: 0.14,
    volatilityMultiplier: 0.9,
    trendMultiplier: 1.2,
  },
  'Volume Breakout': {
    name: 'Volume Breakout',
    baseWeight: 0.13,
    volatilityMultiplier: 1.1,
    trendMultiplier: 1.1,
  },
  'Fibonacci Retracement': {
    name: 'Fibonacci Retracement',
    baseWeight: 0.11,
    volatilityMultiplier: 0.7,
    trendMultiplier: 1.3,
  },
  'Ichimoku Cloud': {
    name: 'Ichimoku Cloud',
    baseWeight: 0.17,
    volatilityMultiplier: 0.85,
    trendMultiplier: 1.4, // Excellent for trends
  },
  'ATR Volatility': {
    name: 'ATR Volatility',
    baseWeight: 0.10,
    volatilityMultiplier: 1.5, // Primary volatility indicator
    trendMultiplier: 0.7,
  },
  'Trend Reversal': {
    name: 'Trend Reversal',
    baseWeight: 0.15,
    volatilityMultiplier: 1.1,
    trendMultiplier: 1.2,
  },
  'MACD Histogram': {
    name: 'MACD Histogram',
    baseWeight: 0.16,
    volatilityMultiplier: 0.95,
    trendMultiplier: 1.3,
  },
  'Support/Resistance': {
    name: 'Support/Resistance',
    baseWeight: 0.19, // Very important
    volatilityMultiplier: 1.0,
    trendMultiplier: 1.0,
  },
  'Red Wick Green Closure': {
    name: 'Red Wick Green Closure',
    baseWeight: 0.14,
    volatilityMultiplier: 1.2,
    trendMultiplier: 0.9,
  },
  'Breakout Retest': {
    name: 'Breakout Retest',
    baseWeight: 0.15,
    volatilityMultiplier: 1.0,
    trendMultiplier: 1.3,
  },
  'Conservative Buy Signal': {
    name: 'Conservative Buy Signal',
    baseWeight: 0.12,
    volatilityMultiplier: 0.7, // Conservative = avoid volatility
    trendMultiplier: 1.1,
  },
  'BTC-ETH Correlation': {
    name: 'BTC-ETH Correlation',
    baseWeight: 0.08,
    volatilityMultiplier: 1.0,
    trendMultiplier: 1.0,
  },
};

// ============================================================================
// UNIFIED DECISION ENGINE
// ============================================================================

export class UnifiedDecisionEngine {
  /**
   * Analyze market condition from price data and strategy signals
   */
  private analyzeMarketCondition(
    priceData: PriceData,
    _signals: StrategySignal[]
  ): MarketCondition {
    // Calculate volatility from 24h price range
    const priceRange = priceData.high24h - priceData.low24h;
    const avgPrice = (priceData.high24h + priceData.low24h) / 2;
    const volatilityPercent = (priceRange / avgPrice) * 100;

    // Normalize volatility to 0-100 scale (0-3% = 0, 10%+ = 100)
    const volatility = Math.min(100, (volatilityPercent / 10) * 100);

    // Determine trend from 24h change
    const changePercent = priceData.changePercent24h;
    let trend: MarketCondition['trend'];
    let strength: number;

    if (changePercent > 5) {
      trend = 'strong_up';
      strength = Math.min(100, changePercent * 10);
    } else if (changePercent > 2) {
      trend = 'up';
      strength = Math.min(100, changePercent * 15);
    } else if (changePercent < -5) {
      trend = 'strong_down';
      strength = Math.min(100, Math.abs(changePercent) * 10);
    } else if (changePercent < -2) {
      trend = 'down';
      strength = Math.min(100, Math.abs(changePercent) * 15);
    } else {
      trend = 'neutral';
      strength = 50;
    }

    return { volatility, trend, strength };
  }

  /**
   * Calculate dynamic weight for a strategy based on market conditions
   */
  private calculateDynamicWeight(
    strategy: StrategyWeight,
    marketCondition: MarketCondition
  ): number {
    let weight = strategy.baseWeight;

    // Adjust for volatility (high volatility = 1.0+, low volatility = 0.5-0.8)
    const volatilityFactor = 0.5 + (marketCondition.volatility / 200);
    weight *= Math.pow(strategy.volatilityMultiplier, volatilityFactor);

    // Adjust for trend strength
    const trendFactor = marketCondition.strength / 100;
    const isTrending = ['strong_up', 'strong_down', 'up', 'down'].includes(marketCondition.trend);
    if (isTrending) {
      weight *= Math.pow(strategy.trendMultiplier, trendFactor);
    }

    return weight;
  }

  /**
   * Calculate weighted votes for each signal type
   */
  private calculateWeightedVotes(
    signals: StrategySignal[],
    marketCondition: MarketCondition
  ) {
    const votes = { BUY: 0, SELL: 0, WAIT: 0, NEUTRAL: 0 };
    const contributions: UnifiedDecision['contributingStrategies'] = [];

    let totalWeight = 0;

    signals.forEach((signal) => {
      const strategyConfig = STRATEGY_WEIGHTS[signal.name] || {
        name: signal.name,
        baseWeight: 0.1,
        volatilityMultiplier: 1.0,
        trendMultiplier: 1.0,
      };

      // Calculate dynamic weight
      const dynamicWeight = this.calculateDynamicWeight(strategyConfig, marketCondition);

      // Weight by confidence (0-100 -> 0-1)
      const confidenceWeight = signal.confidence / 100;

      // Combined weight
      const finalWeight = dynamicWeight * confidenceWeight;
      totalWeight += finalWeight;

      // Add vote
      votes[signal.signal] += finalWeight;

      // Track contribution
      contributions.push({
        name: signal.name,
        signal: signal.signal,
        confidence: signal.confidence,
        weight: dynamicWeight,
        contribution: finalWeight,
      });
    });

    // Normalize votes to percentages
    if (totalWeight > 0) {
      votes.BUY = (votes.BUY / totalWeight) * 100;
      votes.SELL = (votes.SELL / totalWeight) * 100;
      votes.WAIT = (votes.WAIT / totalWeight) * 100;
      votes.NEUTRAL = (votes.NEUTRAL / totalWeight) * 100;
    }

    // Sort contributions by impact
    contributions.sort((a, b) => b.contribution - a.contribution);

    return { votes, contributions };
  }

  /**
   * Detect conflicts between strategies
   */
  private detectConflicts(
    signals: StrategySignal[],
    weightedVotes: UnifiedDecision['weightedVotes']
  ): string[] {
    const warnings: string[] = [];

    // High BUY and SELL votes at same time
    if (weightedVotes.BUY > 30 && weightedVotes.SELL > 30) {
      warnings.push(
        `Strong disagreement: ${weightedVotes.BUY.toFixed(1)}% BUY vs ${weightedVotes.SELL.toFixed(1)}% SELL`
      );
    }

    // Check for opposing high-confidence signals
    const highConfidenceBuy = signals.filter((s) => s.signal === 'BUY' && s.confidence > 75);
    const highConfidenceSell = signals.filter((s) => s.signal === 'SELL' && s.confidence > 75);

    if (highConfidenceBuy.length > 0 && highConfidenceSell.length > 0) {
      warnings.push(
        `Conflicting high-confidence signals: ${highConfidenceBuy.map((s) => s.name).join(', ')} say BUY but ${highConfidenceSell.map((s) => s.name).join(', ')} say SELL`
      );
    }

    // Low consensus (all votes spread out)
    const maxVote = Math.max(weightedVotes.BUY, weightedVotes.SELL, weightedVotes.WAIT, weightedVotes.NEUTRAL);
    if (maxVote < 40) {
      warnings.push(`Low consensus: No clear majority (max ${maxVote.toFixed(1)}%)`);
    }

    return warnings;
  }

  /**
   * Calculate consensus score (how much strategies agree)
   */
  private calculateConsensus(weightedVotes: UnifiedDecision['weightedVotes']): number {
    const maxVote = Math.max(weightedVotes.BUY, weightedVotes.SELL, weightedVotes.WAIT, weightedVotes.NEUTRAL);
    return maxVote; // Consensus = percentage of dominant vote
  }

  /**
   * Determine final recommendation based on weighted votes
   */
  private determineRecommendation(
    weightedVotes: UnifiedDecision['weightedVotes'],
    consensus: number
  ): { signal: SignalType; recommendation: UnifiedDecision['recommendation']; confidence: number } {
    const maxVote = Math.max(weightedVotes.BUY, weightedVotes.SELL, weightedVotes.WAIT, weightedVotes.NEUTRAL);

    let signal: SignalType;
    let recommendation: UnifiedDecision['recommendation'];

    // Determine dominant signal
    if (weightedVotes.BUY === maxVote) {
      signal = 'BUY';
      if (consensus >= 70 && weightedVotes.BUY >= 60) {
        recommendation = 'STRONG_BUY';
      } else {
        recommendation = 'BUY';
      }
    } else if (weightedVotes.SELL === maxVote) {
      signal = 'SELL';
      if (consensus >= 70 && weightedVotes.SELL >= 60) {
        recommendation = 'STRONG_SELL';
      } else {
        recommendation = 'SELL';
      }
    } else if (weightedVotes.WAIT === maxVote) {
      signal = 'WAIT';
      recommendation = 'WAIT';
    } else {
      signal = 'NEUTRAL';
      recommendation = 'NEUTRAL';
    }

    // Calculate confidence (combination of consensus and vote strength)
    const confidence = Math.min(100, (consensus + maxVote) / 2);

    return { signal, recommendation, confidence };
  }

  /**
   * Main decision engine - processes all strategies and returns unified decision
   */
  public makeDecision(priceData: PriceData, signals: StrategySignal[]): UnifiedDecision {
    // 1. Analyze market conditions
    const marketCondition = this.analyzeMarketCondition(priceData, signals);

    // 2. Calculate weighted votes
    const { votes: weightedVotes, contributions } = this.calculateWeightedVotes(signals, marketCondition);

    // 3. Calculate consensus
    const consensus = this.calculateConsensus(weightedVotes);

    // 4. Detect conflicts
    const conflictWarnings = this.detectConflicts(signals, weightedVotes);

    // 5. Determine final recommendation
    const { signal, recommendation, confidence } = this.determineRecommendation(weightedVotes, consensus);

    // 6. Determine market condition label
    let marketConditionLabel: UnifiedDecision['marketCondition'];
    if (marketCondition.volatility > 60) {
      marketConditionLabel = 'volatile';
    } else if (['strong_up', 'strong_down'].includes(marketCondition.trend)) {
      marketConditionLabel = 'trending';
    } else if (marketCondition.trend === 'neutral' && marketCondition.volatility < 30) {
      marketConditionLabel = 'ranging';
    } else {
      marketConditionLabel = 'uncertain';
    }

    return {
      signal,
      confidence,
      recommendation,
      consensus,
      weightedVotes,
      contributingStrategies: contributions,
      conflictWarnings,
      marketCondition: marketConditionLabel,
    };
  }

  /**
   * Get strategy weight configuration (for transparency/auditing)
   */
  public getStrategyWeights(): Record<string, StrategyWeight> {
    return { ...STRATEGY_WEIGHTS };
  }

  /**
   * Update strategy weight (for dynamic adjustment)
   * White-hat: All weight changes are logged and auditable
   */
  public updateStrategyWeight(
    strategyName: string,
    updates: Partial<Omit<StrategyWeight, 'name'>>
  ): void {
    if (STRATEGY_WEIGHTS[strategyName]) {
      STRATEGY_WEIGHTS[strategyName] = {
        ...STRATEGY_WEIGHTS[strategyName],
        ...updates,
      };
      console.log(`[UnifiedDecisionEngine] Updated weights for ${strategyName}:`, updates);
    }
  }
}

// Singleton instance
export const unifiedDecisionEngine = new UnifiedDecisionEngine();
