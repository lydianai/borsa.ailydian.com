/**
 * RISK MANAGEMENT MODULE
 * Advanced risk controls for trading signals
 */

interface RiskLimits {
  maxPositionSize: number;
  dailyLossLimit: number;
  weeklyLossLimit: number;
  maxOpenPositions: number;
  maxCorrelatedPositions: number;
  minConfidence: number;
  maxRiskScore: number;
}

interface PositionRisk {
  symbol: string;
  positionSize: number;
  currentPrice: number;
  entryPrice: number;
  unrealizedPnL: number;
  riskScore: number;
  stopLoss: number;
  takeProfit: number;
}

interface RiskCheckResult {
  approved: boolean;
  reason?: string;
  suggestedPositionSize?: number;
  warnings: string[];
}

export class RiskManagementModule {
  private limits: RiskLimits = {
    maxPositionSize: 10000, // $10,000 per position
    dailyLossLimit: 1000, // $1,000 daily loss limit
    weeklyLossLimit: 5000, // $5,000 weekly loss limit
    maxOpenPositions: 10,
    maxCorrelatedPositions: 3, // Max 3 positions in correlated assets
    minConfidence: 0.70, // Minimum 70% confidence
    maxRiskScore: 0.75 // Maximum 75% risk score
  };

  private dailyPnL: number = 0;
  private weeklyPnL: number = 0;
  private openPositions: PositionRisk[] = [];
  private lastResetTime: number = Date.now();

  /**
   * Check if a signal meets risk criteria
   */
  checkSignalRisk(signal: {
    symbol: string;
    action: string;
    confidence: number;
    riskScore: number;
    price: number;
  }): RiskCheckResult {
    const warnings: string[] = [];

    // 1. Confidence check
    if (signal.confidence < this.limits.minConfidence) {
      return {
        approved: false,
        reason: `Confidence ${(signal.confidence * 100).toFixed(1)}% below minimum ${(this.limits.minConfidence * 100)}%`,
        warnings
      };
    }

    // 2. Risk score check
    if (signal.riskScore > this.limits.maxRiskScore) {
      return {
        approved: false,
        reason: `Risk score ${(signal.riskScore * 100).toFixed(1)}% exceeds maximum ${(this.limits.maxRiskScore * 100)}%`,
        warnings
      };
    }

    // 3. Daily loss limit check
    if (this.dailyPnL < -this.limits.dailyLossLimit) {
      return {
        approved: false,
        reason: `Daily loss limit reached ($${Math.abs(this.dailyPnL).toFixed(2)})`,
        warnings
      };
    }

    // 4. Weekly loss limit check
    if (this.weeklyPnL < -this.limits.weeklyLossLimit) {
      return {
        approved: false,
        reason: `Weekly loss limit reached ($${Math.abs(this.weeklyPnL).toFixed(2)})`,
        warnings
      };
    }

    // 5. Max open positions check
    if (this.openPositions.length >= this.limits.maxOpenPositions) {
      return {
        approved: false,
        reason: `Maximum ${this.limits.maxOpenPositions} open positions reached`,
        warnings
      };
    }

    // 6. Correlation check
    const correlatedPositions = this.getCorrelatedPositions(signal.symbol);
    if (correlatedPositions.length >= this.limits.maxCorrelatedPositions) {
      warnings.push(`Already ${correlatedPositions.length} correlated positions`);
    }

    // 7. Calculate suggested position size
    const suggestedSize = this.calculatePositionSize(signal);

    // 8. Volatility warning
    if (signal.riskScore > 0.6) {
      warnings.push('High volatility detected');
    }

    // 9. Confidence warning
    if (signal.confidence < 0.80) {
      warnings.push('Moderate confidence level');
    }

    return {
      approved: true,
      suggestedPositionSize: suggestedSize,
      warnings
    };
  }

  /**
   * Calculate optimal position size based on risk
   */
  private calculatePositionSize(signal: {
    confidence: number;
    riskScore: number;
    price: number;
  }): number {
    // Kelly Criterion-inspired sizing
    // Size = (Confidence - RiskScore) / RiskScore

    const baseSize = this.limits.maxPositionSize;
    const riskAdjustment = signal.confidence / (1 + signal.riskScore);

    // Conservative sizing: 10-50% of max position
    const suggestedSize = baseSize * riskAdjustment * 0.5;

    return Math.min(suggestedSize, this.limits.maxPositionSize);
  }

  /**
   * Get correlated positions (same sector/category)
   */
  private getCorrelatedPositions(symbol: string): PositionRisk[] {
    // Simplified: BTC-related coins are correlated
    const btcRelated = ['BTC', 'BCH', 'BSV', 'BTT'];
    const ethRelated = ['ETH', 'ETC', 'ETHO'];
    const defiCoins = ['UNI', 'AAVE', 'COMP', 'SUSHI'];

    let correlatedGroup: string[] = [];

    if (btcRelated.includes(symbol)) {
      correlatedGroup = btcRelated;
    } else if (ethRelated.includes(symbol)) {
      correlatedGroup = ethRelated;
    } else if (defiCoins.includes(symbol)) {
      correlatedGroup = defiCoins;
    }

    return this.openPositions.filter(p => correlatedGroup.includes(p.symbol));
  }

  /**
   * Add position to tracking
   */
  addPosition(position: PositionRisk) {
    this.openPositions.push(position);
    console.log(`✅ Position added: ${position.symbol} ($${position.positionSize})`);
  }

  /**
   * Close position and update PnL
   */
  closePosition(symbol: string, exitPrice: number): { realized: number; profit: boolean } {
    const positionIndex = this.openPositions.findIndex(p => p.symbol === symbol);

    if (positionIndex === -1) {
      console.warn(`Position ${symbol} not found`);
      return { realized: 0, profit: false };
    }

    const position = this.openPositions[positionIndex];
    const realized = (exitPrice - position.entryPrice) * (position.positionSize / position.entryPrice);

    this.dailyPnL += realized;
    this.weeklyPnL += realized;

    this.openPositions.splice(positionIndex, 1);

    console.log(`🔒 Position closed: ${symbol} P/L: $${realized.toFixed(2)}`);

    return { realized, profit: realized > 0 };
  }

  /**
   * Get current exposure
   */
  getTotalExposure(): number {
    return this.openPositions.reduce((sum, p) => sum + p.positionSize, 0);
  }

  /**
   * Get portfolio risk score
   */
  getPortfolioRiskScore(): number {
    if (this.openPositions.length === 0) return 0;

    const avgRisk = this.openPositions.reduce((sum, p) => sum + p.riskScore, 0) / this.openPositions.length;
    const concentrationRisk = this.openPositions.length / this.limits.maxOpenPositions;

    return Math.min((avgRisk + concentrationRisk) / 2, 1.0);
  }

  /**
   * Reset daily/weekly counters
   */
  resetCounters() {
    const now = Date.now();
    const daysSinceReset = (now - this.lastResetTime) / (1000 * 60 * 60 * 24);

    if (daysSinceReset >= 1) {
      this.dailyPnL = 0;
      console.log('🔄 Daily P/L reset');
    }

    if (daysSinceReset >= 7) {
      this.weeklyPnL = 0;
      console.log('🔄 Weekly P/L reset');
    }

    if (daysSinceReset >= 1 || daysSinceReset >= 7) {
      this.lastResetTime = now;
    }
  }

  /**
   * Get risk report
   */
  getRiskReport() {
    return {
      limits: this.limits,
      dailyPnL: this.dailyPnL,
      weeklyPnL: this.weeklyPnL,
      openPositions: this.openPositions.length,
      totalExposure: this.getTotalExposure(),
      portfolioRiskScore: this.getPortfolioRiskScore(),
      dailyLimitUsage: Math.abs(this.dailyPnL) / this.limits.dailyLossLimit,
      weeklyLimitUsage: Math.abs(this.weeklyPnL) / this.limits.weeklyLossLimit,
      positionSlots: {
        used: this.openPositions.length,
        available: this.limits.maxOpenPositions - this.openPositions.length
      }
    };
  }

  /**
   * Update limits (admin only)
   */
  updateLimits(newLimits: Partial<RiskLimits>) {
    this.limits = { ...this.limits, ...newLimits };
    console.log('⚙️ Risk limits updated:', newLimits);
  }
}

export const riskManager = new RiskManagementModule();