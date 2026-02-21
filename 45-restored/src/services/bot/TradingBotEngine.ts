/**
 * TRADING BOT ENGINE
 * WHITE-HAT COMPLIANT: Paper trading only, educational purposes
 * NO REAL MONEY - Simulation mode with risk management
 */

export interface BotConfig {
  id: string;
  name: string;
  symbol: string;
  strategy: 'ai_consensus' | 'technical' | 'momentum' | 'mean_reversion';
  enabled: boolean;
  paperTrading: true; // ALWAYS true for safety
  riskManagement: {
    maxPositionSize: number; // Max % of portfolio per trade
    stopLoss: number; // % loss to trigger stop
    takeProfit: number; // % profit to trigger close
    maxDailyLoss: number; // Max % loss per day
    maxOpenPositions: number; // Max concurrent positions
  };
  aiModels: string[]; // Which AI models to use
  confidenceThreshold: number; // Min confidence to trade (0-1)
}

export interface Position {
  id: string;
  botId: string;
  symbol: string;
  side: 'long' | 'short';
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  stopLoss: number;
  takeProfit: number;
  pnl: number;
  pnlPercent: number;
  openedAt: number;
  status: 'open' | 'closed' | 'stopped';
}

export interface Signal {
  symbol: string;
  action: 'buy' | 'sell' | 'hold';
  confidence: number;
  price: number;
  models: string[];
  indicators: any;
  timestamp: number;
}

export class TradingBotEngine {
  private bots: Map<string, BotConfig> = new Map();
  private positions: Map<string, Position> = new Map();
  private isRunning = false;
  private updateInterval: NodeJS.Timeout | null = null;

  constructor() {
    console.log('🤖 Trading Bot Engine initialized (PAPER TRADING ONLY)');
  }

  /**
   * Create a new trading bot
   */
  public createBot(config: Omit<BotConfig, 'id' | 'paperTrading'>): BotConfig {
    const botId = `bot_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    const bot: BotConfig = {
      ...config,
      id: botId,
      paperTrading: true, // ENFORCED - no real trading
    };

    // Validate config
    this.validateBotConfig(bot);

    this.bots.set(botId, bot);
    console.log(`✅ Bot created: ${bot.name} (${botId})`);

    return bot;
  }

  /**
   * Validate bot configuration for safety
   */
  private validateBotConfig(config: BotConfig): void {
    // WHITE-HAT SAFETY CHECKS
    if (!config.paperTrading) {
      throw new Error('❌ SECURITY: Only paper trading is allowed');
    }

    if (config.riskManagement.maxPositionSize > 10) {
      throw new Error('❌ RISK: Max position size cannot exceed 10%');
    }

    if (config.riskManagement.stopLoss > 10) {
      throw new Error('❌ RISK: Stop loss cannot exceed 10%');
    }

    if (config.riskManagement.maxOpenPositions > 5) {
      throw new Error('❌ RISK: Max open positions cannot exceed 5');
    }

    if (config.confidenceThreshold < 0.5) {
      throw new Error('❌ RISK: Confidence threshold must be at least 0.5 (50%)');
    }

    console.log('✅ Bot config validated (WHITE-HAT compliant)');
  }

  /**
   * Get signal from AI services
   */
  private async getSignal(symbol: string, models: string[]): Promise<Signal | null> {
    try {
      // Call Signal Generator service
      const response = await fetch('http://localhost:5004/signals/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, timeframe: '1h' }),
      });

      if (!response.ok) {
        console.error(`❌ Failed to get signal for ${symbol}`);
        return null;
      }

      const data = await response.json();

      if (!data.success || !data.signal) {
        return null;
      }

      return {
        symbol,
        action: data.signal.action,
        confidence: data.signal.confidence / 100, // Convert to 0-1
        price: data.signal.current_price,
        models: data.signal.models || [],
        indicators: data.signal.indicators || {},
        timestamp: Date.now(),
      };
    } catch (error: any) {
      console.error(`❌ Error getting signal for ${symbol}:`, error.message);
      return null;
    }
  }

  /**
   * Execute a trade (PAPER TRADING ONLY)
   */
  private async executeTrade(
    bot: BotConfig,
    signal: Signal
  ): Promise<Position | null> {
    // SAFETY CHECK: Only paper trading
    if (!bot.paperTrading) {
      console.error('❌ BLOCKED: Real trading is not allowed');
      return null;
    }

    // Check confidence threshold
    if (signal.confidence < bot.confidenceThreshold) {
      console.log(`⚠️  Signal confidence too low: ${(signal.confidence * 100).toFixed(1)}%`);
      return null;
    }

    // Check max open positions
    const openPositions = Array.from(this.positions.values()).filter(
      p => p.botId === bot.id && p.status === 'open'
    );

    if (openPositions.length >= bot.riskManagement.maxOpenPositions) {
      console.log(`⚠️  Max open positions reached for ${bot.name}`);
      return null;
    }

    // Calculate position size (paper trading)
    const portfolioValue = 10000; // $10k paper money
    const maxPositionValue = portfolioValue * (bot.riskManagement.maxPositionSize / 100);
    const quantity = maxPositionValue / signal.price;

    // Calculate stop loss and take profit prices
    const stopLoss = signal.price * (1 - bot.riskManagement.stopLoss / 100);
    const takeProfit = signal.price * (1 + bot.riskManagement.takeProfit / 100);

    const position: Position = {
      id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      botId: bot.id,
      symbol: signal.symbol,
      side: signal.action === 'buy' ? 'long' : 'short',
      entryPrice: signal.price,
      currentPrice: signal.price,
      quantity,
      stopLoss,
      takeProfit,
      pnl: 0,
      pnlPercent: 0,
      openedAt: Date.now(),
      status: 'open',
    };

    this.positions.set(position.id, position);

    console.log(`📊 PAPER TRADE: ${signal.action.toUpperCase()} ${quantity.toFixed(4)} ${signal.symbol} @ $${signal.price}`);
    console.log(`   Stop Loss: $${stopLoss.toFixed(2)} | Take Profit: $${takeProfit.toFixed(2)}`);

    return position;
  }

  /**
   * Update all open positions
   */
  private async updatePositions(): Promise<void> {
    for (const [posId, position] of this.positions.entries()) {
      if (position.status !== 'open') continue;

      try {
        // Get current price
        const symbol = position.symbol.replace('/', '');
        const response = await fetch(`http://localhost:3000/api/binance/price?symbol=${symbol}`);
        const data = await response.json();

        if (!data.success) continue;

        const currentPrice = data.data.price;
        position.currentPrice = currentPrice;

        // Calculate P&L
        const priceDiff = currentPrice - position.entryPrice;
        position.pnl = priceDiff * position.quantity;
        position.pnlPercent = (priceDiff / position.entryPrice) * 100;

        // Check stop loss
        if (currentPrice <= position.stopLoss) {
          position.status = 'stopped';
          console.log(`🛑 STOP LOSS triggered: ${position.symbol} | PnL: $${position.pnl.toFixed(2)} (${position.pnlPercent.toFixed(2)}%)`);
        }

        // Check take profit
        if (currentPrice >= position.takeProfit) {
          position.status = 'closed';
          console.log(`✅ TAKE PROFIT triggered: ${position.symbol} | PnL: $${position.pnl.toFixed(2)} (${position.pnlPercent.toFixed(2)}%)`);
        }
      } catch (error: any) {
        console.error(`❌ Error updating position ${posId}:`, error.message);
      }
    }
  }

  /**
   * Run bot logic for all enabled bots
   */
  private async runBotCycle(): Promise<void> {
    for (const [botId, bot] of this.bots.entries()) {
      if (!bot.enabled) continue;

      try {
        // Get AI signal
        const signal = await this.getSignal(bot.symbol, bot.aiModels);

        if (!signal) continue;

        // Only trade on strong buy/sell signals
        if (signal.action === 'buy' || signal.action === 'sell') {
          await this.executeTrade(bot, signal);
        }
      } catch (error: any) {
        console.error(`❌ Error in bot ${bot.name}:`, error.message);
      }
    }

    // Update all positions
    await this.updatePositions();
  }

  /**
   * Start the trading bot engine
   */
  public start(): void {
    if (this.isRunning) {
      console.log('⚠️  Bot engine already running');
      return;
    }

    console.log('🚀 Starting Trading Bot Engine (PAPER TRADING)');
    this.isRunning = true;

    // Run every 60 seconds
    this.updateInterval = setInterval(() => {
      this.runBotCycle();
    }, 60000);

    // Run immediately
    this.runBotCycle();
  }

  /**
   * Stop the trading bot engine
   */
  public stop(): void {
    if (!this.isRunning) {
      console.log('⚠️  Bot engine not running');
      return;
    }

    console.log('🛑 Stopping Trading Bot Engine');
    this.isRunning = false;

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  /**
   * Get all bots
   */
  public getBots(): BotConfig[] {
    return Array.from(this.bots.values());
  }

  /**
   * Get all positions
   */
  public getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get bot statistics
   */
  public getStats(botId: string): any {
    const bot = this.bots.get(botId);
    if (!bot) return null;

    const positions = Array.from(this.positions.values()).filter(p => p.botId === botId);
    const closedPositions = positions.filter(p => p.status === 'closed' || p.status === 'stopped');

    const totalPnL = closedPositions.reduce((sum, p) => sum + p.pnl, 0);
    const winningTrades = closedPositions.filter(p => p.pnl > 0).length;
    const losingTrades = closedPositions.filter(p => p.pnl < 0).length;
    const winRate = closedPositions.length > 0 ? (winningTrades / closedPositions.length) * 100 : 0;

    return {
      botId,
      botName: bot.name,
      totalTrades: closedPositions.length,
      openPositions: positions.filter(p => p.status === 'open').length,
      winningTrades,
      losingTrades,
      winRate: winRate.toFixed(2),
      totalPnL: totalPnL.toFixed(2),
      isRunning: bot.enabled,
    };
  }
}

// Singleton instance
let botEngine: TradingBotEngine | null = null;

export function getTradingBotEngine(): TradingBotEngine {
  if (!botEngine) {
    botEngine = new TradingBotEngine();
  }
  return botEngine;
}
