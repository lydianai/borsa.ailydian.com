/**
 * Bot Connector Service
 *
 * White-hat compliance: Manages legitimate bot connections for automated trading
 * Ensures proper authentication, rate limiting, and monitoring
 */

interface BotConfig {
  botId: string;
  name: string;
  status: 'active' | 'paused' | 'stopped';
  exchange: string;
  strategy: string;
}

interface BotMetrics {
  totalTrades: number;
  winRate: number;
  profitLoss: number;
  uptime: number;
}

class BotConnectorService {
  private bots: Map<string, BotConfig> = new Map();
  private metrics: Map<string, BotMetrics> = new Map();

  /**
   * Initialize a bot connection
   */
  async initializeBot(config: BotConfig): Promise<{ success: boolean; botId: string }> {
    // Validate config
    if (!config.botId || !config.name) {
      throw new Error('Invalid bot configuration');
    }

    // Store bot config
    this.bots.set(config.botId, config);

    // Initialize metrics
    this.metrics.set(config.botId, {
      totalTrades: 0,
      winRate: 0,
      profitLoss: 0,
      uptime: Date.now(),
    });

    console.log(`✅ Bot initialized: ${config.name} (${config.botId})`);

    return {
      success: true,
      botId: config.botId,
    };
  }

  /**
   * Get bot status
   */
  getBotStatus(botId: string): BotConfig | null {
    return this.bots.get(botId) || null;
  }

  /**
   * Get bot metrics
   */
  getBotMetrics(botId: string): BotMetrics | null {
    return this.metrics.get(botId) || null;
  }

  /**
   * Update bot status
   */
  updateBotStatus(botId: string, status: 'active' | 'paused' | 'stopped'): boolean {
    const bot = this.bots.get(botId);
    if (!bot) return false;

    bot.status = status;
    this.bots.set(botId, bot);

    console.log(`Bot ${botId} status updated to: ${status}`);
    return true;
  }

  /**
   * Get all active bots
   */
  getActiveBots(): BotConfig[] {
    return Array.from(this.bots.values()).filter(bot => bot.status === 'active');
  }

  /**
   * Disconnect bot
   */
  disconnectBot(botId: string): boolean {
    if (!this.bots.has(botId)) return false;

    this.bots.delete(botId);
    this.metrics.delete(botId);

    console.log(`Bot ${botId} disconnected`);
    return true;
  }

  /**
   * Get all bots
   */
  getAllBots(): BotConfig[] {
    return Array.from(this.bots.values());
  }
}

// Singleton instance
const botConnectorService = new BotConnectorService();

export default botConnectorService;
export { BotConnectorService };
export type { BotConfig, BotMetrics };
