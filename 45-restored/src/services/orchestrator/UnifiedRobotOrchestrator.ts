import { EventEmitter } from 'events';

export enum OrchestratorEvent {
  MARKET_DATA_UPDATED = 'market:data:updated',
  SIGNAL_GENERATED = 'signal:generated',
  BOT_HEALTH_CHECK = 'bot:health',
  ERROR_OCCURRED = 'error:occurred',
  CONSENSUS_READY = 'consensus:ready',
  BOT_REGISTERED = 'bot:registered',
  BOT_UNREGISTERED = 'bot:unregistered',
  BOT_STATUS_CHANGED = 'bot:status:changed'
}

export type BotType = 'LSTM' | 'GRU' | 'Transformer' | 'XGBoost' | 'LightGBM' | 'CatBoost' | 'RL' | 'Quantum' | 'Hybrid' | 'CNN';
export type BotStatus = 'active' | 'paused' | 'error' | 'stopped' | 'initializing';
export type SignalAction = 'BUY' | 'SELL' | 'HOLD';

export interface BotPerformance {
  avgInferenceTime: number;
  accuracy: number;
  totalSignals: number;
  successfulSignals: number;
  failedSignals: number;
  uptime: number;
  lastInferenceTime: number;
}

export interface BotRegistration {
  id: string;
  name: string;
  type: BotType;
  status: BotStatus;
  healthScore: number;
  lastHeartbeat: number;
  performance: BotPerformance;
  metadata?: Record<string, any>;
  generateSignal?: (marketData: MarketData) => Promise<BotSignal>;
  healthCheck?: () => Promise<HealthCheckResult>;
}

export interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  high24h: number;
  low24h: number;
  change24h: number;
  timestamp: number;
  indicators?: TechnicalIndicators;
}

export interface TechnicalIndicators {
  rsi?: number;
  macd?: { value: number; signal: number; histogram: number };
  bollingerBands?: { upper: number; middle: number; lower: number };
  ema?: { short: number; long: number };
  sma?: { short: number; long: number };
  vwap?: number;
  atr?: number;
}

export interface BotSignal {
  botId: string;
  botName: string;
  botType: BotType;
  symbol: string;
  action: SignalAction;
  confidence: number;
  targetPrice?: number;
  stopLoss?: number;
  reasoning?: string[];
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface ConsensusSignal {
  symbol: string;
  action: SignalAction;
  confidence: number;
  quality: 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR';
  targetPrice: number;
  stopLoss: number;
  riskReward: number;
  entryPrice: number;
  botSignals: BotSignal[];
  consensus: {
    agreement: number;
    totalBots: number;
    buyVotes: number;
    sellVotes: number;
    holdVotes: number;
    conflictResolution: string;
  };
  timestamp: number;
}

export interface HealthCheckResult {
  botId: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  healthScore: number;
  message?: string;
  timestamp: number;
}

class SharedStateManager {
  private marketCache: Map<string, MarketData> = new Map();
  private indicatorCache: Map<string, TechnicalIndicators> = new Map();
  private signalCache: Map<string, BotSignal[]> = new Map();

  getMarketData(symbol: string): MarketData | undefined {
    return this.marketCache.get(symbol);
  }

  setMarketData(symbol: string, data: MarketData): void {
    this.marketCache.set(symbol, data);
  }

  getAllMarketData(): Map<string, MarketData> {
    return new Map(this.marketCache);
  }

  getIndicators(symbol: string): TechnicalIndicators | undefined {
    return this.indicatorCache.get(symbol);
  }

  setIndicators(symbol: string, indicators: TechnicalIndicators): void {
    this.indicatorCache.set(symbol, indicators);
  }

  getSignals(symbol: string): BotSignal[] {
    return this.signalCache.get(symbol) || [];
  }

  addSignal(symbol: string, signal: BotSignal): void {
    const signals = this.getSignals(symbol);
    signals.push(signal);
    this.signalCache.set(symbol, signals);
  }

  clearSignals(symbol: string): void {
    this.signalCache.delete(symbol);
  }

  clearAllCaches(): void {
    this.marketCache.clear();
    this.indicatorCache.clear();
    this.signalCache.clear();
  }
}

class BotRegistry {
  private bots: Map<string, BotRegistration> = new Map();

  register(bot: BotRegistration): void {
    this.bots.set(bot.id, bot);
  }

  unregister(botId: string): boolean {
    return this.bots.delete(botId);
  }

  get(botId: string): BotRegistration | undefined {
    return this.bots.get(botId);
  }

  getAll(): BotRegistration[] {
    return Array.from(this.bots.values());
  }

  getActive(): BotRegistration[] {
    return this.getAll().filter(bot => bot.status === 'active');
  }

  getByType(type: BotType): BotRegistration[] {
    return this.getAll().filter(bot => bot.type === type);
  }

  updateStatus(botId: string, status: BotStatus): void {
    const bot = this.bots.get(botId);
    if (bot) {
      bot.status = status;
    }
  }

  updateHealthScore(botId: string, score: number): void {
    const bot = this.bots.get(botId);
    if (bot) {
      bot.healthScore = Math.max(0, Math.min(100, score));
    }
  }

  updateHeartbeat(botId: string): void {
    const bot = this.bots.get(botId);
    if (bot) {
      bot.lastHeartbeat = Date.now();
    }
  }

  count(): number {
    return this.bots.size;
  }

  countByStatus(status: BotStatus): number {
    return this.getAll().filter(bot => bot.status === status).length;
  }
}

export class UnifiedRobotOrchestrator extends EventEmitter {
  private registry: BotRegistry;
  private stateManager: SharedStateManager;
  private isRunning: boolean = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.registry = new BotRegistry();
    this.stateManager = new SharedStateManager();
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.on(OrchestratorEvent.ERROR_OCCURRED, (error) => {
      console.error('[Orchestrator] Error:', error);
    });

    this.on(OrchestratorEvent.BOT_REGISTERED, (bot: BotRegistration) => {
      console.log(`[Orchestrator] Bot registered: ${bot.name} (${bot.type})`);
    });

    this.on(OrchestratorEvent.BOT_UNREGISTERED, (botId: string) => {
      console.log(`[Orchestrator] Bot unregistered: ${botId}`);
    });

    this.on(OrchestratorEvent.BOT_STATUS_CHANGED, ({ botId, status }) => {
      console.log(`[Orchestrator] Bot ${botId} status changed to: ${status}`);
    });
  }

  registerBot(bot: BotRegistration): void {
    try {
      this.registry.register(bot);
      this.emit(OrchestratorEvent.BOT_REGISTERED, bot);
    } catch (error) {
      this.emit(OrchestratorEvent.ERROR_OCCURRED, {
        context: 'registerBot',
        error,
        botId: bot.id
      });
      throw error;
    }
  }

  unregisterBot(botId: string): boolean {
    try {
      const success = this.registry.unregister(botId);
      if (success) {
        this.emit(OrchestratorEvent.BOT_UNREGISTERED, botId);
      }
      return success;
    } catch (error) {
      this.emit(OrchestratorEvent.ERROR_OCCURRED, {
        context: 'unregisterBot',
        error,
        botId
      });
      return false;
    }
  }

  getBot(botId: string): BotRegistration | undefined {
    return this.registry.get(botId);
  }

  getAllBots(): BotRegistration[] {
    return this.registry.getAll();
  }

  getActiveBots(): BotRegistration[] {
    return this.registry.getActive();
  }

  getBotsByType(type: BotType): BotRegistration[] {
    return this.registry.getByType(type);
  }

  updateMarketData(symbol: string, data: MarketData): void {
    try {
      this.stateManager.setMarketData(symbol, data);
      this.emit(OrchestratorEvent.MARKET_DATA_UPDATED, { symbol, data });
    } catch (error) {
      this.emit(OrchestratorEvent.ERROR_OCCURRED, {
        context: 'updateMarketData',
        error,
        symbol
      });
    }
  }

  getMarketData(symbol: string): MarketData | undefined {
    return this.stateManager.getMarketData(symbol);
  }

  async generateConsensusSignal(symbol: string): Promise<ConsensusSignal | null> {
    try {
      const marketData = this.stateManager.getMarketData(symbol);
      if (!marketData) {
        throw new Error(`No market data available for ${symbol}`);
      }

      const activeBots = this.registry.getActive();
      if (activeBots.length === 0) {
        throw new Error('No active bots available');
      }

      const botSignals: BotSignal[] = [];

      for (const bot of activeBots) {
        try {
          if (bot.generateSignal) {
            const signal = await bot.generateSignal(marketData);
            botSignals.push(signal);
            this.stateManager.addSignal(symbol, signal);
            
            bot.performance.totalSignals++;
            bot.performance.successfulSignals++;
            this.registry.updateHeartbeat(bot.id);
          }
        } catch (error) {
          console.error(`[Orchestrator] Bot ${bot.id} failed to generate signal:`, error);
          bot.performance.failedSignals++;
          this.emit(OrchestratorEvent.ERROR_OCCURRED, {
            context: 'generateSignal',
            error,
            botId: bot.id,
            symbol
          });
        }
      }

      if (botSignals.length === 0) {
        return null;
      }

      this.emit(OrchestratorEvent.SIGNAL_GENERATED, { symbol, signals: botSignals });

      const consensus = this.aggregateSignals(botSignals, marketData);
      this.emit(OrchestratorEvent.CONSENSUS_READY, consensus);

      return consensus;
    } catch (error) {
      this.emit(OrchestratorEvent.ERROR_OCCURRED, {
        context: 'generateConsensusSignal',
        error,
        symbol
      });
      return null;
    }
  }

  private aggregateSignals(signals: BotSignal[], marketData: MarketData): ConsensusSignal {
    const buyVotes: number[] = [];
    const sellVotes: number[] = [];
    const holdVotes: number[] = [];

    const botWeights: Record<BotType, number> = {
      'Transformer': 1.4,
      'GRU': 1.3,
      'LSTM': 1.2,
      'XGBoost': 1.1,
      'LightGBM': 1.1,
      'CatBoost': 1.1,
      'RL': 1.0,
      'Quantum': 1.0,
      'Hybrid': 1.0,
      'CNN': 1.0
    };

    for (const signal of signals) {
      const weight = botWeights[signal.botType] || 1.0;
      const weightedConfidence = signal.confidence * weight;

      if (signal.action === 'BUY') {
        buyVotes.push(weightedConfidence);
      } else if (signal.action === 'SELL') {
        sellVotes.push(weightedConfidence);
      } else {
        holdVotes.push(weightedConfidence);
      }
    }

    const totalBots = signals.length;
    const buyStrength = buyVotes.length > 0 ? buyVotes.reduce((a, b) => a + b, 0) / totalBots : 0;
    const sellStrength = sellVotes.length > 0 ? sellVotes.reduce((a, b) => a + b, 0) / totalBots : 0;
    const holdStrength = holdVotes.length > 0 ? holdVotes.reduce((a, b) => a + b, 0) / totalBots : 0;

    const strengths = { BUY: buyStrength, SELL: sellStrength, HOLD: holdStrength };
    const action = Object.keys(strengths).reduce((a, b) => 
      strengths[a as SignalAction] > strengths[b as SignalAction] ? a : b
    ) as SignalAction;

    const confidence = strengths[action];
    const entryPrice = marketData.price;

    let targetPrice: number;
    let stopLoss: number;

    if (action === 'BUY') {
      targetPrice = entryPrice * 1.02;
      stopLoss = entryPrice * 0.99;
    } else if (action === 'SELL') {
      targetPrice = entryPrice * 0.98;
      stopLoss = entryPrice * 1.01;
    } else {
      targetPrice = entryPrice;
      stopLoss = entryPrice * 0.995;
    }

    const risk = Math.abs(entryPrice - stopLoss);
    const reward = Math.abs(targetPrice - entryPrice);
    const riskReward = risk > 0 ? reward / risk : 0;

    const agreement = action === 'BUY' ? buyVotes.length / totalBots :
                      action === 'SELL' ? sellVotes.length / totalBots :
                      holdVotes.length / totalBots;

    const quality = this.calculateQuality(confidence, agreement, riskReward);

    return {
      symbol: marketData.symbol,
      action,
      confidence: confidence * 100,
      quality,
      targetPrice,
      stopLoss,
      riskReward,
      entryPrice,
      botSignals: signals,
      consensus: {
        agreement,
        totalBots,
        buyVotes: buyVotes.length,
        sellVotes: sellVotes.length,
        holdVotes: holdVotes.length,
        conflictResolution: `${action} selected with ${(agreement * 100).toFixed(1)}% agreement`
      },
      timestamp: Date.now()
    };
  }

  private calculateQuality(confidence: number, agreement: number, riskReward: number): 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR' {
    const score = confidence * 0.4 + agreement * 0.4 + Math.min(riskReward / 3.0, 1.0) * 0.2;

    if (score >= 0.8) return 'EXCELLENT';
    if (score >= 0.7) return 'GOOD';
    if (score >= 0.6) return 'FAIR';
    return 'POOR';
  }

  async performHealthChecks(): Promise<HealthCheckResult[]> {
    const results: HealthCheckResult[] = [];
    const allBots = this.registry.getAll();

    for (const bot of allBots) {
      try {
        let result: HealthCheckResult;

        if (bot.healthCheck) {
          result = await bot.healthCheck();
        } else {
          const timeSinceHeartbeat = Date.now() - bot.lastHeartbeat;
          const isHealthy = timeSinceHeartbeat < 60000;

          result = {
            botId: bot.id,
            status: isHealthy ? 'healthy' : 'degraded',
            healthScore: isHealthy ? 100 : 50,
            message: isHealthy ? 'OK' : 'No recent heartbeat',
            timestamp: Date.now()
          };
        }

        results.push(result);
        this.registry.updateHealthScore(bot.id, result.healthScore);

        if (result.status === 'unhealthy') {
          this.registry.updateStatus(bot.id, 'error');
          this.emit(OrchestratorEvent.BOT_STATUS_CHANGED, {
            botId: bot.id,
            status: 'error'
          });
        }

        this.emit(OrchestratorEvent.BOT_HEALTH_CHECK, result);
      } catch (error) {
        const errorResult: HealthCheckResult = {
          botId: bot.id,
          status: 'unhealthy',
          healthScore: 0,
          message: error instanceof Error ? error.message : 'Health check failed',
          timestamp: Date.now()
        };

        results.push(errorResult);
        this.registry.updateStatus(bot.id, 'error');
        this.emit(OrchestratorEvent.ERROR_OCCURRED, {
          context: 'healthCheck',
          error,
          botId: bot.id
        });
      }
    }

    return results;
  }

  start(): void {
    if (this.isRunning) {
      console.warn('[Orchestrator] Already running');
      return;
    }

    this.isRunning = true;
    console.log('[Orchestrator] Starting...');

    this.healthCheckInterval = setInterval(async () => {
      await this.performHealthChecks();
    }, 30000);

    console.log('[Orchestrator] Started successfully');
  }

  stop(): void {
    if (!this.isRunning) {
      console.warn('[Orchestrator] Not running');
      return;
    }

    this.isRunning = false;

    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }

    console.log('[Orchestrator] Stopped');
  }

  getStatus(): {
    isRunning: boolean;
    totalBots: number;
    activeBots: number;
    errorBots: number;
    pausedBots: number;
    avgHealthScore: number;
  } {
    const allBots = this.registry.getAll();
    const avgHealthScore = allBots.length > 0
      ? allBots.reduce((sum, bot) => sum + bot.healthScore, 0) / allBots.length
      : 0;

    return {
      isRunning: this.isRunning,
      totalBots: this.registry.count(),
      activeBots: this.registry.countByStatus('active'),
      errorBots: this.registry.countByStatus('error'),
      pausedBots: this.registry.countByStatus('paused'),
      avgHealthScore
    };
  }

  clearCaches(): void {
    this.stateManager.clearAllCaches();
  }
}

let orchestratorInstance: UnifiedRobotOrchestrator | null = null;

export function getOrchestrator(): UnifiedRobotOrchestrator {
  if (!orchestratorInstance) {
    orchestratorInstance = new UnifiedRobotOrchestrator();
  }
  return orchestratorInstance;
}
