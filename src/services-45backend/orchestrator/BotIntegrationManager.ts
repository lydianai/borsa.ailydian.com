import {
  getOrchestrator,
  BotRegistration,
  MarketData,
  BotSignal,
  HealthCheckResult
} from './UnifiedRobotOrchestrator';
import { getMarketDataCache } from './SharedMarketDataCache';
import axios from 'axios';

const AI_MODELS_SERVICE = process.env.AI_MODELS_SERVICE_URL || 'http://localhost:5003';
const _SIGNAL_GEN_SERVICE = process.env.SIGNAL_GEN_SERVICE_URL || 'http://localhost:5004';

export class BotIntegrationManager {
  private orchestrator = getOrchestrator();
  private marketCache = getMarketDataCache();
  private isInitialized = false;

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.warn('[BotIntegration] Already initialized');
      return;
    }

    console.log('[BotIntegration] Initializing bot integrations...');

    try {
      await this.registerTypeScriptBots();
      
      await this.registerPythonBots();

      const defaultSymbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
        'XRP/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT'
      ];
      await this.marketCache.warmup(defaultSymbols);

      this.marketCache.start();
      this.orchestrator.start();

      this.setupMarketDataListener();

      this.isInitialized = true;
      console.log('[BotIntegration] Initialization complete');
    } catch (error) {
      console.error('[BotIntegration] Initialization failed:', error);
      throw error;
    }
  }

  private async registerTypeScriptBots(): Promise<void> {
    console.log('[BotIntegration] TypeScript bots disabled (TensorFlow dependencies removed)');
    console.log('[BotIntegration] Using Python bots only');
  }

  private async registerPythonBots(): Promise<void> {
    console.log('[BotIntegration] Registering Python bots...');

    const pythonModels = [
      { id: 'py-lstm-standard', name: 'LSTM Standard', type: 'LSTM' as const, model: 'lstm_standard' },
      { id: 'py-lstm-bidirectional', name: 'LSTM Bidirectional', type: 'LSTM' as const, model: 'lstm_bidirectional' },
      { id: 'py-gru-attention', name: 'GRU Attention', type: 'GRU' as const, model: 'gru_attention' },
      { id: 'py-transformer-standard', name: 'Transformer Standard', type: 'Transformer' as const, model: 'transformer_standard' },
      { id: 'py-xgboost', name: 'XGBoost', type: 'XGBoost' as const, model: 'xgboost' },
      { id: 'py-lightgbm', name: 'LightGBM', type: 'LightGBM' as const, model: 'lightgbm' },
      { id: 'py-catboost', name: 'CatBoost', type: 'CatBoost' as const, model: 'catboost' }
    ];

    for (const model of pythonModels) {
      this.orchestrator.registerBot({
        id: model.id,
        name: model.name,
        type: model.type,
        status: 'active',
        healthScore: 100,
        lastHeartbeat: Date.now(),
        performance: {
          avgInferenceTime: 0,
          accuracy: 0,
          totalSignals: 0,
          successfulSignals: 0,
          failedSignals: 0,
          uptime: 0,
          lastInferenceTime: 0
        },
        generateSignal: async (marketData: MarketData): Promise<BotSignal> => {
          const startTime = Date.now();
          try {
            const response = await axios.post(
              `${AI_MODELS_SERVICE}/predict/single`,
              {
                symbol: marketData.symbol.replace('/', ''),
                timeframe: '1h',
                model: model.model
              },
              { timeout: 10000 }
            );

            if (!response.data?.success) {
              throw new Error('Prediction failed');
            }

            const prediction = response.data.prediction;

            return {
              botId: model.id,
              botName: model.name,
              botType: model.type,
              symbol: marketData.symbol,
              action: prediction.action,
              confidence: prediction.confidence * 100,
              timestamp: Date.now(),
              metadata: {
                inferenceTime: Date.now() - startTime,
                prediction: prediction.prediction
              }
            };
          } catch (error) {
            throw new Error(`${model.name} signal generation failed: ${error}`);
          }
        },
        healthCheck: async (): Promise<HealthCheckResult> => {
          try {
            const response = await axios.get(`${AI_MODELS_SERVICE}/health`, { timeout: 3000 });
            const isHealthy = response.data?.status === 'healthy';

            return {
              botId: model.id,
              status: isHealthy ? 'healthy' : 'unhealthy',
              healthScore: isHealthy ? 100 : 0,
              timestamp: Date.now()
            };
          } catch (error) {
            return {
              botId: model.id,
              status: 'unhealthy',
              healthScore: 0,
              message: 'AI Models service unreachable',
              timestamp: Date.now()
            };
          }
        }
      });
    }

    console.log(`[BotIntegration] Python bots registered: ${pythonModels.length}`);
  }

  private setupMarketDataListener(): void {
    this.marketCache.on('data:updated', ({ symbol, data }) => {
      this.orchestrator.updateMarketData(symbol, {
        symbol,
        price: data.price,
        volume: data.volume,
        high24h: data.high24h,
        low24h: data.low24h,
        change24h: data.change24h,
        timestamp: data.timestamp,
        indicators: data.indicators
      });
    });

    console.log('[BotIntegration] Market data listener configured');
  }

  async shutdown(): Promise<void> {
    console.log('[BotIntegration] Shutting down...');
    
    this.orchestrator.stop();
    this.marketCache.stop();
    
    this.isInitialized = false;
    console.log('[BotIntegration] Shutdown complete');
  }

  getStatus(): {
    initialized: boolean;
    totalBots: number;
    activeBots: number;
    cacheSize: number;
  } {
    return {
      initialized: this.isInitialized,
      totalBots: this.orchestrator.getAllBots().length,
      activeBots: this.orchestrator.getActiveBots().length,
      cacheSize: this.marketCache.getStats().cacheSize
    };
  }
}

let integrationManager: BotIntegrationManager | null = null;

export function getBotIntegrationManager(): BotIntegrationManager {
  if (!integrationManager) {
    integrationManager = new BotIntegrationManager();
  }
  return integrationManager;
}
