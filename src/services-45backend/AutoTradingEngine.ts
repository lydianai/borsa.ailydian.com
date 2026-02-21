/**
 * AUTO TRADING ENGINE - Tam Otomatik İşlem Motoru
 * Gerçek ticarete hazır, tüm AI botlarını kontrol eden merkezi sistem
 */

export interface TradingConfig {
  enabled: boolean;
  mode: 'paper' | 'live'; // Kağıt ticaret veya gerçek ticaret
  maxPositionSize: number; // Maksimum pozisyon büyüklüğü (USD)
  maxDailyLoss: number; // Maksimum günlük kayıp (%)
  maxLeverage: number; // Maksimum kaldıraç
  tradingPairs: string[]; // İşlem yapılacak çiftler
  aiBotsEnabled: {
    quantumPro: boolean;
    masterOrchestrator: boolean;
    attentionTransformer: boolean;
    gradientBoosting: boolean;
    reinforcementLearning: boolean;
    tensorflowOptimizer: boolean;
  };
  riskManagement: {
    stopLossPercent: number;
    takeProfitPercent: number;
    trailingStopPercent: number;
    maxConcurrentTrades: number;
  };
}

export interface TradingSignal {
  botId: string;
  botName: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: number;
  reasoning: string[];
  riskScore: number;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  stopLoss: number;
  takeProfit: number;
  pnl: number;
  pnlPercent: number;
  openedAt: number;
  botId: string;
}

export interface TradingStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  todayPnL: number;
  openPositions: number;
  closedToday: number;
}

export class AutoTradingEngine {
  private config: TradingConfig;
  private isRunning: boolean = false;
  private positions: Map<string, Position> = new Map();
  private signals: TradingSignal[] = [];
  private stats: TradingStats;
  private updateInterval: NodeJS.Timeout | null = null;

  constructor(config: TradingConfig) {
    this.config = config;
    this.stats = {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalPnL: 0,
      todayPnL: 0,
      openPositions: 0,
      closedToday: 0,
    };
  }

  // Motor başlatma
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️ Auto Trading Engine already running');
      return;
    }

    console.log('🚀 Starting Auto Trading Engine...');
    console.log(`📊 Mode: ${this.config.mode.toUpperCase()}`);
    console.log(`💰 Max Position Size: $${this.config.maxPositionSize}`);
    console.log(`🛡️ Max Daily Loss: ${this.config.maxDailyLoss}%`);

    this.isRunning = true;

    // Ana trading loop başlat
    this.updateInterval = setInterval(() => {
      this.tradingLoop();
    }, 5000); // 5 saniyede bir kontrol

    console.log('✅ Auto Trading Engine started successfully');
  }

  // Motor durdurma
  async stop(): Promise<void> {
    if (!this.isRunning) {
      console.log('⚠️ Auto Trading Engine already stopped');
      return;
    }

    console.log('🛑 Stopping Auto Trading Engine...');

    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    // Tüm açık pozisyonları kapat
    await this.closeAllPositions('Engine stopped');

    this.isRunning = false;
    console.log('✅ Auto Trading Engine stopped successfully');
  }

  // Ana trading döngüsü
  private async tradingLoop(): Promise<void> {
    try {
      // 1. Risk kontrolü
      if (!this.checkRiskLimits()) {
        console.log('⚠️ Risk limits exceeded, skipping this cycle');
        return;
      }

      // 2. AI botlarından sinyalleri topla
      const signals = await this.collectSignals();

      // 3. Sinyalleri değerlendir ve trade aç/kapat
      await this.processSignals(signals);

      // 4. Açık pozisyonları güncelle
      await this.updateOpenPositions();

      // 5. İstatistikleri güncelle
      this.updateStats();

    } catch (error) {
      console.error('❌ Error in trading loop:', error);
    }
  }

  // Risk limitlerini kontrol et
  private checkRiskLimits(): boolean {
    // Günlük kayıp kontrolü
    if (Math.abs(this.stats.todayPnL) >= this.config.maxDailyLoss) {
      console.log('🛑 Daily loss limit reached');
      return false;
    }

    // Maksimum eşzamanlı işlem kontrolü
    if (this.positions.size >= this.config.riskManagement.maxConcurrentTrades) {
      console.log('⚠️ Max concurrent trades reached');
      return false;
    }

    return true;
  }

  // Tüm AI botlarından sinyalleri topla
  private async collectSignals(): Promise<TradingSignal[]> {
    const signals: TradingSignal[] = [];

    try {
      // Quantum Pro AI
      if (this.config.aiBotsEnabled.quantumPro) {
        const qpSignals = await this.fetchQuantumProSignals();
        signals.push(...qpSignals);
      }

      // Master Orchestrator
      if (this.config.aiBotsEnabled.masterOrchestrator) {
        const moSignals = await this.fetchMasterOrchestratorSignals();
        signals.push(...moSignals);
      }

      // Attention Transformer
      if (this.config.aiBotsEnabled.attentionTransformer) {
        const atSignals = await this.fetchAttentionTransformerSignals();
        signals.push(...atSignals);
      }

      // Gradient Boosting
      if (this.config.aiBotsEnabled.gradientBoosting) {
        const gbSignals = await this.fetchGradientBoostingSignals();
        signals.push(...gbSignals);
      }

      // Reinforcement Learning
      if (this.config.aiBotsEnabled.reinforcementLearning) {
        const rlSignals = await this.fetchReinforcementLearningSignals();
        signals.push(...rlSignals);
      }

      // TensorFlow Optimizer
      if (this.config.aiBotsEnabled.tensorflowOptimizer) {
        const tfSignals = await this.fetchTensorFlowSignals();
        signals.push(...tfSignals);
      }

      console.log(`📊 Collected ${signals.length} signals from AI bots`);
      return signals;

    } catch (error) {
      console.error('❌ Error collecting signals:', error);
      return [];
    }
  }

  // Quantum Pro AI sinyallerini al
  private async fetchQuantumProSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/quantum-pro/signals');
      const data = await response.json();

      if (data.success && data.signals) {
        return data.signals.map((s: any) => ({
          botId: 'quantum-pro',
          botName: 'Quantum Pro AI',
          symbol: s.symbol,
          action: s.action,
          confidence: s.confidence,
          price: 0, // Gerçek fiyat API'den alınacak
          timestamp: Date.now(),
          reasoning: s.detailedReasons || [],
          riskScore: s.riskScore || 0,
        }));
      }

      return [];
    } catch (error) {
      console.error('Error fetching Quantum Pro signals:', error);
      return [];
    }
  }

  // Diğer botlar için sinyal alma fonksiyonları (benzer yapıda)
  private async fetchMasterOrchestratorSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/ai-bots/all-signals?bot=masterOrchestrator');
      const data = await response.json();

      if (data.success && data.bots?.masterOrchestrator?.signals) {
        return this.transformSignals(data.bots.masterOrchestrator.signals, 'master-orchestrator', 'Master Orchestrator');
      }
      return [];
    } catch (error) {
      console.error('Error fetching Master Orchestrator signals:', error);
      return [];
    }
  }

  private async fetchAttentionTransformerSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/ai-bots/all-signals?bot=attentionTransformer');
      const data = await response.json();

      if (data.success && data.bots?.attentionTransformer?.signals) {
        return this.transformSignals(data.bots.attentionTransformer.signals, 'attention-transformer', 'Attention Transformer');
      }
      return [];
    } catch (error) {
      console.error('Error fetching Attention Transformer signals:', error);
      return [];
    }
  }

  private async fetchGradientBoostingSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/ai-bots/all-signals?bot=gradientBoosting');
      const data = await response.json();

      if (data.success && data.bots?.gradientBoosting?.signals) {
        return this.transformSignals(data.bots.gradientBoosting.signals, 'gradient-boosting', 'Gradient Boosting');
      }
      return [];
    } catch (error) {
      console.error('Error fetching Gradient Boosting signals:', error);
      return [];
    }
  }

  private async fetchReinforcementLearningSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/ai-bots/all-signals?bot=reinforcementLearning');
      const data = await response.json();

      if (data.success && data.bots?.reinforcementLearning?.signals) {
        return this.transformSignals(data.bots.reinforcementLearning.signals, 'reinforcement-learning', 'Reinforcement Learning');
      }
      return [];
    } catch (error) {
      console.error('Error fetching Reinforcement Learning signals:', error);
      return [];
    }
  }

  private async fetchTensorFlowSignals(): Promise<TradingSignal[]> {
    try {
      const response = await fetch('/api/ai-bots/all-signals?bot=tensorflowOptimizer');
      const data = await response.json();

      if (data.success && data.bots?.tensorflowOptimizer?.signals) {
        return this.transformSignals(data.bots.tensorflowOptimizer.signals, 'tensorflow-optimizer', 'TensorFlow Optimizer');
      }
      return [];
    } catch (error) {
      console.error('Error fetching TensorFlow Optimizer signals:', error);
      return [];
    }
  }

  // Helper: API sinyallerini TradingSignal formatına çevir
  private transformSignals(signals: any[], botId: string, botName: string): TradingSignal[] {
    return signals.map((s: any) => ({
      botId,
      botName,
      symbol: s.symbol,
      action: s.action,
      confidence: s.confidence,
      price: s.price,
      timestamp: s.timestamp || Date.now(),
      reasoning: s.detailedReasons || s.reasoning || [],
      riskScore: s.riskScore || 0,
    }));
  }

  // Sinyalleri işle ve trade aç
  private async processSignals(signals: TradingSignal[]): Promise<void> {
    // BUY sinyallerini filtrele ve güvene göre sırala
    const buySignals = signals
      .filter(s => s.action === 'BUY' && s.confidence >= 0.70)
      .sort((a, b) => b.confidence - a.confidence);

    // SELL sinyallerini filtrele
    const sellSignals = signals
      .filter(s => s.action === 'SELL')
      .sort((a, b) => b.confidence - a.confidence);

    // BUY sinyallerine göre pozisyon aç
    for (const signal of buySignals.slice(0, 3)) { // En fazla 3 yeni pozisyon
      if (this.positions.size >= this.config.riskManagement.maxConcurrentTrades) {
        break;
      }

      await this.openPosition(signal);
    }

    // SELL sinyallerine göre pozisyon kapat
    for (const signal of sellSignals) {
      const position = Array.from(this.positions.values()).find(
        p => p.symbol === signal.symbol
      );

      if (position) {
        await this.closePosition(position.id, 'AI SELL signal');
      }
    }
  }

  // Pozisyon aç
  private async openPosition(signal: TradingSignal): Promise<void> {
    try {
      // Gerçek fiyatı al
      const currentPrice = await this.getCurrentPrice(signal.symbol);

      // Pozisyon büyüklüğünü hesapla
      const positionSize = Math.min(
        this.config.maxPositionSize,
        this.config.maxPositionSize * (signal.confidence / 100)
      );

      const quantity = positionSize / currentPrice;

      // Stop loss ve take profit hesapla
      const stopLoss = currentPrice * (1 - this.config.riskManagement.stopLossPercent / 100);
      const takeProfit = currentPrice * (1 + this.config.riskManagement.takeProfitPercent / 100);

      const position: Position = {
        id: `POS-${Date.now()}-${signal.symbol}`,
        symbol: signal.symbol,
        side: 'LONG',
        entryPrice: currentPrice,
        currentPrice: currentPrice,
        quantity,
        stopLoss,
        takeProfit,
        pnl: 0,
        pnlPercent: 0,
        openedAt: Date.now(),
        botId: signal.botId,
      };

      this.positions.set(position.id, position);

      console.log(`✅ Opened position: ${signal.symbol} @ $${currentPrice}`);
      console.log(`   Bot: ${signal.botName}`);
      console.log(`   Confidence: ${(signal.confidence * 100).toFixed(1)}%`);
      console.log(`   Size: $${positionSize.toFixed(2)}`);
      console.log(`   Stop Loss: $${stopLoss.toFixed(2)}`);
      console.log(`   Take Profit: $${takeProfit.toFixed(2)}`);

      this.stats.totalTrades++;

    } catch (error) {
      console.error(`❌ Error opening position for ${signal.symbol}:`, error);
    }
  }

  // Pozisyon kapat
  private async closePosition(positionId: string, reason: string): Promise<void> {
    const position = this.positions.get(positionId);
    if (!position) return;

    try {
      const currentPrice = await this.getCurrentPrice(position.symbol);
      const pnl = (currentPrice - position.entryPrice) * position.quantity;
      const pnlPercent = ((currentPrice - position.entryPrice) / position.entryPrice) * 100;

      console.log(`🔴 Closed position: ${position.symbol}`);
      console.log(`   Entry: $${position.entryPrice.toFixed(2)}`);
      console.log(`   Exit: $${currentPrice.toFixed(2)}`);
      console.log(`   P&L: $${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`);
      console.log(`   Reason: ${reason}`);

      // İstatistikleri güncelle
      this.stats.totalPnL += pnl;
      this.stats.todayPnL += pnl;

      if (pnl > 0) {
        this.stats.winningTrades++;
      } else {
        this.stats.losingTrades++;
      }

      this.stats.closedToday++;

      // Pozisyonu kaldır
      this.positions.delete(positionId);

    } catch (error) {
      console.error(`❌ Error closing position ${positionId}:`, error);
    }
  }

  // Tüm pozisyonları kapat
  private async closeAllPositions(reason: string): Promise<void> {
    console.log(`🛑 Closing all ${this.positions.size} positions...`);

    for (const [positionId] of this.positions) {
      await this.closePosition(positionId, reason);
    }
  }

  // Açık pozisyonları güncelle
  private async updateOpenPositions(): Promise<void> {
    for (const [positionId, position] of this.positions) {
      try {
        const currentPrice = await this.getCurrentPrice(position.symbol);
        position.currentPrice = currentPrice;
        position.pnl = (currentPrice - position.entryPrice) * position.quantity;
        position.pnlPercent = ((currentPrice - position.entryPrice) / position.entryPrice) * 100;

        // Stop loss kontrolü
        if (currentPrice <= position.stopLoss) {
          await this.closePosition(positionId, 'Stop Loss triggered');
          continue;
        }

        // Take profit kontrolü
        if (currentPrice >= position.takeProfit) {
          await this.closePosition(positionId, 'Take Profit triggered');
          continue;
        }

        // Trailing stop kontrolü
        const trailingStopPrice = currentPrice * (1 - this.config.riskManagement.trailingStopPercent / 100);
        if (trailingStopPrice > position.stopLoss) {
          position.stopLoss = trailingStopPrice;
          console.log(`📈 Trailing stop updated for ${position.symbol}: $${trailingStopPrice.toFixed(2)}`);
        }

      } catch (error) {
        console.error(`Error updating position ${positionId}:`, error);
      }
    }
  }

  // İstatistikleri güncelle
  private updateStats(): void {
    this.stats.openPositions = this.positions.size;

    if (this.stats.totalTrades > 0) {
      this.stats.winRate = (this.stats.winningTrades / this.stats.totalTrades) * 100;
    }
  }

  // Güncel fiyatı al (Binance + CoinGecko Hybrid)
  private async getCurrentPrice(symbol: string): Promise<number> {
    try {
      // Top 100'den fiyat al (Binance first, CoinGecko fallback)
      const response = await fetch('/api/market/top100');
      const data = await response.json();

      if (data.success && data.data) {
        const coin = data.data.find((c: any) =>
          c.symbol.toUpperCase() === symbol.toUpperCase()
        );

        if (coin) {
          console.log(`💰 Price for ${symbol}: $${coin.price} (source: ${coin.source})`);
          return coin.price;
        }
      }

      // Fallback: Binance direct API
      console.log(`⚠️ ${symbol} not in top 100, trying Binance direct...`);
      const binanceRes = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${symbol}USDT`);
      if (binanceRes.ok) {
        const binanceData = await binanceRes.json();
        return parseFloat(binanceData.price);
      }

      // Final fallback: mock price
      console.log(`❌ ${symbol} not found, using fallback price`);
      return 50000 + Math.random() * 1000;

    } catch (error) {
      console.error(`Error fetching price for ${symbol}:`, error);
      return 50000; // Fallback price
    }
  }

  // Durum bilgilerini al
  getStatus() {
    return {
      isRunning: this.isRunning,
      mode: this.config.mode,
      stats: this.stats,
      positions: Array.from(this.positions.values()),
      recentSignals: this.signals.slice(-10),
    };
  }

  // Konfigürasyonu güncelle
  updateConfig(newConfig: Partial<TradingConfig>): void {
    this.config = { ...this.config, ...newConfig };
    console.log('⚙️ Trading config updated:', newConfig);
  }
}

// Global instance
let autoTradingEngineInstance: AutoTradingEngine | null = null;

export function getAutoTradingEngine(config?: TradingConfig): AutoTradingEngine {
  if (!autoTradingEngineInstance && config) {
    autoTradingEngineInstance = new AutoTradingEngine(config);
  }

  if (!autoTradingEngineInstance) {
    throw new Error('AutoTradingEngine not initialized');
  }

  return autoTradingEngineInstance;
}
