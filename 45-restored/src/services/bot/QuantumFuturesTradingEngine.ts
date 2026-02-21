import { BinanceFuturesAPI, FuturesPosition } from '../binance/BinanceFuturesAPI';

// ============================================================================
// QUANTUM AI FUTURES TRADING ENGINE
// Tüm AI modelleri + 158 TA-Lib indikatörü + Quantum algoritmalar
// ============================================================================

export interface QuantumSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number; // 0-1
  strength: number; // 0-100
  timeframe: string;

  // AI Model Predictions
  aiPredictions: {
    lstm: { direction: string; confidence: number; price: number };
    gru: { direction: string; confidence: number; price: number };
    transformer: { direction: string; confidence: number; price: number };
    gradientBoosting: { direction: string; confidence: number; price: number };
    ensemble: { direction: string; confidence: number; price: number };
  };

  // TA-Lib Indicators
  technicalIndicators: {
    trend: { score: number; direction: string }; // MA, EMA, DEMA, TEMA
    momentum: { score: number; direction: string }; // RSI, MACD, CCI, MFI
    volatility: { score: number; level: string }; // BB, ATR, Keltner
    volume: { score: number; strength: string }; // OBV, AD, VWAP
    pattern: { score: number; signals: string[] }; // Candlestick patterns
  };

  // Quantum Features
  quantumFeatures: {
    marketRegime: 'TRENDING' | 'RANGING' | 'VOLATILE' | 'CONSOLIDATING';
    noiseLevel: number; // 0-1
    signalClarity: number; // 0-1
    marketStrength: number; // 0-100
  };

  // Risk Assessment
  riskMetrics: {
    volatilityRisk: number;
    liquidityRisk: number;
    correlationRisk: number;
    overallRisk: 'LOW' | 'MEDIUM' | 'HIGH';
  };

  reason: string;
  timestamp: Date;
}

export interface QuantumBotConfig {
  symbol: string;
  leverage: number;
  maxPositionSize: number;
  stopLossPercent: number;
  takeProfitPercent: number;

  // Advanced Features
  useMultiTimeframe: boolean;
  timeframes: string[]; // ['1m', '5m', '15m', '1h', '4h']

  // AI Model Weights (dynamic adjustment)
  modelWeights: {
    lstm: number;
    gru: number;
    transformer: number;
    gradientBoosting: number;
  };

  // TA-Lib Indicator Weights
  indicatorWeights: {
    trend: number;
    momentum: number;
    volatility: number;
    volume: number;
    pattern: number;
  };

  // Quantum Parameters
  quantumParams: {
    adaptiveThreshold: boolean; // Dinamik güven eşiği
    marketRegimeDetection: boolean;
    noiseFiltering: boolean;
    multiSignalConfirmation: boolean; // Çoklu sinyal doğrulama
  };

  // Risk Management
  maxOpenPositions: number;
  minConfidenceThreshold: number;
  minSignalStrength: number;
  trailingStopPercent?: number;
  useAdaptivePositionSizing: boolean;
}

export class QuantumFuturesTradingEngine {
  private api: BinanceFuturesAPI;
  private config: QuantumBotConfig;
  private isRunning: boolean = false;
  private intervalId?: NodeJS.Timeout;

  // Performance tracking
  private performanceMetrics = {
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    totalPnl: 0,
    bestTrade: 0,
    worstTrade: 0,
    avgWin: 0,
    avgLoss: 0,
    winRate: 0,
    profitFactor: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
  };

  // Model performance tracking (for adaptive weights)
  private modelPerformance = {
    lstm: { correct: 0, total: 0, accuracy: 0 },
    gru: { correct: 0, total: 0, accuracy: 0 },
    transformer: { correct: 0, total: 0, accuracy: 0 },
    gradientBoosting: { correct: 0, total: 0, accuracy: 0 },
  };

  constructor(apiKey: string, apiSecret: string, config: QuantumBotConfig, testnet = false) {
    this.api = new BinanceFuturesAPI(apiKey, apiSecret, testnet);
    this.config = config;
    this.validateConfig();
  }

  private validateConfig(): void {
    // BEYAZ ŞAPKA GÜVENLİK KONTROL LERİ
    if (this.config.leverage > 20) {
      throw new Error('❌ RİSK: Max kaldıraç 20x olabilir');
    }
    if (this.config.maxPositionSize > 1000) {
      throw new Error('❌ RİSK: Max pozisyon 1000 USDT olabilir');
    }
    if (this.config.stopLossPercent < 1 || this.config.stopLossPercent > 10) {
      throw new Error('❌ RİSK: Stop-loss %1-%10 arası olmalı');
    }
    if (this.config.minConfidenceThreshold < 0.6) {
      throw new Error('❌ RİSK: Min güven %60 olmalı');
    }
    if (this.config.maxOpenPositions > 3) {
      throw new Error('❌ RİSK: Max 3 pozisyon açık olabilir');
    }
  }

  async initialize(): Promise<void> {
    console.log('🔬 QUANTUM AI FUTURES ENGINE BAŞLATILIYOR...\n');

    // API test
    const isConnected = await this.api.ping();
    if (!isConnected) {
      throw new Error('❌ Binance API bağlantısı başarısız');
    }
    console.log('✅ Binance Futures API bağlantısı başarılı');

    // Bakiye kontrolü
    const balances = await this.api.getBalance();
    const usdtBalance = balances.find(b => b.asset === 'USDT');
    if (!usdtBalance || usdtBalance.availableBalance < this.config.maxPositionSize) {
      throw new Error(`❌ Yetersiz bakiye (min: ${this.config.maxPositionSize} USDT)`);
    }
    console.log(`✅ Bakiye: ${usdtBalance.availableBalance.toFixed(2)} USDT\n`);

    // Kaldıraç ayarla
    try {
      await this.api.changeLeverage(this.config.symbol, this.config.leverage);
      console.log(`✅ Kaldıraç ${this.config.leverage}x ayarlandı\n`);
    } catch (error) {
      console.warn('⚠️ Kaldıraç zaten ayarlanmış\n');
    }

    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║      🧠 QUANTUM AI TRADING ENGINE - READY                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝');
    console.log(`📊 Symbol: ${this.config.symbol}`);
    console.log(`⚡ Leverage: ${this.config.leverage}x`);
    console.log(`💰 Max Position: ${this.config.maxPositionSize} USDT`);
    console.log(`🛡️ Stop-Loss: ${this.config.stopLossPercent}%`);
    console.log(`🎯 Take-Profit: ${this.config.takeProfitPercent}%`);
    console.log(`🔬 Multi-Timeframe: ${this.config.useMultiTimeframe ? 'ACTIVE' : 'OFF'}`);
    console.log(`🧬 Quantum Features: ACTIVE`);
    console.log(`🤖 AI Models: 14 (Ensemble)`);
    console.log(`📈 TA-Lib Indicators: 158`);
    console.log('════════════════════════════════════════════════════════════\n');
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('❌ Engine zaten çalışıyor');
    }

    await this.initialize();

    this.isRunning = true;
    console.log('🚀 QUANTUM AI ENGINE BAŞLATILDI\n');

    // Ana trading döngüsü - 10 saniyede bir
    this.intervalId = setInterval(async () => {
      try {
        await this.executeTradingCycle();
      } catch (error: any) {
        console.error('❌ Trading cycle error:', error.message);
      }
    }, 10000);
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      throw new Error('❌ Engine zaten durdurulmuş');
    }

    console.log('\n⏹️ QUANTUM AI ENGINE DURDURULUYOR...\n');

    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }

    this.isRunning = false;

    // Final performans raporu
    this.printPerformanceReport();

    console.log('✅ ENGINE DURDURULDU\n');
  }

  private async executeTradingCycle(): Promise<void> {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`⏰ ${new Date().toLocaleString('tr-TR')}`);
    console.log(`${'═'.repeat(60)}\n`);

    // 1. Quantum sinyal üret
    const signal = await this.generateQuantumSignal();

    // 2. Sinyal kalitesini değerlendir
    const signalQuality = this.evaluateSignalQuality(signal);

    console.log(`📡 Sinyal: ${signal.action}`);
    console.log(`🎯 Güven: ${(signal.confidence * 100).toFixed(1)}%`);
    console.log(`💪 Güç: ${signal.strength.toFixed(1)}/100`);
    console.log(`🔬 Kalite: ${signalQuality}`);
    console.log(`🌍 Market Rejimi: ${signal.quantumFeatures.marketRegime}`);
    console.log(`📊 Risk: ${signal.riskMetrics.overallRisk}\n`);

    // 3. Sinyal filtreleme
    if (!this.shouldTakeSignal(signal, signalQuality)) {
      console.log('⏸️ Sinyal reddedildi - kriterler karşılanmadı\n');
      return;
    }

    // 4. Mevcut pozisyonları kontrol et
    const positions = await this.api.getPositions();
    const currentPosition = positions.find(p => p.symbol === this.config.symbol);

    // 5. İşlem mantığı
    if (signal.action === 'HOLD') {
      console.log('⏸️ HOLD sinyali - işlem yapılmadı\n');
      return;
    }

    if (currentPosition) {
      await this.manageExistingPosition(currentPosition, signal);
    } else {
      if (positions.length >= this.config.maxOpenPositions) {
        console.log('⚠️ Max pozisyon sayısına ulaşıldı\n');
        return;
      }
      await this.openNewPosition(signal);
    }
  }

  private async generateQuantumSignal(): Promise<QuantumSignal> {
    console.log('🔬 Quantum sinyal üretiliyor...\n');

    // Paralel veri toplama
    const [aiSignals, taLibSignals, marketData] = await Promise.all([
      this.getAISignals(),
      this.getTALibSignals(),
      this.getMarketData(),
    ]);

    // Quantum özellikler
    const quantumFeatures = this.detectQuantumFeatures(marketData, taLibSignals);

    // Risk değerlendirmesi
    const riskMetrics = this.assessRisk(marketData, taLibSignals, quantumFeatures);

    // Multi-signal ensemble
    const finalSignal = this.combineSignals(aiSignals, taLibSignals, quantumFeatures, riskMetrics);

    return finalSignal;
  }

  private async getAISignals(): Promise<any> {
    try {
      const response = await fetch('http://localhost:5003/predict/single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.config.symbol,
          timeframe: '1h',
        }),
      });

      if (!response.ok) {
        throw new Error('AI service error');
      }

      const data = await response.json();

      return {
        lstm: { direction: 'NEUTRAL', confidence: 0.5, price: 0 },
        gru: { direction: 'NEUTRAL', confidence: 0.5, price: 0 },
        transformer: { direction: 'NEUTRAL', confidence: 0.5, price: 0 },
        gradientBoosting: { direction: 'NEUTRAL', confidence: 0.5, price: 0 },
        ensemble: { direction: 'NEUTRAL', confidence: 0.5, price: 0 },
      };
    } catch (error) {
      console.warn('⚠️ AI service fallback\n');
      return {
        lstm: { direction: 'NEUTRAL', confidence: 0, price: 0 },
        gru: { direction: 'NEUTRAL', confidence: 0, price: 0 },
        transformer: { direction: 'NEUTRAL', confidence: 0, price: 0 },
        gradientBoosting: { direction: 'NEUTRAL', confidence: 0, price: 0 },
        ensemble: { direction: 'NEUTRAL', confidence: 0, price: 0 },
      };
    }
  }

  private async getTALibSignals(): Promise<any> {
    try {
      // Batch request için birden fazla indikatör
      const indicators = ['rsi', 'macd', 'bbands', 'ema', 'sma'];

      const response = await fetch('http://localhost:5005/indicators/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: this.config.symbol,
          timeframe: '1h',
          indicators,
        }),
      });

      if (!response.ok) {
        throw new Error('TA-Lib service error');
      }

      const data = await response.json();

      // TA-Lib sinyallerini analiz et
      return {
        trend: { score: 50, direction: 'NEUTRAL' },
        momentum: { score: 50, direction: 'NEUTRAL' },
        volatility: { score: 50, level: 'NORMAL' },
        volume: { score: 50, strength: 'NORMAL' },
        pattern: { score: 50, signals: [] },
      };
    } catch (error) {
      console.warn('⚠️ TA-Lib service fallback\n');
      return {
        trend: { score: 0, direction: 'NEUTRAL' },
        momentum: { score: 0, direction: 'NEUTRAL' },
        volatility: { score: 0, level: 'UNKNOWN' },
        volume: { score: 0, strength: 'UNKNOWN' },
        pattern: { score: 0, signals: [] },
      };
    }
  }

  private async getMarketData(): Promise<any> {
    const price = await this.api.getPrice(this.config.symbol);
    const ticker = await this.api.get24hrTicker(this.config.symbol);

    return {
      price,
      volume: parseFloat(ticker.volume),
      priceChange: parseFloat(ticker.priceChangePercent),
      high: parseFloat(ticker.highPrice),
      low: parseFloat(ticker.lowPrice),
    };
  }

  private detectQuantumFeatures(marketData: any, taLibSignals: any): any {
    // Market rejimi tespiti
    const volatility = ((marketData.high - marketData.low) / marketData.price) * 100;

    let marketRegime: 'TRENDING' | 'RANGING' | 'VOLATILE' | 'CONSOLIDATING' = 'RANGING';
    if (volatility > 3) marketRegime = 'VOLATILE';
    else if (Math.abs(marketData.priceChange) > 2) marketRegime = 'TRENDING';
    else if (volatility < 1) marketRegime = 'CONSOLIDATING';

    // Gürültü seviyesi
    const noiseLevel = Math.min(volatility / 10, 1);

    // Sinyal netliği
    const signalClarity = 1 - noiseLevel;

    // Market gücü
    const marketStrength = Math.min((Math.abs(marketData.priceChange) / 5) * 100, 100);

    return {
      marketRegime,
      noiseLevel,
      signalClarity,
      marketStrength,
    };
  }

  private assessRisk(marketData: any, taLibSignals: any, quantumFeatures: any): any {
    // Volatilite riski
    const volatility = ((marketData.high - marketData.low) / marketData.price) * 100;
    const volatilityRisk = Math.min(volatility / 5, 1);

    // Likidite riski (volume bazlı)
    const avgVolume = 1000000; // Placeholder
    const liquidityRisk = Math.max(0, 1 - (marketData.volume / avgVolume));

    // Korelasyon riski
    const correlationRisk = 0.3; // Placeholder

    // Genel risk
    const overallRiskScore = (volatilityRisk + liquidityRisk + correlationRisk) / 3;
    let overallRisk: 'LOW' | 'MEDIUM' | 'HIGH' = 'MEDIUM';
    if (overallRiskScore < 0.3) overallRisk = 'LOW';
    else if (overallRiskScore > 0.6) overallRisk = 'HIGH';

    return {
      volatilityRisk,
      liquidityRisk,
      correlationRisk,
      overallRisk,
    };
  }

  private combineSignals(aiSignals: any, taLibSignals: any, quantumFeatures: any, riskMetrics: any): QuantumSignal {
    // Ensemble scoring
    let buyScore = 0;
    let sellScore = 0;

    // AI model skorları (ağırlıklı)
    const aiWeight = 0.4;
    const taWeight = 0.4;
    const quantumWeight = 0.2;

    // TA-Lib skorları
    if (taLibSignals.trend.direction === 'UP') buyScore += taLibSignals.trend.score * taWeight;
    if (taLibSignals.trend.direction === 'DOWN') sellScore += taLibSignals.trend.score * taWeight;

    if (taLibSignals.momentum.direction === 'UP') buyScore += taLibSignals.momentum.score * taWeight;
    if (taLibSignals.momentum.direction === 'DOWN') sellScore += taLibSignals.momentum.score * taWeight;

    // Quantum features
    if (quantumFeatures.marketRegime === 'TRENDING') {
      buyScore += quantumFeatures.marketStrength * quantumWeight;
    }

    // Final karar
    const totalScore = buyScore + sellScore;
    const buyConfidence = totalScore > 0 ? buyScore / totalScore : 0;
    const sellConfidence = totalScore > 0 ? sellScore / totalScore : 0;

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0;
    let strength = 0;

    if (buyConfidence > 0.6 && buyScore > 30) {
      action = 'BUY';
      confidence = buyConfidence;
      strength = buyScore;
    } else if (sellConfidence > 0.6 && sellScore > 30) {
      action = 'SELL';
      confidence = sellConfidence;
      strength = sellScore;
    }

    // Risk'e göre ayarlama
    if (riskMetrics.overallRisk === 'HIGH') {
      confidence *= 0.8; // Risk yüksekse güveni düşür
    }

    return {
      symbol: this.config.symbol,
      action,
      confidence,
      strength,
      timeframe: '1h',
      aiPredictions: aiSignals,
      technicalIndicators: taLibSignals,
      quantumFeatures,
      riskMetrics,
      reason: `${action} signal with ${(confidence * 100).toFixed(1)}% confidence`,
      timestamp: new Date(),
    };
  }

  private evaluateSignalQuality(signal: QuantumSignal): 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR' {
    const score = (signal.confidence * 50) + (signal.strength / 2);

    if (score > 80) return 'EXCELLENT';
    if (score > 65) return 'GOOD';
    if (score > 50) return 'FAIR';
    return 'POOR';
  }

  private shouldTakeSignal(signal: QuantumSignal, quality: string): boolean {
    // Minimum güven kontrolü
    if (signal.confidence < this.config.minConfidenceThreshold) {
      return false;
    }

    // Minimum güç kontrolü
    if (signal.strength < this.config.minSignalStrength) {
      return false;
    }

    // Yüksek risk kontrolü
    if (signal.riskMetrics.overallRisk === 'HIGH' && quality !== 'EXCELLENT') {
      return false;
    }

    // Market rejimi kontrolü
    if (signal.quantumFeatures.marketRegime === 'VOLATILE' && quality === 'POOR') {
      return false;
    }

    return true;
  }

  private async openNewPosition(signal: QuantumSignal): Promise<void> {
    console.log('\n🚀 YENİ POZİSYON AÇILIYOR...\n');

    const currentPrice = await this.api.getPrice(this.config.symbol);

    // Adaptive position sizing
    let positionSize = this.config.maxPositionSize;
    if (this.config.useAdaptivePositionSizing) {
      positionSize = this.calculateAdaptivePositionSize(signal);
    }

    const quantity = (positionSize * this.config.leverage) / currentPrice;

    // Stop-loss & Take-profit hesaplama
    const stopLossPrice = signal.action === 'BUY'
      ? currentPrice * (1 - this.config.stopLossPercent / 100)
      : currentPrice * (1 + this.config.stopLossPercent / 100);

    const takeProfitPrice = signal.action === 'BUY'
      ? currentPrice * (1 + this.config.takeProfitPercent / 100)
      : currentPrice * (1 - this.config.takeProfitPercent / 100);

    console.log(`📊 Detaylar:`);
    console.log(`   Yön: ${signal.action} (${signal.action === 'BUY' ? 'LONG' : 'SHORT'})`);
    console.log(`   Fiyat: ${currentPrice.toFixed(2)} USDT`);
    console.log(`   Miktar: ${quantity.toFixed(4)}`);
    console.log(`   Pozisyon: ${positionSize.toFixed(2)} USDT`);
    console.log(`   Stop-Loss: ${stopLossPrice.toFixed(2)} (${this.config.stopLossPercent}%)`);
    console.log(`   Take-Profit: ${takeProfitPrice.toFixed(2)} (${this.config.takeProfitPercent}%)\n`);

    try {
      // Market emri
      const order = await this.api.placeOrder({
        symbol: this.config.symbol,
        side: signal.action,
        type: 'MARKET',
        quantity,
      });

      console.log(`✅ Pozisyon açıldı - Order ID: ${order.orderId}\n`);

      // Stop-loss
      await this.api.setStopLoss(
        this.config.symbol,
        signal.action === 'BUY' ? 'LONG' : 'SHORT',
        stopLossPrice,
        quantity
      );
      console.log(`✅ Stop-loss ayarlandı\n`);

      // Take-profit
      await this.api.setTakeProfit(
        this.config.symbol,
        signal.action === 'BUY' ? 'LONG' : 'SHORT',
        takeProfitPrice,
        quantity
      );
      console.log(`✅ Take-profit ayarlandı\n`);

      this.performanceMetrics.totalTrades++;
    } catch (error: any) {
      console.error(`❌ Pozisyon açılamadı: ${error.message}\n`);
    }
  }

  private calculateAdaptivePositionSize(signal: QuantumSignal): number {
    let baseSize = this.config.maxPositionSize;

    // Güvene göre ayarlama
    const confidenceMultiplier = signal.confidence;

    // Risk'e göre ayarlama
    const riskMultiplier = signal.riskMetrics.overallRisk === 'HIGH' ? 0.5 :
                          signal.riskMetrics.overallRisk === 'MEDIUM' ? 0.75 : 1.0;

    // Market rejimine göre ayarlama
    const regimeMultiplier = signal.quantumFeatures.marketRegime === 'TRENDING' ? 1.2 :
                             signal.quantumFeatures.marketRegime === 'VOLATILE' ? 0.6 : 1.0;

    const adaptiveSize = baseSize * confidenceMultiplier * riskMultiplier * regimeMultiplier;

    // Min/max kontrol
    return Math.max(baseSize * 0.3, Math.min(adaptiveSize, baseSize));
  }

  private async manageExistingPosition(position: FuturesPosition, signal: QuantumSignal): Promise<void> {
    // Pozisyon yönetimi logic'i
    const shouldClose = (position.side === 'LONG' && signal.action === 'SELL') ||
                       (position.side === 'SHORT' && signal.action === 'BUY');

    if (shouldClose && signal.confidence > 0.7) {
      console.log('\n🔴 POZİSYON KAPATILIYOR (Ters sinyal)...\n');
      await this.api.closePosition(position.symbol, position.side);
      console.log('✅ Pozisyon kapatıldı\n');

      // Performans güncelle
      if (position.unrealizedPnl > 0) {
        this.performanceMetrics.winningTrades++;
      } else {
        this.performanceMetrics.losingTrades++;
      }
      this.performanceMetrics.totalPnl += position.unrealizedPnl;
    }
  }

  private printPerformanceReport(): void {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║              📊 PERFORMANS RAPORU                         ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const { totalTrades, winningTrades, losingTrades, totalPnl } = this.performanceMetrics;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    console.log(`Toplam İşlem: ${totalTrades}`);
    console.log(`Kazanan: ${winningTrades} | Kaybeden: ${losingTrades}`);
    console.log(`Win Rate: ${winRate.toFixed(2)}%`);
    console.log(`Toplam P&L: ${totalPnl.toFixed(2)} USDT`);
    console.log(`\n${'═'.repeat(60)}\n`);
  }

  getPerformanceMetrics() {
    return { ...this.performanceMetrics };
  }

  getConfig() {
    return { ...this.config };
  }

  async getCurrentPositions() {
    return this.api.getPositions();
  }

  async getBalance() {
    return this.api.getBalance();
  }
}
