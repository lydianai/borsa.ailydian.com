import { BinanceFuturesAPI, FuturesOrder, FuturesPosition } from '../binance/BinanceFuturesAPI';

export interface BotConfig {
  symbol: string;
  leverage: number;
  maxPositionSize: number; // USDT cinsinden
  stopLossPercent: number; // Yüzde olarak
  takeProfitPercent: number; // Yüzde olarak
  confidenceThreshold: number; // 0-1 arası (0.7 = %70 güven)
  maxOpenPositions: number;
  trailingStopPercent?: number; // İsteğe bağlı trailing stop
}

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'CLOSE' | 'HOLD';
  confidence: number; // 0-1 arası
  predictedPrice?: number;
  reason?: string;
}

export interface BotStatus {
  isRunning: boolean;
  activePositions: number;
  totalPnl: number;
  totalTrades: number;
  winRate: number;
  lastSignal?: TradingSignal;
  lastAction?: string;
  lastActionTime?: Date;
}

export class FuturesTradingBot {
  private api: BinanceFuturesAPI;
  private config: BotConfig;
  private isRunning: boolean = false;
  private intervalId?: NodeJS.Timeout;
  private status: BotStatus;

  constructor(apiKey: string, apiSecret: string, config: BotConfig, testnet: boolean = false) {
    this.api = new BinanceFuturesAPI(apiKey, apiSecret, testnet);
    this.config = config;
    this.status = {
      isRunning: false,
      activePositions: 0,
      totalPnl: 0,
      totalTrades: 0,
      winRate: 0,
    };

    this.validateConfig();
  }

  private validateConfig(): void {
    // RİSK YÖNETİMİ GÜVENLİK KONTROLLARI
    if (this.config.leverage > 20) {
      throw new Error('❌ RİSK: Kaldıraç 20x üzerinde olamaz (tavsiye edilen max 10x)');
    }

    if (this.config.maxPositionSize > 1000) {
      throw new Error('❌ RİSK: Pozisyon başına max 1000 USDT olmalı');
    }

    if (this.config.stopLossPercent < 1 || this.config.stopLossPercent > 10) {
      throw new Error('❌ RİSK: Stop-loss %1-%10 arasında olmalı');
    }

    if (this.config.takeProfitPercent < 1 || this.config.takeProfitPercent > 20) {
      throw new Error('❌ RİSK: Take-profit %1-%20 arasında olmalı');
    }

    if (this.config.confidenceThreshold < 0.6) {
      throw new Error('❌ RİSK: Güven eşiği minimum %60 olmalı');
    }

    if (this.config.maxOpenPositions > 3) {
      throw new Error('❌ RİSK: Aynı anda max 3 pozisyon açık olabilir');
    }
  }

  async initialize(): Promise<void> {
    console.log('🔧 Bot başlatılıyor...');

    // API bağlantısını test et
    const isConnected = await this.api.ping();
    if (!isConnected) {
      throw new Error('❌ Binance API bağlantısı başarısız');
    }
    console.log('✅ Binance API bağlantısı başarılı');

    // Bakiyeyi kontrol et
    const balances = await this.api.getBalance();
    const usdtBalance = balances.find(b => b.asset === 'USDT');
    if (!usdtBalance || usdtBalance.availableBalance < this.config.maxPositionSize) {
      throw new Error(`❌ Yetersiz bakiye. Minimum ${this.config.maxPositionSize} USDT gerekli.`);
    }
    console.log(`✅ Bakiye: ${usdtBalance.availableBalance.toFixed(2)} USDT`);

    // Kaldıraç ayarla
    try {
      await this.api.changeLeverage(this.config.symbol, this.config.leverage);
      console.log(`✅ Kaldıraç ${this.config.leverage}x olarak ayarlandı`);
    } catch (error) {
      console.warn('⚠️ Kaldıraç ayarlanamadı (zaten ayarlanmış olabilir)');
    }

    console.log('✅ Bot hazır!');
  }

  async start(signalGenerator: () => Promise<TradingSignal>): Promise<void> {
    if (this.isRunning) {
      throw new Error('❌ Bot zaten çalışıyor');
    }

    await this.initialize();

    this.isRunning = true;
    this.status.isRunning = true;

    console.log('🤖 BOT BAŞLATILDI - Otomatik trading aktif');
    console.log(`📊 Sembol: ${this.config.symbol}`);
    console.log(`⚡ Kaldıraç: ${this.config.leverage}x`);
    console.log(`💰 Max Pozisyon: ${this.config.maxPositionSize} USDT`);
    console.log(`🛡️ Stop-Loss: ${this.config.stopLossPercent}%`);
    console.log(`🎯 Take-Profit: ${this.config.takeProfitPercent}%`);

    // Ana bot döngüsü - her 10 saniyede bir çalışır
    this.intervalId = setInterval(async () => {
      try {
        await this.executeStrategy(signalGenerator);
      } catch (error: any) {
        console.error('❌ Bot hatası:', error.message);
      }
    }, 10000); // 10 saniye
  }

  async stop(): Promise<void> {
    if (!this.isRunning) {
      throw new Error('❌ Bot zaten durdurulmuş');
    }

    console.log('⏹️ Bot durduruluyor...');

    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }

    this.isRunning = false;
    this.status.isRunning = false;

    console.log('✅ Bot durduruldu');
  }

  private async executeStrategy(signalGenerator: () => Promise<TradingSignal>): Promise<void> {
    // 1. AI'dan sinyal al
    const signal = await signalGenerator();
    this.status.lastSignal = signal;

    console.log(`\n📡 Sinyal: ${signal.action} (Güven: ${(signal.confidence * 100).toFixed(1)}%)`);

    // 2. Güven eşiğini kontrol et
    if (signal.confidence < this.config.confidenceThreshold) {
      console.log(`⚠️ Düşük güven - işlem yapılmadı (min: ${(this.config.confidenceThreshold * 100).toFixed(0)}%)`);
      return;
    }

    // 3. Mevcut pozisyonları kontrol et
    const positions = await this.api.getPositions();
    const currentPosition = positions.find(p => p.symbol === this.config.symbol);
    this.status.activePositions = positions.length;

    // 4. Sinyal tipine göre işlem yap
    if (signal.action === 'HOLD') {
      console.log('⏸️ Bekle sinyali - işlem yapılmadı');
      return;
    }

    if (signal.action === 'CLOSE' && currentPosition) {
      await this.closePosition(currentPosition);
      return;
    }

    if (signal.action === 'BUY' || signal.action === 'SELL') {
      // Aynı yönde pozisyon varsa işlem yapma
      if (currentPosition) {
        const sameDirection =
          (signal.action === 'BUY' && currentPosition.side === 'LONG') ||
          (signal.action === 'SELL' && currentPosition.side === 'SHORT');

        if (sameDirection) {
          console.log('⚠️ Aynı yönde pozisyon zaten açık - yeni işlem yapılmadı');
          return;
        } else {
          // Ters yönde pozisyon varsa önce kapat
          console.log('🔄 Ters yönde pozisyon var - kapatılıyor...');
          await this.closePosition(currentPosition);
        }
      }

      // Max pozisyon sayısını kontrol et
      if (positions.length >= this.config.maxOpenPositions) {
        console.log('⚠️ Max pozisyon sayısına ulaşıldı - yeni işlem yapılmadı');
        return;
      }

      // Yeni pozisyon aç
      await this.openPosition(signal);
    }
  }

  private async openPosition(signal: TradingSignal): Promise<void> {
    const currentPrice = await this.api.getPrice(this.config.symbol);

    // Pozisyon büyüklüğünü hesapla
    const quantity = (this.config.maxPositionSize * this.config.leverage) / currentPrice;

    // Stop-loss ve take-profit fiyatlarını hesapla
    const stopLossPrice =
      signal.action === 'BUY'
        ? currentPrice * (1 - this.config.stopLossPercent / 100)
        : currentPrice * (1 + this.config.stopLossPercent / 100);

    const takeProfitPrice =
      signal.action === 'BUY'
        ? currentPrice * (1 + this.config.takeProfitPercent / 100)
        : currentPrice * (1 - this.config.takeProfitPercent / 100);

    console.log(`\n🚀 YENİ POZİSYON AÇILIYOR`);
    console.log(`Yön: ${signal.action} ${signal.action === 'BUY' ? 'LONG' : 'SHORT'}`);
    console.log(`Fiyat: ${currentPrice.toFixed(2)} USDT`);
    console.log(`Miktar: ${quantity.toFixed(3)}`);
    console.log(`Stop-Loss: ${stopLossPrice.toFixed(2)} USDT (${this.config.stopLossPercent}%)`);
    console.log(`Take-Profit: ${takeProfitPrice.toFixed(2)} USDT (${this.config.takeProfitPercent}%)`);

    try {
      // Market emri ver
      const order = await this.api.placeOrder({
        symbol: this.config.symbol,
        side: signal.action,
        type: 'MARKET',
        quantity: quantity,
      });

      console.log(`✅ Pozisyon açıldı - Order ID: ${order.orderId}`);

      // Stop-loss ayarla
      await this.api.setStopLoss(
        this.config.symbol,
        signal.action === 'BUY' ? 'LONG' : 'SHORT',
        stopLossPrice,
        quantity
      );
      console.log(`✅ Stop-loss ayarlandı: ${stopLossPrice.toFixed(2)} USDT`);

      // Take-profit ayarla
      await this.api.setTakeProfit(
        this.config.symbol,
        signal.action === 'BUY' ? 'LONG' : 'SHORT',
        takeProfitPrice,
        quantity
      );
      console.log(`✅ Take-profit ayarlandı: ${takeProfitPrice.toFixed(2)} USDT`);

      this.status.totalTrades++;
      this.status.lastAction = `Opened ${signal.action} position`;
      this.status.lastActionTime = new Date();
    } catch (error: any) {
      console.error(`❌ Pozisyon açılamadı: ${error.message}`);
      throw error;
    }
  }

  private async closePosition(position: FuturesPosition): Promise<void> {
    console.log(`\n🔴 POZİSYON KAPATILIYOR`);
    console.log(`Sembol: ${position.symbol}`);
    console.log(`Yön: ${position.side}`);
    console.log(`P&L: ${position.unrealizedPnl.toFixed(2)} USDT (${position.unrealizedPnlPercent.toFixed(2)}%)`);

    try {
      await this.api.closePosition(position.symbol, position.side);
      console.log(`✅ Pozisyon kapatıldı`);

      // P&L'yi güncelle
      this.status.totalPnl += position.unrealizedPnl;
      this.status.lastAction = `Closed ${position.side} position`;
      this.status.lastActionTime = new Date();

      // Win rate'i hesapla
      if (position.unrealizedPnl > 0) {
        this.status.winRate =
          (this.status.winRate * (this.status.totalTrades - 1) + 1) / this.status.totalTrades;
      } else {
        this.status.winRate =
          (this.status.winRate * (this.status.totalTrades - 1)) / this.status.totalTrades;
      }
    } catch (error: any) {
      console.error(`❌ Pozisyon kapatılamadı: ${error.message}`);
      throw error;
    }
  }

  async closeAllPositions(): Promise<void> {
    console.log('🔴 TÜM POZİSYONLAR KAPATILIYOR...');

    const positions = await this.api.getPositions();

    for (const position of positions) {
      await this.closePosition(position);
    }

    console.log('✅ Tüm pozisyonlar kapatıldı');
  }

  getStatus(): BotStatus {
    return { ...this.status };
  }

  getConfig(): BotConfig {
    return { ...this.config };
  }

  async getCurrentPositions(): Promise<FuturesPosition[]> {
    return this.api.getPositions();
  }

  async getBalance(): Promise<any> {
    return this.api.getBalance();
  }
}
