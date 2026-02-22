/**
 * 🧠 AI MEMORY STORE - LyTrade LYDIAN ÖĞRENME SİSTEMİ
 *
 * Redis'te her coin ve strateji için performans geçmişi saklar.
 * Otonom AI Agent'ın geçmişten öğrenmesini sağlar.
 *
 * Saklanan Veriler:
 * - Coin-strateji başarı oranları
 * - Signal sonuçları (başarılı/başarısız)
 * - Ortalama kar/zarar
 * - En iyi performans gösteren zaman dilimleri
 * - Adaptif ağırlık önerileri
 */

import { getRedisClient } from '../queue/redis-client';

// Memory key formatları
const MEMORY_KEYS = {
  // Strategy performance per coin: "memory:strategy:{strategyName}:{symbol}"
  strategyPerformance: (strategy: string, symbol: string) =>
    `memory:strategy:${strategy}:${symbol}`,

  // Overall coin performance: "memory:coin:{symbol}"
  coinPerformance: (symbol: string) => `memory:coin:${symbol}`,

  // Global strategy stats: "memory:global:strategy:{strategyName}"
  globalStrategyStats: (strategy: string) => `memory:global:strategy:${strategy}`,

  // Adaptive weights: "memory:weights:{symbol}"
  adaptiveWeights: (symbol: string) => `memory:weights:${symbol}`,
};

/**
 * Strateji performans verisi
 */
export interface StrategyPerformance {
  strategyName: string;
  symbol: string;
  totalSignals: number;
  successfulSignals: number;
  failedSignals: number;
  pendingSignals: number;
  successRate: number; // 0-100
  avgConfidence: number; // Ortalama güven skoru
  avgProfit: number; // Ortalama kar (%)
  bestTimeframe: string; // En iyi performans gösterdiği timeframe
  lastUpdated: string; // ISO timestamp
  recentSignals: SignalResult[]; // Son 50 sinyal
}

/**
 * Sinyal sonucu
 */
export interface SignalResult {
  timestamp: string;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  price: number;
  outcome?: 'success' | 'failure' | 'pending';
  profitPercent?: number;
  timeframe: string;
  exitPrice?: number;
  exitTimestamp?: string;
}

/**
 * Coin performansı (tüm stratejiler toplamı)
 */
export interface CoinPerformance {
  symbol: string;
  totalSignals: number;
  overallSuccessRate: number;
  bestStrategy: string; // En başarılı strateji
  worstStrategy: string; // En kötü strateji
  volatility: number;
  avgVolume24h: number;
  lastAnalyzed: string;
}

/**
 * Adaptif ağırlıklar
 */
export interface AdaptiveWeights {
  symbol: string;
  weights: {
    [strategyName: string]: number; // 0.5 - 2.0 arası
  };
  reasoning: {
    [strategyName: string]: string;
  };
  lastCalculated: string;
  validUntil: string; // 4 saat sonra yeniden hesaplanmalı
}

/**
 * Strateji performansını kaydet veya güncelle
 */
export async function recordStrategyPerformance(data: {
  strategyName: string;
  symbol: string;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  price: number;
  timeframe: string;
}) {
  const redis = getRedisClient();
  const key = MEMORY_KEYS.strategyPerformance(data.strategyName, data.symbol);

  try {
    // Mevcut performansı al
    const existingData = await redis.get(key);
    let performance: StrategyPerformance;

    if (existingData) {
      performance = JSON.parse(existingData);
    } else {
      // İlk kayıt
      performance = {
        strategyName: data.strategyName,
        symbol: data.symbol,
        totalSignals: 0,
        successfulSignals: 0,
        failedSignals: 0,
        pendingSignals: 0,
        successRate: 0,
        avgConfidence: 0,
        avgProfit: 0,
        bestTimeframe: data.timeframe,
        lastUpdated: new Date().toISOString(),
        recentSignals: [],
      };
    }

    // Yeni sinyali ekle
    const newSignal: SignalResult = {
      timestamp: new Date().toISOString(),
      signal: data.signal,
      confidence: data.confidence,
      price: data.price,
      outcome: 'pending',
      timeframe: data.timeframe,
    };

    performance.recentSignals.unshift(newSignal);

    // Son 50 sinyali sakla
    if (performance.recentSignals.length > 50) {
      performance.recentSignals = performance.recentSignals.slice(0, 50);
    }

    performance.totalSignals++;
    performance.pendingSignals++;
    performance.lastUpdated = new Date().toISOString();

    // Ortalama güveni güncelle
    const totalConfidence = performance.recentSignals.reduce((sum, s) => sum + s.confidence, 0);
    performance.avgConfidence = totalConfidence / performance.recentSignals.length;

    // Redis'e kaydet (24 saat TTL)
    await redis.setex(key, 24 * 3600, JSON.stringify(performance));

    console.log(`✅ Memory: Recorded ${data.strategyName} signal for ${data.symbol}`);

    return performance;
  } catch (error: any) {
    console.error(`❌ Memory: Error recording performance:`, error.message);
    throw error;
  }
}

/**
 * Sinyal sonucunu güncelle (başarılı/başarısız)
 */
export async function updateSignalOutcome(data: {
  strategyName: string;
  symbol: string;
  signalTimestamp: string;
  outcome: 'success' | 'failure';
  profitPercent: number;
  exitPrice: number;
}) {
  const redis = getRedisClient();
  const key = MEMORY_KEYS.strategyPerformance(data.strategyName, data.symbol);

  try {
    const existingData = await redis.get(key);
    if (!existingData) {
      console.warn(`⚠️ Memory: No performance data found for ${data.strategyName} on ${data.symbol}`);
      return null;
    }

    const performance: StrategyPerformance = JSON.parse(existingData);

    // İlgili sinyali bul
    const signalIndex = performance.recentSignals.findIndex(
      (s) => s.timestamp === data.signalTimestamp
    );

    if (signalIndex === -1) {
      console.warn(`⚠️ Memory: Signal not found with timestamp ${data.signalTimestamp}`);
      return null;
    }

    // Sinyali güncelle
    const signal = performance.recentSignals[signalIndex];
    signal.outcome = data.outcome;
    signal.profitPercent = data.profitPercent;
    signal.exitPrice = data.exitPrice;
    signal.exitTimestamp = new Date().toISOString();

    // İstatistikleri güncelle
    if (data.outcome === 'success') {
      performance.successfulSignals++;
    } else {
      performance.failedSignals++;
    }
    performance.pendingSignals--;

    // Başarı oranını hesapla
    const completedSignals = performance.successfulSignals + performance.failedSignals;
    performance.successRate = completedSignals > 0
      ? (performance.successfulSignals / completedSignals) * 100
      : 0;

    // Ortalama karı hesapla
    const profitableSignals = performance.recentSignals.filter(
      (s) => s.outcome && s.profitPercent !== undefined
    );
    if (profitableSignals.length > 0) {
      const totalProfit = profitableSignals.reduce((sum, s) => sum + (s.profitPercent || 0), 0);
      performance.avgProfit = totalProfit / profitableSignals.length;
    }

    performance.lastUpdated = new Date().toISOString();

    // Redis'e kaydet
    await redis.setex(key, 24 * 3600, JSON.stringify(performance));

    console.log(
      `✅ Memory: Updated ${data.strategyName} outcome for ${data.symbol} - ${data.outcome} (${data.profitPercent.toFixed(2)}%)`
    );

    return performance;
  } catch (error: any) {
    console.error(`❌ Memory: Error updating signal outcome:`, error.message);
    throw error;
  }
}

/**
 * Coin için tüm strateji performanslarını al
 */
export async function getCoinStrategyPerformances(symbol: string): Promise<StrategyPerformance[]> {
  const redis = getRedisClient();

  try {
    // Tüm strateji adlarını tanımla (gerçek sistemdeki tüm stratejiler)
    const strategies = [
      'conservative-buy-signal',
      'breakout-retest',
      'volume-spike',
      'ma-crossover-pullback',
      'rsi-divergence',
      'ma7-pullback',
      'bollinger-squeeze',
      'ema-ribbon',
      'ichimoku-cloud',
      'macd-histogram',
      'fibonacci-retracement',
      'atr-volatility',
      'trend-reversal',
      'volume-profile',
      'red-wick-green-closure',
    ];

    const performances: StrategyPerformance[] = [];

    for (const strategy of strategies) {
      const key = MEMORY_KEYS.strategyPerformance(strategy, symbol);
      const data = await redis.get(key);

      if (data) {
        performances.push(JSON.parse(data));
      }
    }

    return performances;
  } catch (error: any) {
    console.error(`❌ Memory: Error fetching coin strategy performances:`, error.message);
    return [];
  }
}

/**
 * Adaptif ağırlıkları kaydet
 */
export async function saveAdaptiveWeights(weights: AdaptiveWeights) {
  const redis = getRedisClient();
  const key = MEMORY_KEYS.adaptiveWeights(weights.symbol);

  try {
    // 4 saat geçerlilik süresi
    await redis.setex(key, 4 * 3600, JSON.stringify(weights));

    console.log(`✅ Memory: Saved adaptive weights for ${weights.symbol}`);
  } catch (error: any) {
    console.error(`❌ Memory: Error saving adaptive weights:`, error.message);
    throw error;
  }
}

/**
 * Adaptif ağırlıkları al
 */
export async function getAdaptiveWeights(symbol: string): Promise<AdaptiveWeights | null> {
  const redis = getRedisClient();
  const key = MEMORY_KEYS.adaptiveWeights(symbol);

  try {
    const data = await redis.get(key);
    if (!data) return null;

    const weights: AdaptiveWeights = JSON.parse(data);

    // Geçerlilik süresini kontrol et
    const validUntil = new Date(weights.validUntil);
    const now = new Date();

    if (now > validUntil) {
      console.log(`⚠️ Memory: Adaptive weights for ${symbol} expired, need recalculation`);
      return null;
    }

    return weights;
  } catch (error: any) {
    console.error(`❌ Memory: Error fetching adaptive weights:`, error.message);
    return null;
  }
}

/**
 * Global strateji istatistiklerini kaydet
 */
export async function saveGlobalStrategyStats(strategyName: string, stats: {
  totalCoins: number;
  avgSuccessRate: number;
  topPerformingCoins: string[];
  poorPerformingCoins: string[];
}) {
  const redis = getRedisClient();
  const key = MEMORY_KEYS.globalStrategyStats(strategyName);

  try {
    const data = {
      ...stats,
      lastUpdated: new Date().toISOString(),
    };

    await redis.setex(key, 24 * 3600, JSON.stringify(data));

    console.log(`✅ Memory: Saved global stats for ${strategyName}`);
  } catch (error: any) {
    console.error(`❌ Memory: Error saving global stats:`, error.message);
    throw error;
  }
}

/**
 * Memory sağlık kontrolü
 */
export async function checkMemoryHealth() {
  const redis = getRedisClient();

  try {
    // Test key yaz ve oku
    const testKey = 'memory:health:test';
    const testValue = { timestamp: new Date().toISOString(), test: true };

    await redis.setex(testKey, 60, JSON.stringify(testValue));
    const readValue = await redis.get(testKey);

    if (!readValue) {
      return { status: 'unhealthy', error: 'Cannot read test key' };
    }

    // Memory kullanımını kontrol et (opsiyonel)
    const info = await redis.info('memory');

    return {
      status: 'healthy',
      memoryInfo: info,
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      error: error.message,
    };
  }
}

export default {
  recordStrategyPerformance,
  updateSignalOutcome,
  getCoinStrategyPerformances,
  saveAdaptiveWeights,
  getAdaptiveWeights,
  saveGlobalStrategyStats,
  checkMemoryHealth,
};
