/**
 * AI MEMORY & LEARNING SYSTEM
 * Tracks coin movements, learns from patterns, and provides alerts
 *
 * Features:
 * - Coin movement tracking (BUY → SELL transitions)
 * - Historical pattern analysis
 * - Self-learning from past signals
 * - Alert system for signal reversals
 */

interface CoinMovement {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  price: number;
  confidence: number;
  strategy: string;
  timestamp: number;
}

interface CoinHistory {
  symbol: string;
  movements: CoinMovement[];
  buyToSellCount: number; // AL → SAT geçişlerinin sayısı
  lastSignal: 'BUY' | 'SELL' | 'HOLD';
  averageBuyPrice: number;
  averageSellPrice: number;
  profitRate: number; // Ortalama kar oranı
  riskScore: number; // 0-100 (yüksek = riskli)
  pattern: string; // Örn: "volatile", "stable", "trending"
}

interface AILearning {
  totalAnalyzed: number;
  successfulSignals: number;
  failedSignals: number;
  averageAccuracy: number;
  learnedPatterns: Map<string, number>; // pattern → confidence
}

class AIMemorySystem {
  private static instance: AIMemorySystem;
  private coinHistories: Map<string, CoinHistory>;
  private learning: AILearning;
  private readonly STORAGE_KEY = 'ai_memory_system';
  private readonly LEARNING_KEY = 'ai_learning_data';
  private readonly MAX_MOVEMENTS_PER_COIN = 100;

  private constructor() {
    this.coinHistories = new Map();
    this.learning = {
      totalAnalyzed: 0,
      successfulSignals: 0,
      failedSignals: 0,
      averageAccuracy: 0,
      learnedPatterns: new Map(),
    };
    this.loadFromStorage();
  }

  static getInstance(): AIMemorySystem {
    if (!AIMemorySystem.instance) {
      AIMemorySystem.instance = new AIMemorySystem();
    }
    return AIMemorySystem.instance;
  }

  /**
   * Yeni bir koin hareketi kaydet
   */
  recordMovement(
    symbol: string,
    signal: 'BUY' | 'SELL' | 'HOLD',
    price: number,
    confidence: number,
    strategy: string
  ): void {
    const movement: CoinMovement = {
      symbol,
      signal,
      price,
      confidence,
      strategy,
      timestamp: Date.now(),
    };

    let history = this.coinHistories.get(symbol);

    if (!history) {
      history = {
        symbol,
        movements: [],
        buyToSellCount: 0,
        lastSignal: signal,
        averageBuyPrice: 0,
        averageSellPrice: 0,
        profitRate: 0,
        riskScore: 50,
        pattern: 'unknown',
      };
      this.coinHistories.set(symbol, history);
    }

    // AL → SAT geçişini tespit et
    if (history.lastSignal === 'BUY' && signal === 'SELL') {
      history.buyToSellCount++;
      console.log(`[AI Memory] 🚨 ${symbol}: AL → SAT GEÇİŞİ TESPIT EDİLDİ! (${history.buyToSellCount}. geçiş)`);
    }

    // Hareketi kaydet
    history.movements.push(movement);
    if (history.movements.length > this.MAX_MOVEMENTS_PER_COIN) {
      history.movements.shift(); // En eski hareketi kaldır
    }

    // Son sinyali güncelle
    history.lastSignal = signal;

    // İstatistikleri güncelle
    this.updateStatistics(history);

    // Pattern analizi yap
    this.analyzePattern(history);

    // Local storage'a kaydet
    this.saveToStorage();

    // AI learning'i güncelle
    this.updateLearning(movement, history);
  }

  /**
   * AL → SAT geçişi olan koinleri getir
   */
  getBuyToSellTransitions(): Array<{
    symbol: string;
    transitionCount: number;
    lastPrice: number;
    riskScore: number;
    pattern: string;
  }> {
    const transitions: Array<any> = [];

    this.coinHistories.forEach((history) => {
      if (history.buyToSellCount > 0) {
        const lastMovement = history.movements[history.movements.length - 1];
        transitions.push({
          symbol: history.symbol,
          transitionCount: history.buyToSellCount,
          lastPrice: lastMovement.price,
          riskScore: history.riskScore,
          pattern: history.pattern,
        });
      }
    });

    return transitions.sort((a, b) => b.transitionCount - a.transitionCount);
  }

  /**
   * Belirli bir coin için uyarı kontrolü yap
   */
  checkAlert(symbol: string, currentSignal: 'BUY' | 'SELL' | 'HOLD'): {
    alert: boolean;
    message: string;
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  } {
    const history = this.coinHistories.get(symbol);

    if (!history) {
      return { alert: false, message: '', riskLevel: 'LOW' };
    }

    // AL → SAT geçişi kontrolü
    if (history.lastSignal === 'BUY' && currentSignal === 'SELL') {
      return {
        alert: true,
        message: `⚠️ ${symbol}: AL sinyalinden SAT sinyaline geçiş! Geçmiş geçişler: ${history.buyToSellCount}`,
        riskLevel: history.riskScore > 70 ? 'HIGH' : history.riskScore > 40 ? 'MEDIUM' : 'LOW',
      };
    }

    // Yüksek risk skoru kontrolü
    if (history.riskScore > 75 && currentSignal === 'BUY') {
      return {
        alert: true,
        message: `⚠️ ${symbol}: Yüksek risk skoru (${history.riskScore}/100) - Dikkatli olun!`,
        riskLevel: 'HIGH',
      };
    }

    return { alert: false, message: '', riskLevel: 'LOW' };
  }

  /**
   * Coin pattern analizi
   */
  private analyzePattern(history: CoinHistory): void {
    if (history.movements.length < 5) {
      history.pattern = 'insufficient_data';
      return;
    }

    const recentMovements = history.movements.slice(-10);
    const buyCount = recentMovements.filter((m) => m.signal === 'BUY').length;
    const sellCount = recentMovements.filter((m) => m.signal === 'SELL').length;

    // Volatilite hesapla
    const prices = recentMovements.map((m) => m.price);
    const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const variance = prices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / prices.length;
    const volatility = Math.sqrt(variance) / avgPrice;

    if (volatility > 0.1) {
      history.pattern = 'volatile';
      history.riskScore = Math.min(95, history.riskScore + 5);
    } else if (buyCount > sellCount * 2) {
      history.pattern = 'trending_up';
      history.riskScore = Math.max(20, history.riskScore - 5);
    } else if (sellCount > buyCount * 2) {
      history.pattern = 'trending_down';
      history.riskScore = Math.min(80, history.riskScore + 10);
    } else {
      history.pattern = 'stable';
      history.riskScore = Math.max(30, history.riskScore - 2);
    }
  }

  /**
   * İstatistikleri güncelle
   */
  private updateStatistics(history: CoinHistory): void {
    const buyMovements = history.movements.filter((m) => m.signal === 'BUY');
    const sellMovements = history.movements.filter((m) => m.signal === 'SELL');

    if (buyMovements.length > 0) {
      history.averageBuyPrice =
        buyMovements.reduce((sum, m) => sum + m.price, 0) / buyMovements.length;
    }

    if (sellMovements.length > 0) {
      history.averageSellPrice =
        sellMovements.reduce((sum, m) => sum + m.price, 0) / sellMovements.length;
    }

    if (history.averageBuyPrice > 0 && history.averageSellPrice > 0) {
      history.profitRate =
        ((history.averageSellPrice - history.averageBuyPrice) / history.averageBuyPrice) * 100;
    }
  }

  /**
   * AI learning sistemini güncelle
   */
  private updateLearning(movement: CoinMovement, history: CoinHistory): void {
    this.learning.totalAnalyzed++;

    // Pattern confidence güncelle
    const patternKey = `${movement.strategy}_${history.pattern}`;
    const currentConfidence = this.learning.learnedPatterns.get(patternKey) || 50;

    // Başarılı pattern ise confidence artır
    if (movement.confidence > 70 && history.profitRate > 0) {
      this.learning.successfulSignals++;
      this.learning.learnedPatterns.set(patternKey, Math.min(95, currentConfidence + 2));
    } else if (history.profitRate < -5) {
      this.learning.failedSignals++;
      this.learning.learnedPatterns.set(patternKey, Math.max(20, currentConfidence - 3));
    }

    // Ortalama doğruluğu hesapla
    const total = this.learning.successfulSignals + this.learning.failedSignals;
    if (total > 0) {
      this.learning.averageAccuracy = (this.learning.successfulSignals / total) * 100;
    }

    this.saveToStorage();
  }

  /**
   * Öğrenilmiş pattern confidence'ı getir
   */
  getPatternConfidence(strategy: string, pattern: string): number {
    const patternKey = `${strategy}_${pattern}`;
    return this.learning.learnedPatterns.get(patternKey) || 50;
  }

  /**
   * AI learning istatistiklerini getir
   */
  getLearningStats(): AILearning {
    return { ...this.learning };
  }

  /**
   * Coin geçmişini getir
   */
  getCoinHistory(symbol: string): CoinHistory | undefined {
    return this.coinHistories.get(symbol);
  }

  /**
   * Tüm coin geçmişlerini getir
   */
  getAllHistories(): Map<string, CoinHistory> {
    return new Map(this.coinHistories);
  }

  /**
   * Local storage'a kaydet
   */
  private saveToStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      const historiesArray = Array.from(this.coinHistories.entries());
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(historiesArray));

      const learningData = {
        ...this.learning,
        learnedPatterns: Array.from(this.learning.learnedPatterns.entries()),
      };
      localStorage.setItem(this.LEARNING_KEY, JSON.stringify(learningData));
    } catch (error) {
      console.error('[AI Memory] Storage save error:', error);
    }
  }

  /**
   * Local storage'dan yükle
   */
  private loadFromStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      const historiesData = localStorage.getItem(this.STORAGE_KEY);
      if (historiesData) {
        const historiesArray = JSON.parse(historiesData);
        this.coinHistories = new Map(historiesArray);
      }

      const learningData = localStorage.getItem(this.LEARNING_KEY);
      if (learningData) {
        const parsed = JSON.parse(learningData);
        this.learning = {
          ...parsed,
          learnedPatterns: new Map(parsed.learnedPatterns),
        };
      }
    } catch (error) {
      console.error('[AI Memory] Storage load error:', error);
    }
  }

  /**
   * Tüm hafızayı temizle
   */
  clearMemory(): void {
    this.coinHistories.clear();
    this.learning = {
      totalAnalyzed: 0,
      successfulSignals: 0,
      failedSignals: 0,
      averageAccuracy: 0,
      learnedPatterns: new Map(),
    };
    this.saveToStorage();
    console.log('[AI Memory] Memory cleared');
  }
}

export const aiMemorySystem = AIMemorySystem.getInstance();
export type { CoinHistory, CoinMovement, AILearning };
