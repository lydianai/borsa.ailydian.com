/**
 * CRITICAL NEWS RISK ANALYZER
 *
 * Kritik haber bazlı otomatik risk yönetim sistemi
 *
 * Özellikler:
 * - Kritik haberleri tespit et
 * - Otomatik pause mekanizması
 * - Pozisyon azaltma
 * - Push notification
 * - Risk scoring
 */

import type {
  CriticalNewsAlert,
  NewsRiskRule,
  NewsRiskScore,
  NewsRiskSystemState,
  TradingPauseState,
  PositionReductionAction,
  PauseInfo,
  CriticalNewsNotification,
} from '@/types/news-risk';
import type { CryptoNewsItemWithTranslation } from '@/types/rapid-api';

// ============================================
// DEFAULT RISK RULES
// ============================================

const DEFAULT_RISK_RULES: NewsRiskRule[] = [
  {
    id: 'regulation-critical',
    category: 'regulation',
    minImpact: 9,
    sentiment: 'negative',
    action: 'pause',
    durationMinutes: 120, // 2 saat
    enabled: true,
    description: 'Kritik düzenleme haberleri - yeni girişleri durdur',
  },
  {
    id: 'regulation-high',
    category: 'regulation',
    minImpact: 7,
    sentiment: 'negative',
    action: 'reduce',
    durationMinutes: 60,
    positionReductionRatio: 0.5, // %50 azalt
    enabled: true,
    description: 'Yüksek etkili düzenleme haberleri - pozisyon azalt',
  },
  {
    id: 'hack-critical',
    category: 'hack',
    minImpact: 8,
    action: 'reduce',
    durationMinutes: 240, // 4 saat
    positionReductionRatio: 0.5,
    enabled: true,
    description: 'Exchange/protocol hack - pozisyon azalt',
  },
  {
    id: 'market-crash',
    category: 'market_crash',
    minImpact: 9,
    sentiment: 'negative',
    action: 'exit',
    durationMinutes: 180, // 3 saat
    positionReductionRatio: 0.7, // %70 azalt
    enabled: true,
    description: 'Market çöküşü haberi - büyük pozisyon azalt',
  },
  {
    id: 'upgrade-major',
    category: 'upgrade',
    minImpact: 9,
    action: 'pause',
    durationMinutes: 1440, // 24 saat
    enabled: true,
    description: 'Major protocol upgrade - volatilite bekleniyor',
  },
];

// ============================================
// NEWS RISK ANALYZER CLASS
// ============================================

class NewsRiskAnalyzer {
  private state: NewsRiskSystemState;
  private rules: NewsRiskRule[];

  constructor() {
    this.state = {
      activeAlerts: [],
      pauseState: {
        pausedSymbols: new Map(),
        globalPause: false,
        pauseStartedAt: null,
        pauseEndsAt: null,
        reason: null,
      },
      recentReductions: [],
      riskScores: new Map(),
      enabled: true,
      lastUpdate: new Date(),
    };

    this.rules = [...DEFAULT_RISK_RULES];
  }

  /**
   * Haberleri analiz et ve kritik olanları tespit et
   */
  analyzeNews(newsItems: CryptoNewsItemWithTranslation[]): CriticalNewsAlert[] {
    if (!this.state.enabled) {
      console.log('[NewsRisk] System disabled, skipping analysis');
      return [];
    }

    const now = new Date();
    const alerts: CriticalNewsAlert[] = [];

    for (const news of newsItems) {
      // Her enabled rule'ı kontrol et
      for (const rule of this.rules.filter(r => r.enabled)) {
        if (this.matchesRule(news, rule)) {
          const alert = this.createAlert(news, rule);
          alerts.push(alert);

          console.log(
            `[NewsRisk] 🔴 CRITICAL ALERT: ${news.titleTR.substring(0, 60)}... (Rule: ${rule.id})`
          );
        }
      }
    }

    // Yeni alertleri state'e ekle
    this.state.activeAlerts = [
      ...this.state.activeAlerts.filter(a => now < a.expiresAt && !a.dismissed),
      ...alerts,
    ];

    this.state.lastUpdate = now;
    return alerts;
  }

  /**
   * Haber bir kurala uyuyor mu?
   */
  private matchesRule(news: CryptoNewsItemWithTranslation, rule: NewsRiskRule): boolean {
    // Impact score kontrolü
    if (news.impactScore < rule.minImpact) {
      return false;
    }

    // Sentiment kontrolü (opsiyonel)
    if (rule.sentiment && news.sentiment !== rule.sentiment) {
      return false;
    }

    // Kategori matching (basit keyword bazlı)
    const newsText = `${news.title} ${news.description} ${news.titleTR} ${news.descriptionTR}`.toLowerCase();

    switch (rule.category) {
      case 'regulation':
        return /sec|regulation|cftc|regulatory|ban|illegal|lawsuit|court/.test(newsText);

      case 'hack':
        return /hack|exploit|stolen|attack|vulnerability|security breach/.test(newsText);

      case 'upgrade':
        return /upgrade|fork|hard fork|merge|update|migration/.test(newsText);

      case 'market_crash':
        return /crash|collapse|plunge|panic|selloff|dump/.test(newsText);

      case 'exchange_issue':
        return /exchange.*down|exchange.*halt|exchange.*suspend|withdrawal.*suspend/.test(newsText);

      default:
        return false;
    }
  }

  /**
   * Alert oluştur
   */
  private createAlert(
    news: CryptoNewsItemWithTranslation,
    rule: NewsRiskRule
  ): CriticalNewsAlert {
    const now = new Date();
    const expiresAt = new Date(now.getTime() + rule.durationMinutes * 60 * 1000);

    // Severity belirleme
    let severity: 'critical' | 'high' | 'medium' = 'medium';
    if (news.impactScore >= 9) severity = 'critical';
    else if (news.impactScore >= 8) severity = 'high';

    // Etkilenen semboller (basit keyword matching)
    const affectedSymbols = this.extractAffectedSymbols(news);

    return {
      id: `alert-${Date.now()}-${Math.random().toString(36).substring(7)}`,
      news,
      rule,
      severity,
      createdAt: now,
      expiresAt,
      actionsExecuted: {
        pausedEntries: false,
        reducedPositions: false,
        sentNotification: false,
      },
      affectedSymbols,
      dismissed: false,
    };
  }

  /**
   * Haberden etkilenen sembolleri çıkar
   */
  private extractAffectedSymbols(news: CryptoNewsItemWithTranslation): string[] {
    const text = `${news.title} ${news.titleTR} ${news.tags.join(' ')}`.toLowerCase();
    const symbols: string[] = [];

    // Common crypto symbols
    const cryptoMap: Record<string, string> = {
      bitcoin: 'BTC',
      btc: 'BTC',
      ethereum: 'ETH',
      eth: 'ETH',
      solana: 'SOL',
      cardano: 'ADA',
      ripple: 'XRP',
      xrp: 'XRP',
      'binance coin': 'BNB',
      bnb: 'BNB',
      dogecoin: 'DOGE',
      polkadot: 'DOT',
    };

    for (const [keyword, symbol] of Object.entries(cryptoMap)) {
      if (text.includes(keyword)) {
        symbols.push(symbol);
      }
    }

    // Eğer hiç sembol bulunamadıysa, genel impact olarak tüm major coins
    if (symbols.length === 0 && news.impactScore >= 9) {
      return ['BTC', 'ETH'];
    }

    return [...new Set(symbols)]; // Unique
  }

  /**
   * Otomatik aksiyonları çalıştır
   */
  async executeAutoActions(alert: CriticalNewsAlert): Promise<void> {
    console.log(`[NewsRisk] Executing auto actions for alert ${alert.id}`);

    switch (alert.rule.action) {
      case 'pause':
        await this.pauseNewEntries(alert);
        break;

      case 'reduce':
        await this.reducePositions(alert);
        break;

      case 'exit':
        await this.reducePositions(alert); // Exit = aggressive reduce
        break;

      case 'alert_only':
        // Sadece notification gönder
        await this.sendNotification(alert);
        break;
    }

    alert.actionsExecuted.sentNotification = true;
  }

  /**
   * Yeni girişleri durdur
   */
  private async pauseNewEntries(alert: CriticalNewsAlert): Promise<void> {
    const symbols = alert.affectedSymbols.length > 0 ? alert.affectedSymbols : ['*']; // * = global

    const endsAt = alert.expiresAt;

    if (symbols.includes('*')) {
      // Global pause
      this.state.pauseState.globalPause = true;
      this.state.pauseState.pauseStartedAt = new Date();
      this.state.pauseState.pauseEndsAt = endsAt;
      this.state.pauseState.reason = alert.news.titleTR;

      console.log(
        `[NewsRisk] ⏸️  GLOBAL PAUSE activated until ${endsAt.toLocaleTimeString('tr-TR')}`
      );
    } else {
      // Symbol-specific pause
      for (const symbol of symbols) {
        const pauseInfo: PauseInfo = {
          symbol,
          reason: alert.news.titleTR,
          startedAt: new Date(),
          endsAt,
          newsAlertId: alert.id,
        };

        this.state.pauseState.pausedSymbols.set(symbol, pauseInfo);

        console.log(
          `[NewsRisk] ⏸️  PAUSED ${symbol} until ${endsAt.toLocaleTimeString('tr-TR')}`
        );
      }
    }

    alert.actionsExecuted.pausedEntries = true;

    // Notification gönder
    await this.sendNotification(alert);
  }

  /**
   * Pozisyonları azalt (simülasyon - gerçek trade entegrasyonu için genişletilebilir)
   */
  private async reducePositions(alert: CriticalNewsAlert): Promise<void> {
    const reductionRatio = alert.rule.positionReductionRatio || 0.5;
    const symbols = alert.affectedSymbols.length > 0 ? alert.affectedSymbols : ['BTC', 'ETH'];

    for (const symbol of symbols) {
      const action: PositionReductionAction = {
        id: `reduction-${Date.now()}-${symbol}`,
        symbol,
        originalSize: 1000, // Placeholder - gerçek pozisyon bilgisi gerekli
        newSize: 1000 * (1 - reductionRatio),
        reductionRatio,
        newsAlertId: alert.id,
        executedAt: new Date(),
        success: true, // Simülasyon - gerçekte trade execute edilmeli
      };

      this.state.recentReductions.push(action);

      console.log(
        `[NewsRisk] 📉 REDUCED ${symbol} position by ${(reductionRatio * 100).toFixed(0)}%`
      );
    }

    alert.actionsExecuted.reducedPositions = true;

    // Notification gönder
    await this.sendNotification(alert);
  }

  /**
   * Push notification gönder
   */
  private async sendNotification(alert: CriticalNewsAlert): Promise<void> {
    const notification: CriticalNewsNotification = {
      id: `notif-${alert.id}`,
      title: `🔴 ${alert.severity.toUpperCase()}: Kritik Haber!`,
      message: alert.news.titleTR,
      newsId: alert.news.id,
      severity: alert.severity,
      sent: true,
      sentAt: new Date(),
      read: false,
      actions: this.getExecutedActions(alert),
    };

    // Browser notification API kullan
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'granted') {
        new Notification(notification.title, {
          body: notification.message,
          icon: '/icons/icon-192x192.png',
          badge: '/icons/icon-96x96.png',
          tag: notification.id,
          requireInteraction: alert.severity === 'critical',
        });
      }
    }

    console.log(`[NewsRisk] 📢 Notification sent: ${notification.message.substring(0, 50)}...`);
  }

  /**
   * Gerçekleştirilen aksiyonları listele
   */
  private getExecutedActions(alert: CriticalNewsAlert): string[] {
    const actions: string[] = [];

    if (alert.actionsExecuted.pausedEntries) {
      actions.push(`Yeni girişler ${alert.rule.durationMinutes} dakika durduruldu`);
    }

    if (alert.actionsExecuted.reducedPositions) {
      const ratio = alert.rule.positionReductionRatio || 0.5;
      actions.push(`Pozisyonlar %${(ratio * 100).toFixed(0)} azaltıldı`);
    }

    return actions;
  }

  /**
   * Bir sembol pause'da mı?
   */
  isPaused(symbol: string): boolean {
    const now = new Date();

    // Global pause kontrolü
    if (this.state.pauseState.globalPause && this.state.pauseState.pauseEndsAt) {
      if (now < this.state.pauseState.pauseEndsAt) {
        return true;
      } else {
        // Global pause süresi doldu
        this.state.pauseState.globalPause = false;
      }
    }

    // Symbol-specific pause kontrolü
    const pauseInfo = this.state.pauseState.pausedSymbols.get(symbol);
    if (pauseInfo && now < pauseInfo.endsAt) {
      return true;
    } else if (pauseInfo) {
      // Pause süresi doldu, kaldır
      this.state.pauseState.pausedSymbols.delete(symbol);
    }

    return false;
  }

  /**
   * Risk skorunu hesapla (son 24 saat)
   */
  calculateRiskScore(symbol: string, newsItems: CryptoNewsItemWithTranslation[]): NewsRiskScore {
    const now = new Date();
    const last24h = newsItems.filter(n => now.getTime() - n.timestamp.getTime() < 24 * 3600000);

    // Symbol ile ilgili haberleri filtrele
    const relevantNews = last24h.filter(n => {
      const text = `${n.title} ${n.titleTR} ${n.tags.join(' ')}`.toLowerCase();
      return text.includes(symbol.toLowerCase()) || text.includes(this.getFullName(symbol));
    });

    let totalRiskScore = 0;
    let positiveCount = 0;
    let negativeCount = 0;
    let highestImpactNews: CryptoNewsItemWithTranslation | null = null;
    let maxImpact = 0;

    for (const news of relevantNews) {
      const sentimentValue = news.sentiment === 'positive' ? 1 : news.sentiment === 'negative' ? -1 : 0;
      const scoreContribution = news.impactScore * sentimentValue;

      totalRiskScore += scoreContribution;

      if (news.sentiment === 'positive') positiveCount++;
      if (news.sentiment === 'negative') negativeCount++;

      if (news.impactScore > maxImpact) {
        maxImpact = news.impactScore;
        highestImpactNews = news;
      }
    }

    // Risk level belirleme
    let riskLevel: 'safe' | 'caution' | 'danger' | 'critical' = 'safe';
    if (totalRiskScore < -15) riskLevel = 'critical';
    else if (totalRiskScore < -8) riskLevel = 'danger';
    else if (totalRiskScore < -3) riskLevel = 'caution';

    // Önerilen aksiyon
    let recommendedAction: 'buy' | 'hold' | 'reduce' | 'exit' = 'hold';
    if (totalRiskScore > 10) recommendedAction = 'buy';
    else if (totalRiskScore < -15) recommendedAction = 'exit';
    else if (totalRiskScore < -8) recommendedAction = 'reduce';

    return {
      symbol,
      overallRiskScore: Math.round(totalRiskScore * 10) / 10,
      positiveNewsCount: positiveCount,
      negativeNewsCount: negativeCount,
      highestImpactNews,
      riskLevel,
      recommendedAction,
      calculatedAt: now,
    };
  }

  /**
   * Sembol kısa adından tam adı getir
   */
  private getFullName(symbol: string): string {
    const map: Record<string, string> = {
      BTC: 'bitcoin',
      ETH: 'ethereum',
      SOL: 'solana',
      ADA: 'cardano',
      XRP: 'ripple',
      BNB: 'binance',
      DOGE: 'dogecoin',
      DOT: 'polkadot',
    };
    return map[symbol] || symbol.toLowerCase();
  }

  /**
   * State'i getir
   */
  getState(): NewsRiskSystemState {
    return { ...this.state };
  }

  /**
   * Aktif alertleri getir
   */
  getActiveAlerts(): CriticalNewsAlert[] {
    const now = new Date();
    return this.state.activeAlerts.filter(a => now < a.expiresAt && !a.dismissed);
  }

  /**
   * Alert'i dismiss et
   */
  dismissAlert(alertId: string): void {
    const alert = this.state.activeAlerts.find(a => a.id === alertId);
    if (alert) {
      alert.dismissed = true;
      console.log(`[NewsRisk] Alert ${alertId} dismissed`);
    }
  }

  /**
   * Sistemi enable/disable et
   */
  setEnabled(enabled: boolean): void {
    this.state.enabled = enabled;
    console.log(`[NewsRisk] System ${enabled ? 'enabled' : 'disabled'}`);
  }

  /**
   * Custom rule ekle
   */
  addRule(rule: NewsRiskRule): void {
    this.rules.push(rule);
    console.log(`[NewsRisk] Added custom rule: ${rule.id}`);
  }

  /**
   * Rule'u enable/disable et
   */
  toggleRule(ruleId: string, enabled: boolean): void {
    const rule = this.rules.find(r => r.id === ruleId);
    if (rule) {
      rule.enabled = enabled;
      console.log(`[NewsRisk] Rule ${ruleId} ${enabled ? 'enabled' : 'disabled'}`);
    }
  }
}

// ============================================
// SINGLETON INSTANCE
// ============================================

export const newsRiskAnalyzer = new NewsRiskAnalyzer();

// ============================================
// EXPORT DEFAULTS
// ============================================

export { DEFAULT_RISK_RULES };
export type { NewsRiskAnalyzer };
