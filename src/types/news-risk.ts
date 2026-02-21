/**
 * CRITICAL NEWS RISK MANAGEMENT TYPES
 *
 * Kritik haber bazlı risk yönetimi ve uyarı sistemi için type tanımları
 */

import type { CryptoNewsItemWithTranslation } from './rapid-api';

// ============================================
// RISK RULES
// ============================================

export type NewsRiskCategory = 'regulation' | 'hack' | 'upgrade' | 'market_crash' | 'exchange_issue';

export type NewsRiskAction = 'pause' | 'reduce' | 'exit' | 'alert_only';

export interface NewsRiskRule {
  /** Rule ID */
  id: string;

  /** Haber kategorisi */
  category: NewsRiskCategory;

  /** Minimum impact score (1-10) */
  minImpact: number;

  /** Required sentiment */
  sentiment?: 'positive' | 'negative' | 'neutral';

  /** Otomatik aksiyon */
  action: NewsRiskAction;

  /** Aksiyon süresi (dakika) */
  durationMinutes: number;

  /** Pozisyon azaltma oranı (0.0-1.0) */
  positionReductionRatio?: number;

  /** Aktif mi? */
  enabled: boolean;

  /** Açıklama */
  description: string;
}

// ============================================
// CRITICAL NEWS ALERT
// ============================================

export interface CriticalNewsAlert {
  /** Unique alert ID */
  id: string;

  /** İlgili haber */
  news: CryptoNewsItemWithTranslation;

  /** Tetiklenen kural */
  rule: NewsRiskRule;

  /** Alert seviyesi */
  severity: 'critical' | 'high' | 'medium';

  /** Oluşturulma zamanı */
  createdAt: Date;

  /** Geçerlilik süresi (timestamp) */
  expiresAt: Date;

  /** Otomatik aksiyonlar alındı mı? */
  actionsExecuted: {
    pausedEntries: boolean;
    reducedPositions: boolean;
    sentNotification: boolean;
  };

  /** Etkilenen semboller */
  affectedSymbols: string[];

  /** Kullanıcı tarafından dismiss edildi mi? */
  dismissed: boolean;
}

// ============================================
// PAUSE SYSTEM
// ============================================

export interface TradingPauseState {
  /** Paused symbols */
  pausedSymbols: Map<string, PauseInfo>;

  /** Global pause (tüm semboller) */
  globalPause: boolean;

  /** Pause başlangıç zamanı */
  pauseStartedAt: Date | null;

  /** Pause bitiş zamanı */
  pauseEndsAt: Date | null;

  /** Pause nedeni */
  reason: string | null;
}

export interface PauseInfo {
  symbol: string;
  reason: string;
  startedAt: Date;
  endsAt: Date;
  newsAlertId: string;
}

// ============================================
// POSITION REDUCTION
// ============================================

export interface PositionReductionAction {
  /** Aksiyon ID */
  id: string;

  /** Sembol */
  symbol: string;

  /** Orijinal pozisyon boyutu */
  originalSize: number;

  /** Yeni pozisyon boyutu */
  newSize: number;

  /** Azaltma oranı */
  reductionRatio: number;

  /** İlgili haber alert ID */
  newsAlertId: string;

  /** Execution zamanı */
  executedAt: Date;

  /** Başarılı mı? */
  success: boolean;

  /** Hata mesajı (varsa) */
  error?: string;
}

// ============================================
// NEWS RISK SCORE
// ============================================

export interface NewsRiskScore {
  /** Sembol (BTC, ETH, etc.) */
  symbol: string;

  /** Son 24 saatteki toplam risk skoru (-10 ile +10 arası) */
  overallRiskScore: number;

  /** Pozitif haber sayısı */
  positiveNewsCount: number;

  /** Negatif haber sayısı */
  negativeNewsCount: number;

  /** En yüksek impact'li haber */
  highestImpactNews: CryptoNewsItemWithTranslation | null;

  /** Risk seviyesi */
  riskLevel: 'safe' | 'caution' | 'danger' | 'critical';

  /** Önerilen aksiyon */
  recommendedAction: 'buy' | 'hold' | 'reduce' | 'exit';

  /** Hesaplama zamanı */
  calculatedAt: Date;
}

// ============================================
// NOTIFICATION
// ============================================

export interface CriticalNewsNotification {
  /** Notification ID */
  id: string;

  /** Başlık */
  title: string;

  /** Mesaj */
  message: string;

  /** İlgili haber */
  newsId: string;

  /** Severity */
  severity: 'critical' | 'high' | 'medium';

  /** Gönderildi mi? */
  sent: boolean;

  /** Gönderilme zamanı */
  sentAt: Date | null;

  /** Kullanıcı okudu mu? */
  read: boolean;

  /** Otomatik aksiyonlar */
  actions: string[];
}

// ============================================
// SYSTEM STATE
// ============================================

export interface NewsRiskSystemState {
  /** Aktif kritik uyarılar */
  activeAlerts: CriticalNewsAlert[];

  /** Pause durumu */
  pauseState: TradingPauseState;

  /** Son pozisyon azaltma aksiyonları */
  recentReductions: PositionReductionAction[];

  /** Sembol bazlı risk skorları */
  riskScores: Map<string, NewsRiskScore>;

  /** Sistem aktif mi? */
  enabled: boolean;

  /** Son güncelleme */
  lastUpdate: Date;
}
