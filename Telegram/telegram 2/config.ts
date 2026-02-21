/**
 * 📋 TELEGRAM NOTIFICATION CONFIG
 * Kullanıcı tercihlerine göre bildirim ayarları
 *
 * Kullanıcı Tercihleri:
 * - Sinyal Tipleri: STRONG_BUY, BUY, SELL, WAIT
 * - Minimum Confidence: %70+
 * - Bildirim: Anlık (Real-time)
 * - Stratejiler: Tüm Stratejiler (16 adet)
 */

// ============================================================================
// TYPES
// ============================================================================

// ✨ Genişletilmiş Sinyal Tipleri (Tüm Kaynaklar + Sistem Bildirimleri)
export type SignalType =
  // Trading Signals (600+ coin strategies)
  | 'STRONG_BUY'
  | 'BUY'
  | 'SELL'
  | 'WAIT'
  | 'NEUTRAL'
  // AI Bot Signals
  | 'AI_SIGNAL'
  | 'AI_STRONG_BUY'
  | 'AI_STRONG_SELL'
  // Onchain & Whale
  | 'WHALE_ALERT'
  | 'ONCHAIN_ALERT'
  | 'EXCHANGE_FLOW'
  | 'GAS_SPIKE'
  // Market Analysis
  | 'CORRELATION'
  | 'DIVERGENCE'
  | 'MARKET_SHIFT'
  // Futures & Derivatives
  | 'FUTURES_PREMIUM'
  | 'FUTURES_DISCOUNT'
  | 'FUNDING_RATE_HIGH'
  | 'FUNDING_RATE_LOW'
  | 'LIQUIDATION_CLUSTER'
  // Traditional Markets
  | 'TRADITIONAL_MARKET'
  | 'STOCK_SIGNAL'
  | 'FOREX_SIGNAL'
  | 'COMMODITY_SIGNAL'
  // ⚠️ System & Error Notifications
  | 'SYSTEM_ERROR'
  | 'SERVICE_DOWN'
  | 'API_ERROR'
  | 'ANALYSIS_FAILED'
  | 'DATA_QUALITY_ISSUE'
  | 'SYSTEM_HEALTH'
  | 'BACKGROUND_SERVICE_ERROR';

export interface TelegramNotificationConfig {
  // Hangi sinyal tiplerini gönderelim
  enabledSignalTypes: SignalType[];

  // Minimum confidence seviyesi (0-100)
  minConfidence: number;

  // Bildirim modu
  notificationMode: 'realtime' | 'batched';

  // Batch mode için interval (ms)
  batchIntervalMs?: number;

  // Aktif strateji filtreleri (boşsa hepsi)
  enabledStrategies: string[];

  // Sadece belirli sembolleri takip et (boşsa hepsi)
  symbolWhitelist: string[];

  // Spam önleme: Aynı sembol için minimum bekleme süresi (ms)
  minTimeBetweenSameSymbol: number;

  // Günlük özet gönderilsin mi
  sendDailySummary: boolean;

  // Günlük özet saatleri (24 saat formatında)
  dailySummaryHours: number[];

  // 🔒 GİZLİ MOD: Sadece belirli chat ID'lere bildirim gönder
  // Boş array = herkese açık, dolu array = sadece listedeki chat ID'ler
  allowedChatIds: number[];
}

// ============================================================================
// USER CONFIGURATION
// ============================================================================

/**
 * Kullanıcı Tercihleri
 * Bu değerler kullanıcının seçimlerine göre ayarlanmıştır
 */
export const TELEGRAM_CONFIG: TelegramNotificationConfig = {
  // Sinyal Tipleri: STRONG_BUY, BUY, SELL, WAIT
  enabledSignalTypes: ['STRONG_BUY', 'BUY', 'SELL', 'WAIT'],

  // Minimum Confidence: %70+
  minConfidence: 70,

  // Bildirim: Anlık (Real-time)
  notificationMode: 'realtime',

  // Stratejiler: Tüm Stratejiler (16 adet)
  // Boş array = tüm stratejiler dahil
  enabledStrategies: [],

  // Semboller: Tüm semboller
  // Boş array = tüm semboller dahil
  symbolWhitelist: [],

  // Spam önleme: Aynı sembol için 5 dakika bekle
  minTimeBetweenSameSymbol: 300000, // 5 minutes

  // Günlük özet: Aktif
  sendDailySummary: true,

  // Günlük özet saatleri: 09:00 ve 18:00
  dailySummaryHours: [9, 18],

  // 🔒 GİZLİ MOD: Sadece senin chat ID'ne bildirim gönder
  // .env dosyasından TELEGRAM_ALLOWED_CHAT_IDS oku
  // Örnek: TELEGRAM_ALLOWED_CHAT_IDS=123456789,987654321
  // Boş bırakırsan herkese açık olur
  allowedChatIds: process.env.TELEGRAM_ALLOWED_CHAT_IDS
    ? process.env.TELEGRAM_ALLOWED_CHAT_IDS.split(',').map((id) => parseInt(id.trim(), 10))
    : [], // Boş = herkese açık
};

// ============================================================================
// SIGNAL FILTERING
// ============================================================================

/**
 * Sinyalin bildirim gönderilmeye uygun olup olmadığını kontrol et
 */
export function shouldNotifySignal(
  signalType: SignalType,
  confidence: number,
  strategyName?: string
): boolean {
  // 1. Sinyal tipi etkin mi?
  if (!TELEGRAM_CONFIG.enabledSignalTypes.includes(signalType)) {
    return false;
  }

  // 2. Confidence yeterli mi?
  if (confidence < TELEGRAM_CONFIG.minConfidence) {
    return false;
  }

  // 3. Strateji filtresi varsa, strateji dahil mi?
  if (
    TELEGRAM_CONFIG.enabledStrategies.length > 0 &&
    strategyName &&
    !TELEGRAM_CONFIG.enabledStrategies.includes(strategyName)
  ) {
    return false;
  }

  return true;
}

/**
 * Sembolün bildirim gönderilmeye uygun olup olmadığını kontrol et
 */
export function shouldNotifySymbol(symbol: string): boolean {
  // Whitelist boşsa hepsine izin ver
  if (TELEGRAM_CONFIG.symbolWhitelist.length === 0) {
    return true;
  }

  return TELEGRAM_CONFIG.symbolWhitelist.includes(symbol);
}

// ============================================================================
// SPAM PREVENTION
// ============================================================================

// Son bildirim zamanlarını tut
const lastNotificationTime = new Map<string, number>();

/**
 * Spam kontrolü: Aynı sembol için çok sık bildirim gönderme
 */
export function canNotifySymbol(symbol: string): boolean {
  const now = Date.now();
  const lastTime = lastNotificationTime.get(symbol);

  if (!lastTime) {
    lastNotificationTime.set(symbol, now);
    return true;
  }

  const timeSinceLastNotification = now - lastTime;

  if (timeSinceLastNotification < TELEGRAM_CONFIG.minTimeBetweenSameSymbol) {
    return false; // Çok erken, spam
  }

  lastNotificationTime.set(symbol, now);
  return true;
}

/**
 * Spam önleme cache'ini temizle (debug only)
 */
export function clearSpamCache(): void {
  lastNotificationTime.clear();
}

// ============================================================================
// PRIVATE MODE (GIZLI MOD)
// ============================================================================

/**
 * 🔒 Chat ID'nin izin listesinde olup olmadığını kontrol et
 *
 * Eğer allowedChatIds boşsa → Herkese açık (herkes kullanabilir)
 * Eğer allowedChatIds doluysa → Sadece listedeki chat ID'ler
 */
export function isAllowedChatId(chatId: number): boolean {
  // Whitelist boşsa herkese açık
  if (TELEGRAM_CONFIG.allowedChatIds.length === 0) {
    return true;
  }

  // Whitelist doluysa sadece listedeki chat ID'ler
  return TELEGRAM_CONFIG.allowedChatIds.includes(chatId);
}

/**
 * Bot'un gizli modda olup olmadığını kontrol et
 */
export function isPrivateMode(): boolean {
  return TELEGRAM_CONFIG.allowedChatIds.length > 0;
}

// ============================================================================
// SIGNAL TYPE EMOJI & COLOR
// ============================================================================

/**
 * Sinyal tipine göre emoji ve renk döndür (GENİŞLETİLMİŞ)
 */
export function getSignalEmoji(signalType: SignalType): {
  icon: string;
  trend: string;
  color: string;
} {
  switch (signalType) {
    // Trading Signals
    case 'STRONG_BUY':
      return { icon: '🟢', trend: '↗↗', color: '#00FF00' };
    case 'BUY':
      return { icon: '🟢', trend: '↗', color: '#00D000' };
    case 'SELL':
      return { icon: '🔴', trend: '↘↘', color: '#FF0000' };
    case 'WAIT':
      return { icon: '🟡', trend: '↔', color: '#FFA500' };
    case 'NEUTRAL':
      return { icon: '⚪', trend: '→', color: '#808080' };

    // AI Bot Signals
    case 'AI_SIGNAL':
      return { icon: '🤖', trend: '🧠', color: '#8B00FF' };
    case 'AI_STRONG_BUY':
      return { icon: '🤖', trend: '↗↗', color: '#00FF00' };
    case 'AI_STRONG_SELL':
      return { icon: '🤖', trend: '↘↘', color: '#FF0000' };

    // Onchain & Whale
    case 'WHALE_ALERT':
      return { icon: '🐋', trend: '🌊', color: '#0080FF' };
    case 'ONCHAIN_ALERT':
      return { icon: '⛓️', trend: '🔗', color: '#FF8C00' };
    case 'EXCHANGE_FLOW':
      return { icon: '💱', trend: '🔄', color: '#FFD700' };
    case 'GAS_SPIKE':
      return { icon: '⛽', trend: '⬆️', color: '#FF4500' };

    // Market Analysis
    case 'CORRELATION':
      return { icon: '🔗', trend: '📊', color: '#4169E1' };
    case 'DIVERGENCE':
      return { icon: '↗️↘️', trend: '⚠️', color: '#FF6347' };
    case 'MARKET_SHIFT':
      return { icon: '🌐', trend: '🔄', color: '#9370DB' };

    // Futures & Derivatives
    case 'FUTURES_PREMIUM':
      return { icon: '📈', trend: '➕', color: '#32CD32' };
    case 'FUTURES_DISCOUNT':
      return { icon: '📉', trend: '➖', color: '#DC143C' };
    case 'FUNDING_RATE_HIGH':
      return { icon: '💰', trend: '⬆️', color: '#FFD700' };
    case 'FUNDING_RATE_LOW':
      return { icon: '💸', trend: '⬇️', color: '#B22222' };
    case 'LIQUIDATION_CLUSTER':
      return { icon: '💥', trend: '⚡', color: '#FF1493' };

    // Traditional Markets
    case 'TRADITIONAL_MARKET':
      return { icon: '🏛️', trend: '📊', color: '#2F4F4F' };
    case 'STOCK_SIGNAL':
      return { icon: '📈', trend: '💼', color: '#4682B4' };
    case 'FOREX_SIGNAL':
      return { icon: '💱', trend: '🌍', color: '#20B2AA' };
    case 'COMMODITY_SIGNAL':
      return { icon: '🌾', trend: '📦', color: '#DAA520' };

    // ⚠️ System & Error Notifications
    case 'SYSTEM_ERROR':
      return { icon: '⚠️', trend: '❌', color: '#FF0000' };
    case 'SERVICE_DOWN':
      return { icon: '🔴', trend: '⛔', color: '#8B0000' };
    case 'API_ERROR':
      return { icon: '🔌', trend: '❌', color: '#FF6347' };
    case 'ANALYSIS_FAILED':
      return { icon: '⚠️', trend: '📊', color: '#FFA500' };
    case 'DATA_QUALITY_ISSUE':
      return { icon: '⚠️', trend: '📉', color: '#FFD700' };
    case 'SYSTEM_HEALTH':
      return { icon: '✅', trend: '💚', color: '#00FF00' };
    case 'BACKGROUND_SERVICE_ERROR':
      return { icon: '🔧', trend: '❌', color: '#DC143C' };

    default:
      return { icon: '⚪', trend: '❓', color: '#808080' };
  }
}

/**
 * Confidence seviyesine göre yıldız döndür
 */
export function getConfidenceStars(confidence: number): string {
  if (confidence >= 90) return '⭐⭐⭐⭐⭐';
  if (confidence >= 80) return '⭐⭐⭐⭐';
  if (confidence >= 70) return '⭐⭐⭐';
  if (confidence >= 60) return '⭐⭐';
  return '⭐';
}

export default TELEGRAM_CONFIG;
