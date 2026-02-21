/**
 * 📢 TELEGRAM NOTIFICATION SERVICE
 * Otomatik sinyal bildirimleri gönder
 *
 * Features:
 * - Signal notifications (automatic push)
 * - Subscriber management (in-memory)
 * - Broadcast messaging
 * - Error handling
 * - White-hat compliant
 *
 * Usage:
 * import { notifyNewSignal, subscribe } from '@/lib/telegram/notifications';
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational purposes only
 * - No trading operations
 * - User can unsubscribe anytime
 * - No spam
 * - Privacy protected (only chat ID stored)
 */

import { bot } from './bot';
import { formatPremiumSignal, formatPremiumDailySummary } from './premium-formatter';
import { isAllowedChatId, isPrivateMode } from './config';

// ============================================================================
// TYPES
// ============================================================================

export interface TradingSignal {
  symbol: string;
  price: string | number;
  action: 'STRONG_BUY' | 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  confidence: number;
  timestamp: number | string;
  reason?: string;
  strategy?: string;
}

export interface DailySummary {
  totalSignals: number;
  strongBuyCount: number;
  buyCount: number;
  sellCount: number;
  waitCount: number;
  topSignals: TradingSignal[];
  date: Date;
}

// ============================================================================
// SUBSCRIBER MANAGEMENT (IN-MEMORY)
// ============================================================================

// ⚠️ PRODUCTION: Use Redis or PostgreSQL for persistent storage
const subscribers = new Set<number>(); // Chat IDs

/**
 * Kullanıcıyı bildirimlere abone et
 * 🔒 Gizli mod aktifse sadece izin verilen chat ID'ler abone olabilir
 */
export function subscribe(chatId: number): boolean {
  if (typeof chatId !== 'number' || chatId <= 0) {
    return false;
  }

  // 🔒 GIZLI MOD: Sadece izin verilen chat ID'ler abone olabilir
  if (!isAllowedChatId(chatId)) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.log(`[Telegram] ❌ Chat ID ${chatId} izin listesinde değil (Gizli Mod aktif)`);
    }
    return false;
  }

  subscribers.add(chatId);
  return true;
}

/**
 * Kullanıcının aboneliğini iptal et
 */
export function unsubscribe(chatId: number): boolean {
  return subscribers.delete(chatId);
}

/**
 * Kullanıcının abone olup olmadığını kontrol et
 */
export function isSubscribed(chatId: number): boolean {
  return subscribers.has(chatId);
}

/**
 * Abone sayısını döndür
 */
export function getSubscriberCount(): number {
  return subscribers.size;
}

/**
 * Tüm aboneleri döndür (debug only)
 */
export function getAllSubscribers(): number[] {
  return Array.from(subscribers);
}

/**
 * Tüm abonelikleri temizle (dev only)
 */
export function clearAllSubscribers(): void {
  const nodeEnv = process.env.NODE_ENV as string;
  if (nodeEnv === 'production') {
    throw new Error('Cannot clear subscribers in production');
  }
  subscribers.clear();
}

// ============================================================================
// NOTIFICATION SENDERS
// ============================================================================

/**
 * Yeni trading sinyali bildirimini tüm abonelere gönder
 * Ultra-premium format ile
 * ✨ DIRECT BOT API - BYPASS SUBSCRIBERS (PM2 restart sonrası kaybolma sorunu için)
 */
export async function notifyNewSignal(signal: TradingSignal): Promise<{
  sent: number;
  failed: number;
  errors: string[];
}> {
  // Premium formatter ile mesaj oluştur
  const normalizedSignal: TradingSignal = {
    ...signal,
    price: typeof signal.price === 'number' ? signal.price.toFixed(2) : signal.price,
  };

  const message = formatPremiumSignal(normalizedSignal);

  let sent = 0;
  let failed = 0;
  const errors: string[] = [];

  // ✨ DIREKT TELEGRAM API KULLAN (subscribers bypass)
  // TELEGRAM_ALLOWED_CHAT_IDS environment variable'dan chat IDs al
  const chatIds = process.env.TELEGRAM_ALLOWED_CHAT_IDS
    ? process.env.TELEGRAM_ALLOWED_CHAT_IDS.split(',').map((id) => parseInt(id.trim(), 10))
    : [];

  // Eğer env var yoksa in-memory subscribers'ı kullan (fallback)
  const targetChatIds = chatIds.length > 0 ? chatIds : Array.from(subscribers);

  if (targetChatIds.length === 0) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.warn('[Telegram] No chat IDs configured (neither env var nor subscribers)');
    }
    return { sent: 0, failed: 0, errors: ['No recipients configured'] };
  }

  // Tüm abonelere gönder
  const sendPromises = targetChatIds.map(async (chatId) => {
    try {
      await bot.api.sendMessage(chatId, message, { parse_mode: 'HTML' });
      sent++;
      const nodeEnv = process.env.NODE_ENV as string;
      if (nodeEnv !== 'production') {
        console.log(`[Telegram] Signal sent to ${chatId}`);
      }
    } catch (error: any) {
      failed++;

      // Kullanıcı botu bloklamışsa abonelikten çıkar (sadece in-memory için)
      if (error.error_code === 403 && subscribers.has(chatId)) {
        subscribers.delete(chatId);
        errors.push(`Chat ${chatId}: Bot blocked by user (unsubscribed)`);
      } else {
        errors.push(`Chat ${chatId}: ${error.message}`);
      }

      const nodeEnv = process.env.NODE_ENV as string;
      if (nodeEnv !== 'production') {
        console.error(`[Telegram Notification] Failed to send to ${chatId}:`, error.message);
      }
    }
  });

  await Promise.allSettled(sendPromises);

  return { sent, failed, errors };
}

/**
 * Günlük piyasa özetini gönder
 * Ultra-premium format ile
 */
export async function sendDailySummary(summary?: DailySummary): Promise<{
  sent: number;
  failed: number;
}> {
  let summaryData: DailySummary;

  if (summary) {
    summaryData = summary;
  } else {
    // API'den özet çek
    try {
      const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
      const response = await fetch(`${baseUrl}/api/signals`);
      const data = await response.json();

      const signals = data.signals || [];
      const totalSignals = signals.length;
      const strongBuyCount = signals.filter((s: any) => s.action === 'STRONG_BUY').length;
      const buyCount = signals.filter((s: any) => s.action === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.action === 'SELL').length;
      const waitCount = signals.filter((s: any) => s.action === 'WAIT').length;

      summaryData = {
        totalSignals,
        strongBuyCount,
        buyCount,
        sellCount,
        waitCount,
        topSignals: signals.slice(0, 5),
        date: new Date(),
      };
    } catch (error) {
      const nodeEnv = process.env.NODE_ENV as string;
      if (nodeEnv !== 'production') {
        console.error('[Telegram] Failed to fetch summary:', error);
      }
      return { sent: 0, failed: 0 };
    }
  }

  // Premium formatter ile mesaj oluştur
  const message = formatPremiumDailySummary(summaryData);

  let sent = 0;
  let failed = 0;

  const sendPromises = Array.from(subscribers).map(async (chatId) => {
    try {
      await bot.api.sendMessage(chatId, message, { parse_mode: 'HTML' });
      sent++;
    } catch (error: any) {
      failed++;

      // Kullanıcı botu bloklamışsa abonelikten çıkar
      if (error.error_code === 403) {
        subscribers.delete(chatId);
      }
    }
  });

  await Promise.allSettled(sendPromises);

  return { sent, failed };
}

/**
 * Özel mesaj gönder (broadcast)
 */
export async function broadcastMessage(
  message: string,
  options?: { parse_mode?: 'Markdown' | 'HTML' }
): Promise<{
  sent: number;
  failed: number;
}> {
  let sent = 0;
  let failed = 0;

  const sendPromises = Array.from(subscribers).map(async (chatId) => {
    try {
      await bot.api.sendMessage(chatId, message, options);
      sent++;
    } catch (error: any) {
      failed++;

      // Kullanıcı botu bloklamışsa abonelikten çıkar
      if (error.error_code === 403) {
        subscribers.delete(chatId);
      }
    }
  });

  await Promise.allSettled(sendPromises);

  return { sent, failed };
}

/**
 * Belirli bir kullanıcıya mesaj gönder
 */
export async function sendMessageToUser(
  chatId: number,
  message: string,
  options?: { parse_mode?: 'Markdown' | 'HTML' }
): Promise<boolean> {
  try {
    await bot.api.sendMessage(chatId, message, options);
    return true;
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error(`[Telegram] Failed to send to ${chatId}:`, error.message);
    }

    // Kullanıcı botu bloklamışsa abonelikten çıkar
    if (error.error_code === 403) {
      subscribers.delete(chatId);
    }

    return false;
  }
}

// ============================================================================
// STATISTICS
// ============================================================================

/**
 * Bildirim istatistiklerini getir
 */
export function getNotificationStats() {
  return {
    subscriberCount: subscribers.size,
    subscribers: Array.from(subscribers),
  };
}

export default {
  subscribe,
  unsubscribe,
  isSubscribed,
  getSubscriberCount,
  getAllSubscribers,
  notifyNewSignal,
  sendDailySummary,
  broadcastMessage,
  sendMessageToUser,
  getNotificationStats,
};
