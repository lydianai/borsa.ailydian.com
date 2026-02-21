/**
 * REALTIME TELEGRAM NOTIFIER
 * Send real-time trading signals to Telegram
 */

import { TechnicalAnalysis } from '../indicators/analyzer';
import { telegramNotifier } from './notifier';

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface LiveSignal {
  symbol: string;
  analysis: TechnicalAnalysis;
}

// ============================================================================
// REALTIME TELEGRAM NOTIFIER
// ============================================================================

export class RealtimeTelegramNotifier {
  private rateLimitCache: Map<string, number> = new Map(); // chatId -> lastMessageTime
  private dedupCache: Map<string, number> = new Map(); // signalKey -> timestamp

  constructor() {
    console.log('[RealtimeTelegramNotifier] Initialized');
  }

  /**
   * Handle live signal and send Telegram notification if conditions are met
   */
  async handleLiveSignal(symbol: string, analysis: TechnicalAnalysis): Promise<void> {
    try {
      // Check if Telegram is enabled
      const enabled = process.env.TELEGRAM_ENABLED === '1';
      if (!enabled) {
        console.log('[RealtimeTelegramNotifier] Telegram disabled, skipping notification');
        return;
      }

      // Check if signal meets criteria
      if ((analysis.signal === 'AL' || analysis.signal === 'SAT') && analysis.confidence >= 70) {
        // Generate signal key for deduplication
        const signalKey = `${symbol}:${analysis.signal}:${Math.floor(Date.now() / 600000)}`; // 10-minute window
        
        // Check deduplication
        if (this.isDuplicate(signalKey)) {
          console.log(`[RealtimeTelegramNotifier] Duplicate signal for ${symbol}, skipping`);
          return;
        }

        // Check rate limit (60 messages per minute)
        if (this.isRateLimited()) {
          console.log('[RealtimeTelegramNotifier] Rate limit exceeded, skipping notification');
          return;
        }

        // Send notification
        await this.sendNotification(symbol, analysis);
        
        // Update caches
        this.updateDedupCache(signalKey);
        this.updateRateLimit();
      }
    } catch (error) {
      console.error('[RealtimeTelegramNotifier] Error handling live signal:', error);
    }
  }

  /**
   * Send Telegram notification
   */
  private async sendNotification(symbol: string, analysis: TechnicalAnalysis): Promise<void> {
    try {
      const { signal, confidence, reason, indicators } = analysis;
      
      // Format message
      const _message = `💹 *${symbol}* — ${signal}  (Confidence: ${confidence}%)\n` +
        `RSI:${indicators.rsi || 'N/A'} EMA:${indicators.ema || 'N/A'} MACD:${indicators.macd || 'N/A'}\n` +
        `Reason: ${reason}\n` +
        `Time: ${new Date().toLocaleTimeString('tr-TR')}`;

      // Create strategy signal object for telegramNotifier
      const strategySignal = {
        name: 'Realtime Analyzer',
        signal: signal,
        score: confidence / 100,
        confidence: confidence,
        reason: reason,
        time: Date.now(),
        indicators: {
          totalScore: confidence,
          fundingBias: 'N/A',
          volatility: 0,
          btcCorrelation: 0
        }
      };

      // Send via telegramNotifier
      await telegramNotifier.notifySignal(symbol, strategySignal);
      
      console.log(`[RealtimeTelegramNotifier] Sent notification for ${symbol}`);
    } catch (error) {
      console.error('[RealtimeTelegramNotifier] Error sending notification:', error);
    }
  }

  /**
   * Check if signal is duplicate
   */
  private isDuplicate(signalKey: string): boolean {
    const lastTimestamp = this.dedupCache.get(signalKey);
    if (!lastTimestamp) return false;
    
    // Check if within 10-minute window
    const now = Date.now();
    return (now - lastTimestamp) < 600000; // 10 minutes
  }

  /**
   * Update deduplication cache
   */
  private updateDedupCache(signalKey: string): void {
    this.dedupCache.set(signalKey, Date.now());
    
    // Clean up old entries (older than 15 minutes)
    const now = Date.now();
    for (const [key, timestamp] of this.dedupCache.entries()) {
      if ((now - timestamp) > 900000) { // 15 minutes
        this.dedupCache.delete(key);
      }
    }
  }

  /**
   * Check rate limit
   */
  private isRateLimited(): boolean {
    const chatId = process.env.TELEGRAM_CHAT_ID || '';
    const lastMessageTime = this.rateLimitCache.get(chatId) || 0;
    const now = Date.now();
    
    // Check if within 1-minute window
    if ((now - lastMessageTime) < 60000) { // 1 minute
      // Count messages in this window
      // For simplicity, we'll just check the time gap
      return true;
    }
    
    return false;
  }

  /**
   * Update rate limit cache
   */
  private updateRateLimit(): void {
    const chatId = process.env.TELEGRAM_CHAT_ID || '';
    this.rateLimitCache.set(chatId, Date.now());
    
    // Clean up old entries (older than 2 minutes)
    const now = Date.now();
    for (const [key, timestamp] of this.rateLimitCache.entries()) {
      if ((now - timestamp) > 120000) { // 2 minutes
        this.rateLimitCache.delete(key);
      }
    }
  }

  /**
   * Force send notification (bypass rate limits and dedup)
   */
  async forceSend(symbol: string, analysis: TechnicalAnalysis): Promise<void> {
    try {
      const { signal, confidence, reason, indicators } = analysis;
      
      // Format message
      const _message = `🚨 *FORCE ALERT* ${symbol} — ${signal}  (Confidence: ${confidence}%)\n` +
        `RSI:${indicators.rsi || 'N/A'} EMA:${indicators.ema || 'N/A'} MACD:${indicators.macd || 'N/A'}\n` +
        `Reason: ${reason}\n` +
        `Time: ${new Date().toLocaleTimeString('tr-TR')}`;

      // Create strategy signal object for telegramNotifier
      const strategySignal = {
        name: 'Realtime Analyzer (Forced)',
        signal: signal,
        score: confidence / 100,
        confidence: confidence,
        reason: reason,
        time: Date.now(),
        indicators: {
          totalScore: confidence,
          fundingBias: 'N/A',
          volatility: 0,
          btcCorrelation: 0
        }
      };

      // Send via telegramNotifier
      await telegramNotifier.notifySignal(symbol, strategySignal);
      
      console.log(`[RealtimeTelegramNotifier] Force sent notification for ${symbol}`);
    } catch (error) {
      console.error('[RealtimeTelegramNotifier] Error force sending notification:', error);
    }
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const realtimeTelegramNotifier = new RealtimeTelegramNotifier();