/**
 * TELEGRAM NOTIFIER
 * Send trading signals and alerts to Telegram
 */

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

export interface StrategySignal {
  name: string;
  signal: 'AL' | 'BEKLE' | 'SAT' | string;
  score?: number;
  confidence?: number;
  reason?: string;
  targets?: number[];
  stopLoss?: number;
  entry?: number;
  time: number;
  indicators?: {
    totalScore?: number;
    fundingBias?: string;
    volatility?: number;
    btcCorrelation?: number;
  };
}

// ============================================================================
// TELEGRAM NOTIFIER
// ============================================================================

export class TelegramNotifier {
  /**
   * Send a trading signal notification to Telegram
   */
  async notifySignal(symbol: string, signal: StrategySignal): Promise<void> {
    try {
      // Check if Telegram is enabled
      const enabled = process.env.TELEGRAM_ENABLED === '1';
      if (!enabled) {
        console.log('[TelegramNotifier] Telegram disabled, skipping notification');
        return;
      }

      // Get configuration
      const botToken = process.env.TELEGRAM_BOT_TOKEN;
      const chatId = process.env.TELEGRAM_CHAT_ID;
      
      if (!botToken || !chatId) {
        console.warn('[TelegramNotifier] Telegram credentials not configured');
        return;
      }

      // Format message
      const confidence = signal.confidence !== undefined ? `${signal.confidence.toFixed(1)}%` : 'N/A';
      const score = signal.score !== undefined ? signal.score.toFixed(2) : 'N/A';
      
      let message = `🚨 *${symbol} Trading Signal*\n\n`;
      message += `>Action: ${signal.signal}\n`;
      message += `>Confidence: ${confidence}\n`;
      message += `>Score: ${score}\n`;
      
      if (signal.entry) {
        message += `>Entry: ${signal.entry.toFixed(2)}\n`;
      }
      
      if (signal.targets && signal.targets.length > 0) {
        message += `>Targets: ${signal.targets.map(t => t.toFixed(2)).join(', ')}\n`;
      }
      
      if (signal.stopLoss) {
        message += `>Stop Loss: ${signal.stopLoss.toFixed(2)}\n`;
      }
      
      if (signal.reason) {
        message += `>Reason: ${signal.reason}\n`;
      }
      
      if (signal.indicators) {
        message += `\n📊 *Indicators:*\n`;
        if (signal.indicators.totalScore !== undefined) {
          message += `>Total Score: ${signal.indicators.totalScore.toFixed(2)}\n`;
        }
        if (signal.indicators.fundingBias) {
          message += `>Funding Bias: ${signal.indicators.fundingBias}\n`;
        }
        if (signal.indicators.volatility !== undefined) {
          message += `>Volatility: ${signal.indicators.volatility.toFixed(2)}\n`;
        }
        if (signal.indicators.btcCorrelation !== undefined) {
          message += `>BTC Correlation: ${signal.indicators.btcCorrelation.toFixed(2)}\n`;
        }
      }
      
      message += `\n_${new Date(signal.time).toLocaleString('tr-TR')}_`;

      // Send message to Telegram
      const url = `https://api.telegram.org/bot${botToken}/sendMessage`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chatId,
          text: message,
          parse_mode: 'Markdown',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('[TelegramNotifier] API error:', errorData);
      } else {
        console.log(`[TelegramNotifier] Signal notification sent for ${symbol}`);
      }
    } catch (error) {
      console.error('[TelegramNotifier] Error sending notification:', error);
    }
  }

  /**
   * Send a test message to verify Telegram configuration
   */
  async sendTestMessage(): Promise<boolean> {
    try {
      const enabled = process.env.TELEGRAM_ENABLED === '1';
      if (!enabled) {
        console.log('[TelegramNotifier] Telegram disabled, cannot send test message');
        return false;
      }

      const botToken = process.env.TELEGRAM_BOT_TOKEN;
      const chatId = process.env.TELEGRAM_CHAT_ID;
      
      if (!botToken || !chatId) {
        console.warn('[TelegramNotifier] Telegram credentials not configured for test');
        return false;
      }

      const testMessage = `✅ *Telegram Test Message*\n\n` +
        `Your LyTrade trading system is now connected!\n\n` +
        `Features:\n` +
        `• Real-time trading signals\n` +
        `• Risk management alerts\n` +
        `• System health monitoring\n\n` +
        `_${new Date().toLocaleString('tr-TR')}_`;

      const url = `https://api.telegram.org/bot${botToken}/sendMessage`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chatId,
          text: testMessage,
          parse_mode: 'Markdown',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('[TelegramNotifier] Test message API error:', errorData);
        return false;
      } else {
        console.log('[TelegramNotifier] Test message sent successfully');
        return true;
      }
    } catch (error) {
      console.error('[TelegramNotifier] Error sending test message:', error);
      return false;
    }
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const telegramNotifier = new TelegramNotifier();