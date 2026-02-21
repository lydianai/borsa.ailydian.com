/**
 * 🎯 TELEGRAM SIGNAL NOTIFIER
 * Strategy Aggregator'dan gelen sinyalleri Telegram'a gönderir
 *
 * Kullanıcı Ayarları:
 * - Sinyal Tipleri: STRONG_BUY, BUY, SELL, WAIT
 * - Min Confidence: %70+
 * - Mod: Real-time (anlık)
 * - Stratejiler: Tüm stratejiler
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational purposes only
 * - No trading operations
 * - User can unsubscribe anytime
 */

import { notifyNewSignal, type TradingSignal } from './notifications';
import {
  shouldNotifySignal,
  shouldNotifySymbol,
  canNotifySymbol,
  getSignalEmoji,
  getConfidenceStars,
  type SignalType,
} from './config';

// ============================================================================
// TYPES
// ============================================================================

interface StrategyAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  overallScore: number;
  recommendation: 'STRONG_BUY' | 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  buyCount: number;
  sellCount: number;
  waitCount: number;
  neutralCount: number;
  strategies: Array<{
    name: string;
    signal: 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
    confidence: number;
    reason: string;
  }>;
  timestamp: string;
}

// ============================================================================
// SIGNAL PROCESSING
// ============================================================================

/**
 * Strategy Analysis'i Telegram notification'a çevir
 */
export async function processAndNotifySignal(
  analysis: StrategyAnalysis
): Promise<{
  notified: boolean;
  reason?: string;
  sent?: number;
  failed?: number;
}> {
  const { symbol, recommendation, overallScore, price, changePercent24h, strategies } = analysis;

  // 1. Sembol kontrolü
  if (!shouldNotifySymbol(symbol)) {
    return {
      notified: false,
      reason: 'Symbol not in whitelist',
    };
  }

  // 2. Sinyal tipi ve confidence kontrolü
  if (!shouldNotifySignal(recommendation, overallScore)) {
    return {
      notified: false,
      reason: `Signal type ${recommendation} or confidence ${overallScore}% not enabled`,
    };
  }

  // 3. Spam kontrolü
  if (!canNotifySymbol(symbol)) {
    return {
      notified: false,
      reason: 'Too soon since last notification for this symbol',
    };
  }

  // 4. Telegram mesajını oluştur ve gönder
  const { icon, trend } = getSignalEmoji(recommendation);
  const stars = getConfidenceStars(overallScore);

  // En güçlü 3 stratejiyi al
  const topStrategies = strategies
    .filter((s) => s.signal === recommendation)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);

  // Mesaj formatı
  const signal: TradingSignal = {
    symbol,
    price: price.toFixed(2),
    action: recommendation,
    confidence: overallScore,
    timestamp: Date.now(),
    reason: formatStrategyReasons(topStrategies),
    strategy: `${analysis.buyCount}/${analysis.strategies.length} strateji BUY`,
  };

  // Bildirimi gönder
  const result = await notifyNewSignal(signal);

  return {
    notified: true,
    sent: result.sent,
    failed: result.failed,
  };
}

/**
 * Strateji sebeplerini formatla
 */
function formatStrategyReasons(
  strategies: Array<{ name: string; signal: string; confidence: number; reason: string }>
): string {
  if (strategies.length === 0) {
    return 'Strateji analizi tamamlandı.';
  }

  return strategies
    .map((s, i) => `${i + 1}. ${s.name} (${s.confidence}%): ${s.reason}`)
    .join('\n');
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/**
 * Birden fazla analizi toplu işle
 */
export async function processAndNotifyBatch(
  analyses: StrategyAnalysis[]
): Promise<{
  totalProcessed: number;
  notified: number;
  skipped: number;
  results: Array<{
    symbol: string;
    notified: boolean;
    reason?: string;
  }>;
}> {
  const results = await Promise.all(
    analyses.map(async (analysis) => {
      const result = await processAndNotifySignal(analysis);
      return {
        symbol: analysis.symbol,
        ...result,
      };
    })
  );

  const notified = results.filter((r) => r.notified).length;
  const skipped = results.filter((r) => !r.notified).length;

  return {
    totalProcessed: results.length,
    notified,
    skipped,
    results,
  };
}

// ============================================================================
// HELPER: Format Signal Message (Enhanced)
// ============================================================================

/**
 * Gelişmiş sinyal mesajı formatı
 * (Backup - notifications.ts'de de kullanılabilir)
 */
export function formatSignalMessage(signal: TradingSignal): string {
  const { icon, trend } = getSignalEmoji(signal.action as SignalType);
  const stars = getConfidenceStars(signal.confidence);

  return `${icon} **YENİ ${signal.action} SİNYALİ** ${trend}

📊 Sembol: **${signal.symbol}**
💰 Fiyat: **$${signal.price}**
🎯 Güven: **${signal.confidence}%** ${stars}
${signal.strategy ? `⚙️ ${signal.strategy}\n` : ''}⏰ ${new Date(signal.timestamp).toLocaleString('tr-TR')}

${signal.reason ? `📝 **En Güçlü Stratejiler:**\n${signal.reason}\n\n` : ''}⚠️ *Eğitim amaçlıdır, finansal tavsiye değildir.*

Detay: ${process.env.NEXT_PUBLIC_APP_URL || 'https://sardag.app'}/trading-signals`;
}

export default {
  processAndNotifySignal,
  processAndNotifyBatch,
  formatSignalMessage,
};
