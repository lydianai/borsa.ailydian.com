/**
 * 🌉 UNIFIED NOTIFICATION BRIDGE
 * Tüm bildirim sistemlerini Telegram'a yönlendirir
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational purposes only
 * - No trading operations
 * - Transparent data flow
 */

import { notifyNewSignal, broadcastMessage } from './notifications';
import { formatPremiumSignal } from './premium-formatter';
import type { TradingSignal } from './notifications';
import {
  recordServiceSuccess,
  recordServiceError,
  recordAPIError,
  recordAnalysisError,
} from './system-monitor';
import { bot } from './bot';

// ============================================================================
// 1️⃣ STRATEGY AGGREGATOR BRIDGE (600+ Coin)
// ============================================================================

/**
 * Strategy Aggregator'dan gelen sinyalleri Telegram'a gönder
 */
export async function notifyStrategySignal(signal: {
  symbol: string;
  recommendation: 'STRONG_BUY' | 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';
  overallScore: number;
  price: number;
  strategies: Array<{ name: string; signal: string; confidence: number }>;
  timestamp: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    // En güçlü 3 stratejiyi al
    const topStrategies = signal.strategies
      .filter((s) => s.signal === signal.recommendation)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3)
      .map((s) => `${s.name} (${s.confidence}%)`)
      .join('\n');

    const tradingSignal: TradingSignal = {
      symbol: signal.symbol,
      action: signal.recommendation,
      confidence: signal.overallScore,
      price: signal.price.toString(),
      reason: topStrategies,
      strategy: `${signal.strategies.length} Strateji Analizi`,
      timestamp: signal.timestamp,
    };

    await notifyNewSignal(tradingSignal);

    recordServiceSuccess('Strategy Aggregator');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Strategy Aggregator', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 2️⃣ AI BOT SIGNALS BRIDGE
// ============================================================================

/**
 * AI Bot sinyallerini Telegram'a gönder
 */
export async function notifyAIBotSignal(signal: {
  botName: string;
  symbol: string;
  action: string;
  confidence: number;
  price: number;
  reason?: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const tradingSignal: TradingSignal = {
      symbol: signal.symbol,
      action: signal.action as any,
      confidence: signal.confidence,
      price: signal.price.toString(),
      reason: signal.reason,
      strategy: `🤖 ${signal.botName}`,
      timestamp: new Date().toISOString(),
    };

    await notifyNewSignal(tradingSignal);

    recordServiceSuccess('AI Bots');

    return { success: true };
  } catch (error: any) {
    recordServiceError('AI Bots', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 3️⃣ ONCHAIN/WHALE ALERTS BRIDGE
// ============================================================================

/**
 * Whale alert'leri Telegram'a gönder
 */
export async function notifyWhaleAlert(alert: {
  amount: number;
  token: string;
  from: string;
  to: string;
  txHash?: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const message = `
╭━━━━━━━━━━━━━━╮
┃ 🐋 <b>WHALE ALERT</b> 🐋
├━━━━━━━━━━━━━━┤
┃ ${alert.amount.toLocaleString()} ${alert.token}
┃ From: <code>${alert.from.slice(0, 10)}...</code>
┃ To: <code>${alert.to.slice(0, 10)}...</code>
${alert.txHash ? `┃ TX: <code>${alert.txHash.slice(0, 10)}...</code>` : ''}
├━━━━━━━━━━━━━━┤
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('Onchain Monitor');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Onchain Monitor', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 4️⃣ TRADITIONAL MARKETS BRIDGE
// ============================================================================

/**
 * Geleneksel piyasa sinyallerini Telegram'a gönder
 */
export async function notifyTraditionalMarketSignal(signal: {
  symbol: string;
  marketType: 'stock' | 'forex' | 'commodity' | 'bond';
  action: string;
  price: number;
  confidence: number;
  reason?: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const marketIcon =
      signal.marketType === 'stock'
        ? '📈'
        : signal.marketType === 'forex'
          ? '💱'
          : signal.marketType === 'commodity'
            ? '🌾'
            : '💰';

    const tradingSignal: TradingSignal = {
      symbol: signal.symbol,
      action: signal.action as any,
      confidence: signal.confidence,
      price: signal.price.toString(),
      reason: signal.reason,
      strategy: `${marketIcon} Traditional Market`,
      timestamp: new Date().toISOString(),
    };

    await notifyNewSignal(tradingSignal);

    recordServiceSuccess('Traditional Markets');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Traditional Markets', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 5️⃣ CORRELATION SIGNALS BRIDGE
// ============================================================================

/**
 * Korelasyon sinyallerini Telegram'a gönder
 */
export async function notifyCorrelationSignal(signal: {
  pair: string; // e.g., "BTC/ETH"
  type: 'correlation' | 'divergence';
  value: number;
  description: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const emoji = signal.type === 'correlation' ? '🔗' : '↗️↘️';

    const message = `
╭━━━━━━━━━━━━━━╮
┃ ${emoji} <b>${signal.type.toUpperCase()}</b> ${emoji}
├━━━━━━━━━━━━━━┤
┃ ${signal.pair}
┃ ${signal.description}
├━━━━━━━━━━━━━━┤
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('Correlation Analysis');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Correlation Analysis', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 6️⃣ FUTURES MATRIX BRIDGE
// ============================================================================

/**
 * Futures sinyallerini Telegram'a gönder
 */
export async function notifyFuturesSignal(signal: {
  symbol: string;
  type: 'premium' | 'discount' | 'funding_rate' | 'liquidation';
  value: number;
  description: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const emoji =
      signal.type === 'premium'
        ? '📈'
        : signal.type === 'discount'
          ? '📉'
          : signal.type === 'funding_rate'
            ? '💰'
            : '💥';

    const message = `
╭━━━━━━━━━━━━━━╮
┃ ${emoji} <b>FUTURES ${signal.type.toUpperCase()}</b>
├━━━━━━━━━━━━━━┤
┃ ${signal.symbol}
┃ ${signal.description}
├━━━━━━━━━━━━━━┤
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('Futures Matrix');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Futures Matrix', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 7️⃣ WEB PUSH → TELEGRAM REDIRECT
// ============================================================================

/**
 * Web push yerine Telegram kullan
 */
export async function sendWebPushRedirect(
  message: string,
  options?: {
    title?: string;
    icon?: string;
    url?: string;
  }
): Promise<{ success: boolean; platform: string }> {
  try {
    const formattedMessage = `
╭━━━━━━━━━━━━━━╮
┃ 🔔 <b>${options?.title || 'BİLDİRİM'}</b>
├━━━━━━━━━━━━━━┤
┃ ${message}
╰━━━━━━━━━━━━━━╯
${options?.url ? `\n⟫ <a href="${options.url}">Detaylı Bilgi</a>` : ''}
    `.trim();

    await broadcastMessage(formattedMessage, { parse_mode: 'HTML' });

    return { success: true, platform: 'telegram' };
  } catch (error: any) {
    console.error('[Notification Bridge] Web push redirect failed:', error);
    return { success: false, platform: 'none' };
  }
}

// ============================================================================
// 8️⃣ HEADER NOTIFICATION → TELEGRAM REDIRECT
// ============================================================================

/**
 * Header bildirimlerini Telegram'a yönlendir
 * ✨ DIRECT BOT API - BYPASS SUBSCRIBERS (subscribers Set'i boş kalma sorunu için)
 */
export async function sendHeaderNotification(
  message: string,
  type: 'success' | 'error' | 'warning' | 'info' = 'info'
): Promise<{ success: boolean }> {
  try {
    const emoji = type === 'success' ? '✅' : type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️';
    const formattedMessage = `${emoji} ${message}`;

    // ✨ DIREKT TELEGRAM API KULLAN (subscribers bypass)
    const chatIds = process.env.TELEGRAM_ALLOWED_CHAT_IDS
      ? process.env.TELEGRAM_ALLOWED_CHAT_IDS.split(',').map((id) => parseInt(id.trim(), 10))
      : [];

    if (chatIds.length === 0) {
      console.warn('[Telegram] No allowed chat IDs configured');
      return { success: false };
    }

    // Her chat ID'ye gönder
    for (const chatId of chatIds) {
      try {
        await bot.api.sendMessage(chatId, formattedMessage, { parse_mode: 'HTML' });
        console.log(`[Telegram] Header notification sent to ${chatId}`);
      } catch (error: any) {
        console.error(`[Telegram] Failed to send to ${chatId}:`, error.message);
      }
    }

    return { success: true };
  } catch (error: any) {
    console.error('[Notification Bridge] Header notification failed:', error);
    return { success: false };
  }
}

// ============================================================================
// 9️⃣ NIRVANA DASHBOARD BRIDGE (TÜM STRATEJİLER)
// ============================================================================

/**
 * Nirvana Dashboard genel özetini Telegram'a gönder
 * Tüm stratejilerin özet bilgisi (Türkçe)
 */
export async function notifyNirvanaOverview(data: {
  totalStrategies: number;
  activeStrategies: number;
  totalSignals: number;
  marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  sentimentScore: number;
  topOpportunities: Array<{
    symbol: string;
    strategy: string;
    signal: string;
    confidence: number;
  }>;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const sentimentEmoji =
      data.marketSentiment === 'BULLISH'
        ? '🟢 YÜKSELİŞ'
        : data.marketSentiment === 'BEARISH'
          ? '🔴 DÜŞÜŞ'
          : '🟡 NÖTR';

    // NaN kontrolü: sentimentScore undefined olabilir
    const sentimentScoreDisplay =
      data.sentimentScore != null && !isNaN(data.sentimentScore)
        ? data.sentimentScore
        : 'N/A';

    const message = `
╭━━━━━━━━━━━━━━━━━━━━╮
┃ 🌟 <b>NİRVANA ÖZET</b> 🌟
├━━━━━━━━━━━━━━━━━━━━┤
┃ 📊 Aktif Strateji: ${data.activeStrategies || 0}/${data.totalStrategies || 0}
┃ 🎯 Toplam Sinyal: ${data.totalSignals || 0}
┃ ${sentimentEmoji} (${sentimentScoreDisplay})
├━━━━━━━━━━━━━━━━━━━━┤
${data.topOpportunities && data.topOpportunities.length > 0 ? `┃ <b>🔝 EN İYİ FIRSATLAR:</b>\n${data.topOpportunities.slice(0, 3).map((opp) => `┃ ${opp.signal === 'BUY' ? '🟢' : opp.signal === 'SELL' ? '🔴' : '🟡'} ${opp.symbol}\n┃   ${opp.strategy}\n┃   Güven: %${opp.confidence || 0}`).join('\n')}\n` : ''}├━━━━━━━━━━━━━━━━━━━━┤
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('Nirvana Dashboard');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Nirvana Dashboard', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 🔟 OMNIPOTENT FUTURES (WYCKOFF) BRIDGE
// ============================================================================

/**
 * Wyckoff analizi sinyallerini Telegram'a gönder (Türkçe)
 */
export async function notifyOmnipotentFuturesSignal(signal: {
  symbol: string;
  price: number;
  wyckoffPhase: 'ACCUMULATION' | 'MARKUP' | 'DISTRIBUTION' | 'MARKDOWN';
  signal: string;
  confidence: number;
  omnipotentScore: number;
  volumeProfile: string;
  reason?: string;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const phaseEmoji =
      signal.wyckoffPhase === 'ACCUMULATION'
        ? '🟢 TOPLAMA'
        : signal.wyckoffPhase === 'MARKUP'
          ? '📈 YUKARI'
          : signal.wyckoffPhase === 'DISTRIBUTION'
            ? '🔴 DAĞITIM'
            : '📉 AŞAĞI';

    // NaN kontrolü: price undefined olabilir
    const priceStr =
      signal.price != null && !isNaN(signal.price)
        ? signal.price.toString()
        : 'Veri yok';

    const tradingSignal: TradingSignal = {
      symbol: signal.symbol,
      action: signal.signal as any,
      confidence: signal.confidence,
      price: priceStr,
      reason: `${phaseEmoji}\nOmnipotent Skor: ${signal.omnipotentScore}/100\nHacim: ${signal.volumeProfile}\n${signal.reason || ''}`,
      strategy: '🎯 Wyckoff Analizi',
      timestamp: new Date().toISOString(),
    };

    await notifyNewSignal(tradingSignal);

    recordServiceSuccess('Omnipotent Futures');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Omnipotent Futures', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 1️⃣1️⃣ BTC-ETH ANALYSIS BRIDGE
// ============================================================================

/**
 * BTC-ETH korelasyon analizini Telegram'a gönder (Türkçe)
 */
export async function notifyBTCETHAnalysis(data: {
  correlation30d: number;
  trend: 'Rising' | 'Falling' | 'Stable';
  signal: string | null;
  divergenceStrength?: number;
}): Promise<{ success: boolean; error?: string }> {
  try {
    const trendEmoji =
      data.trend === 'Rising'
        ? '📈 YÜKSELIŞ'
        : data.trend === 'Falling'
          ? '📉 DÜŞÜŞ'
          : '📊 STABIL';

    // NaN kontrolü: correlation30d undefined veya null olabilir
    const correlationPercent =
      data.correlation30d != null && !isNaN(data.correlation30d)
        ? (data.correlation30d * 100).toFixed(1)
        : 'Veri yok';

    const message = `
╭━━━━━━━━━━━━━━━━━━━━╮
┃ 🔗 <b>BTC-ETH ANALİZ</b> 🔗
├━━━━━━━━━━━━━━━━━━━━┤
┃ 📊 30 Günlük Korelasyon: ${correlationPercent === 'Veri yok' ? correlationPercent : `%${correlationPercent}`}
┃ ${trendEmoji}
${data.signal ? `┃ 🎯 Sinyal: ${data.signal}\n` : ''}${data.divergenceStrength != null && !isNaN(data.divergenceStrength) ? `┃ ⚠️ Sapma Gücü: ${data.divergenceStrength.toFixed(2)}\n` : ''}├━━━━━━━━━━━━━━━━━━━━┤
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('BTC-ETH Analysis');

    return { success: true };
  } catch (error: any) {
    recordServiceError('BTC-ETH Analysis', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 1️⃣2️⃣ MARKET CORRELATION DETAIL BRIDGE
// ============================================================================

/**
 * Detaylı market correlation sinyalini Telegram'a gönder (Türkçe)
 */
export async function notifyMarketCorrelationDetail(signal: {
  symbol: string;
  price: number;
  btcCorrelation: number;
  omnipotentScore: number;
  marketPhase: string;
  trend: string;
  signal: string;
  confidence: number;
  fundingBias?: string;
  liquidationRisk?: number;
}): Promise<{ success: boolean; error?: string }> {
  try {
    // NaN kontrolü: price ve btcCorrelation undefined olabilir
    const priceStr =
      signal.price != null && !isNaN(signal.price)
        ? signal.price.toString()
        : 'Veri yok';

    const btcCorrelationPercent =
      signal.btcCorrelation != null && !isNaN(signal.btcCorrelation)
        ? `${(signal.btcCorrelation * 100).toFixed(1)}%`
        : 'Veri yok';

    const tradingSignal: TradingSignal = {
      symbol: signal.symbol,
      action: signal.signal as any,
      confidence: signal.confidence,
      price: priceStr,
      reason: `🔗 BTC Korelasyon: ${btcCorrelationPercent}\n⚡ Omnipotent Skor: ${signal.omnipotentScore}/100\n📊 Faz: ${signal.marketPhase}\n📈 Trend: ${signal.trend}${signal.fundingBias ? `\n💰 Funding: ${signal.fundingBias}` : ''}${signal.liquidationRisk != null && !isNaN(signal.liquidationRisk) ? `\n⚠️ Likidite Risk: ${signal.liquidationRisk}%` : ''}`,
      strategy: '🌐 Market Korelasyon',
      timestamp: new Date().toISOString(),
    };

    await notifyNewSignal(tradingSignal);

    recordServiceSuccess('Market Correlation');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Market Correlation', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// 1️⃣3️⃣ CRYPTO NEWS BRIDGE
// ============================================================================

/**
 * Kripto haberlerini Telegram'a gönder (Türkçe)
 */
export async function notifyCryptoNews(news: {
  title: string;
  titleTR: string;
  descriptionTR: string;
  url: string;
  impactScore: number;
  category: string;
  sentiment: string;
  tags: string[];
}): Promise<{ success: boolean; error?: string }> {
  try {
    const categoryEmoji =
      news.category === 'bitcoin'
        ? '₿'
        : news.category === 'ethereum'
          ? '⟠'
          : news.category === 'regulation'
            ? '⚖️'
            : news.category === 'defi'
              ? '🏦'
              : '📰';

    const sentimentEmoji =
      news.sentiment === 'positive' ? '🟢' : news.sentiment === 'negative' ? '🔴' : '🟡';

    const message = `
╭━━━━━━━━━━━━━━━━━━━━╮
┃ ${categoryEmoji} <b>KRİPTO HABER</b> ${sentimentEmoji}
├━━━━━━━━━━━━━━━━━━━━┤
┃ <b>${news.titleTR}</b>
├━━━━━━━━━━━━━━━━━━━━┤
┃ ${news.descriptionTR.substring(0, 200)}${news.descriptionTR.length > 200 ? '...' : ''}
├━━━━━━━━━━━━━━━━━━━━┤
┃ 🎯 Etki Skoru: ${news.impactScore}/10
┃ 🏷️ ${news.tags.slice(0, 3).join(', ')}
├━━━━━━━━━━━━━━━━━━━━┤
┃ 🔗 <a href="${news.url}">Haberi Oku</a>
┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
╰━━━━━━━━━━━━━━━━━━━━╯
    `.trim();

    await broadcastMessage(message, { parse_mode: 'HTML' });

    recordServiceSuccess('Crypto News');

    return { success: true };
  } catch (error: any) {
    recordServiceError('Crypto News', error.message);
    return { success: false, error: error.message };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  // Signal Sources
  notifyStrategySignal,
  notifyAIBotSignal,
  notifyWhaleAlert,
  notifyTraditionalMarketSignal,
  notifyCorrelationSignal,
  notifyFuturesSignal,

  // New Integrations (Türkçe)
  notifyNirvanaOverview,
  notifyOmnipotentFuturesSignal,
  notifyBTCETHAnalysis,
  notifyMarketCorrelationDetail,
  notifyCryptoNews,

  // Web UI Redirects
  sendWebPushRedirect,
  sendHeaderNotification,
};
