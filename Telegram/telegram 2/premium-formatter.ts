/**
 * 💎 TELEGRAM PREMIUM FORMATTER
 * Ultra-compact, renkli, premium modern tasarım
 *
 * Features:
 * - Kompakt kare format
 * - Renk vurgusu (karakter yoğunluğu)
 * - HTML formatı
 * - Premium Unicode ikonlar
 * - 600+ coin + traditional markets desteği
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational purposes only
 * - No trading operations
 */

import type { TradingSignal } from './notifications';
import { getSignalEmoji, getConfidenceStars, type SignalType } from './config';

// ============================================================================
// ULTRA-COMPACT DESIGN PALETTE (Renkli - Karakter Yoğunluğu)
// ============================================================================

interface ColorScheme {
  icon: string; // Modern Unicode ikon
  gradient: string; // Gradient efekt (KOMPAKT)
  border: string; // Kenar çizgisi
  headerBg: string; // Header arka plan (renk simülasyonu)
  indicator: string; // Trend göstergesi
  bullet: string; // Liste bullet
}

const COMPACT_SCHEMES: Partial<Record<SignalType, ColorScheme>> = {
  STRONG_BUY: {
    icon: '◆',
    gradient: '━━━━━━━━━━━━━━',
    border: '┃',
    headerBg: '🟢', // Yeşil emoji (güçlü alım)
    indicator: '↗↗',
    bullet: '▸',
  },
  BUY: {
    icon: '◇',
    gradient: '━━━━━━━━━━━━━━',
    border: '│',
    headerBg: '🟢', // Yeşil emoji (alım)
    indicator: '↗',
    bullet: '▹',
  },
  SELL: {
    icon: '◈',
    gradient: '━━━━━━━━━━━━━━',
    border: '┃',
    headerBg: '🔴', // Kırmızı emoji (satım)
    indicator: '↘↘',
    bullet: '▸',
  },
  WAIT: {
    icon: '◊',
    gradient: '━━━━━━━━━━━━━━',
    border: '│',
    headerBg: '🟡', // Sarı emoji (bekleme)
    indicator: '↔',
    bullet: '▹',
  },
  NEUTRAL: {
    icon: '○',
    gradient: '━━━━━━━━━━━━━━',
    border: '│',
    headerBg: '⚪', // Beyaz emoji (nötr)
    indicator: '→',
    bullet: '▹',
  },
};

// ============================================================================
// COMPACT UNICODE ART COMPONENTS
// ============================================================================

/**
 * Get scheme with fallback to NEUTRAL
 */
function getScheme(signalType: SignalType): ColorScheme {
  return COMPACT_SCHEMES[signalType] || COMPACT_SCHEMES.NEUTRAL!;
}

/**
 * Kompakt header bar
 */
function createCompactHeader(signalType: SignalType): string {
  const scheme = getScheme(signalType);
  return `╭${scheme.gradient}╮`;
}

/**
 * Kompakt footer bar
 */
function createCompactFooter(signalType: SignalType): string {
  const scheme = getScheme(signalType);
  return `╰${scheme.gradient}╯`;
}

/**
 * Kompakt divider
 */
function createCompactDivider(signalType: SignalType): string {
  const scheme = getScheme(signalType);
  return `├${scheme.gradient}┤`;
}

// ============================================================================
// COMPACT SIGNAL LABELS (Renkli Başlıklar)
// ============================================================================

const COMPACT_LABELS: Partial<Record<SignalType, string>> = {
  STRONG_BUY: 'GÜÇLÜ ALIM',
  BUY: 'ALIM SİNYALİ',
  SELL: 'SATIM SİNYALİ',
  WAIT: 'BEKLEME',
  NEUTRAL: 'NÖTR',
};

// ============================================================================
// CONFIDENCE VISUALIZATION
// ============================================================================

/**
 * Ultra-modern confidence bar (geometric shapes)
 */
function createConfidenceBar(confidence: number): string {
  const filled = Math.round(confidence / 10);
  const empty = 10 - filled;

  // Modern geometric progress bar
  return '■'.repeat(filled) + '□'.repeat(empty);
}

/**
 * Ultra-modern confidence label (no emojis)
 */
function getConfidenceLabel(confidence: number): string {
  if (confidence >= 90) return '◆ MAXIMUM';
  if (confidence >= 80) return '▲ YÜKSEK';
  if (confidence >= 70) return '● GÜÇLÜ';
  if (confidence >= 60) return '▸ ORTA';
  return '▹ DÜŞÜK';
}

/**
 * Ultra-modern confidence stars (geometric shapes)
 */
function getModernStars(confidence: number): string {
  if (confidence >= 90) return '◆◆◆◆◆';
  if (confidence >= 80) return '◆◆◆◆◇';
  if (confidence >= 70) return '◆◆◆◇◇';
  if (confidence >= 60) return '◆◆◇◇◇';
  return '◆◇◇◇◇';
}

// ============================================================================
// MARKET TYPE DETECTION
// ============================================================================

/**
 * Ultra-modern piyasa tipi algılama (Unicode ikonlar)
 */
function detectMarketType(symbol: string): {
  type: 'crypto' | 'forex' | 'stock' | 'commodity' | 'index';
  icon: string;
  label: string;
} {
  // Crypto (USDT, BUSD, BTC pairs)
  if (symbol.includes('USDT') || symbol.includes('BUSD') || symbol.includes('BTC')) {
    return { type: 'crypto', icon: '₿', label: 'CRYPTO' };
  }

  // Forex (currency pairs)
  const forexPairs = ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD'];
  if (forexPairs.some((pair) => symbol.includes(pair))) {
    return { type: 'forex', icon: '¤', label: 'FOREX' }; // Currency sign
  }

  // Stock indices
  const indices = ['SPX', 'NDX', 'DJI', 'FTSE', 'DAX', 'NIKKEI'];
  if (indices.some((idx) => symbol.includes(idx))) {
    return { type: 'index', icon: '∑', label: 'INDEX' }; // Sigma (sum)
  }

  // Commodities
  const commodities = ['GOLD', 'SILVER', 'OIL', 'GAS'];
  if (commodities.some((com) => symbol.toUpperCase().includes(com))) {
    return { type: 'commodity', icon: '◉', label: 'COMMODITY' };
  }

  // Default: stock
  return { type: 'stock', icon: '∆', label: 'STOCK' }; // Delta (change)
}

// ============================================================================
// ULTRA-COMPACT PREMIUM FORMATTER (Kare Format)
// ============================================================================

/**
 * Ultra-kompakt sinyal formatı
 * Kare layout, renkli başlıklar, HTML format
 */
export function formatPremiumSignal(signal: TradingSignal): string {
  const signalType = signal.action as SignalType;
  const scheme = getScheme(signalType);
  const market = detectMarketType(signal.symbol);
  const confidenceBar = createConfidenceBar(signal.confidence);
  const confidenceLabel = getConfidenceLabel(signal.confidence);
  const modernStars = getModernStars(signal.confidence);

  // Time formatting (kompakt)
  const time = new Date(signal.timestamp).toLocaleString('tr-TR', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });

  // Reason/Strategy (ilk 3 satır)
  const topIndicators = signal.reason
    ? signal.reason
        .split('\n')
        .slice(0, 3)
        .map((line) => `${scheme.bullet} ${line}`)
        .join('\n')
    : '';

  // Ultra-kompakt HTML formatı (Renkli Başlık)
  const parts = [
    // ═══════ HEADER (Renkli Emoji) ═══════
    createCompactHeader(signalType),
    `${scheme.border} ${scheme.headerBg} <b>${COMPACT_LABELS[signalType] || signalType}</b> ${scheme.headerBg}`,
    createCompactDivider(signalType),

    // ═══════ MARKET INFO ═══════
    `${scheme.border} ${market.icon} <code>${signal.symbol}</code> ${scheme.indicator}`,
    `${scheme.border} $ <b>${signal.price}</b>`,
    createCompactDivider(signalType),

    // ═══════ CONFIDENCE ═══════
    `${scheme.border} ◎ ${signal.confidence}% ${modernStars}`,
    `${scheme.border} ${confidenceBar} ${confidenceLabel}`,

    // ═══════ TOP INDICATORS ═══════
    topIndicators
      ? [createCompactDivider(signalType), `${scheme.border} <i>EN GÜÇLÜ:</i>`, topIndicators].join('\n')
      : '',

    // ═══════ FOOTER ═══════
    createCompactDivider(signalType),
    `${scheme.border} ⌚ ${time}`,
    createCompactFooter(signalType),

    // ═══════ LINK & DISCLAIMER ═══════
    '',
    `<a href="${process.env.NEXT_PUBLIC_APP_URL || 'https://sardag.app'}/trading-signals">⟫ Detaylı Analiz</a>`,
    `<i>※ Eğitim amaçlı</i>`,
  ];

  return parts.filter(Boolean).join('\n');
}

// ============================================================================
// COMPACT DAILY SUMMARY FORMATTER
// ============================================================================

/**
 * Ultra-kompakt günlük özet formatı
 */
export function formatPremiumDailySummary(summary: {
  totalSignals: number;
  strongBuyCount: number;
  buyCount: number;
  sellCount: number;
  waitCount: number;
  topSignals: TradingSignal[];
  date: Date;
}): string {
  // Kompakt tarih formatı
  const dateStr = summary.date.toLocaleDateString('tr-TR', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });

  const parts = [
    // Header (Renkli)
    '╭━━━━━━━━━━━━━━━━━━━╮',
    '│ 📊 <b>GÜNLÜK ÖZET</b> 📊 │',
    '├━━━━━━━━━━━━━━━━━━━┤',
    `│ ⌚ ${dateStr}       │`,
    '├━━━━━━━━━━━━━━━━━━━┤',

    // Statistics (renkli emoji)
    `│ ◎ Toplam: ${summary.totalSignals}         │`,
    `│ 🟢 Güçlü Alım: ${summary.strongBuyCount}      │`,
    `│ 🟢 Alım: ${summary.buyCount}             │`,
    `│ 🔴 Satım: ${summary.sellCount}            │`,
    `│ 🟡 Bekleme: ${summary.waitCount}          │`,

    // Top 3 Signals (kompakt)
    summary.topSignals.length > 0
      ? [
          '├━━━━━━━━━━━━━━━━━━━┤',
          '│ <i>EN İYİ 3:</i>           │',
          ...summary.topSignals.slice(0, 3).map((s, i) => {
            const scheme = getScheme(s.action as SignalType);
            return `│ ${i + 1}. ${scheme.icon} ${s.symbol} ${scheme.indicator} ${s.confidence}%`;
          }),
        ].join('\n')
      : '',

    // Footer
    '╰━━━━━━━━━━━━━━━━━━━╯',
    '',
    `<a href="${process.env.NEXT_PUBLIC_APP_URL || 'https://sardag.app'}">⟫ Tüm Sinyaller</a>`,
    '<i>※ Eğitim amaçlı</i>',
  ];

  return parts.filter(Boolean).join('\n');
}

// ============================================================================
// ⚠️ SYSTEM ERROR & HEALTH NOTIFICATION FORMATTER
// ============================================================================

/**
 * Sistem hatası/uyarısı için özel format
 * Arka plan servisleri, API hataları, analiz sorunları vb.
 */
export function formatSystemError(error: {
  type: 'error' | 'warning' | 'info';
  service: string;
  message: string;
  details?: string;
  timestamp?: Date;
}): string {
  const typeEmoji = error.type === 'error' ? '⚠️' : error.type === 'warning' ? '🟡' : 'ℹ️';
  const typeLabel =
    error.type === 'error' ? 'SYSTEM ERROR' : error.type === 'warning' ? 'WARNING' : 'INFO';

  const time = error.timestamp
    ? new Date(error.timestamp).toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      })
    : new Date().toLocaleString('tr-TR', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      });

  const parts = [
    '╭━━━━━━━━━━━━━━╮',
    `┃ ${typeEmoji} <b>${typeLabel}</b> ${typeEmoji}`,
    '├━━━━━━━━━━━━━━┤',
    `┃ 🔧 <b>${error.service}</b>`,
    `┃ ${error.message}`,
  ];

  if (error.details) {
    parts.push('├━━━━━━━━━━━━━━┤');
    parts.push(`┃ <i>${error.details}</i>`);
  }

  parts.push('├━━━━━━━━━━━━━━┤');
  parts.push(`┃ ⌚ ${time}`);
  parts.push('╰━━━━━━━━━━━━━━╯');

  return parts.join('\n');
}

/**
 * Arka plan servis hatası için bildirim
 */
export function formatBackgroundServiceError(service: {
  name: string;
  error: string;
  lastSuccessTime?: Date;
}): string {
  return formatSystemError({
    type: 'error',
    service: service.name,
    message: 'Servis çalışmıyor',
    details: `Hata: ${service.error}${service.lastSuccessTime ? `\nSon başarılı: ${service.lastSuccessTime.toLocaleString('tr-TR')}` : ''}`,
  });
}

/**
 * API hatası için bildirim
 */
export function formatAPIError(api: { endpoint: string; error: string; statusCode?: number }): string {
  return formatSystemError({
    type: 'error',
    service: 'API',
    message: `${api.endpoint} başarısız`,
    details: `${api.statusCode ? `HTTP ${api.statusCode}: ` : ''}${api.error}`,
  });
}

/**
 * Analiz hatası için bildirim
 */
export function formatAnalysisError(analysis: {
  strategy: string;
  symbol: string;
  error: string;
}): string {
  return formatSystemError({
    type: 'warning',
    service: 'Analysis Engine',
    message: `${analysis.strategy} - ${analysis.symbol}`,
    details: `Analiz başarısız: ${analysis.error}`,
  });
}

/**
 * Data kalite uyarısı
 */
export function formatDataQualityWarning(warning: {
  source: string;
  issue: string;
  affectedSymbols?: string[];
}): string {
  return formatSystemError({
    type: 'warning',
    service: 'Data Quality',
    message: `${warning.source} - Veri sorunu`,
    details: `${warning.issue}${warning.affectedSymbols ? `\nEtkilenen: ${warning.affectedSymbols.slice(0, 5).join(', ')}${warning.affectedSymbols.length > 5 ? '...' : ''}` : ''}`,
  });
}

/**
 * Sistem sağlık raporu (her şey OK)
 */
export function formatSystemHealthy(services: string[]): string {
  return [
    '╭━━━━━━━━━━━━━━╮',
    '┃ ✅ <b>SYSTEM HEALTHY</b> ✅',
    '├━━━━━━━━━━━━━━┤',
    `┃ ${services.length} servis aktif`,
    '├━━━━━━━━━━━━━━┤',
    ...services.slice(0, 5).map((s) => `┃ ✅ ${s}`),
    services.length > 5 ? `┃ ... ve ${services.length - 5} diğer` : '',
    '├━━━━━━━━━━━━━━┤',
    `┃ ⌚ ${new Date().toLocaleString('tr-TR', { hour: '2-digit', minute: '2-digit' })}`,
    '╰━━━━━━━━━━━━━━╯',
  ]
    .filter(Boolean)
    .join('\n');
}

export default {
  formatPremiumSignal,
  formatPremiumDailySummary,
  detectMarketType,
  createConfidenceBar,
  getConfidenceLabel,
  // System & Error Formatters
  formatSystemError,
  formatBackgroundServiceError,
  formatAPIError,
  formatAnalysisError,
  formatDataQualityWarning,
  formatSystemHealthy,
};
