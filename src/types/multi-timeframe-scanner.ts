/**
 * 🔍 MULTI-TIMEFRAME SCANNER - TYPE DEFINITIONS
 *
 * Complete TypeScript interfaces for multi-timeframe market scanning system
 * - Indicator validation results
 * - Candlestick pattern recognition
 * - Signal scoring and consensus
 * - Real-time notifications
 *
 * WHITE-HAT PRINCIPLES:
 * - Type-safe operations
 * - Transparent scoring logic
 * - Educational purpose only
 */

// ============================================================================
// TIMEFRAME TYPES
// ============================================================================

export type Timeframe = '1h' | '4h' | '1d';

export interface TimeframeConfig {
  interval: Timeframe;
  label: string;
  weight: number; // Importance in consensus (1h=30%, 4h=70%)
}

// ============================================================================
// CANDLESTICK PATTERN TYPES
// ============================================================================

export type CandlestickPattern =
  // Yükseliş Formasyonları (Bullish)
  | 'hammer'              // Çekiç
  | 'inverted_hammer'     // Ters Çekiç
  | 'bullish_engulfing'   // Yükseliş Yutma
  | 'morning_star'        // Sabah Yıldızı
  | 'three_white_soldiers' // Üç Beyaz Asker
  | 'piercing_line'       // Delici Hat
  | 'bullish_harami'      // Yükseliş Harami
  | 'tweezer_bottom'      // Makas Dibi

  // Düşüş Formasyonları (Bearish)
  | 'shooting_star'       // Kayan Yıldız
  | 'bearish_engulfing'   // Düşüş Yutma
  | 'evening_star'        // Akşam Yıldızı
  | 'three_black_crows'   // Üç Kara Karga
  | 'dark_cloud_cover'    // Kara Bulut Örtüsü
  | 'bearish_harami'      // Düşüş Harami
  | 'tweezer_top';        // Makas Tepe

export interface PatternDetection {
  pattern: CandlestickPattern;
  name: string;           // Türkçe isim
  direction: 'bullish' | 'bearish';
  confidence: number;     // 0-100
  position: number;       // Kaç mum önce oluştu (0 = en son mum)
  description: string;    // Türkçe açıklama
}

// ============================================================================
// INDICATOR VALIDATION TYPES
// ============================================================================

export interface RSIValidation {
  value: number;
  zone: 'oversold' | 'neutral' | 'overbought'; // <30, 30-70, >70
  divergence?: 'bullish' | 'bearish';
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface MFIValidation {
  value: number;
  zone: 'oversold' | 'neutral' | 'overbought';
  moneyFlowTrend: 'increasing' | 'decreasing' | 'stable';
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface BollingerValidation {
  upper: number;
  middle: number;
  lower: number;
  currentPrice: number;
  position: 'above_upper' | 'near_upper' | 'middle' | 'near_lower' | 'below_lower';
  bandwidth: number;       // Genişlik (volatilite göstergesi)
  squeeze: boolean;        // Daralma var mı?
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface VWAPValidation {
  value: number;
  currentPrice: number;
  position: 'above' | 'at' | 'below';
  deviation: number;       // VWAP'tan sapma %
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface FVGValidation {
  gaps: Array<{
    high: number;
    low: number;
    type: 'bullish' | 'bearish';
    filled: boolean;       // Gap dolduruldu mu?
    age: number;           // Kaç mum önce oluştu
  }>;
  nearestGap?: {
    distance: number;      // Mevcut fiyata uzaklık %
    type: 'bullish' | 'bearish';
  };
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface OrderBlockValidation {
  blocks: Array<{
    high: number;
    low: number;
    type: 'bullish' | 'bearish';
    tested: boolean;       // Test edildi mi?
    strength: number;      // 0-100
    age: number;
  }>;
  nearestBlock?: {
    distance: number;
    type: 'bullish' | 'bearish';
    strength: number;
  };
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface LiquidityValidation {
  pools: Array<{
    price: number;
    type: 'buy_side' | 'sell_side';
    volume: number;
    distance: number;      // Mevcut fiyata uzaklık %
  }>;
  nearestPool?: {
    type: 'buy_side' | 'sell_side';
    distance: number;
    volume: number;
  };
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface SupportResistanceValidation {
  levels: Array<{
    price: number;
    type: 'support' | 'resistance';
    strength: number;      // 0-100 (kaç kez test edildi)
    distance: number;      // Mevcut fiyata uzaklık %
  }>;
  nearestSupport?: number;
  nearestResistance?: number;
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface FibonacciValidation {
  levels: {
    '0': number;
    '23.6': number;
    '38.2': number;
    '50': number;
    '61.8': number;
    '78.6': number;
    '100': number;
  };
  currentLevel?: '0' | '23.6' | '38.2' | '50' | '61.8' | '78.6' | '100';
  nearestLevel?: {
    level: string;
    price: number;
    distance: number;
  };
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface PremiumDiscountValidation {
  equilibrium: number;    // %50 seviyesi
  currentPrice: number;
  zone: 'premium' | 'equilibrium' | 'discount'; // >50%, ~50%, <50%
  percentage: number;     // Hangi seviyede (0-100%)
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

export interface MarketStructureValidation {
  trend: 'bullish' | 'bearish' | 'ranging';
  higherHighs: number;    // Yükselen zirveler sayısı
  higherLows: number;     // Yükselen dipler sayısı
  lowerHighs: number;     // Düşen zirveler sayısı
  lowerLows: number;      // Düşen dipler sayısı
  choch?: {               // Change of Character
    detected: boolean;
    type: 'bullish' | 'bearish';
    position: number;
  };
  bos?: {                 // Break of Structure
    detected: boolean;
    type: 'bullish' | 'bearish';
    position: number;
  };
  signal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;
  reason: string;
}

// ============================================================================
// COMPREHENSIVE INDICATOR ANALYSIS
// ============================================================================

export interface IndicatorAnalysis {
  rsi: RSIValidation;
  mfi: MFIValidation;
  bollinger: BollingerValidation;
  vwap: VWAPValidation;
  fvg: FVGValidation;
  orderBlocks: OrderBlockValidation;
  liquidity: LiquidityValidation;
  supportResistance: SupportResistanceValidation;
  fibonacci: FibonacciValidation;
  premiumDiscount: PremiumDiscountValidation;
  marketStructure: MarketStructureValidation;
}

// ============================================================================
// MULTI-TIMEFRAME ANALYSIS
// ============================================================================

export interface TimeframeAnalysis {
  timeframe: Timeframe;
  timestamp: number;

  // Fiyat Verileri
  price: {
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  };

  // İndikatör Analizi
  indicators: IndicatorAnalysis;

  // Mum Formasyonları
  patterns: PatternDetection[];

  // Genel Sinyal
  overallSignal: 'BUY' | 'SELL' | 'NEUTRAL';
  confidence: number;        // 0-100

  // BUY sinyali veren indikatör sayısı
  buyIndicatorCount: number;
  totalIndicators: number;

  // BUY formasyonu sayısı
  bullishPatternCount: number;
}

export interface MultiTimeframeAnalysis {
  symbol: string;
  timestamp: number;

  // Her timeframe için analiz
  timeframes: {
    '1h': TimeframeAnalysis;
    '4h': TimeframeAnalysis;
  };

  // Konsensüs Skoru (ağırlıklı ortalama)
  consensusScore: number;    // 0-100

  // LONG sinyali uygunluğu
  longSignalQuality: 'excellent' | 'good' | 'moderate' | 'poor' | 'none';

  // Gereksinimler karşılandı mı?
  requirements: {
    allIndicatorsBuy: boolean;        // Tüm indikatörler BUY mu?
    bullishPatternsPresent: boolean;  // Yükseliş formasyonları var mı?
    multiTimeframeConfirm: boolean;   // Çoklu zaman dilimi onayı var mı?
    minimumConfidence: boolean;       // Minimum güven eşiği aşıldı mı?
  };

  // Bildirim gönderilmeli mi?
  shouldNotify: boolean;

  // Özet açıklama (Türkçe)
  summary: string;
}

// ============================================================================
// SCANNER CONFIGURATION
// ============================================================================

export interface ScannerConfig {
  // Tarama ayarları
  symbols: string[];                 // Taranacak semboller (ör: ['BTCUSDT', 'ETHUSDT'])
  timeframes: Timeframe[];           // Analiz edilecek zaman dilimleri
  interval: number;                  // Tarama aralığı (ms) - varsayılan 60000 (1 dakika)

  // Filtre kriterleri
  filters: {
    minConsensusScore: number;       // Minimum konsensüs skoru (0-100)
    requireAllIndicators: boolean;   // Tüm indikatörler BUY olmalı mı?
    requirePatterns: boolean;        // Mum formasyonu şart mı?
    minBullishPatterns: number;      // Minimum yükseliş formasyonu sayısı
  };

  // Bildirim ayarları
  notifications: {
    enabled: boolean;
    channels: ('push' | 'email' | 'webhook')[];
    cooldown: number;                // Aynı sembol için bildirim aralığı (ms)
  };

  // Rate limiting
  rateLimit: {
    maxRequestsPerMinute: number;
    batchSize: number;
    batchDelay: number;
  };
}

// ============================================================================
// SCANNER STATE & RESULTS
// ============================================================================

export interface ScannerState {
  isRunning: boolean;
  startedAt?: number;
  lastScanAt?: number;
  totalScans: number;
  totalSignalsFound: number;
  errors: number;
}

export interface ScanResult {
  timestamp: number;
  symbol: string;
  analysis: MultiTimeframeAnalysis;
  notificationSent: boolean;
}

export interface ScannerStats {
  scannedSymbols: number;
  qualifiedSignals: number;
  notificationsSent: number;
  averageScanTime: number;
  lastError?: string;
}

// ============================================================================
// NOTIFICATION PAYLOAD
// ============================================================================

export interface SignalNotification {
  type: 'LONG_SIGNAL';
  symbol: string;
  timestamp: number;

  data: {
    consensusScore: number;
    timeframes: string[];            // ['1h', '4h']
    indicators: string[];            // BUY sinyali veren indikatörler
    patterns: string[];              // Tespit edilen formasyonlar (Türkçe)
    summary: string;                 // Kısa özet (Türkçe)
  };

  priority: 'high' | 'medium' | 'low';
}

// ============================================================================
// CACHE TYPES
// ============================================================================

export interface CachedAnalysis {
  symbol: string;
  timeframe: Timeframe;
  analysis: TimeframeAnalysis;
  cachedAt: number;
  expiresAt: number;
}

export interface ScannerCache {
  analyses: Map<string, CachedAnalysis>; // key: `${symbol}_${timeframe}`
  notifications: Map<string, number>;    // key: symbol, value: last notification timestamp
}
