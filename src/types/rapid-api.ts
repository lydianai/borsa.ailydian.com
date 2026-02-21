/**
 * RapidAPI TypeScript Type Definitions
 * - Crypto News API
 * - Harem Altın API
 */

// ============================================
// CRYPTO NEWS API TYPES
// ============================================

export interface RawCryptoNewsItem {
  title: string;
  description?: string;
  url: string;
  image?: string;
  published_at: string;
  source: {
    name: string;
    url?: string;
  };
}

export interface CryptoNewsItemWithTranslation {
  id: string;
  title: string;
  titleTR: string; // Groq AI çevirisi
  description: string;
  descriptionTR: string; // Groq AI çevirisi
  url: string;
  image: string;
  publishedAt: Date;
  source: {
    name: string;
    url?: string;
  };
  // AI Analysis
  impactScore: number; // 1-10 (Groq AI tarafından belirlenir)
  category: 'bitcoin' | 'ethereum' | 'regulation' | 'defi' | 'market' | 'technology' | 'other';
  sentiment: 'positive' | 'negative' | 'neutral';
  tags: string[];
  // Meta
  language: string; // Original language
  timestamp: Date; // Cache için
}

export interface CryptoNewsAPIResponse {
  success: boolean;
  data: CryptoNewsItemWithTranslation[];
  cached: boolean;
  cacheAge?: number; // seconds
  error?: string;
}

// ============================================
// HAREM ALTIN API TYPES
// ============================================

export interface HaremGoldPriceRaw {
  // API response structure (belirsiz, API'den gelince güncelleyeceğiz)
  [key: string]: any;
}

export interface GoldPriceData {
  symbol: string;
  name: string;
  priceTRY: number; // TL fiyat
  priceUSD?: number; // USD fiyat (varsa)
  carat22TRY?: number; // 22 ayar (gram)
  carat24TRY?: number; // 24 ayar (gram)
  change24h: number; // % değişim
  timestamp: Date;
}

export interface HaremGoldAPIResponse {
  success: boolean;
  data: GoldPriceData[];
  error?: string;
}

// ============================================
// GROQ AI TRANSLATION TYPES
// ============================================

export interface GroqTranslationRequest {
  text: string;
  fromLang: string;
  toLang: string;
}

export interface GroqNewsAnalysisRequest {
  title: string;
  description: string;
  source: string;
}

export interface GroqNewsAnalysisResponse {
  titleTR: string;
  descriptionTR: string;
  impactScore: number; // 1-10
  category: string;
  sentiment: 'positive' | 'negative' | 'neutral';
  tags: string[];
  reasoning: string; // Why this score?
}

// ============================================
// CACHE TYPES
// ============================================

export interface CacheEntry<T> {
  data: T;
  timestamp: Date;
  expiresAt: Date;
}

export interface CacheOptions {
  ttl: number; // Time to live (seconds)
  key: string;
}
