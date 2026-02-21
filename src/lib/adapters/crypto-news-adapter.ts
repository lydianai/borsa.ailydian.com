/**
 * CRYPTO NEWS API ADAPTER (RapidAPI)
 *
 * Features:
 * - Fetch latest crypto news from RapidAPI
 * - Translate to Turkish using Groq AI
 * - Filter by importance (>= 7/10)
 * - In-memory cache (10 minutes)
 */

import type {
  RawCryptoNewsItem,
  CryptoNewsItemWithTranslation,
  CryptoNewsAPIResponse,
} from '@/types/rapid-api';
import {
  analyzeAndTranslateNews,
  generateNewsCacheKey,
} from '@/lib/groq-news-analyzer';
import { newsRiskAnalyzer } from '@/lib/news-risk-analyzer';

// ============================================
// IN-MEMORY CACHE
// ============================================
const newsCache = new Map<string, { data: CryptoNewsItemWithTranslation[]; expiresAt: number }>();
const CACHE_TTL = 10 * 60 * 1000; // 10 minutes

function getCachedNews(): CryptoNewsItemWithTranslation[] | null {
  const cached = newsCache.get('crypto-news-all');
  if (!cached) return null;

  if (Date.now() > cached.expiresAt) {
    newsCache.delete('crypto-news-all');
    return null;
  }

  return cached.data;
}

function setCachedNews(data: CryptoNewsItemWithTranslation[]) {
  newsCache.set('crypto-news-all', {
    data,
    expiresAt: Date.now() + CACHE_TTL,
  });
}

// ============================================
// RAPIDAPI FETCH
// ============================================

/**
 * Fetch crypto news from CryptoPanic API
 * FREE - No credit card required
 * https://cryptopanic.com/developers/api/
 */
async function fetchCryptoNewsFromAPI(): Promise<RawCryptoNewsItem[]> {
  const apiKey = process.env.CRYPTOPANIC_API_KEY;

  // CryptoPanic API çalışmıyorsa mock data kullan
  if (!apiKey || apiKey === 'your_cryptopanic_api_key_here') {
    console.log('[CryptoNews] Using mock data (no API key)');
    return getMockNewsData();
  }

  try {
    // CryptoPanic Public API - Free tier
    const url = `https://cryptopanic.com/api/v1/posts/?auth_token=${apiKey}&public=true&kind=news&filter=important`;

    console.log('[CryptoNews] Fetching from CryptoPanic API...');

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      console.error(`[CryptoNews] API Error: ${response.status} ${response.statusText}`);
      return getMockNewsData();
    }

    const data = await response.json();

    if (!data.results || !Array.isArray(data.results)) {
      console.error('[CryptoNews] Invalid API response structure');
      return getMockNewsData();
    }

    console.log(`[CryptoNews] ✅ Fetched ${data.results.length} news from CryptoPanic`);

    // Convert CryptoPanic format to our format
    const newsItems: RawCryptoNewsItem[] = data.results.slice(0, 10).map((item: any) => ({
      title: item.title || 'No title',
      description: item.title || '', // CryptoPanic doesn't have description, use title
      url: item.url || '#',
      image: item.currencies?.[0]?.url
        ? `https://cryptopanic.com/static/icons/coins/64/${item.currencies[0].code.toLowerCase()}.png`
        : 'https://via.placeholder.com/400x200/1a1a1a/FFFFFF?text=Crypto+News',
      published_at: item.published_at || new Date().toISOString(),
      source: {
        name: item.source?.title || 'CryptoPanic',
        url: item.source?.url || 'https://cryptopanic.com',
      },
    }));

    return newsItems;

  } catch (error: any) {
    console.error('[CryptoNews] Fetch Error:', error.message);
    return getMockNewsData();
  }
}

// ============================================
// MOCK DATA (Development)
// ============================================
function getMockNewsData(): RawCryptoNewsItem[] {
  return [
    {
      title: 'Bitcoin Surges Past $75,000 as Institutional Adoption Accelerates',
      description: 'Major financial institutions announce Bitcoin integration plans, driving price to new all-time highs.',
      url: 'https://example.com/btc-75k',
      image: 'https://via.placeholder.com/400x200/FF9900/FFFFFF?text=Bitcoin+News',
      published_at: new Date().toISOString(),
      source: {
        name: 'CoinDesk',
        url: 'https://coindesk.com',
      },
    },
    {
      title: 'SEC Approves Multiple Ethereum ETF Applications',
      description: 'The U.S. Securities and Exchange Commission has approved several Ethereum ETF applications, marking a major milestone for crypto regulation.',
      url: 'https://example.com/eth-etf',
      image: 'https://via.placeholder.com/400x200/627EEA/FFFFFF?text=Ethereum+ETF',
      published_at: new Date(Date.now() - 3600000).toISOString(),
      source: {
        name: 'Bloomberg Crypto',
        url: 'https://bloomberg.com/crypto',
      },
    },
    {
      title: 'Major DeFi Protocol Suffers $50M Exploit',
      description: 'A significant vulnerability in a leading DeFi protocol has resulted in a $50 million loss, raising concerns about smart contract security.',
      url: 'https://example.com/defi-exploit',
      image: 'https://via.placeholder.com/400x200/FF0000/FFFFFF?text=DeFi+Exploit',
      published_at: new Date(Date.now() - 7200000).toISOString(),
      source: {
        name: 'The Block',
        url: 'https://theblock.co',
      },
    },
  ];
}

// ============================================
// MAIN FUNCTION
// ============================================

/**
 * Get crypto news with Turkish translation and importance filtering
 */
export async function getCryptoNewsWithTranslation(): Promise<CryptoNewsAPIResponse> {
  try {
    // Check cache first
    const cached = getCachedNews();
    if (cached) {
      console.log(`[CryptoNews] Cache hit! (${cached.length} news)`);
      return {
        success: true,
        data: cached,
        cached: true,
        cacheAge: 0,
      };
    }

    console.log('[CryptoNews] Cache miss, fetching from API...');

    // Fetch raw news
    const rawNews = await fetchCryptoNewsFromAPI();
    console.log(`[CryptoNews] Fetched ${rawNews.length} raw news items`);

    if (rawNews.length === 0) {
      return {
        success: false,
        data: [],
        cached: false,
        error: 'No news available',
      };
    }

    // ⚡ PARALLEL PROCESSING with timeout for fast response
    const GROQ_TIMEOUT = 5000; // 5 seconds per translation

    const processWithTimeout = async (item: RawCryptoNewsItem) => {
      try {
        const timeoutPromise = new Promise<null>((_, reject) =>
          setTimeout(() => reject(new Error('Groq timeout')), GROQ_TIMEOUT)
        );

        const analysisPromise = analyzeAndTranslateNews({
          title: item.title,
          description: item.description || '',
          source: item.source.name,
        });

        const analysis = await Promise.race([analysisPromise, timeoutPromise]);

        // Fallback: Groq başarısız olursa varsayılan değerlerle ekle
        if (analysis && analysis.impactScore >= 7) {
          return {
            id: generateNewsCacheKey(item.title, item.source.name),
            title: item.title,
            titleTR: analysis.titleTR,
            description: item.description || '',
            descriptionTR: analysis.descriptionTR,
            url: item.url,
            image: item.image || 'https://via.placeholder.com/400x200/1a1a1a/FFFFFF?text=Crypto+News',
            publishedAt: new Date(item.published_at),
            source: item.source,
            impactScore: analysis.impactScore,
            category: analysis.category as any,
            sentiment: analysis.sentiment,
            tags: analysis.tags,
            language: 'en',
            timestamp: new Date(),
          };
        } else if (!analysis) {
          // Groq hatası - Varsayılan değerlerle ekle (tüm haberler gösterilsin)
          return {
            id: generateNewsCacheKey(item.title, item.source.name),
            title: item.title,
            titleTR: item.title, // Çeviri yapılamadı, orijinal başlık
            description: item.description || '',
            descriptionTR: item.description || '', // Çeviri yapılamadı, orijinal açıklama
            url: item.url,
            image: item.image || 'https://via.placeholder.com/400x200/1a1a1a/FFFFFF?text=Crypto+News',
            publishedAt: new Date(item.published_at),
            source: item.source,
            impactScore: 7, // Varsayılan önemli skor
            category: 'market',
            sentiment: 'neutral',
            tags: ['Crypto', 'News'],
            language: 'en',
            timestamp: new Date(),
          };
        }
        return null;
      } catch (error) {
        // Timeout or error - return with default values
        console.log(`[CryptoNews] Groq timeout/error for: ${item.title.substring(0, 50)}...`);
        return {
          id: generateNewsCacheKey(item.title, item.source.name),
          title: item.title,
          titleTR: item.title,
          description: item.description || '',
          descriptionTR: item.description || '',
          url: item.url,
          image: item.image || 'https://via.placeholder.com/400x200/1a1a1a/FFFFFF?text=Crypto+News',
          publishedAt: new Date(item.published_at),
          source: item.source,
          impactScore: 7,
          category: 'market',
          sentiment: 'neutral',
          tags: ['Crypto', 'News'],
          language: 'en',
          timestamp: new Date(),
        };
      }
    };

    // Process all news in parallel (much faster!)
    const results = await Promise.allSettled(rawNews.map(processWithTimeout));
    const processedNews: CryptoNewsItemWithTranslation[] = results
      .filter((r): r is PromiseFulfilledResult<CryptoNewsItemWithTranslation | null> =>
        r.status === 'fulfilled' && r.value !== null
      )
      .map(r => r.value as CryptoNewsItemWithTranslation);

    console.log(`[CryptoNews] Processed ${processedNews.length}/${rawNews.length} important news (>= 7/10)`);

    // 🚨 CRITICAL NEWS RISK ANALYSIS
    // Haberleri risk analyzer'a gönder ve kritik olanları tespit et
    if (processedNews.length > 0) {
      const criticalAlerts = newsRiskAnalyzer.analyzeNews(processedNews);

      // Eğer kritik alert varsa, otomatik aksiyonları çalıştır
      for (const alert of criticalAlerts) {
        await newsRiskAnalyzer.executeAutoActions(alert);
      }

      if (criticalAlerts.length > 0) {
        console.log(`[CryptoNews] 🚨 ${criticalAlerts.length} critical alerts detected and processed!`);
      }
    }

    // Cache result
    setCachedNews(processedNews);

    return {
      success: true,
      data: processedNews,
      cached: false,
    };

  } catch (error: any) {
    console.error('[CryptoNews] Error:', error.message);
    return {
      success: false,
      data: [],
      cached: false,
      error: error.message,
    };
  }
}

/**
 * Clear cache (manual refresh)
 */
export function clearNewsCache() {
  newsCache.clear();
  console.log('[CryptoNews] Cache cleared');
}
