/**
 * 🏆 HAREM ALTIN API ADAPTER
 * Live Turkish Gold Price Data Integration
 *
 * Features:
 * - 10-minute cache (gold prices update slower than crypto)
 * - RapidAPI integration
 * - TL price formatting
 * - Error handling with fallback
 * - Multi-timeframe support
 */

import type { HaremAltinResponse, FormattedGoldPrice } from '@/types/harem-altin';

// Re-export types for convenience
export type { FormattedGoldPrice };

interface CacheEntry {
  data: FormattedGoldPrice[];
  timestamp: number;
}

// 10-minute cache for gold prices
const CACHE_DURATION = 10 * 60 * 1000;
let cache: CacheEntry | null = null;

/**
 * Generate mock gold prices (fallback when API is not available)
 * Updated with realistic Turkish market prices (Kasım 2024)
 */
function generateMockGoldPrices(): FormattedGoldPrice[] {
  const now = new Date();
  const baseGramPrice = 5950 + (Math.random() * 100 - 50); // ~5900-6000 TL per gram (realistic current price)

  return [
    {
      symbol: 'GRAM_ALTIN',
      name: 'Gram Altın',
      price: baseGramPrice,
      change24h: (Math.random() * 4) - 2, // -2% to +2%
      buyPrice: baseGramPrice * 0.995,
      sellPrice: baseGramPrice * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'CEYREK_ALTIN',
      name: 'Çeyrek Altın',
      price: baseGramPrice * 1.75, // ~10400 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 1.75 * 0.995,
      sellPrice: baseGramPrice * 1.75 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'YARIM_ALTIN',
      name: 'Yarım Altın',
      price: baseGramPrice * 3.5, // ~20825 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 3.5 * 0.995,
      sellPrice: baseGramPrice * 3.5 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'TAM_ALTIN',
      name: 'Tam Altın',
      price: baseGramPrice * 7, // ~41650 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 7 * 0.995,
      sellPrice: baseGramPrice * 7 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'CUMHURIYET',
      name: 'Cumhuriyet Altını',
      price: baseGramPrice * 7.2, // ~42840 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 7.2 * 0.995,
      sellPrice: baseGramPrice * 7.2 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'ATA_ALTIN',
      name: 'Ata Altın',
      price: baseGramPrice * 7.3, // ~43435 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 7.3 * 0.995,
      sellPrice: baseGramPrice * 7.3 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: 'GREMSE_ALTIN',
      name: 'Gremse Altın',
      price: baseGramPrice * 14.5, // ~86275 TL
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 14.5 * 0.995,
      sellPrice: baseGramPrice * 14.5 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: '22_AYAR_BILEZIK',
      name: '22 Ayar Bilezik (gr)',
      price: baseGramPrice * 0.92, // 22 ayar = %92 saflık
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 0.92 * 0.995,
      sellPrice: baseGramPrice * 0.92 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
    {
      symbol: '14_AYAR_ALTIN',
      name: '14 Ayar Altın (gr)',
      price: baseGramPrice * 0.585, // 14 ayar = %58.5 saflık
      change24h: (Math.random() * 4) - 2,
      buyPrice: baseGramPrice * 0.585 * 0.995,
      sellPrice: baseGramPrice * 0.585 * 1.005,
      lastUpdate: now,
      category: 'gold',
      currency: 'TRY',
    },
  ];
}

/**
 * Fetch live gold prices from Harem Altın API
 */
export async function fetchGoldPrices(): Promise<FormattedGoldPrice[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_DURATION) {
    console.log('✅ [Harem Altın] Using cached data');
    return cache.data;
  }

  const apiKey = process.env.RAPIDAPI_KEY;
  const host = process.env.RAPIDAPI_HAREM_HOST || 'harem-altin-live-gold-price-data.p.rapidapi.com';

  if (!apiKey || apiKey === 'your_rapidapi_key_here') {
    console.warn('⚠️ [Harem Altın] RAPIDAPI_KEY not configured, using mock data');
    const mockData = generateMockGoldPrices();
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }

  // Correct endpoint for Harem Altın API
  const endpoints = [
    '/harem_altin/prices',
  ];

  let _lastError: Error | null = null;

  for (const endpoint of endpoints) {
    try {
      console.log(`🔍 [Harem Altın] Trying endpoint: ${endpoint}`);

      const response = await fetch(`https://${host}${endpoint}`, {
        method: 'GET',
        headers: {
          'X-RapidAPI-Key': apiKey,
          'X-RapidAPI-Host': host,
        },
        next: { revalidate: 600 }, // 10 minutes
      });

      if (response.status === 403) {
        console.warn('⚠️ [Harem Altın] API subscription required (403). Using mock data.');
        const mockData = generateMockGoldPrices();
        cache = { data: mockData, timestamp: Date.now() };
        return mockData;
      }

      if (response.status === 404) {
        console.warn(`⚠️ [Harem Altın] ${endpoint} not found (404)`);
        continue;
      }

      if (!response.ok) {
        console.warn(`⚠️ [Harem Altın] ${endpoint} returned ${response.status}`);
        continue;
      }

      const rawData = await response.json();
      console.log('✅ [Harem Altın] Raw API response:', JSON.stringify(rawData).substring(0, 200));

      // Parse and format the data
      const formattedPrices = parseGoldData(rawData);

      if (formattedPrices.length > 0) {
        // Update cache
        cache = {
          data: formattedPrices,
          timestamp: Date.now(),
        };

        console.log(`✅ [Harem Altın] Successfully fetched ${formattedPrices.length} gold prices`);
        return formattedPrices;
      }
    } catch (error) {
      lastError = error as Error;
      console.error(`❌ [Harem Altın] Error on ${endpoint}:`, error);
    }
  }

  console.warn('⚠️ [Harem Altın] All endpoints failed, using mock data as fallback');
  const mockData = generateMockGoldPrices();
  cache = { data: mockData, timestamp: Date.now() };
  return mockData;
}

/**
 * Parse Turkish price format to number
 * Example: "5.855,92" -> 5855.92
 */
function parseTurkishPrice(priceStr: string): number {
  if (!priceStr) return 0;
  // Remove thousand separators (dots) and replace decimal comma with dot
  const cleanPrice = priceStr.replace(/\./g, '').replace(',', '.');
  return parseFloat(cleanPrice) || 0;
}

/**
 * Parse gold price data from Harem Altın API response
 * Format: { success: true, data: [{key, buy, sell, percent, arrow, last_update}] }
 */
function parseGoldData(data: any): FormattedGoldPrice[] {
  const prices: FormattedGoldPrice[] = [];

  try {
    // Harem Altın API format
    if (data.success && Array.isArray(data.data)) {
      data.data.forEach((item: any) => {
        const key = item.key;
        const buyPrice = parseTurkishPrice(item.buy);
        const sellPrice = parseTurkishPrice(item.sell);
        const changePercent = parseFloat(item.percent) || 0;

        // Only include gold products (exclude silver, platinum, etc.)
        const goldKeywords = ['ALTIN', 'GRAM', 'ÇEYREK', 'YARIM', 'TAM', 'ATA', 'GREMSE', '14 AYAR', '22 AYAR', 'Has Altın'];
        const isGold = goldKeywords.some(keyword => key.includes(keyword));

        if (isGold && buyPrice > 0 && sellPrice > 0) {
          // Use sell price as the main price (more relevant for buyers)
          const price = sellPrice;

          prices.push({
            symbol: key.toUpperCase().replace(/\s+/g, '_'),
            name: key,
            price: price,
            change24h: changePercent,
            buyPrice: buyPrice,
            sellPrice: sellPrice,
            lastUpdate: new Date(),
            category: 'gold',
            currency: 'TRY',
          });
        }
      });
    }
  } catch (error) {
    console.error('❌ [Harem Altın] Error parsing data:', error);
  }

  console.log(`[Harem Altın] Parsed ${prices.length} gold products from API`);
  return prices;
}


/**
 * Get gold prices with optional filtering
 */
export async function getGoldPrices(options?: {
  symbols?: string[];
  minPrice?: number;
  maxPrice?: number;
}): Promise<FormattedGoldPrice[]> {
  let prices = await fetchGoldPrices();

  // Apply filters
  if (options?.symbols && options.symbols.length > 0) {
    prices = prices.filter(p => options.symbols!.includes(p.symbol));
  }

  if (options?.minPrice) {
    prices = prices.filter(p => p.price >= options.minPrice!);
  }

  if (options?.maxPrice) {
    prices = prices.filter(p => p.price <= options.maxPrice!);
  }

  return prices;
}

/**
 * Clear cache (useful for testing)
 */
export function clearGoldCache(): void {
  cache = null;
  console.log('🗑️ [Harem Altın] Cache cleared');
}

/**
 * Get cache status
 */
export function getGoldCacheStatus() {
  if (!cache) {
    return { cached: false, age: 0 };
  }

  const age = Date.now() - cache.timestamp;
  return {
    cached: true,
    age,
    remaining: Math.max(0, CACHE_DURATION - age),
    count: cache.data.length,
  };
}
