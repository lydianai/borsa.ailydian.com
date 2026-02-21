/**
 * 🌾 AGRICULTURAL COMMODITIES ADAPTER
 * Agricultural Products Price Data Integration
 *
 * Features:
 * - 10-minute cache (agricultural prices update slower than crypto)
 * - Commodities API integration (fallback to Alpha Vantage)
 * - TL price conversion with live USD/TRY rate
 * - Error handling with fallback
 * - Wheat, Corn, Soybeans, Coffee, Sugar
 */

interface AgriculturalCommodity {
  symbol: string;
  name: string;
  price: number;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  timestamp: Date;
  category: 'agriculture';
  currency: 'TRY';
  unit: string;
}

interface CacheEntry {
  data: AgriculturalCommodity[];
  timestamp: number;
}

// 10-minute cache for agricultural commodities
const CACHE_DURATION = 10 * 60 * 1000;
let cache: CacheEntry | null = null;

/**
 * Get live USD/TRY exchange rate
 */
async function getUSDTRYRate(): Promise<number> {
  try {
    const response = await fetch('https://api.exchangerate-api.com/v4/latest/USD');
    const data = await response.json();
    return data.rates.TRY || 42.0;
  } catch (error) {
    console.warn('⚠️ [Agriculture] Failed to fetch USD/TRY rate, using fallback 42.0');
    return 42.0;
  }
}

/**
 * Generate mock agricultural commodity prices (fallback when API is not available)
 * Updated with realistic market prices (October 2025)
 */
function generateMockAgriculturePrices(usdTryRate: number): AgriculturalCommodity[] {
  const now = new Date();

  return [
    {
      symbol: 'WHEAT',
      name: 'Buğday',
      priceUSD: 6.50 + (Math.random() * 0.4 - 0.2), // ~$6.30-6.70 per bushel
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 4) - 2, // -2% to +2%
      timestamp: now,
      category: 'agriculture' as const,
      currency: 'TRY' as const,
      unit: 'bushel',
    },
    {
      symbol: 'CORN',
      name: 'Mısır',
      priceUSD: 4.85 + (Math.random() * 0.3 - 0.15), // ~$4.70-5.00 per bushel
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 4) - 2,
      timestamp: now,
      category: 'agriculture' as const,
      currency: 'TRY' as const,
      unit: 'bushel',
    },
    {
      symbol: 'SOYBEAN',
      name: 'Soya Fasulyesi',
      priceUSD: 12.50 + (Math.random() * 0.8 - 0.4), // ~$12.10-12.90 per bushel
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 4) - 2,
      timestamp: now,
      category: 'agriculture' as const,
      currency: 'TRY' as const,
      unit: 'bushel',
    },
    {
      symbol: 'COFFEE',
      name: 'Kahve',
      priceUSD: 2.15 + (Math.random() * 0.2 - 0.1), // ~$2.05-2.25 per lb
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 5) - 2.5, // More volatile
      timestamp: now,
      category: 'agriculture' as const,
      currency: 'TRY' as const,
      unit: 'lb',
    },
    {
      symbol: 'SUGAR',
      name: 'Şeker',
      priceUSD: 0.21 + (Math.random() * 0.02 - 0.01), // ~$0.20-0.22 per lb
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 4) - 2,
      timestamp: now,
      category: 'agriculture' as const,
      currency: 'TRY' as const,
      unit: 'lb',
    },
  ].map(commodity => ({
    ...commodity,
    price: commodity.priceUSD,
    priceTRY: commodity.priceUSD * usdTryRate,
  }));
}

/**
 * Fetch live agricultural commodity prices
 */
export async function fetchAgriculturalCommodities(): Promise<AgriculturalCommodity[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_DURATION) {
    console.log('✅ [Agriculture] Using cached data');
    return cache.data;
  }

  const commoditiesApiKey = process.env.COMMODITIES_API_KEY;
  const alphaVantageApiKey = process.env.ALPHA_VANTAGE_API_KEY;

  // If no API keys configured, use mock data
  if (
    (!commoditiesApiKey || commoditiesApiKey === 'your_commodities_api_key_here') &&
    (!alphaVantageApiKey || alphaVantageApiKey === 'demo' || alphaVantageApiKey === 'your_alpha_vantage_key_here')
  ) {
    console.warn('⚠️ [Agriculture] No API keys configured, using mock data');
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockAgriculturePrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }

  try {
    const usdTryRate = await getUSDTRYRate();
    console.log(`💱 [Agriculture] USD/TRY rate: ${usdTryRate.toFixed(4)}`);

    // Try Commodities API first, fallback to Alpha Vantage
    let results: AgriculturalCommodity[] = [];

    if (commoditiesApiKey && commoditiesApiKey !== 'your_commodities_api_key_here') {
      results = await fetchFromCommoditiesAPI(commoditiesApiKey, usdTryRate);
    }

    // If Commodities API failed, try Alpha Vantage
    if (results.length === 0 && alphaVantageApiKey && alphaVantageApiKey !== 'demo') {
      results = await fetchFromAlphaVantage(alphaVantageApiKey, usdTryRate);
    }

    // If we got at least one successful result, cache it
    if (results.length > 0) {
      cache = {
        data: results,
        timestamp: Date.now(),
      };
      console.log(`✅ [Agriculture] Successfully fetched ${results.length} commodities`);
      return results;
    }

    // All APIs failed, use mock data
    console.warn('⚠️ [Agriculture] All endpoints failed, using mock data as fallback');
    const mockData = generateMockAgriculturePrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  } catch (error) {
    console.error('❌ [Agriculture] Fatal error:', error);
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockAgriculturePrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }
}

/**
 * Fetch from Commodities API
 */
async function fetchFromCommoditiesAPI(
  apiKey: string,
  usdTryRate: number
): Promise<AgriculturalCommodity[]> {
  const commodities = [
    { symbol: 'WHEAT', name: 'Buğday', code: 'WHEAT', unit: 'bushel' },
    { symbol: 'CORN', name: 'Mısır', code: 'CORN', unit: 'bushel' },
    { symbol: 'SOYBEAN', name: 'Soya Fasulyesi', code: 'SOYBEAN', unit: 'bushel' },
    { symbol: 'COFFEE', name: 'Kahve', code: 'COFFEE', unit: 'lb' },
    { symbol: 'SUGAR', name: 'Şeker', code: 'SUGAR', unit: 'lb' },
  ];

  const results: AgriculturalCommodity[] = [];

  for (const commodity of commodities) {
    try {
      const url = `https://commodities-api.com/api/latest?access_key=${apiKey}&symbols=${commodity.code}`;
      console.log(`🔍 [Agriculture] Fetching ${commodity.name} from Commodities API...`);

      const response = await fetch(url, {
        next: { revalidate: 600 },
      });

      if (!response.ok) {
        console.warn(`⚠️ [Agriculture] ${commodity.name} returned ${response.status}`);
        continue;
      }

      const data = await response.json();

      if (data.success && data.rates && data.rates[commodity.code]) {
        const price = data.rates[commodity.code];

        results.push({
          symbol: commodity.symbol,
          name: commodity.name,
          priceUSD: price,
          price,
          priceTRY: price * usdTryRate,
          change24h: (Math.random() * 4) - 2, // API doesn't provide change, use mock
          timestamp: new Date(),
          category: 'agriculture' as const,
          currency: 'TRY' as const,
          unit: commodity.unit,
        });

        console.log(`✅ [Agriculture] ${commodity.name}: $${price.toFixed(2)}`);
      }

      // Small delay to respect rate limits
      await new Promise(resolve => setTimeout(resolve, 200));
    } catch (error) {
      console.error(`❌ [Agriculture] Error fetching ${commodity.name}:`, error);
    }
  }

  return results;
}

/**
 * Fetch from Alpha Vantage (fallback)
 */
async function fetchFromAlphaVantage(
  apiKey: string,
  usdTryRate: number
): Promise<AgriculturalCommodity[]> {
  const commodities = [
    { symbol: 'WHEAT', name: 'Buğday', function: 'WHEAT', unit: 'bushel' },
    { symbol: 'CORN', name: 'Mısır', function: 'CORN', unit: 'bushel' },
    { symbol: 'SOYBEAN', name: 'Soya Fasulyesi', function: 'SOYBEAN', unit: 'bushel' },
    { symbol: 'COFFEE', name: 'Kahve', function: 'COFFEE', unit: 'lb' },
    { symbol: 'SUGAR', name: 'Şeker', function: 'SUGAR', unit: 'lb' },
  ];

  const results: AgriculturalCommodity[] = [];

  for (const commodity of commodities) {
    try {
      const url = `https://www.alphavantage.co/query?function=${commodity.function}&interval=daily&apikey=${apiKey}`;
      console.log(`🔍 [Agriculture] Fetching ${commodity.name} from Alpha Vantage...`);

      const response = await fetch(url, {
        next: { revalidate: 600 },
      });

      if (!response.ok) {
        console.warn(`⚠️ [Agriculture] ${commodity.name} returned ${response.status}`);
        continue;
      }

      const data = await response.json();

      if (data['Note'] || data['Error Message']) {
        console.warn(`⚠️ [Agriculture] API Error: ${data['Note'] || data['Error Message']}`);
        continue;
      }

      if (data.data && Array.isArray(data.data) && data.data.length > 0) {
        const latest = data.data[0];
        const previous = data.data[1];

        const currentPrice = parseFloat(latest.value);
        const previousPrice = previous ? parseFloat(previous.value) : currentPrice;
        const change24h = previousPrice > 0 ? ((currentPrice - previousPrice) / previousPrice) * 100 : 0;

        results.push({
          symbol: commodity.symbol,
          name: commodity.name,
          priceUSD: currentPrice,
          price: currentPrice,
          priceTRY: currentPrice * usdTryRate,
          change24h,
          timestamp: new Date(latest.date),
          category: 'agriculture' as const,
          currency: 'TRY' as const,
          unit: commodity.unit,
        });

        console.log(`✅ [Agriculture] ${commodity.name}: $${currentPrice.toFixed(2)}`);
      }

      // Add delay to respect rate limits
      await new Promise(resolve => setTimeout(resolve, 500));
    } catch (error) {
      console.error(`❌ [Agriculture] Error fetching ${commodity.name}:`, error);
    }
  }

  return results;
}

/**
 * Get agricultural commodities with optional filtering
 */
export async function getAgriculturalCommodities(options?: {
  symbols?: string[];
  minPrice?: number;
  maxPrice?: number;
}): Promise<AgriculturalCommodity[]> {
  let commodities = await fetchAgriculturalCommodities();

  // Apply filters
  if (options?.symbols && options.symbols.length > 0) {
    commodities = commodities.filter(c => options.symbols!.includes(c.symbol));
  }

  if (options?.minPrice) {
    commodities = commodities.filter(c => c.priceTRY >= options.minPrice!);
  }

  if (options?.maxPrice) {
    commodities = commodities.filter(c => c.priceTRY <= options.maxPrice!);
  }

  return commodities;
}

/**
 * Clear cache (useful for testing)
 */
export function clearAgricultureCache(): void {
  cache = null;
  console.log('🗑️ [Agriculture] Cache cleared');
}

/**
 * Get cache status
 */
export function getAgricultureCacheStatus() {
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

export type { AgriculturalCommodity };
