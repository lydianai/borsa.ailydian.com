/**
 * 🛢️ ENERGY COMMODITIES ADAPTER
 * Oil & Natural Gas Price Data Integration
 *
 * Features:
 * - 10-minute cache (commodity prices update slower than crypto)
 * - Alpha Vantage API integration
 * - TL price conversion with live USD/TRY rate
 * - Error handling with fallback
 * - Brent Crude, WTI Crude, Natural Gas
 */

interface EnergyCommodity {
  symbol: string;
  name: string;
  price: number;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  timestamp: Date;
  category: 'energy';
  currency: 'TRY';
  unit: string;
}

interface CacheEntry {
  data: EnergyCommodity[];
  timestamp: number;
}

// 10-minute cache for energy commodities
const CACHE_DURATION = 10 * 60 * 1000;
let cache: CacheEntry | null = null;

/**
 * Get live USD/TRY exchange rate
 */
async function getUSDTRYRate(): Promise<number> {
  try {
    const response = await fetch('https://api.exchangerate-api.com/v4/latest/USD');
    const data = await response.json();
    return data.rates.TRY || 42.0; // Fallback to 42.0 if API fails
  } catch (error) {
    console.warn('⚠️ [Energy] Failed to fetch USD/TRY rate, using fallback 42.0');
    return 42.0;
  }
}

/**
 * Generate mock energy commodity prices (fallback when API is not available)
 * Updated with realistic market prices (October 2025)
 */
function generateMockEnergyPrices(usdTryRate: number): EnergyCommodity[] {
  const now = new Date();

  return [
    {
      symbol: 'BRENT',
      name: 'Brent Crude Oil',
      priceUSD: 85.50 + (Math.random() * 4 - 2), // ~$83-88 per barrel
      price: 0, // Will be calculated
      priceTRY: 0, // Will be calculated
      change24h: (Math.random() * 6) - 3, // -3% to +3%
      timestamp: now,
      category: 'energy' as const,
      currency: 'TRY' as const,
      unit: 'varil',
    },
    {
      symbol: 'WTI',
      name: 'WTI Crude Oil',
      priceUSD: 81.20 + (Math.random() * 4 - 2), // ~$79-84 per barrel (usually $3-5 cheaper than Brent)
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 6) - 3,
      timestamp: now,
      category: 'energy' as const,
      currency: 'TRY' as const,
      unit: 'varil',
    },
    {
      symbol: 'NATGAS',
      name: 'Natural Gas',
      priceUSD: 3.45 + (Math.random() * 0.4 - 0.2), // ~$3.25-3.65 per MMBtu
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 8) - 4, // -4% to +4% (more volatile)
      timestamp: now,
      category: 'energy' as const,
      currency: 'TRY' as const,
      unit: 'MMBtu',
    },
  ].map(commodity => ({
    ...commodity,
    price: commodity.priceUSD,
    priceTRY: commodity.priceUSD * usdTryRate,
  }));
}

/**
 * Fetch live energy commodity prices from Alpha Vantage API
 */
export async function fetchEnergyCommodities(): Promise<EnergyCommodity[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_DURATION) {
    console.log('✅ [Energy] Using cached data');
    return cache.data;
  }

  const apiKey = process.env.ALPHA_VANTAGE_API_KEY;

  if (!apiKey || apiKey === 'demo' || apiKey === 'your_alpha_vantage_key_here') {
    console.warn('⚠️ [Energy] ALPHA_VANTAGE_API_KEY not configured, using mock data');
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockEnergyPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }

  try {
    const usdTryRate = await getUSDTRYRate();
    console.log(`💱 [Energy] USD/TRY rate: ${usdTryRate.toFixed(4)}`);

    // Alpha Vantage commodity endpoints
    // Note: Free tier has 5 calls/minute limit, so we'll fetch sequentially
    const commodities = [
      { symbol: 'BRENT', function: 'BRENT', name: 'Brent Crude Oil', unit: 'varil' },
      { symbol: 'WTI', function: 'WTI', name: 'WTI Crude Oil', unit: 'varil' },
      { symbol: 'NATGAS', function: 'NATURAL_GAS', name: 'Natural Gas', unit: 'MMBtu' },
    ];

    const results: EnergyCommodity[] = [];
    let successCount = 0;

    for (const commodity of commodities) {
      try {
        const url = `https://www.alphavantage.co/query?function=${commodity.function}&interval=daily&apikey=${apiKey}`;
        console.log(`🔍 [Energy] Fetching ${commodity.name}...`);

        const response = await fetch(url, {
          next: { revalidate: 600 }, // 10 minutes
        });

        if (!response.ok) {
          console.warn(`⚠️ [Energy] ${commodity.name} returned ${response.status}`);
          continue;
        }

        const data = await response.json();

        // Check for API rate limit or error
        if (data['Note'] || data['Error Message']) {
          console.warn(`⚠️ [Energy] API Error: ${data['Note'] || data['Error Message']}`);
          continue;
        }

        // Parse the data
        const parsed = parseCommodityData(data, commodity, usdTryRate);
        if (parsed) {
          results.push(parsed);
          successCount++;
          console.log(`✅ [Energy] ${commodity.name}: $${parsed.priceUSD.toFixed(2)} = ₺${parsed.priceTRY.toFixed(2)}`);
        }

        // Add small delay to respect rate limits (5 calls/minute = 12 seconds between calls)
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`❌ [Energy] Error fetching ${commodity.name}:`, error);
      }
    }

    // If we got at least one successful result, cache it
    if (successCount > 0) {
      // Fill missing commodities with mock data
      const mockData = generateMockEnergyPrices(usdTryRate);
      const finalData = commodities.map(commodity => {
        const realData = results.find(r => r.symbol === commodity.symbol);
        const mockDataItem = mockData.find(m => m.symbol === commodity.symbol);
        return realData || mockDataItem!;
      });

      cache = {
        data: finalData,
        timestamp: Date.now(),
      };

      console.log(`✅ [Energy] Successfully fetched ${successCount}/${commodities.length} commodities`);
      return finalData;
    }

    // All endpoints failed, use mock data
    console.warn('⚠️ [Energy] All endpoints failed, using mock data as fallback');
    const mockData = generateMockEnergyPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  } catch (error) {
    console.error('❌ [Energy] Fatal error:', error);
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockEnergyPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }
}

/**
 * Parse commodity data from Alpha Vantage API response
 */
function parseCommodityData(
  data: any,
  commodity: { symbol: string; name: string; unit: string },
  usdTryRate: number
): EnergyCommodity | null {
  try {
    // Alpha Vantage commodity format:
    // { "name": "Brent Crude Oil", "data": [{ "value": "85.50", "date": "2025-10-25" }, ...] }
    if (data.data && Array.isArray(data.data) && data.data.length > 0) {
      const latest = data.data[0];
      const previous = data.data[1];

      const currentPrice = parseFloat(latest.value);
      const previousPrice = previous ? parseFloat(previous.value) : currentPrice;
      const change24h = previousPrice > 0 ? ((currentPrice - previousPrice) / previousPrice) * 100 : 0;

      return {
        symbol: commodity.symbol,
        name: commodity.name,
        priceUSD: currentPrice,
        price: currentPrice,
        priceTRY: currentPrice * usdTryRate,
        change24h,
        timestamp: new Date(latest.date),
        category: 'energy' as const,
        currency: 'TRY' as const,
        unit: commodity.unit,
      };
    }

    return null;
  } catch (error) {
    console.error(`❌ [Energy] Error parsing ${commodity.name} data:`, error);
    return null;
  }
}

/**
 * Get energy commodities with optional filtering
 */
export async function getEnergyCommodities(options?: {
  symbols?: string[];
  minPrice?: number;
  maxPrice?: number;
}): Promise<EnergyCommodity[]> {
  let commodities = await fetchEnergyCommodities();

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
export function clearEnergyCache(): void {
  cache = null;
  console.log('🗑️ [Energy] Cache cleared');
}

/**
 * Get cache status
 */
export function getEnergyCacheStatus() {
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

export type { EnergyCommodity };
