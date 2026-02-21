/**
 * 📈 STOCK INDICES ADAPTER
 * Major Stock Market Indices Data Integration
 *
 * Features:
 * - 10-minute cache (market indices update slower than crypto)
 * - Alpha Vantage API integration
 * - TL price conversion with live USD/TRY rate
 * - Error handling with fallback
 * - S&P 500, NASDAQ Composite, Dow Jones
 */

interface StockIndex {
  symbol: string;
  name: string;
  price: number;
  priceUSD: number;
  priceTRY: number;
  change24h: number;
  changePercent: number;
  timestamp: Date;
  category: 'index';
  currency: 'TRY';
  marketCap?: string;
}

interface CacheEntry {
  data: StockIndex[];
  timestamp: number;
}

// 10-minute cache for stock indices
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
    console.warn('⚠️ [Indices] Failed to fetch USD/TRY rate, using fallback 42.0');
    return 42.0;
  }
}

/**
 * Generate mock stock index prices (fallback when API is not available)
 * Updated with realistic market values (October 2025)
 */
function generateMockIndexPrices(usdTryRate: number): StockIndex[] {
  const now = new Date();

  return [
    {
      symbol: 'SPX',
      name: 'S&P 500',
      priceUSD: 5850 + (Math.random() * 100 - 50), // ~5800-5900
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 2) - 1, // -1% to +1%
      changePercent: 0,
      timestamp: now,
      category: 'index' as const,
      currency: 'TRY' as const,
      marketCap: '$45T+',
    },
    {
      symbol: 'NDX',
      name: 'NASDAQ Composite',
      priceUSD: 18500 + (Math.random() * 200 - 100), // ~18400-18600
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 3) - 1.5, // -1.5% to +1.5% (more volatile)
      changePercent: 0,
      timestamp: now,
      category: 'index' as const,
      currency: 'TRY' as const,
      marketCap: '$22T+',
    },
    {
      symbol: 'DJI',
      name: 'Dow Jones Industrial Average',
      priceUSD: 42500 + (Math.random() * 300 - 150), // ~42350-42650
      price: 0,
      priceTRY: 0,
      change24h: (Math.random() * 2) - 1,
      changePercent: 0,
      timestamp: now,
      category: 'index' as const,
      currency: 'TRY' as const,
      marketCap: '$14T+',
    },
  ].map(index => {
    const priceTRY = index.priceUSD * usdTryRate;
    return {
      ...index,
      price: index.priceUSD,
      priceTRY,
      changePercent: index.change24h,
    };
  });
}

/**
 * Fetch live stock index prices from Alpha Vantage API
 */
export async function fetchStockIndices(): Promise<StockIndex[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_DURATION) {
    console.log('✅ [Indices] Using cached data');
    return cache.data;
  }

  const apiKey = process.env.ALPHA_VANTAGE_API_KEY;

  if (!apiKey || apiKey === 'demo' || apiKey === 'your_alpha_vantage_key_here') {
    console.warn('⚠️ [Indices] ALPHA_VANTAGE_API_KEY not configured, using mock data');
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockIndexPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }

  try {
    const usdTryRate = await getUSDTRYRate();
    console.log(`💱 [Indices] USD/TRY rate: ${usdTryRate.toFixed(4)}`);

    // Alpha Vantage uses ETFs as proxies for indices
    // SPY = S&P 500, QQQ = NASDAQ, DIA = Dow Jones
    const indices = [
      {
        symbol: 'SPX',
        etfSymbol: 'SPY',
        name: 'S&P 500',
        marketCap: '$45T+',
        multiplier: 10 // SPY is ~1/10 of SPX
      },
      {
        symbol: 'NDX',
        etfSymbol: 'QQQ',
        name: 'NASDAQ Composite',
        marketCap: '$22T+',
        multiplier: 40 // QQQ is ~1/40 of NDX
      },
      {
        symbol: 'DJI',
        etfSymbol: 'DIA',
        name: 'Dow Jones Industrial Average',
        marketCap: '$14T+',
        multiplier: 100 // DIA is ~1/100 of DJI
      },
    ];

    const results: StockIndex[] = [];
    let successCount = 0;

    for (const index of indices) {
      try {
        const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${index.etfSymbol}&apikey=${apiKey}`;
        console.log(`🔍 [Indices] Fetching ${index.name} via ${index.etfSymbol}...`);

        const response = await fetch(url, {
          next: { revalidate: 600 }, // 10 minutes
        });

        if (!response.ok) {
          console.warn(`⚠️ [Indices] ${index.name} returned ${response.status}`);
          continue;
        }

        const data = await response.json();

        // Check for API rate limit or error
        if (data['Note'] || data['Error Message']) {
          console.warn(`⚠️ [Indices] API Error: ${data['Note'] || data['Error Message']}`);
          continue;
        }

        // Parse the data
        const parsed = parseIndexData(data, index, usdTryRate);
        if (parsed) {
          results.push(parsed);
          successCount++;
          console.log(`✅ [Indices] ${index.name}: $${parsed.priceUSD.toFixed(2)} = ₺${parsed.priceTRY.toFixed(2)}`);
        }

        // Add delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`❌ [Indices] Error fetching ${index.name}:`, error);
      }
    }

    // If we got at least one successful result, cache it
    if (successCount > 0) {
      // Fill missing indices with mock data
      const mockData = generateMockIndexPrices(usdTryRate);
      const finalData = indices.map(index => {
        const realData = results.find(r => r.symbol === index.symbol);
        const mockDataItem = mockData.find(m => m.symbol === index.symbol);
        return realData || mockDataItem!;
      });

      cache = {
        data: finalData,
        timestamp: Date.now(),
      };

      console.log(`✅ [Indices] Successfully fetched ${successCount}/${indices.length} indices`);
      return finalData;
    }

    // All endpoints failed, use mock data
    console.warn('⚠️ [Indices] All endpoints failed, using mock data as fallback');
    const mockData = generateMockIndexPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  } catch (error) {
    console.error('❌ [Indices] Fatal error:', error);
    const usdTryRate = await getUSDTRYRate();
    const mockData = generateMockIndexPrices(usdTryRate);
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }
}

/**
 * Parse stock index data from Alpha Vantage API response
 */
function parseIndexData(
  data: any,
  index: { symbol: string; name: string; marketCap: string; multiplier: number },
  usdTryRate: number
): StockIndex | null {
  try {
    // Alpha Vantage GLOBAL_QUOTE format:
    // { "Global Quote": { "05. price": "585.23", "10. change percent": "0.45%" } }
    const quote = data['Global Quote'];

    if (!quote || !quote['05. price']) {
      console.warn(`⚠️ [Indices] Invalid data structure for ${index.name}`);
      return null;
    }

    const etfPrice = parseFloat(quote['05. price']);
    const changePercentStr = quote['10. change percent'] || '0%';
    const changePercent = parseFloat(changePercentStr.replace('%', ''));

    // Convert ETF price to index value using multiplier
    const indexPrice = etfPrice * index.multiplier;

    return {
      symbol: index.symbol,
      name: index.name,
      priceUSD: indexPrice,
      price: indexPrice,
      priceTRY: indexPrice * usdTryRate,
      change24h: changePercent,
      changePercent,
      timestamp: new Date(),
      category: 'index' as const,
      currency: 'TRY' as const,
      marketCap: index.marketCap,
    };
  } catch (error) {
    console.error(`❌ [Indices] Error parsing ${index.name} data:`, error);
    return null;
  }
}

/**
 * Get stock indices with optional filtering
 */
export async function getStockIndices(options?: {
  symbols?: string[];
  minPrice?: number;
  maxPrice?: number;
}): Promise<StockIndex[]> {
  let indices = await fetchStockIndices();

  // Apply filters
  if (options?.symbols && options.symbols.length > 0) {
    indices = indices.filter(i => options.symbols!.includes(i.symbol));
  }

  if (options?.minPrice) {
    indices = indices.filter(i => i.priceTRY >= options.minPrice!);
  }

  if (options?.maxPrice) {
    indices = indices.filter(i => i.priceTRY <= options.maxPrice!);
  }

  return indices;
}

/**
 * Clear cache (useful for testing)
 */
export function clearIndicesCache(): void {
  cache = null;
  console.log('🗑️ [Indices] Cache cleared');
}

/**
 * Get cache status
 */
export function getIndicesCacheStatus() {
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

export type { StockIndex };
