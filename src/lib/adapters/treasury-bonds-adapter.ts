/**
 * 📊 TREASURY BONDS ADAPTER
 * US Treasury Bonds Yield Data Integration
 *
 * Features:
 * - 10-minute cache (bond yields update slower than equities)
 * - Alpha Vantage API integration
 * - Yield percentage display
 * - Error handling with fallback
 * - US 10-Year, 2-Year, 30-Year Treasury Bonds
 */

interface TreasuryBond {
  symbol: string;
  name: string;
  yield: number; // Yield percentage (e.g., 4.25%)
  price: number; // Bond price (typically around 100)
  change24h: number;
  timestamp: Date;
  category: 'bond';
  currency: 'USD';
  maturity: string;
}

interface CacheEntry {
  data: TreasuryBond[];
  timestamp: number;
}

// 10-minute cache for treasury bonds
const CACHE_DURATION = 10 * 60 * 1000;
let cache: CacheEntry | null = null;

/**
 * Generate mock treasury bond data (fallback when API is not available)
 * Updated with realistic yield rates (October 2025)
 */
function generateMockBondData(): TreasuryBond[] {
  const now = new Date();

  return [
    {
      symbol: 'US2Y',
      name: '2-Year Treasury',
      yield: 4.15 + (Math.random() * 0.2 - 0.1), // ~4.05-4.25%
      price: 98.5 + (Math.random() * 2 - 1),
      change24h: (Math.random() * 0.2) - 0.1, // -0.1% to +0.1%
      timestamp: now,
      category: 'bond',
      currency: 'USD',
      maturity: '2Y',
    },
    {
      symbol: 'US10Y',
      name: '10-Year Treasury',
      yield: 4.45 + (Math.random() * 0.2 - 0.1), // ~4.35-4.55%
      price: 97.2 + (Math.random() * 2 - 1),
      change24h: (Math.random() * 0.2) - 0.1,
      timestamp: now,
      category: 'bond',
      currency: 'USD',
      maturity: '10Y',
    },
    {
      symbol: 'US30Y',
      name: '30-Year Treasury',
      yield: 4.65 + (Math.random() * 0.2 - 0.1), // ~4.55-4.75%
      price: 95.8 + (Math.random() * 2 - 1),
      change24h: (Math.random() * 0.2) - 0.1,
      timestamp: now,
      category: 'bond',
      currency: 'USD',
      maturity: '30Y',
    },
  ];
}

/**
 * Fetch live treasury bond yields from Alpha Vantage API
 */
export async function fetchTreasuryBonds(): Promise<TreasuryBond[]> {
  // Check cache first
  if (cache && Date.now() - cache.timestamp < CACHE_DURATION) {
    console.log('✅ [Bonds] Using cached data');
    return cache.data;
  }

  const apiKey = process.env.ALPHA_VANTAGE_API_KEY;

  if (!apiKey || apiKey === 'demo' || apiKey === 'your_alpha_vantage_key_here') {
    console.warn('⚠️ [Bonds] ALPHA_VANTAGE_API_KEY not configured, using mock data');
    const mockData = generateMockBondData();
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }

  try {
    // Alpha Vantage treasury yield endpoints
    const bonds = [
      {
        symbol: 'US2Y',
        name: '2-Year Treasury',
        function: 'TREASURY_YIELD',
        interval: 'daily',
        maturity: '2year',
        displayMaturity: '2Y',
      },
      {
        symbol: 'US10Y',
        name: '10-Year Treasury',
        function: 'TREASURY_YIELD',
        interval: 'daily',
        maturity: '10year',
        displayMaturity: '10Y',
      },
      {
        symbol: 'US30Y',
        name: '30-Year Treasury',
        function: 'TREASURY_YIELD',
        interval: 'daily',
        maturity: '30year',
        displayMaturity: '30Y',
      },
    ];

    const results: TreasuryBond[] = [];
    let successCount = 0;

    for (const bond of bonds) {
      try {
        const url = `https://www.alphavantage.co/query?function=${bond.function}&interval=${bond.interval}&maturity=${bond.maturity}&apikey=${apiKey}`;
        console.log(`🔍 [Bonds] Fetching ${bond.name}...`);

        const response = await fetch(url, {
          next: { revalidate: 600 }, // 10 minutes
        });

        if (!response.ok) {
          console.warn(`⚠️ [Bonds] ${bond.name} returned ${response.status}`);
          continue;
        }

        const data = await response.json();

        // Check for API rate limit or error
        if (data['Note'] || data['Error Message']) {
          console.warn(`⚠️ [Bonds] API Error: ${data['Note'] || data['Error Message']}`);
          continue;
        }

        // Parse the data
        const parsed = parseBondData(data, bond);
        if (parsed) {
          results.push(parsed);
          successCount++;
          console.log(`✅ [Bonds] ${bond.name}: ${parsed.yield.toFixed(2)}% yield`);
        }

        // Add delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 500));
      } catch (error) {
        console.error(`❌ [Bonds] Error fetching ${bond.name}:`, error);
      }
    }

    // If we got at least one successful result, cache it
    if (successCount > 0) {
      // Fill missing bonds with mock data
      const mockData = generateMockBondData();
      const finalData = bonds.map(bond => {
        const realData = results.find(r => r.symbol === bond.symbol);
        const mockDataItem = mockData.find(m => m.symbol === bond.symbol);
        return realData || mockDataItem!;
      });

      cache = {
        data: finalData,
        timestamp: Date.now(),
      };

      console.log(`✅ [Bonds] Successfully fetched ${successCount}/${bonds.length} bonds`);
      return finalData;
    }

    // All endpoints failed, use mock data
    console.warn('⚠️ [Bonds] All endpoints failed, using mock data as fallback');
    const mockData = generateMockBondData();
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  } catch (error) {
    console.error('❌ [Bonds] Fatal error:', error);
    const mockData = generateMockBondData();
    cache = { data: mockData, timestamp: Date.now() };
    return mockData;
  }
}

/**
 * Parse treasury bond data from Alpha Vantage API response
 */
function parseBondData(
  data: any,
  bond: { symbol: string; name: string; displayMaturity: string }
): TreasuryBond | null {
  try {
    // Alpha Vantage treasury yield format:
    // { "name": "10-year", "data": [{ "date": "2025-10-25", "value": "4.45" }, ...] }
    if (data.data && Array.isArray(data.data) && data.data.length > 0) {
      const latest = data.data[0];
      const previous = data.data[1];

      const currentYield = parseFloat(latest.value);
      const previousYield = previous ? parseFloat(previous.value) : currentYield;
      const change24h = currentYield - previousYield;

      // Calculate approximate bond price from yield
      // Simplified formula: Price ≈ 100 - (yield - 4.0) * 5
      // This is a rough approximation; real bond pricing is more complex
      const baseYield = 4.0;
      const price = 100 - (currentYield - baseYield) * 5;

      return {
        symbol: bond.symbol,
        name: bond.name,
        yield: currentYield,
        price: Math.max(90, Math.min(110, price)), // Clamp between 90-110
        change24h,
        timestamp: new Date(latest.date),
        category: 'bond',
        currency: 'USD',
        maturity: bond.displayMaturity,
      };
    }

    return null;
  } catch (error) {
    console.error(`❌ [Bonds] Error parsing ${bond.name} data:`, error);
    return null;
  }
}

/**
 * Get treasury bonds with optional filtering
 */
export async function getTreasuryBonds(options?: {
  symbols?: string[];
  minYield?: number;
  maxYield?: number;
}): Promise<TreasuryBond[]> {
  let bonds = await fetchTreasuryBonds();

  // Apply filters
  if (options?.symbols && options.symbols.length > 0) {
    bonds = bonds.filter(b => options.symbols!.includes(b.symbol));
  }

  if (options?.minYield) {
    bonds = bonds.filter(b => b.yield >= options.minYield!);
  }

  if (options?.maxYield) {
    bonds = bonds.filter(b => b.yield <= options.maxYield!);
  }

  return bonds;
}

/**
 * Clear cache (useful for testing)
 */
export function clearBondsCache(): void {
  cache = null;
  console.log('🗑️ [Bonds] Cache cleared');
}

/**
 * Get cache status
 */
export function getBondsCacheStatus() {
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

export type { TreasuryBond };
