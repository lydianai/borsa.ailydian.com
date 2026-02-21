/**
 * PRECIOUS METALS API ADAPTER
 * Altin, Gumus, Paladyum, Bakir fiyatlari (TL/gram)
 *
 * Features:
 * - Multi-source data fetching (resilient fallback)
 * - 22 & 24 carat gold calculation
 * - TL conversion with current USD/TRY rate
 * - Turkish gold prices (Gram, Çeyrek, Yarım, Tam, Cumhuriyet) via Harem Altın API
 * - 1-hour caching
 * - White-hat error handling
 */

import circuitBreakerManager from '../resilience/circuit-breaker';
import { fetchGoldPrices, type FormattedGoldPrice } from '../adapters/harem-altin-adapter';

// ============================================================================
// TYPES
// ============================================================================

export interface PreciousMetalPrice {
  symbol: string;
  name: string;
  priceUSD: number;      // USD per troy ounce
  priceTRY: number;      // TRY per troy ounce
  pricePerGramTRY: number; // TRY per gram
  change24h: number;     // Percentage
  lastUpdated: Date;
}

export interface GoldPrice extends PreciousMetalPrice {
  carat22PerGramTRY: number; // 22 ayar TL/gram
  carat24PerGramTRY: number; // 24 ayar TL/gram (pure)
}

export interface PreciousMetalsData {
  gold: GoldPrice;
  silver: PreciousMetalPrice;
  palladium: PreciousMetalPrice;
  copper: PreciousMetalPrice;
  turkishGold?: FormattedGoldPrice[]; // Harem Altın API: Gram, Çeyrek, Yarım, Tam, Cumhuriyet
  usdTryRate: number;
  lastUpdated: Date;
}

interface CachedData {
  data: PreciousMetalsData;
  timestamp: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const CACHE_TTL = 3600000; // 1 hour
const TROY_OUNCE_TO_GRAM = 31.1034768; // 1 troy ounce = 31.1034768 grams
const CARAT_22_PURITY = 0.9167; // 22/24 = 0.9167
const CARAT_24_PURITY = 1.0;    // 24/24 = 1.0 (pure gold)

// Free API endpoints (multiple sources for reliability)
const API_SOURCES = {
  // MetalpriceAPI (free tier: 50 requests/month)
  metalpriceapi: 'https://api.metalpriceapi.com/v1/latest',

  // GoldAPI (free tier: 100 requests/month)
  goldapi: 'https://www.goldapi.io/api',

  // Fallback: Use current approximate prices (updated weekly)
  fallback: {
    gold: 2050,    // USD per troy ounce
    silver: 24,
    palladium: 1050,
    copper: 0.27,  // USD per pound, will convert
  },
};

// ============================================================================
// CACHE
// ============================================================================

let cache: CachedData | null = null;

// Previous price cache for change24h calculation
interface PreviousPrices {
  gold: number;
  silver: number;
  palladium: number;
  copper: number;
  timestamp: number;
}

let previousPrices: PreviousPrices | null = null;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get current USD/TRY exchange rate
 */
async function getUSDTRYRate(): Promise<number> {
  try {
    // Free exchange rate API
    const response = await fetch('https://api.exchangerate-api.com/v4/latest/USD');

    if (!response.ok) {
      throw new Error(`Exchange rate API failed: ${response.status}`);
    }

    const data = await response.json();
    return data.rates.TRY || 32.5; // Fallback to approximate rate
  } catch (error) {
    console.warn('[PreciousMetals] USD/TRY rate fetch failed, using fallback:', error);
    return 32.5; // Approximate fallback rate
  }
}

/**
 * Calculate price per gram in TRY
 */
function calculatePricePerGram(priceUSD: number, usdTryRate: number): number {
  const priceTRY = priceUSD * usdTryRate;
  const pricePerGramTRY = priceTRY / TROY_OUNCE_TO_GRAM;
  return Math.round(pricePerGramTRY * 100) / 100; // 2 decimal places
}

/**
 * Calculate carat-specific gold prices
 */
function calculateGoldCaratPrices(
  pureGoldPricePerGram: number
): { carat22: number; carat24: number } {
  return {
    carat22: Math.round(pureGoldPricePerGram * CARAT_22_PURITY * 100) / 100,
    carat24: Math.round(pureGoldPricePerGram * CARAT_24_PURITY * 100) / 100,
  };
}

/**
 * Fetch prices from primary API (MetalpriceAPI)
 */
async function fetchFromMetalPriceAPI(): Promise<Partial<PreciousMetalsData> | null> {
  try {
    // Note: This requires API key for production
    // For development, we'll use fallback
    const apiKey = process.env.METAL_PRICE_API_KEY;

    if (!apiKey) {
      console.log('[PreciousMetals] No MetalpriceAPI key, using fallback');
      return null;
    }

    const response = await fetch(
      `${API_SOURCES.metalpriceapi}?api_key=${apiKey}&base=USD&currencies=XAU,XAG,XPD,XCU`
    );

    if (!response.ok) {
      throw new Error(`MetalpriceAPI failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.rates) {
      return null;
    }

    // Convert rates (they give USD per metal unit)
    return {
      // Rates are inverted (USD per XAU), so we invert them
      gold: {
        priceUSD: 1 / data.rates.XAU,
      } as any,
      silver: {
        priceUSD: 1 / data.rates.XAG,
      } as any,
      palladium: {
        priceUSD: 1 / data.rates.XPD,
      } as any,
      copper: {
        priceUSD: (1 / data.rates.XCU) / 0.45359237, // Convert to per pound
      } as any,
    };
  } catch (error) {
    console.error('[PreciousMetals] MetalpriceAPI error:', error);
    return null;
  }
}

/**
 * Fetch prices using fallback data
 */
function getFallbackPrices(): Partial<PreciousMetalsData> {
  console.log('[PreciousMetals] Using fallback prices (approximate)');

  return {
    gold: {
      priceUSD: API_SOURCES.fallback.gold,
    } as any,
    silver: {
      priceUSD: API_SOURCES.fallback.silver,
    } as any,
    palladium: {
      priceUSD: API_SOURCES.fallback.palladium,
    } as any,
    copper: {
      priceUSD: API_SOURCES.fallback.copper,
    } as any,
  };
}

// ============================================================================
// MAIN FUNCTIONS
// ============================================================================

/**
 * Fetch precious metals prices
 */
async function fetchPreciousMetalsPrices(): Promise<PreciousMetalsData> {
  // Try primary API first
  let partialData = await fetchFromMetalPriceAPI();

  // Fallback if primary fails
  if (!partialData) {
    partialData = getFallbackPrices();
  }

  // Get USD/TRY rate
  const usdTryRate = await getUSDTRYRate();

  // Calculate TRY prices for each metal
  const goldPriceUSD = partialData.gold?.priceUSD || API_SOURCES.fallback.gold;
  const silverPriceUSD = partialData.silver?.priceUSD || API_SOURCES.fallback.silver;
  const palladiumPriceUSD = partialData.palladium?.priceUSD || API_SOURCES.fallback.palladium;
  const copperPriceUSD = partialData.copper?.priceUSD || API_SOURCES.fallback.copper;

  // Calculate per gram prices
  const goldPerGramTRY = calculatePricePerGram(goldPriceUSD, usdTryRate);
  const silverPerGramTRY = calculatePricePerGram(silverPriceUSD, usdTryRate);
  const palladiumPerGramTRY = calculatePricePerGram(palladiumPriceUSD, usdTryRate);
  const copperPerGramTRY = calculatePricePerGram(copperPriceUSD * 453.592, usdTryRate); // Copper is per pound

  // Calculate gold carat prices
  const goldCaratPrices = calculateGoldCaratPrices(goldPerGramTRY);

  const now = new Date();

  // Calculate change24h from previous prices (if available)
  const goldChange = previousPrices
    ? ((goldPriceUSD - previousPrices.gold) / previousPrices.gold) * 100
    : 0;
  const silverChange = previousPrices
    ? ((silverPriceUSD - previousPrices.silver) / previousPrices.silver) * 100
    : 0;
  const palladiumChange = previousPrices
    ? ((palladiumPriceUSD - previousPrices.palladium) / previousPrices.palladium) * 100
    : 0;
  const copperChange = previousPrices
    ? ((copperPriceUSD - previousPrices.copper) / previousPrices.copper) * 100
    : 0;

  // Store current prices for next comparison (if cache is old enough)
  if (!previousPrices || Date.now() - previousPrices.timestamp > CACHE_TTL) {
    previousPrices = {
      gold: goldPriceUSD,
      silver: silverPriceUSD,
      palladium: palladiumPriceUSD,
      copper: copperPriceUSD,
      timestamp: Date.now(),
    };
  }

  // Fetch Turkish gold prices from Harem Altın API
  let turkishGoldPrices: FormattedGoldPrice[] = [];
  try {
    turkishGoldPrices = await fetchGoldPrices();
    console.log(`[PreciousMetals] Fetched ${turkishGoldPrices.length} Turkish gold prices from Harem Altın`);
  } catch (error) {
    console.warn('[PreciousMetals] Failed to fetch Turkish gold prices:', error);
    // Continue without Turkish gold prices (they are optional)
  }

  const result: PreciousMetalsData = {
    gold: {
      symbol: 'XAU',
      name: 'Gold',
      priceUSD: goldPriceUSD,
      priceTRY: goldPriceUSD * usdTryRate,
      pricePerGramTRY: goldPerGramTRY,
      carat22PerGramTRY: goldCaratPrices.carat22,
      carat24PerGramTRY: goldCaratPrices.carat24,
      change24h: Math.round(goldChange * 100) / 100,
      lastUpdated: now,
    },
    silver: {
      symbol: 'XAG',
      name: 'Silver',
      priceUSD: silverPriceUSD,
      priceTRY: silverPriceUSD * usdTryRate,
      pricePerGramTRY: silverPerGramTRY,
      change24h: Math.round(silverChange * 100) / 100,
      lastUpdated: now,
    },
    palladium: {
      symbol: 'XPD',
      name: 'Palladium',
      priceUSD: palladiumPriceUSD,
      priceTRY: palladiumPriceUSD * usdTryRate,
      pricePerGramTRY: palladiumPerGramTRY,
      change24h: Math.round(palladiumChange * 100) / 100,
      lastUpdated: now,
    },
    copper: {
      symbol: 'XCU',
      name: 'Copper',
      priceUSD: copperPriceUSD,
      priceTRY: copperPriceUSD * usdTryRate,
      pricePerGramTRY: copperPerGramTRY,
      change24h: Math.round(copperChange * 100) / 100,
      lastUpdated: now,
    },
    turkishGold: turkishGoldPrices.length > 0 ? turkishGoldPrices : undefined,
    usdTryRate,
    lastUpdated: now,
  };

  return result;
}

/**
 * Get precious metals data with caching
 */
export async function getPreciousMetalsData(
  forceRefresh: boolean = false
): Promise<PreciousMetalsData> {
  // Check cache
  if (!forceRefresh && cache && Date.now() - cache.timestamp < CACHE_TTL) {
    console.log('[PreciousMetals] Returning cached data');
    return cache.data;
  }

  console.log('[PreciousMetals] Fetching fresh data...');

  // Fetch with circuit breaker
  const breaker = circuitBreakerManager.getBreaker('precious-metals-api', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 15000,
  });

  try {
    const data = await breaker.execute(
      async () => await fetchPreciousMetalsPrices(),
      async () => {
        // Fallback: Return stale cache if available
        if (cache) {
          console.warn('[PreciousMetals] Using stale cache as fallback');
          return cache.data;
        }
        // Ultimate fallback: Return with fallback prices
        return fetchPreciousMetalsPrices();
      }
    );

    // Update cache
    cache = {
      data,
      timestamp: Date.now(),
    };

    return data;
  } catch (error: any) {
    console.error('[PreciousMetals] Failed to fetch data:', error);

    // Return stale cache or throw
    if (cache) {
      console.warn('[PreciousMetals] Returning stale cache due to error');
      return cache.data;
    }

    throw new Error(`Failed to fetch precious metals data: ${error.message}`);
  }
}

/**
 * Clear cache (for testing)
 */
export function clearPreciousMetalsCache(): void {
  cache = null;
  previousPrices = null;
  console.log('[PreciousMetals] Cache cleared');
}

/**
 * Get cache statistics
 */
export function getPreciousMetalsCacheStats() {
  return {
    isCached: cache !== null,
    age: cache ? Date.now() - cache.timestamp : 0,
    ttl: CACHE_TTL,
    isStale: cache ? Date.now() - cache.timestamp > CACHE_TTL : true,
  };
}
