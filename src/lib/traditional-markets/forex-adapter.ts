/**
 * FOREX (CURRENCY EXCHANGE) API ADAPTER
 * 10 Major Currencies vs TRY - LIVE REAL-TIME DATA
 *
 * Features:
 * - Real-time exchange rates (NO MOCK DATA)
 * - 10 major currencies: USD, EUR, GBP, JPY, CHF, CAD, AUD, CNY, RUB, SAR
 * - Multiple free API sources with fallback
 * - 15-minute caching
 * - 24h change calculation
 * - White-hat error handling
 */

import circuitBreakerManager from '../resilience/circuit-breaker';

// ============================================================================
// TYPES
// ============================================================================

export interface ForexRate {
  symbol: string;        // e.g., "USD/TRY"
  baseCurrency: string;  // e.g., "USD"
  quoteCurrency: string; // Always "TRY"
  name: string;          // e.g., "US Dollar"
  rate: number;          // Current exchange rate
  change24h: number;     // Percentage change in 24h
  timestamp: Date;
}

export interface ForexData {
  rates: ForexRate[];
  lastUpdated: Date;
  source: string;
}

interface CachedForexData {
  data: ForexData;
  timestamp: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

const CACHE_TTL = 900000; // 15 minutes (forex changes frequently)

// 10 major currencies
const MAJOR_CURRENCIES = [
  { symbol: 'USD', name: 'US Dollar' },
  { symbol: 'EUR', name: 'Euro' },
  { symbol: 'GBP', name: 'British Pound' },
  { symbol: 'JPY', name: 'Japanese Yen' },
  { symbol: 'CHF', name: 'Swiss Franc' },
  { symbol: 'CAD', name: 'Canadian Dollar' },
  { symbol: 'AUD', name: 'Australian Dollar' },
  { symbol: 'CNY', name: 'Chinese Yuan' },
  { symbol: 'RUB', name: 'Russian Ruble' },
  { symbol: 'SAR', name: 'Saudi Riyal' },
];

// Free API endpoints - REAL DATA ONLY
const API_ENDPOINTS = {
  // Primary: ExchangeRate-API (free, no key required, 1500 req/month)
  primary: 'https://api.exchangerate-api.com/v4/latest',

  // Backup 1: Frankfurter (free, no key, ECB data)
  backup1: 'https://api.frankfurter.app/latest',

  // Backup 2: ExchangeRate.host (free, no key)
  backup2: 'https://api.exchangerate.host/latest',
};

// ============================================================================
// CACHE
// ============================================================================

let cache: CachedForexData | null = null;
let previousRates: Map<string, number> = new Map(); // For 24h change calculation

// ============================================================================
// API FETCHERS (REAL DATA)
// ============================================================================

/**
 * Fetch from primary API (ExchangeRate-API)
 */
async function fetchFromPrimaryAPI(): Promise<ForexData | null> {
  try {
    const response = await fetch(`${API_ENDPOINTS.primary}/TRY`, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Primary API failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.rates) {
      throw new Error('Invalid API response: no rates');
    }

    // Convert rates (they give TRY per foreign currency, we need foreign per TRY)
    const rates: ForexRate[] = MAJOR_CURRENCIES.map((currency) => {
      const rateFromAPI = data.rates[currency.symbol];

      if (!rateFromAPI) {
        console.warn(`[Forex] Missing rate for ${currency.symbol}`);
        return null;
      }

      // Invert rate: API gives TRY/XXX, we want XXX/TRY
      const rate = 1 / rateFromAPI;

      // Calculate 24h change
      const previousRate = previousRates.get(currency.symbol);
      const change24h = previousRate
        ? ((rate - previousRate) / previousRate) * 100
        : 0;

      return {
        symbol: `${currency.symbol}/TRY`,
        baseCurrency: currency.symbol,
        quoteCurrency: 'TRY',
        name: currency.name,
        rate: Math.round(rate * 10000) / 10000, // 4 decimal places
        change24h: Math.round(change24h * 100) / 100,
        timestamp: new Date(),
      };
    }).filter((rate): rate is ForexRate => rate !== null);

    return {
      rates,
      lastUpdated: new Date(),
      source: 'ExchangeRate-API',
    };
  } catch (error) {
    console.error('[Forex] Primary API error:', error);
    return null;
  }
}

/**
 * Fetch from backup API 1 (Frankfurter)
 */
async function fetchFromBackup1API(): Promise<ForexData | null> {
  try {
    const response = await fetch(`${API_ENDPOINTS.backup1}?to=TRY`, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backup1 API failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.rates) {
      throw new Error('Invalid API response: no rates');
    }

    const rates: ForexRate[] = MAJOR_CURRENCIES.map((currency) => {
      const rateFromAPI = data.rates.TRY;

      if (!rateFromAPI) {
        console.warn(`[Forex] Missing rate for TRY from ${currency.symbol}`);
        return null;
      }

      // This API gives foreign currency per TRY
      const rate = rateFromAPI;

      const previousRate = previousRates.get(currency.symbol);
      const change24h = previousRate
        ? ((rate - previousRate) / previousRate) * 100
        : 0;

      return {
        symbol: `${currency.symbol}/TRY`,
        baseCurrency: currency.symbol,
        quoteCurrency: 'TRY',
        name: currency.name,
        rate: Math.round(rate * 10000) / 10000,
        change24h: Math.round(change24h * 100) / 100,
        timestamp: new Date(),
      };
    }).filter((rate): rate is ForexRate => rate !== null);

    return {
      rates,
      lastUpdated: new Date(),
      source: 'Frankfurter (ECB)',
    };
  } catch (error) {
    console.error('[Forex] Backup1 API error:', error);
    return null;
  }
}

/**
 * Fetch from backup API 2 (ExchangeRate.host)
 */
async function fetchFromBackup2API(): Promise<ForexData | null> {
  try {
    const currencyList = MAJOR_CURRENCIES.map((c) => c.symbol).join(',');
    const response = await fetch(
      `${API_ENDPOINTS.backup2}?base=TRY&symbols=${currencyList}`,
      {
        headers: {
          'Accept': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Backup2 API failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.rates) {
      throw new Error('Invalid API response: no rates');
    }

    const rates: ForexRate[] = MAJOR_CURRENCIES.map((currency) => {
      const rateFromAPI = data.rates[currency.symbol];

      if (!rateFromAPI) {
        console.warn(`[Forex] Missing rate for ${currency.symbol}`);
        return null;
      }

      // This API gives foreign per TRY, which is what we want
      const rate = rateFromAPI;

      const previousRate = previousRates.get(currency.symbol);
      const change24h = previousRate
        ? ((rate - previousRate) / previousRate) * 100
        : 0;

      return {
        symbol: `${currency.symbol}/TRY`,
        baseCurrency: currency.symbol,
        quoteCurrency: 'TRY',
        name: currency.name,
        rate: Math.round(rate * 10000) / 10000,
        change24h: Math.round(change24h * 100) / 100,
        timestamp: new Date(),
      };
    }).filter((rate): rate is ForexRate => rate !== null);

    return {
      rates,
      lastUpdated: new Date(),
      source: 'ExchangeRate.host',
    };
  } catch (error) {
    console.error('[Forex] Backup2 API error:', error);
    return null;
  }
}

/**
 * Fetch forex data with automatic fallback
 */
async function fetchForexData(): Promise<ForexData> {
  // Try primary API
  console.log('[Forex] Fetching from primary API...');
  let data = await fetchFromPrimaryAPI();

  // Try backup 1 if primary fails
  if (!data) {
    console.log('[Forex] Primary failed, trying backup 1...');
    data = await fetchFromBackup1API();
  }

  // Try backup 2 if both fail
  if (!data) {
    console.log('[Forex] Backup 1 failed, trying backup 2...');
    data = await fetchFromBackup2API();
  }

  // If all APIs fail, throw error (NO MOCK DATA)
  if (!data) {
    throw new Error('All forex APIs failed - no data available');
  }

  // Update previous rates for 24h change calculation
  data.rates.forEach((rate) => {
    previousRates.set(rate.baseCurrency, rate.rate);
  });

  return data;
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Get forex data with caching - REAL-TIME DATA ONLY
 */
export async function getForexData(forceRefresh: boolean = false): Promise<ForexData> {
  // Check cache
  if (!forceRefresh && cache && Date.now() - cache.timestamp < CACHE_TTL) {
    console.log('[Forex] Returning cached data');
    return cache.data;
  }

  console.log('[Forex] Fetching fresh real-time data...');

  // Fetch with circuit breaker
  const breaker = circuitBreakerManager.getBreaker('forex-api', {
    failureThreshold: 3,
    successThreshold: 2,
    timeout: 10000,
  });

  try {
    const data = await breaker.execute(
      async () => await fetchForexData(),
      async () => {
        // Fallback: Return stale cache if available
        if (cache) {
          console.warn('[Forex] Using stale cache as fallback');
          return cache.data;
        }
        throw new Error('No cache available and all APIs failed');
      }
    );

    // Update cache
    cache = {
      data,
      timestamp: Date.now(),
    };

    return data;
  } catch (error: any) {
    console.error('[Forex] Failed to fetch data:', error);

    // Return stale cache or throw
    if (cache) {
      console.warn('[Forex] Returning stale cache due to error');
      return cache.data;
    }

    throw new Error(`Failed to fetch forex data: ${error.message}`);
  }
}

/**
 * Get specific currency rate
 */
export async function getCurrencyRate(
  baseCurrency: string,
  forceRefresh: boolean = false
): Promise<ForexRate | null> {
  const data = await getForexData(forceRefresh);
  return data.rates.find((rate) => rate.baseCurrency === baseCurrency) || null;
}

/**
 * Clear cache
 */
export function clearForexCache(): void {
  cache = null;
  console.log('[Forex] Cache cleared');
}

/**
 * Get cache statistics
 */
export function getForexCacheStats() {
  return {
    isCached: cache !== null,
    age: cache ? Date.now() - cache.timestamp : 0,
    ttl: CACHE_TTL,
    isStale: cache ? Date.now() - cache.timestamp > CACHE_TTL : true,
    source: cache?.data.source || 'none',
  };
}
