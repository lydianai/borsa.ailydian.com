/**
 * WATCHLIST & FILTERS API
 * Manages watchlist, blacklist, and advanced filtering settings
 *
 * Features:
 * - Watchlist (favorite coins)
 * - Blacklist (ignored coins)
 * - Price range filters
 * - Volume filters
 * - Market cap filters
 * - Signal strength filters
 * - Volatility filters
 * - Timeframe selection
 * - Exchange selection
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { watchlistFiltersDB } from '@/lib/database';

// Watchlist & Filters Interface
interface WatchlistFilters {
  watchlist: {
    enabled: boolean;
    coins: string[]; // ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    onlyShowWatchlist: boolean; // Hide all others when enabled
  };

  blacklist: {
    enabled: boolean;
    coins: string[]; // Coins to ignore
    autoHideLowVolume: boolean;
    volumeThreshold: number; // Min 24h volume in USDT
  };

  priceFilter: {
    enabled: boolean;
    minPrice: number;
    maxPrice: number;
    currency: 'USDT' | 'USD';
  };

  volumeFilter: {
    enabled: boolean;
    min24hVolume: number; // In USDT
    min24hTrades: number;
  };

  marketCapFilter: {
    enabled: boolean;
    minMarketCap: number; // In millions
    maxMarketCap: number;
    categories: string[]; // ["DeFi", "Layer1", "Meme", etc.]
  };

  signalFilter: {
    enabled: boolean;
    minConfidence: number; // 0-100
    signalTypes: ('BUY' | 'SELL' | 'STRONG_BUY' | 'STRONG_SELL')[];
    strategies: string[]; // Which strategies to include
  };

  volatilityFilter: {
    enabled: boolean;
    minVolatility: number; // Min % change in 24h
    maxVolatility: number; // Max % change
    period: '1h' | '4h' | '24h';
  };

  timeframeSelection: {
    enabled: boolean;
    timeframes: ('1m' | '5m' | '15m' | '1h' | '4h' | '1d')[];
    defaultTimeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  };

  exchangeFilter: {
    enabled: boolean;
    exchanges: ('Binance' | 'Bybit' | 'OKX' | 'KuCoin')[];
    spotOnly: boolean;
    futuresOnly: boolean;
  };

  quickFilters: {
    showGainersOnly: boolean; // Only positive % change
    showLosersOnly: boolean; // Only negative % change
    showHighVolumeOnly: boolean; // Top 100 by volume
    showNewListings: boolean; // Listed in last 30 days
    showTopMovers: boolean; // Highest volatility
  };
}

// Default Configuration
const DEFAULT_WATCHLIST_FILTERS: WatchlistFilters = {
  watchlist: {
    enabled: false,
    coins: [],
    onlyShowWatchlist: false,
  },
  blacklist: {
    enabled: false,
    coins: [],
    autoHideLowVolume: false,
    volumeThreshold: 100000, // $100K
  },
  priceFilter: {
    enabled: false,
    minPrice: 0,
    maxPrice: 1000000,
    currency: 'USDT',
  },
  volumeFilter: {
    enabled: false,
    min24hVolume: 0,
    min24hTrades: 0,
  },
  marketCapFilter: {
    enabled: false,
    minMarketCap: 0,
    maxMarketCap: 1000000, // $1T
    categories: [],
  },
  signalFilter: {
    enabled: false,
    minConfidence: 70,
    signalTypes: ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL'],
    strategies: [],
  },
  volatilityFilter: {
    enabled: false,
    minVolatility: 0,
    maxVolatility: 100,
    period: '24h',
  },
  timeframeSelection: {
    enabled: true,
    timeframes: ['1h', '4h', '1d'],
    defaultTimeframe: '1h',
  },
  exchangeFilter: {
    enabled: false,
    exchanges: ['Binance', 'Bybit'],
    spotOnly: false,
    futuresOnly: false,
  },
  quickFilters: {
    showGainersOnly: false,
    showLosersOnly: false,
    showHighVolumeOnly: false,
    showNewListings: false,
    showTopMovers: false,
  },
};

// Database storage (persistent, encrypted)
// Old: const watchlistFiltersStore = new Map<string, WatchlistFilters>();
// Now using: watchlistFiltersDB from @/lib/database

/**
 * Get session ID from cookies
 */
async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  return sessionId;
}

/**
 * Validate filters
 */
function validateFilters(filters: Partial<WatchlistFilters>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Validate price filter
  if (filters.priceFilter?.enabled) {
    if (filters.priceFilter.minPrice < 0) {
      errors.push('Min price cannot be negative');
    }
    if (filters.priceFilter.maxPrice < filters.priceFilter.minPrice) {
      errors.push('Max price must be greater than min price');
    }
  }

  // Validate volume filter
  if (filters.volumeFilter?.enabled) {
    if (filters.volumeFilter.min24hVolume < 0) {
      errors.push('Min volume cannot be negative');
    }
    if (filters.volumeFilter.min24hTrades < 0) {
      errors.push('Min trades cannot be negative');
    }
  }

  // Validate market cap filter
  if (filters.marketCapFilter?.enabled) {
    if (filters.marketCapFilter.minMarketCap < 0) {
      errors.push('Min market cap cannot be negative');
    }
    if (filters.marketCapFilter.maxMarketCap < filters.marketCapFilter.minMarketCap) {
      errors.push('Max market cap must be greater than min market cap');
    }
  }

  // Validate signal filter
  if (filters.signalFilter?.enabled) {
    if (filters.signalFilter.minConfidence < 0 || filters.signalFilter.minConfidence > 100) {
      errors.push('Min confidence must be between 0 and 100');
    }
  }

  // Validate volatility filter
  if (filters.volatilityFilter?.enabled) {
    if (filters.volatilityFilter.minVolatility < 0) {
      errors.push('Min volatility cannot be negative');
    }
    if (filters.volatilityFilter.maxVolatility < filters.volatilityFilter.minVolatility) {
      errors.push('Max volatility must be greater than min volatility');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * GET - Retrieve watchlist and filters
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Get filters or use defaults
    let filters = watchlistFiltersDB.get(sessionId);
    if (!filters) {
      filters = DEFAULT_WATCHLIST_FILTERS;
      watchlistFiltersDB.set(sessionId, filters);
    }

    // Calculate stats
    const stats = {
      watchlistCount: filters.watchlist.coins.length,
      blacklistCount: filters.blacklist.coins.length,
      activeFiltersCount:
        (filters.priceFilter.enabled ? 1 : 0) +
        (filters.volumeFilter.enabled ? 1 : 0) +
        (filters.marketCapFilter.enabled ? 1 : 0) +
        (filters.signalFilter.enabled ? 1 : 0) +
        (filters.volatilityFilter.enabled ? 1 : 0) +
        (filters.exchangeFilter.enabled ? 1 : 0),
      quickFiltersActive: Object.values(filters.quickFilters).filter(Boolean).length,
    };

    return NextResponse.json({
      success: true,
      data: {
        filters,
        stats,
      },
    });
  } catch (error) {
    console.error('[Watchlist Filters API] GET Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get filters',
      },
      { status: 500 }
    );
  }
}

/**
 * POST - Update watchlist and filters
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    // Handle specific actions
    if (body.action === 'add_to_watchlist') {
      const currentFilters = watchlistFiltersDB.get(sessionId) || DEFAULT_WATCHLIST_FILTERS;
      const coin = body.coin?.toUpperCase();

      if (!coin) {
        return NextResponse.json({ success: false, error: 'Coin symbol required' }, { status: 400 });
      }

      if (!currentFilters.watchlist.coins.includes(coin)) {
        currentFilters.watchlist.coins.push(coin);
        watchlistFiltersDB.set(sessionId, currentFilters);
      }

      return NextResponse.json({
        success: true,
        message: `${coin} added to watchlist`,
        data: currentFilters,
      });
    }

    if (body.action === 'remove_from_watchlist') {
      const currentFilters = watchlistFiltersDB.get(sessionId) || DEFAULT_WATCHLIST_FILTERS;
      const coin = body.coin?.toUpperCase();

      currentFilters.watchlist.coins = currentFilters.watchlist.coins.filter((c: string) => c !== coin);
      watchlistFiltersDB.set(sessionId, currentFilters);

      return NextResponse.json({
        success: true,
        message: `${coin} removed from watchlist`,
        data: currentFilters,
      });
    }

    if (body.action === 'add_to_blacklist') {
      const currentFilters = watchlistFiltersDB.get(sessionId) || DEFAULT_WATCHLIST_FILTERS;
      const coin = body.coin?.toUpperCase();

      if (!coin) {
        return NextResponse.json({ success: false, error: 'Coin symbol required' }, { status: 400 });
      }

      if (!currentFilters.blacklist.coins.includes(coin)) {
        currentFilters.blacklist.coins.push(coin);
        watchlistFiltersDB.set(sessionId, currentFilters);
      }

      return NextResponse.json({
        success: true,
        message: `${coin} added to blacklist`,
        data: currentFilters,
      });
    }

    if (body.action === 'remove_from_blacklist') {
      const currentFilters = watchlistFiltersDB.get(sessionId) || DEFAULT_WATCHLIST_FILTERS;
      const coin = body.coin?.toUpperCase();

      currentFilters.blacklist.coins = currentFilters.blacklist.coins.filter((c: string) => c !== coin);
      watchlistFiltersDB.set(sessionId, currentFilters);

      return NextResponse.json({
        success: true,
        message: `${coin} removed from blacklist`,
        data: currentFilters,
      });
    }

    // Validate input
    const validation = validateFilters(body);
    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid filter settings',
          details: validation.errors,
        },
        { status: 400 }
      );
    }

    // Get current filters or defaults
    const currentFilters = watchlistFiltersDB.get(sessionId) || DEFAULT_WATCHLIST_FILTERS;

    // Deep merge with new filters
    const updatedFilters: WatchlistFilters = {
      watchlist: { ...currentFilters.watchlist, ...(body.watchlist || {}) },
      blacklist: { ...currentFilters.blacklist, ...(body.blacklist || {}) },
      priceFilter: { ...currentFilters.priceFilter, ...(body.priceFilter || {}) },
      volumeFilter: { ...currentFilters.volumeFilter, ...(body.volumeFilter || {}) },
      marketCapFilter: { ...currentFilters.marketCapFilter, ...(body.marketCapFilter || {}) },
      signalFilter: { ...currentFilters.signalFilter, ...(body.signalFilter || {}) },
      volatilityFilter: { ...currentFilters.volatilityFilter, ...(body.volatilityFilter || {}) },
      timeframeSelection: { ...currentFilters.timeframeSelection, ...(body.timeframeSelection || {}) },
      exchangeFilter: { ...currentFilters.exchangeFilter, ...(body.exchangeFilter || {}) },
      quickFilters: { ...currentFilters.quickFilters, ...(body.quickFilters || {}) },
    };

    // Save to database
    watchlistFiltersDB.set(sessionId, updatedFilters);

    // Create response with Set-Cookie header
    const response = NextResponse.json({
      success: true,
      message: 'Filters updated successfully',
      data: updatedFilters,
    });

    // Set session cookie
    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return response;
  } catch (error) {
    console.error('[Watchlist Filters API] POST Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update filters',
      },
      { status: 500 }
    );
  }
}

/**
 * PUT - Reset to defaults
 */
export async function PUT(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Reset to defaults
    watchlistFiltersDB.set(sessionId, DEFAULT_WATCHLIST_FILTERS);

    return NextResponse.json({
      success: true,
      message: 'Filters reset to defaults',
      data: DEFAULT_WATCHLIST_FILTERS,
    });
  } catch (error) {
    console.error('[Watchlist Filters API] PUT Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset filters',
      },
      { status: 500 }
    );
  }
}
