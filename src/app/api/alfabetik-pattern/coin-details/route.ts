/**
 * ALFABETIK PATTERN - COIN TRADING DETAILS API
 * Returns comprehensive trading signal details for a specific coin
 *
 * Features:
 * - Real-time price integration
 * - Entry/Exit/TP/SL calculation
 * - Leverage recommendation
 * - Risk/Reward analysis
 * - Pattern-based signal generation
 */

import { NextRequest, NextResponse } from 'next/server';
import { generateTradingSignal, type CoinTradingDetails } from '@/lib/trading-signal-calculator';
import { analyzeAlfabetikPattern, enrichWithHistoricalData, type CoinData } from '@/lib/alfabetik-pattern-analyzer';

// Cache for trading details (5 minutes)
interface CoinDetailsCache {
  data: CoinTradingDetails;
  timestamp: number;
}

const coinDetailsCache = new Map<string, CoinDetailsCache>();
const ALFABETIK_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

/**
 * Fetch real-time coin data from Binance
 */
async function fetchCoinData(symbol: string): Promise<CoinData | null> {
  try {
    // Fetch from our Binance futures endpoint
    const response = await fetch('http://localhost:3000/api/binance/futures', {
      next: { revalidate: 60 } // Cache for 1 minute
    });

    if (!response.ok) {
      throw new Error(`Binance API failed: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success || !data.data?.all) {
      throw new Error('Invalid Binance response format');
    }

    // Find the specific coin
    const coin = data.data.all.find((c: any) => c.symbol === symbol);

    if (!coin) {
      return null;
    }

    // Return in our CoinData format
    return {
      symbol: coin.symbol,
      price: parseFloat(coin.price),
      changePercent24h: parseFloat(coin.changePercent24h || '0'),
      changePercent7d: parseFloat(coin.changePercent7d || '0'),
      changePercent30d: parseFloat(coin.changePercent30d || '0'),
      volume24h: parseFloat(coin.volume24h || '0'),
      category: coin.category || 'other'
    };
  } catch (error) {
    console.error(`Error fetching coin data for ${symbol}:`, error);
    return null;
  }
}

/**
 * Get pattern data for the coin's letter group
 */
async function getPatternForCoin(symbol: string, allCoinData: CoinData[]): Promise<{
  patternSignal: "STRONG_BUY" | "BUY" | "SELL" | "HOLD";
  patternConfidence: number;
  patternLetter: string;
  momentum: "YUKARIDA" | "ASAGIDA" | "YATAY";
} | null> {
  try {
    // Get the first letter
    const firstLetter = symbol.charAt(0).toUpperCase();

    // Enrich with historical data
    const enrichedData = await enrichWithHistoricalData(allCoinData);

    // Run pattern analysis
    const analysis = analyzeAlfabetikPattern(enrichedData);

    // Find the pattern for this letter
    const pattern = analysis.patterns.find(p => p.harf === firstLetter);

    if (!pattern) {
      return null;
    }

    return {
      patternSignal: pattern.signal,
      patternConfidence: pattern.confidence,
      patternLetter: pattern.harf,
      momentum: pattern.momentum
    };
  } catch (error) {
    console.error(`Error getting pattern for ${symbol}:`, error);
    return null;
  }
}

/**
 * GET /api/alfabetik-pattern/coin-details?symbol=BTCUSDT
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Get symbol from query params
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol')?.toUpperCase();

    // Validate input
    if (!symbol) {
      return NextResponse.json(
        {
          success: false,
          error: 'Symbol parameter is required',
          message: 'Please provide a symbol parameter (e.g., ?symbol=BTCUSDT)'
        },
        { status: 400 }
      );
    }

    // Check cache first
    const cached = coinDetailsCache.get(symbol);
    if (cached && Date.now() - cached.timestamp < ALFABETIK_CACHE_TTL) {
      const cacheAge = Math.floor((Date.now() - cached.timestamp) / 1000);
      return NextResponse.json({
        success: true,
        data: cached.data,
        cached: true,
        cacheAge,
        processingTime: Date.now() - startTime
      });
    }

    // Fetch all coin data for pattern analysis
    const allCoinsResponse = await fetch('http://localhost:3000/api/binance/futures', {
      next: { revalidate: 60 }
    });

    if (!allCoinsResponse.ok) {
      throw new Error('Failed to fetch market data');
    }

    const allCoinsData = await allCoinsResponse.json();
    const allCoinData: CoinData[] = allCoinsData.data.all.map((c: any) => ({
      symbol: c.symbol,
      price: parseFloat(c.price),
      changePercent24h: parseFloat(c.changePercent24h || '0'),
      changePercent7d: parseFloat(c.changePercent7d || '0'),
      changePercent30d: parseFloat(c.changePercent30d || '0'),
      volume24h: parseFloat(c.volume24h || '0'),
      category: c.category || 'other'
    }));

    // Get specific coin data
    const coinData = await fetchCoinData(symbol);

    if (!coinData) {
      return NextResponse.json(
        {
          success: false,
          error: 'Coin not found',
          message: `Symbol ${symbol} not found in Binance futures markets`
        },
        { status: 404 }
      );
    }

    // Get pattern data
    let patternData = await getPatternForCoin(symbol, allCoinData);

    if (!patternData) {
      // Fallback to HOLD if pattern not found
      patternData = {
        patternSignal: 'HOLD',
        patternConfidence: 50,
        patternLetter: symbol.charAt(0).toUpperCase(),
        momentum: 'YATAY'
      };
    }

    // Generate trading signal
    const tradingDetails = generateTradingSignal(
      coinData.symbol,
      coinData.price,
      patternData.patternSignal,
      patternData.patternConfidence,
      coinData.changePercent24h,
      coinData.changePercent7d || 0,
      coinData.changePercent30d || 0,
      coinData.volume24h,
      patternData.momentum,
      patternData.patternLetter
    );

    // Cache the result
    coinDetailsCache.set(symbol, {
      data: tradingDetails,
      timestamp: Date.now()
    });

    // Clean old cache entries (older than 10 minutes)
    const cleanupThreshold = Date.now() - (10 * 60 * 1000);
    for (const [key, value] of coinDetailsCache.entries()) {
      if (value.timestamp < cleanupThreshold) {
        coinDetailsCache.delete(key);
      }
    }

    return NextResponse.json({
      success: true,
      data: tradingDetails,
      cached: false,
      processingTime: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Error in coin-details API:', error);

    return NextResponse.json(
      {
        success: false,
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'Unknown error occurred',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

/**
 * Clear cache endpoint (for development/testing)
 * DELETE /api/alfabetik-pattern/coin-details
 */
export async function DELETE() {
  try {
    const cacheSize = coinDetailsCache.size;
    coinDetailsCache.clear();

    return NextResponse.json({
      success: true,
      message: `Cache cleared: ${cacheSize} entries removed`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to clear cache',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
