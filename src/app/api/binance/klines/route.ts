/**
 * 📊 BINANCE KLINES (HISTORICAL CANDLES) API
 *
 * Provides historical OHLCV (Open, High, Low, Close, Volume) candlestick data
 * from Binance Futures for multi-timeframe analysis.
 *
 * Timeframes: 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
 * Default Limit: 100 candles
 * Cache: 5 minutes
 */

import { NextRequest, NextResponse } from 'next/server';

// Cache configuration
const BINANCE_KLINES_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
const cache = new Map<string, { data: any; timestamp: number }>();

export const dynamic = 'force-dynamic';

interface KlineData {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
  quoteVolume: string;
  trades: number;
  takerBuyBaseVolume: string;
  takerBuyQuoteVolume: string;
}

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: number;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol')?.toUpperCase() || 'BTCUSDT';
    const interval = searchParams.get('interval') || '4h';
    const limit = parseInt(searchParams.get('limit') || '100');

    // Validate parameters
    const validIntervals = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'];
    if (!validIntervals.includes(interval)) {
      return NextResponse.json(
        {
          success: false,
          error: `Geçersiz interval. Kullanılabilir: ${validIntervals.join(', ')}`,
        },
        { status: 400 }
      );
    }

    if (limit < 1 || limit > 1000) {
      return NextResponse.json(
        {
          success: false,
          error: 'Limit 1-1000 arasında olmalıdır',
        },
        { status: 400 }
      );
    }

    // Check cache
    const cacheKey = `${symbol}_${interval}_${limit}`;
    const cached = cache.get(cacheKey);
    const now = Date.now();

    if (cached && (now - cached.timestamp) < BINANCE_KLINES_CACHE_TTL) {
      console.log(`[Klines API] Cache hit: ${cacheKey}`);
      return NextResponse.json({
        success: true,
        data: cached.data,
        cached: true,
        cacheAge: Math.round((now - cached.timestamp) / 1000),
      });
    }

    // Fetch from Binance
    console.log(`[Klines API] Fetching ${symbol} ${interval} (${limit} candles)`);

    const binanceUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`;
    const response = await fetch(binanceUrl);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status} ${response.statusText}`);
    }

    const rawKlines: any[] = await response.json();

    // Transform to friendly format
    const candles: Candle[] = rawKlines.map((k) => ({
      timestamp: k[0],
      open: parseFloat(k[1]),
      high: parseFloat(k[2]),
      low: parseFloat(k[3]),
      close: parseFloat(k[4]),
      volume: parseFloat(k[5]),
      closeTime: k[6],
    }));

    // Calculate additional metrics
    const firstCandle = candles[0];
    const lastCandle = candles[candles.length - 1];
    const priceChange = lastCandle.close - firstCandle.open;
    const priceChangePercent = ((priceChange / firstCandle.open) * 100);
    const avgVolume = candles.reduce((sum, c) => sum + c.volume, 0) / candles.length;
    const highestPrice = Math.max(...candles.map(c => c.high));
    const lowestPrice = Math.min(...candles.map(c => c.low));

    const result = {
      symbol,
      interval,
      candles,
      stats: {
        count: candles.length,
        firstTimestamp: firstCandle.timestamp,
        lastTimestamp: lastCandle.timestamp,
        firstPrice: firstCandle.open,
        lastPrice: lastCandle.close,
        priceChange,
        priceChangePercent: parseFloat(priceChangePercent.toFixed(2)),
        highestPrice,
        lowestPrice,
        priceRange: highestPrice - lowestPrice,
        avgVolume: parseFloat(avgVolume.toFixed(2)),
        totalVolume: parseFloat(candles.reduce((sum, c) => sum + c.volume, 0).toFixed(2)),
      },
    };

    // Update cache
    cache.set(cacheKey, { data: result, timestamp: now });

    return NextResponse.json({
      success: true,
      data: result,
      cached: false,
    });

  } catch (error) {
    console.error('[Klines API Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Kline verisi alınamadı',
      },
      { status: 500 }
    );
  }
}
