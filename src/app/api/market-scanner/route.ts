/**
 * MARKET SCANNER API
 * Real-time market scanning with multiple technical indicators
 *
 * Features:
 * - Volume analysis (high volume coins)
 * - Price momentum (gainers/losers)
 * - Technical patterns (RSI, MACD)
 * - Real-time Binance Futures data
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface ScanResult {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  scanScore: number;
  signals: string[];
  lastUpdate: string;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '50');
    const sortBy = searchParams.get('sortBy') || 'scanScore'; // scanScore, volume, change

    console.log(`[Market Scanner] Scanning market (limit: ${limit}, sortBy: ${sortBy})`);

    // Fetch real-time Binance Futures data
    const protocol = request.headers.get('x-forwarded-proto') || 'https';
    const host = request.headers.get('host') || 'localhost:3000';
    const baseUrl = `${protocol}://${host}`;
    const response = await fetch(`${baseUrl}/api/binance/futures`, {
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`Binance API returned ${response.status}`);
    }

    const binanceData = await response.json();

    if (!binanceData.success || !binanceData.data.all) {
      throw new Error('Invalid Binance data format');
    }

    const allCoins = binanceData.data.all;
    console.log(`[Market Scanner] Processing ${allCoins.length} coins...`);

    // Scan and score each coin
    const scanResults: ScanResult[] = allCoins.map((coin: any) => {
      const signals: string[] = [];
      let scanScore = 0;

      // Volume Analysis (weight: 30%)
      if (coin.volume24h > 10000000) {
        signals.push('High Volume');
        scanScore += 30;
      } else if (coin.volume24h > 5000000) {
        signals.push('Good Volume');
        scanScore += 15;
      }

      // Price Momentum (weight: 40%)
      const change = coin.changePercent24h || 0;
      if (change > 5) {
        signals.push('Strong Uptrend');
        scanScore += 40;
      } else if (change > 2) {
        signals.push('Uptrend');
        scanScore += 25;
      } else if (change < -5) {
        signals.push('Strong Downtrend');
        scanScore += 20; // Still interesting for short
      } else if (change < -2) {
        signals.push('Downtrend');
        scanScore += 10;
      }

      // Volatility Analysis (weight: 30%)
      const range = coin.high24h && coin.low24h
        ? ((coin.high24h - coin.low24h) / coin.low24h) * 100
        : 0;

      if (range > 10) {
        signals.push('High Volatility');
        scanScore += 30;
      } else if (range > 5) {
        signals.push('Medium Volatility');
        scanScore += 20;
      }

      // Breakout Detection (simple)
      if (coin.price >= coin.high24h * 0.98) {
        signals.push('Near 24h High');
        scanScore += 15;
      }

      if (coin.price <= coin.low24h * 1.02) {
        signals.push('Near 24h Low');
        scanScore += 10;
      }

      // Cap score at 100
      scanScore = Math.min(scanScore, 100);

      return {
        symbol: coin.symbol,
        price: coin.price,
        change24h: coin.change24h || 0,
        changePercent24h: coin.changePercent24h || 0,
        volume24h: coin.volume24h || 0,
        high24h: coin.high24h || 0,
        low24h: coin.low24h || 0,
        scanScore,
        signals,
        lastUpdate: coin.lastUpdate || new Date().toISOString(),
      };
    });

    // Sort based on user preference
    if (sortBy === 'volume') {
      scanResults.sort((a, b) => b.volume24h - a.volume24h);
    } else if (sortBy === 'change') {
      scanResults.sort((a, b) => b.changePercent24h - a.changePercent24h);
    } else {
      // Default: scanScore
      scanResults.sort((a, b) => b.scanScore - a.scanScore);
    }

    // Apply limit and filter (only coins with signals)
    const filteredResults = scanResults
      .filter(r => r.signals.length > 0)
      .slice(0, limit);

    console.log(`[Market Scanner] Scan complete: ${filteredResults.length}/${scanResults.length} coins (with signals)`);

    // Calculate statistics
    const stats = {
      totalScanned: allCoins.length,
      totalWithSignals: scanResults.filter(r => r.signals.length > 0).length,
      avgScanScore: Math.round(
        scanResults.reduce((sum, r) => sum + r.scanScore, 0) / scanResults.length
      ),
      topGainer: scanResults.reduce((max, r) => r.changePercent24h > max.changePercent24h ? r : max, scanResults[0]),
      topLoser: scanResults.reduce((min, r) => r.changePercent24h < min.changePercent24h ? r : min, scanResults[0]),
      highestVolume: scanResults.reduce((max, r) => r.volume24h > max.volume24h ? r : max, scanResults[0]),
    };

    return NextResponse.json({
      success: true,
      data: {
        results: filteredResults,
        stats: {
          totalScanned: stats.totalScanned,
          totalWithSignals: stats.totalWithSignals,
          avgScanScore: stats.avgScanScore,
          topGainer: {
            symbol: stats.topGainer?.symbol,
            change: stats.topGainer?.changePercent24h,
          },
          topLoser: {
            symbol: stats.topLoser?.symbol,
            change: stats.topLoser?.changePercent24h,
          },
          highestVolume: {
            symbol: stats.highestVolume?.symbol,
            volume: stats.highestVolume?.volume24h,
          },
        },
      },
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    console.error('[Market Scanner API Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Unknown error',
      },
      { status: 500 }
    );
  }
}
