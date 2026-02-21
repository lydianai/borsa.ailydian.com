/**
 * 📋 BINANCE FUTURES SYMBOLS API
 *
 * Fetches all available USDT-M perpetual futures symbols from Binance
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only data fetch
 * - No trading execution
 * - Educational purpose
 *
 * USAGE:
 * GET /api/bot-analysis/symbols
 */

import { NextResponse } from 'next/server';
import type { BotAnalysisAPIResponse } from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 3600; // Cache for 1 hour

interface BinanceSymbol {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  status: string;
  contractType: string;
}

export async function GET() {
  const startTime = Date.now();

  try {
    console.log('[Symbols API] Fetching Binance Futures symbols...');

    // Fetch exchange info from Binance Futures
    const response = await fetch('https://fapi.binance.com/fapi/v1/exchangeInfo');

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    // Filter USDT-M perpetual futures only
    const usdtPerpetuals = data.symbols
      .filter((s: BinanceSymbol) =>
        s.quoteAsset === 'USDT' &&
        s.contractType === 'PERPETUAL' &&
        s.status === 'TRADING'
      )
      .map((s: BinanceSymbol) => ({
        symbol: s.symbol,
        baseAsset: s.baseAsset,
        displayName: s.baseAsset
      }))
      .sort((a: any, b: any) => a.baseAsset.localeCompare(b.baseAsset));

    const duration = Date.now() - startTime;

    console.log(`[Symbols API] Fetched ${usdtPerpetuals.length} USDT-M perpetual symbols in ${duration}ms`);

    const apiResponse: BotAnalysisAPIResponse<{
      symbols: Array<{ symbol: string; baseAsset: string; displayName: string }>;
      count: number;
    }> = {
      success: true,
      data: {
        symbols: usdtPerpetuals,
        count: usdtPerpetuals.length
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(apiResponse);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Symbols API] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fetch symbols',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
