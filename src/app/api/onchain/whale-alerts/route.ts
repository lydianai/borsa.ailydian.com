/**
 * ON-CHAIN WHALE ALERTS API
 * GET /api/onchain/whale-alerts
 *
 * Returns whale transaction data and activity analysis
 *
 * Query params:
 * - symbol: Get data for specific symbol (optional)
 * - refresh: Force cache refresh (optional)
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getAllOnChainData,
  getWhaleActivityForSymbol,
  clearAllOnChainCache,
} from '@/lib/onchain';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get('symbol');
  const refresh = searchParams.get('refresh') === 'true';

  try {
    // Force refresh if requested
    if (refresh) {
      clearAllOnChainCache();
      console.log('[API /onchain/whale-alerts] Cache cleared - fetching fresh data');
    }

    // Get specific symbol or all data
    if (symbol) {
      const whaleActivity = await getWhaleActivityForSymbol(symbol.toUpperCase());

      if (!whaleActivity) {
        return NextResponse.json(
          {
            success: false,
            error: `No whale activity found for ${symbol}`,
            data: null,
          },
          { status: 404 }
        );
      }

      return NextResponse.json({
        success: true,
        data: {
          symbol: whaleActivity.symbol,
          activity: whaleActivity.activity,
          confidence: whaleActivity.confidence,
          riskScore: whaleActivity.riskScore,
          exchangeNetflow: whaleActivity.exchangeNetflow,
          summary: whaleActivity.summary,
          recentTransactions: whaleActivity.recentTransactions.slice(0, 5), // Last 5
          timestamp: whaleActivity.timestamp,
        },
      });
    }

    // Get all on-chain data
    const onChainData = await getAllOnChainData();

    // Convert Map to array for JSON response
    const whaleActivityArray = Array.from(onChainData.whaleActivity.values()).map((activity) => ({
      symbol: activity.symbol,
      activity: activity.activity,
      confidence: activity.confidence,
      riskScore: activity.riskScore,
      exchangeNetflow: activity.exchangeNetflow,
      summary: activity.summary,
      recentTransactionsCount: activity.recentTransactions.length,
    }));

    return NextResponse.json({
      success: true,
      data: {
        whaleActivity: whaleActivityArray,
        summary: onChainData.summary,
        timestamp: onChainData.timestamp,
      },
    });
  } catch (error: any) {
    console.error('[API /onchain/whale-alerts] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch whale alerts',
        data: null,
      },
      { status: 500 }
    );
  }
}
