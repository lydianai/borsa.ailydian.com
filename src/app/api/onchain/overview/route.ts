/**
 * ON-CHAIN MARKET OVERVIEW API
 * GET /api/onchain/overview
 *
 * Returns market-wide on-chain analysis and trending whale activities
 */

import { NextRequest, NextResponse } from 'next/server';
import { getOnChainOverview } from '@/lib/onchain';

export const dynamic = 'force-dynamic';

export async function GET(_request: NextRequest) {
  try {
    const overview = await getOnChainOverview();

    return NextResponse.json({
      success: true,
      data: {
        trending: overview.trending,
        mostAccumulated: overview.mostAccumulated,
        mostDistributed: overview.mostDistributed,
        marketSentiment: overview.marketSentiment,
        insights: {
          bullish:
            overview.marketSentiment === 'bullish'
              ? 'Strong whale accumulation detected across multiple assets'
              : overview.marketSentiment === 'neutral'
              ? 'Mixed whale activity - no clear market direction'
              : 'Limited whale accumulation activity',
          bearish:
            overview.marketSentiment === 'bearish'
              ? 'Significant whale distribution detected - caution advised'
              : overview.marketSentiment === 'neutral'
              ? 'Mixed whale activity - no clear market direction'
              : 'Limited whale selling activity',
        },
      },
    });
  } catch (error: any) {
    console.error('[API /onchain/overview] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch on-chain overview',
        data: null,
      },
      { status: 500 }
    );
  }
}
