/**
 * CRYPTO NEWS API ROUTE
 * GET /api/crypto-news
 *
 * Returns Turkish-translated important crypto news (impact >= 7/10)
 */

import { NextRequest, NextResponse } from 'next/server';
import { getCryptoNewsWithTranslation, clearNewsCache } from '@/lib/adapters/crypto-news-adapter';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

export async function GET(request: NextRequest) {
  try {
    // Check for refresh param
    const { searchParams } = new URL(request.url);
    const refresh = searchParams.get('refresh');

    if (refresh === 'true') {
      clearNewsCache();
      console.log('[API] Cache cleared, fetching fresh news...');
    }

    const response = await getCryptoNewsWithTranslation();

    return NextResponse.json(response, {
      headers: {
        'Cache-Control': 'public, s-maxage=600, stale-while-revalidate=300',
      },
    });

  } catch (error: any) {
    console.error('[API] Crypto News Error:', error);
    return NextResponse.json(
      {
        success: false,
        data: [],
        error: error.message || 'Unknown error',
      },
      { status: 500 }
    );
  }
}
