/**
 * 📊 SENTIMENT ANALYSIS API
 * Multi-source market sentiment aggregation
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const SENTIMENT_SERVICE_URL = process.env.SENTIMENT_SERVICE_URL || 'http://localhost:5017';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Call Python sentiment analysis service
    const response = await fetch(`${SENTIMENT_SERVICE_URL}/analyze?symbol=${symbol}`, {
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      throw new Error(`Sentiment service error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Sentiment Analysis API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Sentiment analysis failed',
      },
      { status: 500 }
    );
  }
}
