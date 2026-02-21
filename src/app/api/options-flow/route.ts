/**
 * 📊 OPTIONS FLOW API
 * Deribit options data analysis and gamma squeeze detection
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const OPTIONS_SERVICE_URL = process.env.OPTIONS_SERVICE_URL || 'http://localhost:5018';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const currency = searchParams.get('currency') || 'BTC';

    // Call Python options flow service (30s timeout for complex analysis)
    const response = await fetch(`${OPTIONS_SERVICE_URL}/analyze/${currency}`, {
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      throw new Error(`Options service error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Options Flow API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Options flow analysis failed',
      },
      { status: 500 }
    );
  }
}
