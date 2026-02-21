/**
 * 💰 FUNDING & DERIVATIVES API
 * Real-time funding rate and derivatives tracking
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const FUNDING_SERVICE_URL = process.env.FUNDING_SERVICE_URL || 'http://localhost:5014';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Call Python funding service
    const response = await fetch(`${FUNDING_SERVICE_URL}/analyze/${symbol}`, {
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Funding service error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Funding Derivatives API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Funding analysis failed',
      },
      { status: 500 }
    );
  }
}
