/**
 * 🎯 12-LAYER CONFIRMATION ENGINE API
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CONFIRMATION_SERVICE_URL = process.env.CONFIRMATION_SERVICE_URL || 'http://localhost:5019';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    const response = await fetch(`${CONFIRMATION_SERVICE_URL}/analyze/${symbol}`, {
      signal: AbortSignal.timeout(20000),
    });

    if (!response.ok) {
      throw new Error(`Confirmation service error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Confirmation Engine API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Confirmation analysis failed',
      },
      { status: 500 }
    );
  }
}
