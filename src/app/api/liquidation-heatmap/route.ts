/**
 * 🔥 LIQUIDATION HEATMAP API
 * Real-time liquidation cluster detection
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const LIQUIDATION_SERVICE_URL = process.env.LIQUIDATION_SERVICE_URL || 'http://localhost:5013';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Call Python liquidation service
    const response = await fetch(`${LIQUIDATION_SERVICE_URL}/analyze/${symbol}`, {
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Liquidation service error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Liquidation Heatmap API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Liquidation analysis failed',
      },
      { status: 500 }
    );
  }
}
