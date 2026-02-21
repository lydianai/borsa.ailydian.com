/**
 * 📈 MACRO CORRELATION MATRIX API
 * BTC/Altcoin correlation analysis and market regime detection
 */

import { NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MACRO_SERVICE_URL = process.env.MACRO_SERVICE_URL || 'http://localhost:5016';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const base = searchParams.get('base') || 'BTCUSDT';

    // Call Python macro correlation service
    const response = await fetch(`${MACRO_SERVICE_URL}/analyze?base=${base}`, {
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      throw new Error(`Macro correlation service error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[Macro Correlation API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Macro correlation analysis failed',
      },
      { status: 500 }
    );
  }
}
