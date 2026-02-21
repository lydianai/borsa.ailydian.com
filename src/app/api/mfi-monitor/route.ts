/**
 * MFI MONITOR API ROUTE
 *
 * Proxies requests to Python MFI Monitor Microservice (Port 5023)
 * Provides Money Flow Index analysis with multi-timeframe support
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MFI_SERVICE_URL = 'http://localhost:5023';

/**
 * POST /api/mfi-monitor
 *
 * Analyze MFI for a symbol across multiple timeframes
 *
 * Request body:
 * {
 *   "symbol": "BTCUSDT",
 *   "timeframes": ["15m", "30m", "1h"]
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol = 'BTCUSDT', timeframes = ['15m', '30m', '1h'] } = body;

    console.log(`[MFI Monitor] Analyzing ${symbol} on timeframes: ${timeframes.join(', ')}`);

    // Call Python MFI Monitor service
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

    const response = await fetch(`${MFI_SERVICE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol,
        timeframes,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      console.error(`[MFI Monitor] Service error: ${response.status}`);
      return NextResponse.json(
        {
          success: false,
          error: `MFI Monitor service returned ${response.status}`,
        },
        { status: response.status }
      );
    }

    const data = await response.json();

    if (!data.success) {
      console.error(`[MFI Monitor] Analysis failed:`, data.error);
      return NextResponse.json(
        {
          success: false,
          error: data.error || 'MFI analysis failed',
        },
        { status: 400 }
      );
    }

    console.log(`[MFI Monitor] Success for ${symbol}. Overall signal: ${data.data.overall.signal}`);

    return NextResponse.json({
      success: true,
      data: data.data,
    });
  } catch (error: any) {
    if (error.name === 'AbortError') {
      console.error('[MFI Monitor] Request timeout');
      return NextResponse.json(
        {
          success: false,
          error: 'MFI Monitor request timeout',
        },
        { status: 504 }
      );
    }

    console.error('[MFI Monitor] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Internal server error',
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/mfi-monitor?symbol=BTCUSDT
 *
 * Quick MFI check for a single symbol (default timeframes)
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';
    const timeframes = ['15m', '30m', '1h'];

    console.log(`[MFI Monitor] Quick check for ${symbol}`);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    const response = await fetch(`${MFI_SERVICE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol,
        timeframes,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      return NextResponse.json(
        {
          success: false,
          error: `MFI Monitor service error: ${response.status}`,
        },
        { status: response.status }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      data: data.data,
    });
  } catch (error: any) {
    if (error.name === 'AbortError') {
      return NextResponse.json(
        {
          success: false,
          error: 'Request timeout',
        },
        { status: 504 }
      );
    }

    console.error('[MFI Monitor] GET Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Internal server error',
      },
      { status: 500 }
    );
  }
}
