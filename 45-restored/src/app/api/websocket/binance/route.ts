/**
 * BINANCE WEBSOCKET API ENDPOINT
 * Server-side WebSocket connection to Binance
 * WHITE-HAT: Read-only public market data
 */

import { NextRequest, NextResponse } from 'next/server';
import { getBinanceWebSocketService } from '@/services/websocket/BinanceWebSocketService';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * Initialize WebSocket connection
 */
export async function POST(request: NextRequest) {
  try {
    const { symbols } = await request.json();

    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Symbols array required' },
        { status: 400 }
      );
    }

    const wsService = getBinanceWebSocketService();
    wsService.connect(symbols);

    return NextResponse.json({
      success: true,
      message: 'WebSocket connection initiated',
      symbols,
    });
  } catch (error: any) {
    console.error('❌ WebSocket connection error:', error.message);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

/**
 * Get WebSocket status
 */
export async function GET() {
  try {
    const wsService = getBinanceWebSocketService();
    const status = wsService.getStatus();

    return NextResponse.json({
      success: true,
      ...status,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

/**
 * Disconnect WebSocket
 */
export async function DELETE() {
  try {
    const wsService = getBinanceWebSocketService();
    wsService.disconnect();

    return NextResponse.json({
      success: true,
      message: 'WebSocket disconnected',
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
