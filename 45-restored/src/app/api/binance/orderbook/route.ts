import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';
    const limit = searchParams.get('limit') || '10';

    // Binance Order Book API
    const response = await fetch(
      `https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=${limit}`
    );

    if (!response.ok) {
      throw new Error('Failed to fetch order book from Binance');
    }

    const data = await response.json();

    // Format: { bids: [[price, quantity], ...], asks: [[price, quantity], ...] }
    return NextResponse.json({
      success: true,
      data: {
        bids: data.bids.map((bid: string[]) => [
          parseFloat(bid[0]),
          parseFloat(bid[1]),
        ]),
        asks: data.asks.map((ask: string[]) => [
          parseFloat(ask[0]),
          parseFloat(ask[1]),
        ]),
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch order book',
      },
      { status: 500 }
    );
  }
}
