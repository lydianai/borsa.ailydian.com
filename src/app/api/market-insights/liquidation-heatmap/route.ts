/**
 * LIQUIDATION HEATMAP API ROUTE
 * With Binance Order Book fallback
 */

import { NextRequest, NextResponse } from 'next/server';

/**
 * Generate fallback liquidation heatmap from Binance Order Book
 * Approximates liquidation zones using order book depth
 */
async function generateBinanceFallback(symbol: string) {
  try {
    // Get current mark price
    const priceResponse = await fetch(
      `https://fapi.binance.com/fapi/v1/ticker/price?symbol=${symbol}`
    );
    const priceData = await priceResponse.json();
    const currentPrice = parseFloat(priceData.price);

    // Get order book depth
    const depthResponse = await fetch(
      `https://fapi.binance.com/fapi/v1/depth?symbol=${symbol}&limit=100`
    );
    const depthData = await depthResponse.json();

    // Calculate approximate liquidation levels from order book
    // Higher volume = potential liquidation zone
    const heatmapPoints: Array<{ price: number; liquidation_amount_usd: number }> = [];

    // Process bids (potential long liquidations below current price)
    for (const [priceStr, qtyStr] of depthData.bids.slice(0, 30)) {
      const price = parseFloat(priceStr);
      const quantity = parseFloat(qtyStr);
      const volumeUSD = price * quantity;

      heatmapPoints.push({
        price,
        liquidation_amount_usd: volumeUSD * 0.5 // Approximate 50% as liquidation risk
      });
    }

    // Process asks (potential short liquidations above current price)
    for (const [priceStr, qtyStr] of depthData.asks.slice(0, 30)) {
      const price = parseFloat(priceStr);
      const quantity = parseFloat(qtyStr);
      const volumeUSD = price * quantity;

      heatmapPoints.push({
        price,
        liquidation_amount_usd: volumeUSD * 0.5
      });
    }

    // Sort by price
    heatmapPoints.sort((a, b) => a.price - b.price);

    return {
      success: true,
      data: {
        symbol,
        current_price: currentPrice,
        heatmap: heatmapPoints,
        source: 'binance_orderbook_fallback'
      }
    };
  } catch (error) {
    console.error('[Liquidation Heatmap] Binance fallback error:', error);
    throw error;
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Try Python service first
    try {
      const response = await fetch(
        `http://localhost:5031/liquidation-heatmap/${symbol}`,
        {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
          cache: 'no-store',
          signal: AbortSignal.timeout(3000) // 3s timeout
        }
      );

      if (response.ok) {
        const data = await response.json();
        return NextResponse.json(data);
      }
    } catch (pythonError) {
      console.warn('[Liquidation Heatmap] Python service unavailable, using Binance fallback');
    }

    // Use Binance fallback
    const fallbackData = await generateBinanceFallback(symbol);
    return NextResponse.json(fallbackData);

  } catch (error: any) {
    console.error('[Liquidation Heatmap API] Error:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Liquidation heatmap unavailable' },
      { status: 500 }
    );
  }
}
