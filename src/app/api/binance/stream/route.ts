/**
 * BINANCE STREAM API
 * Real-time price data streaming endpoint
 */

import { NextResponse } from 'next/server';
import liveFeedManager from '@/lib/data/live-feed';
import binanceWebSocketService from '@/lib/data-service/binance-websocket';

// ============================================================================
// API ROUTE HANDLERS
// ============================================================================

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    
    if (symbol) {
      // Get specific symbol data
      const priceData = liveFeedManager.getPrice(symbol);
      if (priceData) {
        return NextResponse.json({
          success: true,
          data: priceData,
          timestamp: Date.now()
        });
      } else {
        return NextResponse.json({
          success: false,
          error: 'Symbol not found',
          timestamp: Date.now()
        }, { status: 404 });
      }
    } else {
      // Get all cached prices
      const allPrices = liveFeedManager.getAllPrices();
      return NextResponse.json({
        success: true,
        data: allPrices,
        count: allPrices.length,
        timestamp: Date.now()
      });
    }
  } catch (error: any) {
    console.error('[BinanceStreamAPI] GET error:', error);
    return NextResponse.json({
      success: false,
      error: error.message,
      timestamp: Date.now()
    }, { status: 500 });
  }
}

// Note: WebSocket connections are handled separately through the BinanceWebSocketService
// This API endpoint is for HTTP-based data retrieval