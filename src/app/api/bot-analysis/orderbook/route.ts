/**
 * 📊 ORDER BOOK ANALYSIS API
 *
 * Fetches and analyzes Binance Futures order book data
 * - Real-time bid/ask depth
 * - Whale wall detection (>$500K orders)
 * - Order book imbalance calculation
 * - Buy/sell pressure analysis
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data
 * - No automated trading
 * - Rate-limited requests
 * - Educational purpose only
 *
 * USAGE:
 * GET /api/bot-analysis/orderbook?symbol=BTCUSDT&depth=100
 */

import { NextRequest, NextResponse } from 'next/server';
import type {
  OrderBookSnapshot,
  OrderBookLevel,
  OrderBookImbalance,
  WhaleLevelDetection,
  BotAnalysisAPIResponse
} from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// CONSTANTS
// ============================================================================

const WHALE_THRESHOLD_USD = 500000; // $500K threshold for whale walls
const DEFAULT_DEPTH = 100; // Default order book depth

// ============================================================================
// HELPER: FETCH BINANCE ORDER BOOK
// ============================================================================

async function fetchBinanceOrderBook(symbol: string, limit: number = 100) {
  try {
    const url = `https://fapi.binance.com/fapi/v1/depth?symbol=${symbol}&limit=${limit}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('[OrderBook API] Failed to fetch Binance data:', error);
    throw error;
  }
}

// ============================================================================
// HELPER: PROCESS ORDER BOOK LEVELS
// ============================================================================

function processOrderBookLevels(
  rawLevels: Array<[string, string]>,
  _side: 'bids' | 'asks'
): OrderBookLevel[] {
  const levels: OrderBookLevel[] = [];
  let cumulative = 0;

  for (const [priceStr, quantityStr] of rawLevels) {
    const price = parseFloat(priceStr);
    const quantity = parseFloat(quantityStr);
    const usdValue = price * quantity;

    cumulative += quantity;

    levels.push({
      price,
      quantity,
      total: cumulative,
      usdValue
    });
  }

  return levels;
}

// ============================================================================
// HELPER: DETECT WHALE LEVELS
// ============================================================================

function detectWhaleLevels(
  bids: OrderBookLevel[],
  asks: OrderBookLevel[]
): WhaleLevelDetection[] {
  const whales: WhaleLevelDetection[] = [];

  // Total order book depth (USD)
  const totalBidValue = bids.reduce((sum, level) => sum + level.usdValue, 0);
  const totalAskValue = asks.reduce((sum, level) => sum + level.usdValue, 0);
  const totalDepth = totalBidValue + totalAskValue;

  // Check bids for whale walls
  for (const bid of bids) {
    if (bid.usdValue >= WHALE_THRESHOLD_USD) {
      whales.push({
        price: bid.price,
        side: 'BID',
        quantity: bid.quantity,
        usdValue: bid.usdValue,
        isWhaleWall: true,
        percentageOfDepth: (bid.usdValue / totalDepth) * 100
      });
    }
  }

  // Check asks for whale walls
  for (const ask of asks) {
    if (ask.usdValue >= WHALE_THRESHOLD_USD) {
      whales.push({
        price: ask.price,
        side: 'ASK',
        quantity: ask.quantity,
        usdValue: ask.usdValue,
        isWhaleWall: true,
        percentageOfDepth: (ask.usdValue / totalDepth) * 100
      });
    }
  }

  return whales;
}

// ============================================================================
// HELPER: CALCULATE ORDER BOOK IMBALANCE
// ============================================================================

function calculateImbalance(
  bids: OrderBookLevel[],
  asks: OrderBookLevel[]
): OrderBookImbalance {
  // Calculate total volumes (top 20 levels for accuracy)
  const topBids = bids.slice(0, 20);
  const topAsks = asks.slice(0, 20);

  const bidVolume = topBids.reduce((sum, level) => sum + level.usdValue, 0);
  const askVolume = topAsks.reduce((sum, level) => sum + level.usdValue, 0);

  // Imbalance ratio (> 1 = bullish, < 1 = bearish)
  const ratio = bidVolume / (askVolume || 1);

  // Determine signal
  let signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
  let confidence = 50;

  if (ratio > 1.5) {
    signal = 'BULLISH';
    confidence = Math.min(95, 50 + (ratio - 1) * 30);
  } else if (ratio < 0.67) {
    signal = 'BEARISH';
    confidence = Math.min(95, 50 + (1 - ratio) * 30);
  }

  // Calculate pressure percentages
  const totalPressure = bidVolume + askVolume;
  const bidPressure = (bidVolume / totalPressure) * 100;
  const askPressure = (askVolume / totalPressure) * 100;

  return {
    symbol: bids[0]?.price ? 'BTCUSDT' : 'UNKNOWN', // Will be replaced with actual symbol
    timestamp: Date.now(),
    ratio,
    signal,
    confidence,
    bidPressure,
    askPressure
  };
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';
  const depth = parseInt(searchParams.get('depth') || String(DEFAULT_DEPTH));

  try {
    console.log(`[OrderBook API] Fetching order book for ${symbol}...`);

    // Fetch raw order book from Binance
    const rawData = await fetchBinanceOrderBook(symbol, depth);

    // Process bids and asks
    const bids = processOrderBookLevels(rawData.bids, 'bids');
    const asks = processOrderBookLevels(rawData.asks, 'asks');

    // Calculate metrics
    const bidVolume = bids.reduce((sum, level) => sum + level.usdValue, 0);
    const askVolume = asks.reduce((sum, level) => sum + level.usdValue, 0);

    const bestBid = bids[0]?.price || 0;
    const bestAsk = asks[0]?.price || 0;
    const midPrice = (bestBid + bestAsk) / 2;
    const spread = bestAsk - bestBid;
    const spreadPercent = (spread / midPrice) * 100;

    // Create order book snapshot
    const snapshot: OrderBookSnapshot = {
      symbol,
      timestamp: Date.now(),
      bids,
      asks,
      bidVolume,
      askVolume,
      spread,
      spreadPercent,
      midPrice
    };

    // Detect whale levels
    const whaleLevels = detectWhaleLevels(bids, asks);

    // Calculate imbalance
    const imbalance = calculateImbalance(bids, asks);
    imbalance.symbol = symbol; // Update symbol

    const duration = Date.now() - startTime;

    console.log(`[OrderBook API] ${symbol} analyzed in ${duration}ms - Whales: ${whaleLevels.length}, Signal: ${imbalance.signal}`);

    const response: BotAnalysisAPIResponse<{
      snapshot: OrderBookSnapshot;
      whaleLevels: WhaleLevelDetection[];
      imbalance: OrderBookImbalance;
    }> = {
      success: true,
      data: {
        snapshot,
        whaleLevels,
        imbalance
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[OrderBook API] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Order book analysis failed',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
