/**
 * Binance Aggregate Trades API
 * Gerçek alıcı/satıcı baskısını almak için kullanılır
 * CVD (Cumulative Volume Delta) için kritik veri
 */

import { NextRequest, NextResponse } from 'next/server';

const BINANCE_BASE_URL = 'https://fapi.binance.com';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const limit = parseInt(searchParams.get('limit') || '500');

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol parameter is required' },
        { status: 400 }
      );
    }

    // Binance Futures Aggregate Trades endpoint
    const url = `${BINANCE_BASE_URL}/fapi/v1/aggTrades?symbol=${symbol}&limit=${limit}`;

    const response = await fetch(url, {
      next: { revalidate: 0 } // Her zaman taze veri
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const trades = await response.json();

    // Her trade'i işle
    const processedTrades = trades.map((trade: any) => ({
      id: trade.a, // Aggregate trade ID
      price: parseFloat(trade.p),
      quantity: parseFloat(trade.q),
      timestamp: trade.T,
      isBuyerMaker: trade.m, // false = alıcı agresif (market buy), true = satıcı agresif
      isBuy: !trade.m, // Basitleştirilmiş: Alım mı satım mı?
      time: new Date(trade.T).toISOString()
    }));

    // Alım/Satım özetini hesapla
    let totalBuyVolume = 0;
    let totalSellVolume = 0;
    let buyCount = 0;
    let sellCount = 0;

    processedTrades.forEach((trade: any) => {
      if (trade.isBuy) {
        totalBuyVolume += trade.quantity;
        buyCount++;
      } else {
        totalSellVolume += trade.quantity;
        sellCount++;
      }
    });

    const delta = totalBuyVolume - totalSellVolume;
    const buyPressure = totalBuyVolume + totalSellVolume > 0
      ? (totalBuyVolume / (totalBuyVolume + totalSellVolume)) * 100
      : 50;

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        trades: processedTrades,
        summary: {
          totalTrades: processedTrades.length,
          buyCount,
          sellCount,
          totalBuyVolume,
          totalSellVolume,
          delta,
          buyPressure: buyPressure.toFixed(2),
          timestamp: new Date().toISOString()
        }
      }
    });

  } catch (error) {
    console.error('Binance aggTrades API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch aggregate trades',
        message: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
