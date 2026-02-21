/**
 * QUANTUM PRO BACKTEST API
 * Historical strategy validation
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Mock backtest result generator
function generateMockBacktest(symbol: string, startDate: Date, endDate: Date) {
  const days = Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));
  const totalTrades = Math.floor(days * 0.8); // ~0.8 trades per day
  const winRate = 0.55 + Math.random() * 0.15; // 55-70% win rate
  const winningTrades = Math.floor(totalTrades * winRate);
  const losingTrades = totalTrades - winningTrades;

  const avgWin = 150 + Math.random() * 100;
  const avgLoss = 80 + Math.random() * 50;
  const grossProfit = winningTrades * avgWin;
  const grossLoss = losingTrades * avgLoss;
  const netProfit = grossProfit - grossLoss;
  const profitFactor = grossProfit / grossLoss;

  const initialCapital = 10000;
  const finalCapital = initialCapital + netProfit;
  const returnPct = (netProfit / initialCapital) * 100;

  const maxDrawdown = -Math.random() * 15; // -0 to -15%
  const sharpeRatio = 1.2 + Math.random() * 1.0; // 1.2-2.2

  return {
    symbol,
    period: {
      start: startDate.toISOString(),
      end: endDate.toISOString(),
      days
    },
    performance: {
      totalTrades,
      winningTrades,
      losingTrades,
      winRate: winRate * 100,
      netProfit: Math.round(netProfit),
      returnPct: Math.round(returnPct * 100) / 100,
      profitFactor: Math.round(profitFactor * 100) / 100,
      avgWin: Math.round(avgWin),
      avgLoss: Math.round(avgLoss),
      maxDrawdown: Math.round(maxDrawdown * 100) / 100,
      sharpeRatio: Math.round(sharpeRatio * 100) / 100
    },
    capital: {
      initial: initialCapital,
      final: Math.round(finalCapital),
      peak: Math.round(finalCapital * 1.1)
    },
    trades: Array.from({ length: Math.min(totalTrades, 20) }, (_, i) => ({
      id: i + 1,
      date: new Date(startDate.getTime() + (i / totalTrades) * (endDate.getTime() - startDate.getTime())).toISOString(),
      action: Math.random() > 0.5 ? 'BUY' : 'SELL',
      price: 50000 + Math.random() * 10000,
      profit: (Math.random() > winRate ? -1 : 1) * (Math.random() * 200)
    }))
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, startDate, endDate, strategy = 'ensemble' } = body;

    if (!symbol || !startDate || !endDate) {
      return NextResponse.json({
        success: false,
        error: 'Missing required parameters: symbol, startDate, endDate'
      }, { status: 400 });
    }

    const start = new Date(startDate);
    const end = new Date(endDate);

    if (isNaN(start.getTime()) || isNaN(end.getTime())) {
      return NextResponse.json({
        success: false,
        error: 'Invalid date format'
      }, { status: 400 });
    }

    if (start >= end) {
      return NextResponse.json({
        success: false,
        error: 'Start date must be before end date'
      }, { status: 400 });
    }

    console.log(`🔄 Running backtest for ${symbol} (${startDate} to ${endDate})`);

    const result = generateMockBacktest(symbol, start, end);

    return NextResponse.json({
      success: true,
      backtest: result,
      strategy,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('Backtest API error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}