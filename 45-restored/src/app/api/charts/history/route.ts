import { NextRequest, NextResponse } from 'next/server';
import DatabaseService from '@/lib/database-service';

/**
 * HISTORICAL CHART DATA API
 * Get real historical data for performance charts
 */

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const botId = searchParams.get('botId') || 'default-bot';
    const timeframe = (searchParams.get('timeframe') || '24H') as '1H' | '24H' | '7D' | '30D';

    const db = DatabaseService.getInstance();

    // Get performance history
    const performanceData = await db.getPerformanceHistory(botId, timeframe);

    // Format for charts
    const labels = performanceData.map(p =>
      p.timestamp.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      })
    );

    const pnlData = performanceData.map(p => p.totalPnL);
    const winRateData = performanceData.map(p => p.winRate);
    const sharpeData = performanceData.map(p => p.sharpeRatio);
    const drawdownData = performanceData.map(p => p.drawdown);

    // Get trade count per time bucket (for activity chart)
    const tradesData = await getTradeActivity(botId, timeframe);

    return NextResponse.json({
      success: true,
      data: {
        labels,
        pnlData,
        winRateData,
        sharpeData,
        drawdownData,
        tradesData,
      },
    });
  } catch (error: any) {
    console.error('Chart history error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch chart history',
      },
      { status: 500 }
    );
  }
}

async function getTradeActivity(botId: string, timeframe: string) {
  const db = DatabaseService.getInstance();

  // Get recent trades
  const limit = {
    '1H': 60,
    '24H': 24,
    '7D': 168,
    '30D': 720,
  }[timeframe] || 24;

  const trades = await db.getTrades(botId, limit);

  // Count trades per time bucket
  const buckets = new Array(limit).fill(0);

  trades.forEach(trade => {
    const now = new Date();
    const tradeTime = new Date(trade.entryTime);
    const diffMs = now.getTime() - tradeTime.getTime();

    let bucketSize = 60000; // 1 minute default
    if (timeframe === '24H') bucketSize = 3600000; // 1 hour
    if (timeframe === '7D') bucketSize = 3600000; // 1 hour
    if (timeframe === '30D') bucketSize = 3600000; // 1 hour

    const bucketIndex = Math.floor(diffMs / bucketSize);
    if (bucketIndex >= 0 && bucketIndex < buckets.length) {
      buckets[buckets.length - 1 - bucketIndex]++;
    }
  });

  return buckets;
}
