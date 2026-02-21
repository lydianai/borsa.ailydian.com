/**
 * PERFORMANCE & ANALYTICS API
 * Track system performance, signal accuracy, and generate reports
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { performanceAnalyticsDB } from '@/lib/database';

interface PerformanceMetrics {
  overview: {
    totalSignals: number;
    winRate: number;
    totalProfit: number;
    avgConfidence: number;
    activeSince: string;
  };
  strategyPerformance: {
    [key: string]: {
      signals: number;
      winRate: number;
      profit: number;
      avgResponseTime: number;
    };
  };
  timeBasedMetrics: {
    hourly: { signals: number; winRate: number }[];
    daily: { date: string; signals: number; profit: number }[];
  };
  recentSignals: {
    symbol: string;
    type: string;
    confidence: number;
    profit: number | null;
    timestamp: string;
  }[];
}

const DEFAULT_METRICS: PerformanceMetrics = {
  overview: {
    totalSignals: 0,
    winRate: 0,
    totalProfit: 0,
    avgConfidence: 0,
    activeSince: new Date().toISOString(),
  },
  strategyPerformance: {},
  timeBasedMetrics: {
    hourly: [],
    daily: [],
  },
  recentSignals: [],
};

// Database storage (persistent, encrypted)
// Old: const metricsStore = new Map<string, PerformanceMetrics>();
// Now using: performanceAnalyticsDB from @/lib/database

async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  return sessionId;
}

export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    let metrics = performanceAnalyticsDB.get(sessionId);

    if (!metrics) {
      // Generate mock data for demo
      metrics = {
        overview: {
          totalSignals: 1247,
          winRate: 72.3,
          totalProfit: 15420.50,
          avgConfidence: 78.5,
          activeSince: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
        },
        strategyPerformance: {
          'ai-signals': { signals: 342, winRate: 75.2, profit: 4230.20, avgResponseTime: 145 },
          'quantum-signals': { signals: 198, winRate: 80.1, profit: 3890.40, avgResponseTime: 220 },
          'conservative-signals': { signals: 156, winRate: 85.3, profit: 2670.80, avgResponseTime: 98 },
        },
        timeBasedMetrics: {
          hourly: Array.from({ length: 24 }, (_, _i) => ({
            signals: Math.floor(Math.random() * 50) + 10,
            winRate: Math.random() * 30 + 60,
          })),
          daily: Array.from({ length: 30 }, (_, i) => ({
            date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            signals: Math.floor(Math.random() * 100) + 20,
            profit: Math.random() * 1000 - 200,
          })),
        },
        recentSignals: Array.from({ length: 10 }, (_, i) => ({
          symbol: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'][Math.floor(Math.random() * 3)],
          type: ['BUY', 'SELL', 'STRONG_BUY'][Math.floor(Math.random() * 3)],
          confidence: Math.floor(Math.random() * 30) + 70,
          profit: Math.random() > 0.3 ? Math.random() * 500 : -Math.random() * 200,
          timestamp: new Date(Date.now() - i * 60 * 60 * 1000).toISOString(),
        })),
      };
      performanceAnalyticsDB.set(sessionId, metrics);
    }

    return NextResponse.json({ success: true, data: metrics });
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Failed to get metrics' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    if (body.action === 'export_report') {
      const metrics = performanceAnalyticsDB.get(sessionId) || DEFAULT_METRICS;
      return NextResponse.json({
        success: true,
        data: metrics,
        message: 'Report exported successfully',
      });
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ success: false, error: 'Failed to update' }, { status: 500 });
  }
}
