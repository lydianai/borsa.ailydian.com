/**
 * SIGNAL MONITOR API
 * Control and query the continuous signal monitoring service
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Mock monitor state
let monitorState = {
  running: false,
  lastScan: new Date().toISOString(),
  scanCount: 0
};

// Mock stats generator
function generateMockStats() {
  return {
    status: monitorState.running ? 'running' : 'stopped',
    uptime: Math.floor(Math.random() * 86400),
    totalScans: monitorState.scanCount,
    lastScan: monitorState.lastScan,
    activeAlerts: Math.floor(Math.random() * 5),
    totalAlerts24h: Math.floor(Math.random() * 20),
    avgScanTime: 1200 + Math.random() * 500,
    watchlist: ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'AVAX', 'MATIC', 'LINK']
  };
}

// Mock active alerts generator
function generateMockAlerts() {
  const symbols = ['BTC', 'ETH', 'BNB', 'SOL'];
  const alertTypes = ['Strong Buy Signal', 'Overbought Warning', 'Support Level Test', 'Volume Spike'];

  return symbols.slice(0, Math.floor(Math.random() * 3) + 1).map(symbol => ({
    id: Math.random().toString(36).substr(2, 9),
    symbol,
    type: alertTypes[Math.floor(Math.random() * alertTypes.length)],
    confidence: 0.75 + Math.random() * 0.20,
    timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
    action: Math.random() > 0.5 ? 'BUY' : 'SELL',
    price: 50000 + Math.random() * 10000
  }));
}

// Mock alert history generator
function generateMockHistory(limit: number) {
  const count = Math.min(limit, 50);
  const symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP'];

  return Array.from({ length: count }, (_, i) => ({
    id: Math.random().toString(36).substr(2, 9),
    symbol: symbols[Math.floor(Math.random() * symbols.length)],
    action: Math.random() > 0.5 ? 'BUY' : 'SELL',
    confidence: 0.70 + Math.random() * 0.25,
    timestamp: new Date(Date.now() - i * 300000).toISOString(),
    result: Math.random() > 0.4 ? 'triggered' : 'expired'
  }));
}

/**
 * GET /api/quantum-pro/monitor
 * Get monitor status and stats
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');

    if (action === 'stats') {
      const stats = generateMockStats();
      return NextResponse.json({
        success: true,
        stats
      });
    }

    if (action === 'active') {
      const alerts = generateMockAlerts();
      return NextResponse.json({
        success: true,
        count: alerts.length,
        alerts
      });
    }

    if (action === 'history') {
      const limit = parseInt(searchParams.get('limit') || '100');
      const history = generateMockHistory(limit);
      return NextResponse.json({
        success: true,
        count: history.length,
        history
      });
    }

    // Default: return stats
    const stats = generateMockStats();
    return NextResponse.json({
      success: true,
      stats
    });

  } catch (error: any) {
    console.error('Monitor GET error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}

/**
 * POST /api/quantum-pro/monitor
 * Control monitor (start/stop/scan)
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action === 'start') {
      monitorState.running = true;
      return NextResponse.json({
        success: true,
        message: 'Signal monitor started'
      });
    }

    if (action === 'stop') {
      monitorState.running = false;
      return NextResponse.json({
        success: true,
        message: 'Signal monitor stopped'
      });
    }

    if (action === 'scan') {
      monitorState.lastScan = new Date().toISOString();
      monitorState.scanCount++;
      return NextResponse.json({
        success: true,
        message: 'Market scan completed',
        results: {
          scanned: 10,
          signalsFound: Math.floor(Math.random() * 5),
          timestamp: monitorState.lastScan
        }
      });
    }

    return NextResponse.json({
      success: false,
      error: 'Invalid action. Use: start, stop, or scan'
    }, { status: 400 });

  } catch (error: any) {
    console.error('Monitor POST error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}