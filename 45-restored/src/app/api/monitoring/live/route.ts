/**
 * LIVE MONITORING API
 * Real-time bot metrics, alerts, performance data
 * Connected to AzurePoweredQuantumBot
 */

import { NextRequest, NextResponse } from 'next/server';
import AlertService from '@/lib/alert-service';
import BotConnectorService from '@/lib/bot-connector';

export const dynamic = 'force-dynamic';

interface LiveMetrics {
  bot: {
    isRunning: boolean;
    status: 'ACTIVE' | 'PAUSED' | 'STOPPED' | 'ERROR';
    uptime: number;
    lastUpdate: string;
  };
  performance: {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    totalPnL: number;
    dailyPnL: number;
    sharpeRatio: number;
    maxDrawdown: number;
    currentDrawdown: number;
  };
  positions: {
    open: number;
    totalValue: number;
    unrealizedPnL: number;
  };
  risk: {
    dailyLoss: number;
    maxDailyLoss: number;
    utilizationPercent: number;
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  };
  compliance: {
    status: 'COMPLIANT' | 'WARNING' | 'VIOLATION';
    violations: string[];
    lastCheck: string;
  };
  alerts: {
    critical: number;
    high: number;
    medium: number;
    low: number;
    recent: any[];
  };
}

// Mock data (will be replaced with real bot data)
let mockMetrics: LiveMetrics = {
  bot: {
    isRunning: false,
    status: 'STOPPED',
    uptime: 0,
    lastUpdate: new Date().toISOString(),
  },
  performance: {
    totalTrades: 0,
    winningTrades: 0,
    losingTrades: 0,
    winRate: 0,
    totalPnL: 0,
    dailyPnL: 0,
    sharpeRatio: 0,
    maxDrawdown: 0,
    currentDrawdown: 0,
  },
  positions: {
    open: 0,
    totalValue: 0,
    unrealizedPnL: 0,
  },
  risk: {
    dailyLoss: 0,
    maxDailyLoss: 1000,
    utilizationPercent: 0,
    riskLevel: 'LOW',
  },
  compliance: {
    status: 'COMPLIANT',
    violations: [],
    lastCheck: new Date().toISOString(),
  },
  alerts: {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
    recent: [],
  },
};

export async function GET(request: NextRequest) {
  try {
    const alertService = AlertService.getInstance();
    const botConnector = BotConnectorService.getInstance();
    const alerts = alertService.getAlerts(10);

    // Count alerts by severity
    const alertCounts = alerts.reduce(
      (acc, alert) => {
        const severity = alert.severity.toLowerCase();
        acc[severity] = (acc[severity] || 0) + 1;
        return acc;
      },
      { critical: 0, high: 0, medium: 0, low: 0 }
    );

    // Get real bot metrics (or fallback to mock)
    const botMetrics = await botConnector.getMetrics();

    const metrics: LiveMetrics = {
      ...botMetrics,
      alerts: {
        ...alertCounts,
        recent: alerts.map(a => ({
          id: a.id,
          type: a.type,
          severity: a.severity,
          title: a.title,
          message: a.message,
          timestamp: a.timestamp,
          acknowledged: a.acknowledged,
        })),
      },
    };

    return NextResponse.json({
      success: true,
      data: metrics,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Live monitoring error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}

// POST endpoint for bot control
export async function POST(request: NextRequest) {
  try {
    const { action, data } = await request.json();
    const botConnector = BotConnectorService.getInstance();
    const alertService = AlertService.getInstance();

    switch (action) {
      case 'start':
        await botConnector.startBot();
        break;

      case 'stop':
        await botConnector.stopBot();
        break;

      case 'emergency_stop':
        await botConnector.emergencyStop();
        break;

      case 'acknowledge_alert':
        if (data?.alertId) {
          alertService.acknowledgeAlert(data.alertId);
        }
        break;

      default:
        return NextResponse.json(
          {
            success: false,
            error: 'Invalid action',
          },
          { status: 400 }
        );
    }

    // Get updated metrics after action
    const metrics = await botConnector.getMetrics();

    return NextResponse.json({
      success: true,
      message: `Action '${action}' executed successfully`,
      data: metrics,
    });
  } catch (error: any) {
    console.error('Monitoring control error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
