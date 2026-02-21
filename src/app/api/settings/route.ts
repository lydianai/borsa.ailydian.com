/**
 * SETTINGS API
 * Quantum Pro configuration management
 * WHITE HAT: Educational settings only
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

interface QuantumProSettings {
  signals: {
    minConfidence: number;
    refreshInterval: number;
    maxSignals: number;
  };
  backtest: {
    defaultPeriod: string;
    showAllStrategies: boolean;
  };
  risk: {
    maxPositionSize: number;
    stopLossPercent: number;
    takeProfitPercent: number;
    dailyLossLimit: number;
  };
  bots: {
    autoStart: boolean;
    maxConcurrentBots: number;
  };
  monitoring: {
    refreshInterval: number;
    showLivePositions: boolean;
  };
}

const defaultSettings: QuantumProSettings = {
  signals: {
    minConfidence: 0.60,
    refreshInterval: 30,
    maxSignals: 50,
  },
  backtest: {
    defaultPeriod: '30days',
    showAllStrategies: true,
  },
  risk: {
    maxPositionSize: 2,
    stopLossPercent: 1.5,
    takeProfitPercent: 3.0,
    dailyLossLimit: 5.0,
  },
  bots: {
    autoStart: false,
    maxConcurrentBots: 5,
  },
  monitoring: {
    refreshInterval: 5,
    showLivePositions: true,
  },
};

export async function GET(_request: NextRequest) {
  try {
    console.log('[Settings] Fetching Quantum Pro settings...');

    return NextResponse.json({
      success: true,
      settings: defaultSettings,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Settings] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { settings } = body;

    console.log('[Settings] Updating Quantum Pro settings...');

    // WHITE HAT: Educational only - settings stored client-side
    return NextResponse.json({
      success: true,
      message: 'Settings updated successfully (Client-side storage)',
      settings,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Settings] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
