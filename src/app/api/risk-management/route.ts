/**
 * RISK MANAGEMENT API
 * Manages user risk settings: stop-loss, daily limits, position sizing
 *
 * Features:
 * - Global stop-loss percentage
 * - Daily loss limits
 * - Position size calculator parameters
 * - Risk/Reward ratio settings
 * - Auto-close settings
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { riskManagementDB } from '@/lib/database';

// Risk Management Settings Interface
interface RiskSettings {
  // Stop-Loss Settings
  globalStopLoss: {
    enabled: boolean;
    percentage: number; // 1-20% default: 2%
    trailingStop: boolean;
    trailingDistance: number; // 0.5-5% default: 1%
  };

  // Daily Limits
  dailyLimits: {
    enabled: boolean;
    maxDailyLoss: number; // USD amount, default: 1000
    maxDailyTrades: number; // 1-100, default: 20
    pauseTradingOnLimit: boolean;
  };

  // Position Sizing
  positionSizing: {
    method: 'fixed' | 'percentage' | 'kelly'; // default: percentage
    fixedAmount: number; // USD, default: 100
    portfolioPercentage: number; // 1-50%, default: 2%
    maxPositionSize: number; // USD, default: 5000
  };

  // Risk/Reward Settings
  riskReward: {
    minRatio: number; // 1-5, default: 2 (2:1 ratio)
    autoCalculate: boolean;
  };

  // Auto-Close Settings
  autoClose: {
    enabled: boolean;
    takeProfit: number; // percentage, default: 5%
    stopLoss: number; // percentage, default: 2%
  };

  // Alerts
  alerts: {
    notifyOnStopLoss: boolean;
    notifyOnDailyLimit: boolean;
    notifyOnLargePosition: boolean;
  };
}

// Default Risk Settings (Conservative)
const DEFAULT_RISK_SETTINGS: RiskSettings = {
  globalStopLoss: {
    enabled: true,
    percentage: 2,
    trailingStop: false,
    trailingDistance: 1,
  },
  dailyLimits: {
    enabled: true,
    maxDailyLoss: 1000,
    maxDailyTrades: 20,
    pauseTradingOnLimit: true,
  },
  positionSizing: {
    method: 'percentage',
    fixedAmount: 100,
    portfolioPercentage: 2,
    maxPositionSize: 5000,
  },
  riskReward: {
    minRatio: 2,
    autoCalculate: true,
  },
  autoClose: {
    enabled: false,
    takeProfit: 5,
    stopLoss: 2,
  },
  alerts: {
    notifyOnStopLoss: true,
    notifyOnDailyLimit: true,
    notifyOnLargePosition: true,
  },
};

// Database storage (persistent, encrypted)
// Old: const riskSettingsStore = new Map<string, RiskSettings>();
// Old: const dailyStatsStore = new Map<string, DailyStats>();
// Now using: riskManagementDB from @/lib/database

// Daily trading stats (for limit tracking)
interface DailyStats {
  date: string;
  totalLoss: number;
  totalProfit: number;
  tradesCount: number;
}

/**
 * Get session ID from cookies or generate new one
 */
async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  return sessionId;
}

/**
 * Validate risk settings input
 */
function validateRiskSettings(settings: Partial<RiskSettings>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Validate stop-loss percentage
  if (settings.globalStopLoss) {
    if (settings.globalStopLoss.percentage < 0.5 || settings.globalStopLoss.percentage > 20) {
      errors.push('Stop-loss percentage must be between 0.5% and 20%');
    }
    if (settings.globalStopLoss.trailingDistance < 0.1 || settings.globalStopLoss.trailingDistance > 10) {
      errors.push('Trailing stop distance must be between 0.1% and 10%');
    }
  }

  // Validate daily limits
  if (settings.dailyLimits) {
    if (settings.dailyLimits.maxDailyLoss < 10 || settings.dailyLimits.maxDailyLoss > 1000000) {
      errors.push('Daily loss limit must be between $10 and $1,000,000');
    }
    if (settings.dailyLimits.maxDailyTrades < 1 || settings.dailyLimits.maxDailyTrades > 1000) {
      errors.push('Max daily trades must be between 1 and 1000');
    }
  }

  // Validate position sizing
  if (settings.positionSizing) {
    if (settings.positionSizing.portfolioPercentage < 0.1 || settings.positionSizing.portfolioPercentage > 100) {
      errors.push('Portfolio percentage must be between 0.1% and 100%');
    }
    if (settings.positionSizing.maxPositionSize < 10 || settings.positionSizing.maxPositionSize > 10000000) {
      errors.push('Max position size must be between $10 and $10,000,000');
    }
  }

  // Validate risk/reward ratio
  if (settings.riskReward) {
    if (settings.riskReward.minRatio < 0.5 || settings.riskReward.minRatio > 10) {
      errors.push('Risk/reward ratio must be between 0.5 and 10');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * GET - Retrieve risk management settings
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Get settings or use defaults
    let settings = riskManagementDB.get(sessionId);
    if (!settings) {
      settings = DEFAULT_RISK_SETTINGS;
      riskManagementDB.set(sessionId, settings);
    }

    // Get daily stats
    const today = new Date().toISOString().split('T')[0];
    const statsKey = `stats_${sessionId}_${today}`;
    let stats = riskManagementDB.get(statsKey);
    if (!stats) {
      stats = {
        date: today,
        totalLoss: 0,
        totalProfit: 0,
        tradesCount: 0,
      };
    }

    // Check if daily limits reached
    const limitsReached = {
      dailyLoss: settings.dailyLimits.enabled && stats.totalLoss >= settings.dailyLimits.maxDailyLoss,
      dailyTrades: settings.dailyLimits.enabled && stats.tradesCount >= settings.dailyLimits.maxDailyTrades,
    };

    return NextResponse.json({
      success: true,
      data: {
        settings,
        dailyStats: stats,
        limitsReached,
        tradingPaused: settings.dailyLimits.pauseTradingOnLimit && (limitsReached.dailyLoss || limitsReached.dailyTrades),
      },
    });
  } catch (error) {
    console.error('[Risk Management API] GET Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get risk settings',
      },
      { status: 500 }
    );
  }
}

/**
 * POST - Update risk management settings
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    // Validate input
    const validation = validateRiskSettings(body);
    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid risk settings',
          details: validation.errors,
        },
        { status: 400 }
      );
    }

    // Get current settings or defaults
    const currentSettings = riskManagementDB.get(sessionId) || DEFAULT_RISK_SETTINGS;

    // Merge with new settings
    const updatedSettings: RiskSettings = {
      globalStopLoss: { ...currentSettings.globalStopLoss, ...(body.globalStopLoss || {}) },
      dailyLimits: { ...currentSettings.dailyLimits, ...(body.dailyLimits || {}) },
      positionSizing: { ...currentSettings.positionSizing, ...(body.positionSizing || {}) },
      riskReward: { ...currentSettings.riskReward, ...(body.riskReward || {}) },
      autoClose: { ...currentSettings.autoClose, ...(body.autoClose || {}) },
      alerts: { ...currentSettings.alerts, ...(body.alerts || {}) },
    };

    // Save to database
    riskManagementDB.set(sessionId, updatedSettings);

    // Create response with Set-Cookie header
    const response = NextResponse.json({
      success: true,
      data: updatedSettings,
      message: 'Risk settings updated successfully',
    });

    // Set session cookie (7 days)
    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return response;
  } catch (error) {
    console.error('[Risk Management API] POST Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update risk settings',
      },
      { status: 500 }
    );
  }
}

/**
 * PUT - Reset to default settings
 */
export async function PUT(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Reset to defaults
    riskManagementDB.set(sessionId, DEFAULT_RISK_SETTINGS);

    return NextResponse.json({
      success: true,
      data: DEFAULT_RISK_SETTINGS,
      message: 'Risk settings reset to defaults',
    });
  } catch (error) {
    console.error('[Risk Management API] PUT Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset risk settings',
      },
      { status: 500 }
    );
  }
}
