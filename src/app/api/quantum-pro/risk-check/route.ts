/**
 * QUANTUM PRO RISK CHECK API
 * Validate signals against risk limits
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Mock risk state
let riskState = {
  dailyPnL: 0,
  weeklyPnL: 0,
  openPositions: 0,
  totalExposure: 0,
  lastReset: new Date().toISOString()
};

// Mock risk check logic
function checkMockRisk(signal: {
  symbol: string;
  action: string;
  confidence: number;
  riskScore: number;
  price: number;
}) {
  const { confidence, riskScore } = signal;

  // Basic approval logic
  const highConfidence = confidence >= 0.75;
  const lowRisk = riskScore <= 50;
  const withinLimits = riskState.openPositions < 10 && Math.abs(riskState.dailyPnL) < 1000;

  const approved = highConfidence && lowRisk && withinLimits;

  let reason = '';
  const warnings: string[] = [];

  if (!approved) {
    if (!highConfidence) reason = 'Confidence too low';
    else if (!lowRisk) reason = 'Risk score too high';
    else if (!withinLimits) reason = 'Risk limits exceeded';
  } else {
    reason = 'Signal approved';
  }

  if (confidence < 0.80) warnings.push('Confidence below optimal threshold');
  if (riskScore > 40) warnings.push('Elevated risk score');
  if (riskState.openPositions > 7) warnings.push('High number of open positions');

  const suggestedPositionSize = approved ? Math.max(0.5, (1 - riskScore / 100) * 2) : 0;

  return {
    approved,
    reason,
    suggestedPositionSize: Math.round(suggestedPositionSize * 100) / 100,
    warnings
  };
}

// Mock risk report generator
function generateMockRiskReport() {
  return {
    dailyPnL: riskState.dailyPnL,
    weeklyPnL: riskState.weeklyPnL,
    openPositions: riskState.openPositions,
    totalExposure: riskState.totalExposure,
    portfolioRiskScore: Math.round((Math.abs(riskState.dailyPnL) / 10 + riskState.openPositions * 5) * 100) / 100,
    limits: {
      dailyLossLimit: -1000,
      weeklyLossLimit: -5000,
      maxOpenPositions: 10,
      maxExposure: 50000
    },
    positions: Array.from({ length: riskState.openPositions }, (_, i) => ({
      symbol: ['BTC', 'ETH', 'BNB', 'SOL'][i % 4],
      side: Math.random() > 0.5 ? 'LONG' : 'SHORT',
      size: Math.round(Math.random() * 5000),
      unrealizedPnL: (Math.random() - 0.5) * 200
    }))
  };
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, action, confidence, riskScore, price } = body;

    if (!symbol || !action || confidence === undefined || riskScore === undefined || !price) {
      return NextResponse.json({
        success: false,
        error: 'Missing required parameters: symbol, action, confidence, riskScore, price'
      }, { status: 400 });
    }

    if (!['BUY', 'SELL'].includes(action)) {
      return NextResponse.json({
        success: false,
        error: 'Action must be BUY or SELL'
      }, { status: 400 });
    }

    // Check signal risk
    const riskCheck = checkMockRisk({
      symbol,
      action,
      confidence,
      riskScore,
      price
    });

    // Get current risk report
    const riskReport = generateMockRiskReport();

    return NextResponse.json({
      success: true,
      approved: riskCheck.approved,
      reason: riskCheck.reason,
      suggestedPositionSize: riskCheck.suggestedPositionSize,
      warnings: riskCheck.warnings,
      limits: {
        dailyPnL: riskReport.dailyPnL,
        dailyLimit: riskReport.limits.dailyLossLimit,
        weeklyPnL: riskReport.weeklyPnL,
        weeklyLimit: riskReport.limits.weeklyLossLimit,
        openPositions: riskReport.openPositions,
        maxPositions: riskReport.limits.maxOpenPositions,
        totalExposure: riskReport.totalExposure,
        portfolioRiskScore: riskReport.portfolioRiskScore
      },
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('Risk check API error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}

export async function GET(_request: NextRequest) {
  try {
    // Get risk report
    const report = generateMockRiskReport();

    return NextResponse.json({
      success: true,
      report,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('Risk report API error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}