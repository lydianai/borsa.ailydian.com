/**
 * 💰 POSITION SIZE CALCULATOR API
 *
 * Calculates optimal position size based on risk management principles
 * - Account balance & risk percentage (1-2% recommendation)
 * - ATR-based volatility adjustment
 * - Leverage recommendations based on signal quality
 * - Stop-loss placement optimization
 * - Kelly Criterion optional calculation
 *
 * WHITE-HAT PRINCIPLES:
 * - Educational risk management
 * - Read-only calculations
 * - No automated trading
 * - Conservative recommendations
 *
 * USAGE:
 * GET /api/bot-analysis/position-calculator?symbol=BTCUSDT&accountBalance=10000&riskPercent=1.5&signalScore=85
 */

import { NextRequest, NextResponse } from 'next/server';
import type { BotAnalysisAPIResponse } from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// CONSTANTS
// ============================================================================

const DEFAULT_ACCOUNT_BALANCE = 10000; // $10,000 default
const DEFAULT_RISK_PERCENT = 1.5; // 1.5% risk per trade (conservative)
const MAX_RISK_PERCENT = 3; // Maximum 3% risk
const _ATR_PERIOD = 14; // 14-period ATR for volatility
const KELLY_MULTIPLIER = 0.25; // Quarter Kelly for safety

// ============================================================================
// TYPES
// ============================================================================

interface PositionCalculation {
  symbol: string;
  timestamp: number;

  // Input parameters
  accountBalance: number;
  riskPercent: number;
  signalScore: number;

  // Calculations
  riskAmount: number; // Dollar amount willing to risk
  optimalPositionSize: number; // USD value of position
  optimalQuantity: number; // Number of contracts/coins

  // Leverage
  recommendedLeverage: number;
  maxLeverage: number;
  leverageWarning: string | null;

  // Stop Loss
  stopLossPrice: number;
  stopLossPercent: number;
  stopLossDistance: number;

  // Take Profit Levels
  takeProfitLevels: Array<{
    level: number;
    price: number;
    percentGain: number;
    percentOfPosition: number;
  }>;

  // Risk/Reward
  riskRewardRatio: number;
  expectedValue: number;

  // Kelly Criterion (optional advanced)
  kellyPercent: number;
  kellyPositionSize: number;

  // Recommendations
  recommendations: string[];
  warnings: string[];
}

// ============================================================================
// HELPER: FETCH CURRENT PRICE
// ============================================================================

async function fetchCurrentPrice(symbol: string): Promise<number> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/ticker/price?symbol=${symbol}`;
    const response = await fetch(url);
    const data = await response.json();
    return parseFloat(data.price);
  } catch (error) {
    console.error('[Position Calculator] Failed to fetch price:', error);
    return 0;
  }
}

// ============================================================================
// HELPER: CALCULATE ATR (AVERAGE TRUE RANGE)
// ============================================================================

async function calculateATR(symbol: string): Promise<number> {
  try {
    // Fetch recent klines for ATR calculation
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=1h&limit=20`;
    const response = await fetch(url);
    const klines = await response.json();

    if (!klines || klines.length === 0) {
      return 0;
    }

    // Calculate True Range for each candle
    const trueRanges: number[] = [];

    for (let i = 1; i < klines.length; i++) {
      const high = parseFloat(klines[i][2]);
      const low = parseFloat(klines[i][3]);
      const prevClose = parseFloat(klines[i - 1][4]);

      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );

      trueRanges.push(tr);
    }

    // Average True Range
    const atr = trueRanges.reduce((sum, tr) => sum + tr, 0) / trueRanges.length;
    return atr;

  } catch (error) {
    console.error('[Position Calculator] Failed to calculate ATR:', error);
    return 0;
  }
}

// ============================================================================
// HELPER: RECOMMEND LEVERAGE BASED ON SIGNAL QUALITY
// ============================================================================

function recommendLeverage(signalScore: number): {
  recommended: number;
  max: number;
  warning: string | null;
} {
  // Higher signal quality = can use slightly higher leverage (still conservative)

  if (signalScore >= 85) {
    // EXCELLENT signal
    return {
      recommended: 3,
      max: 5,
      warning: null
    };
  } else if (signalScore >= 70) {
    // GOOD signal
    return {
      recommended: 2,
      max: 3,
      warning: 'Orta-yüksek güven seviyesi. Dikkatli leverage kullanın.'
    };
  } else if (signalScore >= 55) {
    // MODERATE signal
    return {
      recommended: 1,
      max: 2,
      warning: 'Zayıf sinyal. Leverage kullanımı risklidir, 1x önerilir.'
    };
  } else {
    // POOR/NONE signal
    return {
      recommended: 1,
      max: 1,
      warning: 'UYARI: Sinyal zayıf. Pozisyon açmak önerilmez. Eğer açacaksanız 1x kullanın.'
    };
  }
}

// ============================================================================
// HELPER: CALCULATE KELLY CRITERION
// ============================================================================

function calculateKelly(
  winRate: number, // Probability of winning (0-1)
  avgWin: number,   // Average win amount
  avgLoss: number   // Average loss amount
): number {
  // Kelly Formula: K = (W * R - L) / R
  // W = win probability, L = loss probability, R = win/loss ratio

  const lossRate = 1 - winRate;
  const winLossRatio = avgWin / avgLoss;

  const kellyPercent = (winRate * winLossRatio - lossRate) / winLossRatio;

  // Return quarter Kelly for safety (full Kelly is too aggressive)
  return Math.max(0, kellyPercent * KELLY_MULTIPLIER);
}

// ============================================================================
// MAIN CALCULATION
// ============================================================================

async function calculatePosition(
  symbol: string,
  accountBalance: number,
  riskPercent: number,
  signalScore: number
): Promise<PositionCalculation> {

  // Fetch current price
  const currentPrice = await fetchCurrentPrice(symbol);

  // Calculate ATR for volatility-based stop loss
  const atr = await calculateATR(symbol);
  const atrPercent = currentPrice > 0 ? (atr / currentPrice) * 100 : 2;

  // Calculate risk amount in dollars
  const riskAmount = accountBalance * (riskPercent / 100);

  // Stop Loss placement (1.5 x ATR below entry)
  const stopLossPercent = Math.max(1.0, atrPercent * 1.5); // Minimum 1% stop
  const stopLossPrice = currentPrice * (1 - stopLossPercent / 100);
  const stopLossDistance = currentPrice - stopLossPrice;

  // Optimal position size = Risk Amount / Stop Loss Distance (in %)
  const optimalPositionSize = riskAmount / (stopLossPercent / 100);
  const optimalQuantity = currentPrice > 0 ? optimalPositionSize / currentPrice : 0;

  // Leverage recommendation
  const { recommended, max, warning } = recommendLeverage(signalScore);

  // Take Profit Levels (based on ATR)
  const takeProfitLevels = [
    {
      level: 1,
      price: currentPrice * (1 + atrPercent * 1.5 / 100),
      percentGain: atrPercent * 1.5,
      percentOfPosition: 30
    },
    {
      level: 2,
      price: currentPrice * (1 + atrPercent * 3 / 100),
      percentGain: atrPercent * 3,
      percentOfPosition: 40
    },
    {
      level: 3,
      price: currentPrice * (1 + atrPercent * 5 / 100),
      percentGain: atrPercent * 5,
      percentOfPosition: 30
    }
  ];

  // Risk/Reward Ratio (average TP vs SL)
  const avgTpGain = (takeProfitLevels[0].percentGain * 0.3 +
                     takeProfitLevels[1].percentGain * 0.4 +
                     takeProfitLevels[2].percentGain * 0.3);
  const riskRewardRatio = avgTpGain / stopLossPercent;

  // Kelly Criterion (assume 60% win rate based on signal quality)
  const winRate = signalScore >= 85 ? 0.65 : signalScore >= 70 ? 0.60 : 0.55;
  const avgWin = avgTpGain;
  const avgLoss = stopLossPercent;
  const kellyPercent = calculateKelly(winRate, avgWin, avgLoss) * 100;
  const kellyPositionSize = accountBalance * (kellyPercent / 100);

  // Expected Value
  const expectedValue = (winRate * avgWin) - ((1 - winRate) * avgLoss);

  // Recommendations
  const recommendations: string[] = [];
  const warnings: string[] = [];

  if (riskPercent > MAX_RISK_PERCENT) {
    warnings.push(`Risk oranı çok yüksek (${riskPercent}%). Maximum ${MAX_RISK_PERCENT}% önerilir.`);
  }

  if (riskRewardRatio < 2) {
    warnings.push(`Risk/Reward oranı düşük (${riskRewardRatio.toFixed(2)}:1). Minimum 2:1 hedeflenmelidir.`);
  }

  if (signalScore >= 85) {
    recommendations.push('Mükemmel sinyal kalitesi. Güvenle pozisyon açabilirsiniz.');
  } else if (signalScore >= 70) {
    recommendations.push('İyi sinyal kalitesi. Orta risk ile giriş yapılabilir.');
  } else if (signalScore < 55) {
    recommendations.push('UYARI: Sinyal zayıf. Pozisyon açmadan önce ek analiz yapın.');
  }

  if (optimalPositionSize > accountBalance * 0.2) {
    warnings.push('Pozisyon büyüklüğü hesap bakiyesinin %20\'sini aşıyor. Daha küçük pozisyon düşünün.');
  }

  recommendations.push(`ATR bazlı stop loss: ${stopLossPercent.toFixed(2)}% (${stopLossPrice.toFixed(2)} USD)`);
  recommendations.push(`Önerilen leverage: ${recommended}x (maksimum ${max}x)`);
  recommendations.push(`3 seviyeli take profit stratejisi kullanın`);

  if (kellyPercent > 0 && kellyPercent < riskPercent) {
    recommendations.push(`Kelly Criterion daha düşük pozisyon önerir (%${kellyPercent.toFixed(2)})`);
  }

  return {
    symbol,
    timestamp: Date.now(),
    accountBalance,
    riskPercent,
    signalScore,
    riskAmount,
    optimalPositionSize,
    optimalQuantity,
    recommendedLeverage: recommended,
    maxLeverage: max,
    leverageWarning: warning,
    stopLossPrice,
    stopLossPercent,
    stopLossDistance,
    takeProfitLevels,
    riskRewardRatio,
    expectedValue,
    kellyPercent,
    kellyPositionSize,
    recommendations,
    warnings
  };
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';
  const accountBalance = parseFloat(searchParams.get('accountBalance') || String(DEFAULT_ACCOUNT_BALANCE));
  const riskPercent = parseFloat(searchParams.get('riskPercent') || String(DEFAULT_RISK_PERCENT));
  const signalScore = parseFloat(searchParams.get('signalScore') || '50');

  try {
    console.log(
      `[Position Calculator] Calculating for ${symbol} - Balance: $${accountBalance}, Risk: ${riskPercent}%, Signal: ${signalScore}`
    );

    // Validate inputs
    if (accountBalance <= 0 || accountBalance > 10000000) {
      throw new Error('Invalid account balance. Must be between $1 and $10,000,000');
    }

    if (riskPercent <= 0 || riskPercent > 10) {
      throw new Error('Invalid risk percent. Must be between 0.1% and 10%');
    }

    if (signalScore < 0 || signalScore > 100) {
      throw new Error('Invalid signal score. Must be between 0 and 100');
    }

    // Calculate position
    const calculation = await calculatePosition(symbol, accountBalance, riskPercent, signalScore);

    const duration = Date.now() - startTime;

    console.log(
      `[Position Calculator] ${symbol} calculated in ${duration}ms - Position Size: $${calculation.optimalPositionSize.toFixed(2)}, R:R: ${calculation.riskRewardRatio.toFixed(2)}:1`
    );

    const response: BotAnalysisAPIResponse<PositionCalculation> = {
      success: true,
      data: calculation,
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Position Calculator] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Position calculation failed',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
