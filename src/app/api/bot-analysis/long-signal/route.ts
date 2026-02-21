/**
 * 🎯 COMPOSITE LONG SIGNAL SCORING API
 *
 * Combines all bot analysis data sources to generate a comprehensive LONG signal score
 *
 * SCORING COMPONENTS:
 * - Order Book Imbalance (25%): Bid/ask pressure analysis
 * - Funding Rate (20%): Funding rate sentiment
 * - Open Interest & CVD (30%): Position flow and volume delta
 * - Whale Walls (15%): Large order detection
 * - Technical Context (10%): Price action confirmation
 *
 * SIGNAL QUALITY:
 * - EXCELLENT (85-100): Strong long opportunity, high confidence
 * - GOOD (70-84): Good long setup, moderate-high confidence
 * - MODERATE (55-69): Weak long signal, low-moderate confidence
 * - POOR (40-54): No clear signal
 * - NONE (0-39): Bearish conditions, avoid long
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only analysis
 * - Educational purpose only
 * - No automated trading execution
 *
 * USAGE:
 * GET /api/bot-analysis/long-signal?symbol=BTCUSDT
 */

import { NextRequest, NextResponse } from 'next/server';
import type {
  LongPositionSignal,
  BotAnalysisAPIResponse
} from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// SCORING WEIGHTS
// ============================================================================

const WEIGHTS = {
  orderbook: 0.25,
  funding: 0.20,
  openInterest: 0.30,
  whale: 0.15,
  technical: 0.10
};

// ============================================================================
// HELPER: FETCH ALL DATA SOURCES
// ============================================================================

async function fetchAllDataSources(symbol: string) {
  const baseUrl = 'http://localhost:3000/api/bot-analysis';

  try {
    const [orderbookRes, fundingRes, oiRes] = await Promise.all([
      fetch(`${baseUrl}/orderbook?symbol=${symbol}`),
      fetch(`${baseUrl}/funding?symbol=${symbol}`),
      fetch(`${baseUrl}/open-interest?symbol=${symbol}`)
    ]);

    const [orderbookData, fundingData, oiData] = await Promise.all([
      orderbookRes.json(),
      fundingRes.json(),
      oiRes.json()
    ]);

    return {
      orderbook: orderbookData.success ? orderbookData.data : null,
      funding: fundingData.success ? fundingData.data : null,
      openInterest: oiData.success ? oiData.data : null
    };
  } catch (error) {
    console.error('[LONG Signal] Failed to fetch data sources:', error);
    throw error;
  }
}

// ============================================================================
// HELPER: CALCULATE ORDER BOOK SCORE
// ============================================================================

function calculateOrderBookScore(orderbookData: any): number {
  if (!orderbookData || !orderbookData.imbalance) return 50;

  const { signal, confidence, bidPressure } = orderbookData.imbalance;

  // BULLISH signal = high score
  if (signal === 'BULLISH') {
    return Math.min(100, 50 + (confidence / 2) + (bidPressure / 2));
  }

  // BEARISH signal = low score
  if (signal === 'BEARISH') {
    return Math.max(0, 50 - (confidence / 2) - (50 - bidPressure) / 2);
  }

  // NEUTRAL
  return 50 + (bidPressure - 50) / 2;
}

// ============================================================================
// HELPER: CALCULATE FUNDING RATE SCORE
// ============================================================================

function calculateFundingScore(fundingData: any): number {
  if (!fundingData || !fundingData.signal) return 50;

  const { signal, confidence } = fundingData.signal;

  // LONG_OPPORTUNITY (negative funding) = high score
  if (signal === 'LONG_OPPORTUNITY') {
    return Math.min(100, 60 + (confidence / 2));
  }

  // EXTREME_LONG opportunity = max score
  if (signal === 'EXTREME_LONG') {
    return 95;
  }

  // SHORT_OPPORTUNITY (positive funding) = low score
  if (signal === 'SHORT_OPPORTUNITY') {
    return Math.max(10, 40 - (confidence / 3));
  }

  // EXTREME_SHORT = very low score
  if (signal === 'EXTREME_SHORT') {
    return 5;
  }

  // NEUTRAL
  return 50;
}

// ============================================================================
// HELPER: CALCULATE OPEN INTEREST & CVD SCORE
// ============================================================================

function calculateOIScore(oiData: any): number {
  if (!oiData || !oiData.interpretation) return 50;

  const { compositeSignal, confidence } = oiData.interpretation;
  const { trend } = oiData.cvd;

  // BULLISH composite (OI + CVD aligned) = high score
  if (compositeSignal === 'BULLISH') {
    return Math.min(100, 60 + (confidence / 2));
  }

  // BEARISH composite = low score
  if (compositeSignal === 'BEARISH') {
    return Math.max(0, 40 - (confidence / 2));
  }

  // NEUTRAL but check CVD trend
  if (trend === 'BULLISH') {
    return 55;
  } else if (trend === 'BEARISH') {
    return 45;
  }

  return 50;
}

// ============================================================================
// HELPER: CALCULATE WHALE SCORE
// ============================================================================

function calculateWhaleScore(orderbookData: any): number {
  if (!orderbookData || !orderbookData.whaleLevels) return 50;

  const whaleLevels = orderbookData.whaleLevels;

  if (whaleLevels.length === 0) return 50;

  // Count bid vs ask whale walls
  const bidWhales = whaleLevels.filter((w: any) => w.side === 'BID').length;
  const askWhales = whaleLevels.filter((w: any) => w.side === 'ASK').length;

  // More bid whales = bullish support = high score
  if (bidWhales > askWhales) {
    const ratio = bidWhales / (askWhales || 1);
    return Math.min(90, 50 + (ratio * 15));
  }

  // More ask whales = resistance = low score
  if (askWhales > bidWhales) {
    const ratio = askWhales / (bidWhales || 1);
    return Math.max(10, 50 - (ratio * 15));
  }

  return 50;
}

// ============================================================================
// HELPER: CALCULATE TECHNICAL SCORE
// ============================================================================

function calculateTechnicalScore(orderbookData: any, oiData: any): number {
  // Simplified technical score based on price momentum
  // In production, this would use actual technical indicators

  let score = 50;

  // If OI is increasing and price is rising = bullish
  if (oiData && oiData.openInterestChange) {
    const { changePercent, priceChange } = oiData.openInterestChange;

    if (changePercent > 0 && priceChange > 0) {
      score += 20;
    } else if (changePercent < 0 && priceChange < 0) {
      score -= 20;
    }
  }

  // Spread analysis
  if (orderbookData && orderbookData.snapshot) {
    const { spreadPercent } = orderbookData.snapshot;

    // Tight spread = good liquidity = slightly bullish
    if (spreadPercent < 0.05) {
      score += 10;
    }
  }

  return Math.max(0, Math.min(100, score));
}

// ============================================================================
// HELPER: CALCULATE ENTRY/EXIT RECOMMENDATIONS
// ============================================================================

function calculateEntryExit(
  _symbol: string,
  overallScore: number,
  orderbookData: any
) {
  const currentPrice = orderbookData?.snapshot?.midPrice || 0;

  // Entry recommendations
  const entry = {
    recommendedPrice: currentPrice,
    priceRange: {
      min: currentPrice * 0.998, // 0.2% below
      max: currentPrice * 1.002  // 0.2% above
    },
    positionSize: overallScore > 70 ? 500 : 300, // USD
    leverage: overallScore > 80 ? 3 : overallScore > 65 ? 2 : 1
  };

  // Exit recommendations (ATR-based)
  const atrPercent = 0.02; // 2% ATR simulation
  const stopLossPercent = 0.015; // 1.5% stop loss
  const tp1 = currentPrice * (1 + atrPercent * 1.5);
  const tp2 = currentPrice * (1 + atrPercent * 3);
  const tp3 = currentPrice * (1 + atrPercent * 5);

  const exit = {
    stopLoss: currentPrice * (1 - stopLossPercent),
    takeProfits: [
      { price: tp1, percentage: 30 },
      { price: tp2, percentage: 40 },
      { price: tp3, percentage: 30 }
    ],
    trailingStop: currentPrice * 0.98 // 2% trailing
  };

  // Risk/Reward
  const potentialGain = ((tp2 - currentPrice) / currentPrice) * 100;
  const potentialLoss = ((currentPrice - exit.stopLoss) / currentPrice) * 100;
  const ratio = potentialGain / potentialLoss;

  return {
    entry,
    exit,
    riskReward: {
      ratio,
      potentialGain,
      potentialLoss,
      confidence: overallScore
    }
  };
}

// ============================================================================
// HELPER: GENERATE SIGNAL QUALITY
// ============================================================================

function getSignalQuality(score: number): 'EXCELLENT' | 'GOOD' | 'MODERATE' | 'POOR' | 'NONE' {
  if (score >= 85) return 'EXCELLENT';
  if (score >= 70) return 'GOOD';
  if (score >= 55) return 'MODERATE';
  if (score >= 40) return 'POOR';
  return 'NONE';
}

// ============================================================================
// HELPER: GENERATE SUMMARY AND REASONS
// ============================================================================

function generateSummary(
  symbol: string,
  quality: string,
  scores: any,
  riskReward: any
): { summary: string; reasons: string[]; warnings: string[] } {
  const reasons: string[] = [];
  const warnings: string[] = [];

  // Analyze component scores
  if (scores.orderbook > 65) {
    reasons.push(`Order book imbalance güçlü alıcı baskısı gösteriyor (${scores.orderbook.toFixed(0)}/100)`);
  }

  if (scores.funding > 65) {
    reasons.push(`Funding rate negatif, short trader'lar fee ödüyor (${scores.funding.toFixed(0)}/100)`);
  }

  if (scores.openInterest > 65) {
    reasons.push(`Open Interest ve CVD long bias gösteriyor (${scores.openInterest.toFixed(0)}/100)`);
  }

  if (scores.whale > 60) {
    reasons.push(`Whale bid wall'ları destek oluşturuyor (${scores.whale.toFixed(0)}/100)`);
  }

  // Warnings
  if (scores.orderbook < 40) {
    warnings.push('Order book satıcı baskısı altında, dikkatli olun');
  }

  if (scores.funding < 35) {
    warnings.push('Funding rate aşırı pozitif, long pozisyonlar riskli');
  }

  if (riskReward.ratio < 2) {
    warnings.push(`Risk/Reward oranı düşük (${riskReward.ratio.toFixed(2)}:1), minimum 2:1 tercih edilir`);
  }

  let summary = '';
  if (quality === 'EXCELLENT') {
    summary = `${symbol} için MÜKEMMEL long fırsatı tespit edildi! Tüm göstergeler bullish, yüksek güven seviyesi.`;
  } else if (quality === 'GOOD') {
    summary = `${symbol} için İYİ long setup var. Çoğu gösterge bullish, orta-yüksek güven seviyesi.`;
  } else if (quality === 'MODERATE') {
    summary = `${symbol} için ORTA seviye long sinyali. Bazı göstergeler bullish, dikkatli giriş önerilir.`;
  } else if (quality === 'POOR') {
    summary = `${symbol} için NET sinyal yok. Bekleme modunda kalın.`;
  } else {
    summary = `${symbol} için BEARISH koşullar tespit edildi. Long pozisyondan kaçının.`;
  }

  return { summary, reasons, warnings };
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';

  try {
    console.log(`[LONG Signal API] Calculating composite score for ${symbol}...`);

    // Fetch all data sources
    const dataSources = await fetchAllDataSources(symbol);

    // Calculate component scores
    const scores = {
      orderbook: calculateOrderBookScore(dataSources.orderbook),
      funding: calculateFundingScore(dataSources.funding),
      liquidation: 0, // Placeholder - liquidation scoring not yet implemented
      openInterest: calculateOIScore(dataSources.openInterest),
      whale: calculateWhaleScore(dataSources.orderbook),
      technical: calculateTechnicalScore(dataSources.orderbook, dataSources.openInterest)
    };

    // Calculate weighted overall score
    const overallScore =
      scores.orderbook * WEIGHTS.orderbook +
      scores.funding * WEIGHTS.funding +
      scores.openInterest * WEIGHTS.openInterest +
      scores.whale * WEIGHTS.whale +
      scores.technical * WEIGHTS.technical;

    // Determine signal quality
    const quality = getSignalQuality(overallScore);
    const shouldNotify = quality === 'EXCELLENT' || quality === 'GOOD';

    // Calculate entry/exit recommendations
    const { entry, exit, riskReward } = calculateEntryExit(
      symbol,
      overallScore,
      dataSources.orderbook
    );

    // Generate summary
    const { summary, reasons, warnings } = generateSummary(
      symbol,
      quality,
      scores,
      riskReward
    );

    const duration = Date.now() - startTime;

    const longSignal: LongPositionSignal = {
      symbol,
      timestamp: Date.now(),
      overallScore,
      quality,
      shouldNotify,
      scores,
      entry,
      exit,
      riskReward,
      summary,
      reasons,
      warnings
    };

    console.log(
      `[LONG Signal API] ${symbol} composite score: ${overallScore.toFixed(1)}/100 (${quality}) in ${duration}ms`
    );

    const response: BotAnalysisAPIResponse<LongPositionSignal> = {
      success: true,
      data: longSignal,
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[LONG Signal API] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'LONG signal calculation failed',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
