/**
 * 💰 FUNDING RATE ANALYSIS API
 *
 * Fetches and analyzes Binance Futures funding rate data
 * - Current funding rate
 * - Historical funding rate trends
 * - LONG/SHORT opportunity detection based on funding extremes
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data
 * - No automated trading
 * - Educational purpose only
 *
 * USAGE:
 * GET /api/bot-analysis/funding?symbol=BTCUSDT
 */

import { NextRequest, NextResponse } from 'next/server';
import type {
  FundingRate,
  FundingRateSignal,
  BotAnalysisAPIResponse
} from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// CONSTANTS
// ============================================================================

// Funding rate thresholds (in decimal form)
const EXTREME_POSITIVE_FUNDING = 0.0003; // 0.03% - Extremely bullish (risky for longs)
const HIGH_POSITIVE_FUNDING = 0.0001; // 0.01% - High bullish sentiment
const _NEUTRAL_THRESHOLD = 0.00005; // 0.005% - Neutral zone
const HIGH_NEGATIVE_FUNDING = -0.0001; // -0.01% - High bearish sentiment (good for longs)
const EXTREME_NEGATIVE_FUNDING = -0.0003; // -0.03% - Extremely bearish (great for longs)

// ============================================================================
// HELPER: FETCH CURRENT FUNDING RATE
// ============================================================================

async function fetchCurrentFundingRate(symbol: string): Promise<FundingRate> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/premiumIndex?symbol=${symbol}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return {
      symbol: data.symbol,
      fundingRate: parseFloat(data.lastFundingRate),
      fundingTime: data.nextFundingTime,
      markPrice: parseFloat(data.markPrice),
      indexPrice: parseFloat(data.indexPrice),
      estimatedSettlePrice: parseFloat(data.estimatedSettlePrice || '0'),
      lastFundingRate: parseFloat(data.lastFundingRate),
      nextFundingTime: data.nextFundingTime,
      interestRate: parseFloat(data.interestRate),
      time: data.time
    };
  } catch (error) {
    console.error('[Funding API] Failed to fetch current funding:', error);
    throw error;
  }
}

// ============================================================================
// HELPER: ANALYZE FUNDING RATE
// ============================================================================

function analyzeFundingRate(funding: FundingRate): FundingRateSignal {
  const rate = funding.fundingRate;
  let signal: FundingRateSignal['signal'] = 'NEUTRAL';
  let confidence = 50;
  let reason = '';
  let recommendation = '';

  // Analyze funding rate
  if (rate >= EXTREME_POSITIVE_FUNDING) {
    signal = 'EXTREME_LONG';
    confidence = Math.min(95, 70 + (rate / EXTREME_POSITIVE_FUNDING) * 25);
    reason = `Aşırı yüksek funding rate (%${(rate * 100).toFixed(4)}). Piyasa aşırı long pozisyonda, düzeltme riski yüksek.`;
    recommendation = 'LONG POZİSYON RİSKLİ! Mevcut long\'ları azaltın veya kısa vadeli SHORT düşünün. Funding fee çok yüksek.';
  } else if (rate > HIGH_POSITIVE_FUNDING) {
    signal = 'SHORT_OPPORTUNITY';
    confidence = Math.min(85, 60 + ((rate - HIGH_POSITIVE_FUNDING) / (EXTREME_POSITIVE_FUNDING - HIGH_POSITIVE_FUNDING)) * 25);
    reason = `Yüksek funding rate (%${(rate * 100).toFixed(4)}). Long trader\'lar short trader\'lara funding fee ödüyor.`;
    recommendation = 'Yeni LONG girişleri için uygun değil. Mevcut long\'larda kısmi kar realizasyonu önerilir. SHORT fırsatı olabilir.';
  } else if (rate <= EXTREME_NEGATIVE_FUNDING) {
    signal = 'LONG_OPPORTUNITY';
    confidence = Math.min(95, 70 + Math.abs(rate / EXTREME_NEGATIVE_FUNDING) * 25);
    reason = `Aşırı negatif funding rate (%${(rate * 100).toFixed(4)}). Piyasa aşırı short pozisyonda, sıkışma (short squeeze) riski var.`;
    recommendation = 'MÜKEMMEL LONG FIRSATI! Short trader\'lar long trader\'lara funding fee ödüyor. Dipten alım için ideal.';
  } else if (rate < HIGH_NEGATIVE_FUNDING) {
    signal = 'LONG_OPPORTUNITY';
    confidence = Math.min(85, 60 + Math.abs((rate - HIGH_NEGATIVE_FUNDING) / (EXTREME_NEGATIVE_FUNDING - HIGH_NEGATIVE_FUNDING)) * 25);
    reason = `Negatif funding rate (%${(rate * 100).toFixed(4)}). Short baskısı var, potansiyel yukarı hareket.`;
    recommendation = 'İYİ LONG FIRSATI. Funding rate negatif, short trader\'lar fee ödüyor. Kademeli long girişi yapılabilir.';
  } else {
    signal = 'NEUTRAL';
    confidence = 50;
    reason = `Nötr funding rate (%${(rate * 100).toFixed(4)}). Piyasa dengede.`;
    recommendation = 'Funding rate bazlı net bir sinyal yok. Diğer göstergeleri inceleyin.';
  }

  return {
    symbol: funding.symbol,
    currentRate: rate,
    signal,
    confidence,
    reason,
    recommendation
  };
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';

  try {
    console.log(`[Funding API] Fetching funding rate for ${symbol}...`);

    // Fetch current funding rate
    const fundingData = await fetchCurrentFundingRate(symbol);

    // Analyze funding rate
    const signal = analyzeFundingRate(fundingData);

    // Calculate time until next funding
    const now = Date.now();
    const timeUntilFunding = fundingData.nextFundingTime - now;
    const hoursUntilFunding = (timeUntilFunding / (1000 * 60 * 60)).toFixed(1);

    const duration = Date.now() - startTime;

    console.log(
      `[Funding API] ${symbol} analyzed in ${duration}ms - Rate: ${(fundingData.fundingRate * 100).toFixed(4)}%, Signal: ${signal.signal}`
    );

    const response: BotAnalysisAPIResponse<{
      fundingData: FundingRate;
      signal: FundingRateSignal;
      nextFundingIn: string;
    }> = {
      success: true,
      data: {
        fundingData,
        signal,
        nextFundingIn: `${hoursUntilFunding} saat`
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Funding API] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Funding rate analysis failed',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
