/**
 * 🎯 PERFECT LONG OPPORTUNITIES SCANNER
 *
 * Scans ALL coins and returns only those with "MÜKEMMEL LONG FIRSATI!" funding rate recommendation
 * - Negative funding rate (shorts paying longs)
 * - Auto-updates every 30 seconds
 * - Disappears when recommendation changes
 *
 * WHY THIS MATTERS:
 * - Short traders are paying funding fees to long traders
 * - Perfect for bottom-fishing / dip buying
 * - High probability of reversal
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market analysis
 * - Educational purpose only
 * - No trading execution
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// TYPES
// ============================================================================

interface PerfectLongOpportunity {
  symbol: string;
  fundingRate: number;
  fundingRatePercent: string;
  recommendation: string;
  quality: 'EXCELLENT';
  timestamp: number;
  price: number | null;
}

interface PerfectLongsResponse {
  success: boolean;
  data?: {
    opportunities: PerfectLongOpportunity[];
    count: number;
    timestamp: number;
    lastUpdate: string;
  };
  error?: string;
  metadata?: {
    duration: number;
    timestamp: number;
  };
}

// ============================================================================
// HELPER: FETCH ALL FUNDING RATES
// ============================================================================

async function fetchAllFundingRates(): Promise<any[]> {
  try {
    const response = await fetch('https://fapi.binance.com/fapi/v1/premiumIndex', {
      headers: { 'Accept': 'application/json' },
      next: { revalidate: 30 } // Cache for 30 seconds
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('[Perfect Longs] Failed to fetch funding rates:', error);
    return [];
  }
}

// ============================================================================
// HELPER: CHECK IF FUNDING RATE IS PERFECT LONG OPPORTUNITY
// ============================================================================

function isPerfectLongOpportunity(fundingRate: number): boolean {
  // Perfect long opportunity: funding rate < -0.01% (shorts paying longs)
  return fundingRate < -0.0001;
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(_request: NextRequest) {
  const startTime = Date.now();

  try {
    console.log('[Perfect Longs] Scanning all coins for perfect long opportunities...');

    // Fetch all funding rates from Binance
    const allFundingRates = await fetchAllFundingRates();

    if (allFundingRates.length === 0) {
      throw new Error('Failed to fetch funding rates from Binance');
    }

    // Filter for perfect long opportunities
    const perfectLongs: PerfectLongOpportunity[] = [];

    for (const item of allFundingRates) {
      const symbol = item.symbol;
      const fundingRate = parseFloat(item.lastFundingRate || '0');
      const markPrice = parseFloat(item.markPrice || '0');

      // Check if this is a perfect long opportunity
      if (isPerfectLongOpportunity(fundingRate)) {
        const fundingRatePercent = (fundingRate * 100).toFixed(4);

        perfectLongs.push({
          symbol,
          fundingRate,
          fundingRatePercent: `${fundingRatePercent}%`,
          recommendation: 'MÜKEMMEL LONG FIRSATI! Short trader\'lar long trader\'lara funding fee ödüyor. Dipten alım için ideal.',
          quality: 'EXCELLENT',
          timestamp: Date.now(),
          price: markPrice > 0 ? markPrice : null
        });
      }
    }

    // Sort by funding rate (most negative first = best opportunity)
    perfectLongs.sort((a, b) => a.fundingRate - b.fundingRate);

    const duration = Date.now() - startTime;

    console.log(
      `[Perfect Longs] Found ${perfectLongs.length} perfect long opportunities in ${duration}ms`
    );

    const response: PerfectLongsResponse = {
      success: true,
      data: {
        opportunities: perfectLongs,
        count: perfectLongs.length,
        timestamp: Date.now(),
        lastUpdate: new Date().toISOString()
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Perfect Longs] Error:', error);

    const errorResponse: PerfectLongsResponse = {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to scan for perfect long opportunities',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
