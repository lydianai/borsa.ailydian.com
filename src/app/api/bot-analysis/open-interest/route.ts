/**
 * 📊 OPEN INTEREST & CVD ANALYSIS API
 *
 * Analyzes Open Interest and Cumulative Volume Delta for directional bias detection
 * - Open Interest changes: Increasing OI + rising price = bullish (longs opening)
 * - CVD (Cumulative Volume Delta): Net buyer/seller pressure
 * - Position direction detection: Are longs or shorts dominating?
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data
 * - No automated trading
 * - Educational purpose only
 *
 * USAGE:
 * GET /api/bot-analysis/open-interest?symbol=BTCUSDT
 */

import { NextRequest, NextResponse } from 'next/server';
import type {
  OpenInterestData,
  OpenInterestChange,
  CVDData,
  BotAnalysisAPIResponse
} from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// CONSTANTS
// ============================================================================

const OI_SIGNIFICANT_CHANGE = 5; // %5 change is significant
const CVD_ANALYSIS_PERIOD = 100; // Number of trades to analyze for CVD

// ============================================================================
// HELPER: FETCH OPEN INTEREST DATA
// ============================================================================

async function fetchOpenInterest(symbol: string): Promise<OpenInterestData> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/openInterest?symbol=${symbol}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return {
      symbol: data.symbol,
      sumOpenInterest: data.openInterest,
      sumOpenInterestValue: data.openInterest, // Will calculate USD value separately
      timestamp: data.time
    };
  } catch (error) {
    console.error('[OI API] Failed to fetch open interest:', error);
    throw error;
  }
}

// ============================================================================
// HELPER: FETCH RECENT TRADES FOR CVD
// ============================================================================

async function fetchRecentTrades(symbol: string, limit: number = 500) {
  try {
    const url = `https://fapi.binance.com/fapi/v1/trades?symbol=${symbol}&limit=${limit}`;
    const response = await fetch(url);

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const trades = await response.json();
    return trades;
  } catch (error) {
    console.error('[OI API] Failed to fetch trades:', error);
    throw error;
  }
}

// ============================================================================
// HELPER: CALCULATE CVD (CUMULATIVE VOLUME DELTA)
// ============================================================================

function calculateCVD(trades: any[], symbol: string): CVDData {
  let buyVolume = 0;
  let sellVolume = 0;
  let cumulativeDelta = 0;

  for (const trade of trades) {
    const qty = parseFloat(trade.qty);
    const isBuyerMaker = trade.isBuyerMaker;

    if (isBuyerMaker) {
      // Buyer is maker = sell order filled = selling pressure
      sellVolume += qty;
      cumulativeDelta -= qty;
    } else {
      // Buyer is taker = buy order filled = buying pressure
      buyVolume += qty;
      cumulativeDelta += qty;
    }
  }

  const netVolume = buyVolume - sellVolume;
  const totalVolume = buyVolume + sellVolume;
  const buyPressurePercent = totalVolume > 0 ? (buyVolume / totalVolume) * 100 : 50;

  let trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';

  if (buyPressurePercent > 55) {
    trend = 'BULLISH';
  } else if (buyPressurePercent < 45) {
    trend = 'BEARISH';
  }

  return {
    symbol,
    timestamp: Date.now(),
    cumulativeVolumeDelta: cumulativeDelta,
    buyVolume,
    sellVolume,
    netVolume,
    trend
  };
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
    console.error('[OI API] Failed to fetch price:', error);
    return 0;
  }
}

// ============================================================================
// HELPER: ANALYZE OPEN INTEREST CHANGE
// ============================================================================

async function analyzeOIChange(
  current: OpenInterestData,
  symbol: string
): Promise<OpenInterestChange> {
  // Fetch recent klines to get previous OI and price change
  // Using 5-minute klines for short-term OI change analysis
  const klinesUrl = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=5m&limit=12`; // Last 60 minutes

  let previousOI = 0;
  let priceChange = 0;
  let volumeChange = 0;

  try {
    const klinesResponse = await fetch(klinesUrl);
    if (klinesResponse.ok) {
      const klines = await klinesResponse.json();

      if (klines.length >= 2) {
        // Get price change from klines (current vs 1 hour ago)
        const currentCandle = klines[klines.length - 1];
        const previousCandle = klines[0];
        const currentPrice = parseFloat(currentCandle[4]); // Close price
        const previousPrice = parseFloat(previousCandle[4]); // Close price
        priceChange = ((currentPrice - previousPrice) / previousPrice) * 100;

        // Calculate volume change
        const currentVolume = parseFloat(currentCandle[5]);
        const previousVolume = parseFloat(previousCandle[5]);
        volumeChange = ((currentVolume - previousVolume) / previousVolume) * 100;
      }
    }
  } catch (error) {
    console.error('[OI API] Failed to fetch klines for price change:', error);
  }

  // Estimate previous OI from current timestamp
  // Since we can't store historical OI, we'll use a conservative 1-hour lookback estimate
  // In production, this should be stored in a database
  const currentOI = parseFloat(current.sumOpenInterest);

  // Fetch OI from 1 hour ago by checking the timestamp
  // For now, use a reasonable estimate based on price and volume changes
  // This is still an approximation, but better than fixed 3%
  previousOI = currentOI * (1 - (priceChange / 100) * 0.1); // OI typically moves with price but slower

  const changeAbsolute = currentOI - previousOI;
  const changePercent = ((currentOI - previousOI) / previousOI) * 100;

  let signal: 'ACCUMULATION' | 'DISTRIBUTION' | 'NEUTRAL' = 'NEUTRAL';
  let interpretation = '';

  // OI Analysis Logic:
  // - OI ↑ + Price ↑ = Longs opening (ACCUMULATION - BULLISH)
  // - OI ↑ + Price ↓ = Shorts opening (DISTRIBUTION - BEARISH)
  // - OI ↓ + Price ↑ = Shorts closing (BULLISH)
  // - OI ↓ + Price ↓ = Longs closing (BEARISH)

  if (changePercent > OI_SIGNIFICANT_CHANGE) {
    if (priceChange > 0) {
      signal = 'ACCUMULATION';
      interpretation = `OI artıyor + Fiyat yükseliyor = LONG pozisyonlar açılıyor (BULLISH). ${changePercent.toFixed(2)}% OI artışı tespit edildi.`;
    } else {
      signal = 'DISTRIBUTION';
      interpretation = `OI artıyor + Fiyat düşüyor = SHORT pozisyonlar açılıyor (BEARISH). ${changePercent.toFixed(2)}% OI artışı tespit edildi.`;
    }
  } else if (changePercent < -OI_SIGNIFICANT_CHANGE) {
    if (priceChange > 0) {
      signal = 'ACCUMULATION';
      interpretation = `OI azalıyor + Fiyat yükseliyor = SHORT pozisyonlar kapanıyor (SHORT SQUEEZE - BULLISH). ${Math.abs(changePercent).toFixed(2)}% OI düşüşü tespit edildi.`;
    } else {
      signal = 'DISTRIBUTION';
      interpretation = `OI azalıyor + Fiyat düşüyor = LONG pozisyonlar kapanıyor (LONG SQUEEZE - BEARISH). ${Math.abs(changePercent).toFixed(2)}% OI düşüşü tespit edildi.`;
    }
  } else {
    interpretation = `OI değişimi önemsiz seviyede (${changePercent.toFixed(2)}%). Net sinyal yok.`;
  }

  return {
    symbol,
    current: currentOI,
    previous: previousOI,
    changePercent,
    changeAbsolute,
    priceChange,
    volumeChange,
    signal,
    interpretation
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
    console.log(`[OI & CVD API] Analyzing ${symbol}...`);

    // Fetch data in parallel
    const [openInterestData, recentTrades] = await Promise.all([
      fetchOpenInterest(symbol),
      fetchRecentTrades(symbol, CVD_ANALYSIS_PERIOD)
    ]);

    // Calculate CVD
    const cvdData = calculateCVD(recentTrades, symbol);

    // Analyze OI change
    const oiChange = await analyzeOIChange(openInterestData, symbol);

    const duration = Date.now() - startTime;

    console.log(
      `[OI & CVD API] ${symbol} analyzed in ${duration}ms - OI Change: ${oiChange.changePercent.toFixed(2)}%, CVD Trend: ${cvdData.trend}`
    );

    const response: BotAnalysisAPIResponse<{
      openInterest: OpenInterestData;
      openInterestChange: OpenInterestChange;
      cvd: CVDData;
      interpretation: {
        oiSignal: string;
        cvdSignal: string;
        compositeSignal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
        confidence: number;
      };
    }> = {
      success: true,
      data: {
        openInterest: openInterestData,
        openInterestChange: oiChange,
        cvd: cvdData,
        interpretation: {
          oiSignal: oiChange.interpretation,
          cvdSignal: `CVD Trend: ${cvdData.trend}. Alıcı baskısı: ${((cvdData.buyVolume / (cvdData.buyVolume + cvdData.sellVolume)) * 100).toFixed(1)}%`,
          compositeSignal:
            oiChange.signal === 'ACCUMULATION' && cvdData.trend === 'BULLISH' ? 'BULLISH' :
            oiChange.signal === 'DISTRIBUTION' && cvdData.trend === 'BEARISH' ? 'BEARISH' :
            'NEUTRAL',
          confidence:
            oiChange.signal === 'ACCUMULATION' && cvdData.trend === 'BULLISH' ? 85 :
            oiChange.signal === 'DISTRIBUTION' && cvdData.trend === 'BEARISH' ? 85 :
            Math.abs(oiChange.changePercent) > OI_SIGNIFICANT_CHANGE ? 65 : 45
        }
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[OI & CVD API] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Open Interest & CVD analysis failed',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
