/**
 * 📊 CVD (CUMULATIVE VOLUME DELTA) ANALYZER
 *
 * The secret weapon of professional traders and institutions
 * Shows REAL buying vs selling pressure that price charts can't reveal
 *
 * WHAT IS CVD?
 * - Tracks cumulative difference between buy volume and sell volume
 * - Positive CVD = More buying pressure (bullish)
 * - Negative CVD = More selling pressure (bearish)
 *
 * WHY IT MATTERS:
 * ✅ Price rising + CVD falling = Institutions selling (FAKE rally)
 * ✅ Price falling + CVD rising = Institutions buying (ACCUMULATION)
 * ✅ See exact trade sizes: retail vs institutional vs whales
 *
 * REAL EXAMPLE:
 * BTC @ $97k (May 2025): Price up, CVD down = Correction followed
 * Professional traders saw this coming, retail didn't
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Educational and research purposes only
 * - No trading execution or financial advice
 * - Transparent algorithmic analysis
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// TYPES
// ============================================================================

interface AggTrade {
  a: number;        // Aggregate trade ID
  p: string;        // Price
  q: string;        // Quantity
  f: number;        // First trade ID
  l: number;        // Last trade ID
  T: number;        // Timestamp
  m: boolean;       // Was buyer the maker? (true = sell, false = buy)
}

interface CVDDataPoint {
  timestamp: number;
  price: number;
  cvd: number;
  buyVolume: number;
  sellVolume: number;
  netVolume: number;
}

interface SizeAnalysis {
  retail: {
    buyVolume: number;
    sellVolume: number;
    netVolume: number;
    percentage: number;
  };
  institutional: {
    buyVolume: number;
    sellVolume: number;
    netVolume: number;
    percentage: number;
  };
  whale: {
    buyVolume: number;
    sellVolume: number;
    netVolume: number;
    percentage: number;
  };
}

interface DivergenceSignal {
  type: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  strength: number; // 0-100
  message: string;
}

interface CVDAnalysis {
  symbol: string;
  currentPrice: number;

  // CVD Data
  cvdData: CVDDataPoint[];
  currentCVD: number;
  cvdChange24h: number;
  cvdTrend: 'RISING' | 'FALLING' | 'NEUTRAL';

  // Volume Analysis
  totalBuyVolume: number;
  totalSellVolume: number;
  buyPressure: number; // Percentage
  sellPressure: number; // Percentage

  // Size-based Analysis
  sizeAnalysis: SizeAnalysis;
  dominantForce: 'RETAIL' | 'INSTITUTIONAL' | 'WHALE';

  // Divergence Detection
  divergence: DivergenceSignal;

  timestamp: number;
}

// ============================================================================
// CONSTANTS
// ============================================================================

// Trade size thresholds (in USDT value)
const RETAIL_THRESHOLD = 100000;        // < $100k = Retail
const INSTITUTIONAL_THRESHOLD = 1000000; // $100k - $1M = Institutional
// > $1M = Whale

// ============================================================================
// DATA FETCHING
// ============================================================================

/**
 * Fetch aggregated trades from Binance
 * Uses last 1000 trades for real-time CVD calculation
 */
async function fetchAggTrades(symbol: string, limit: number = 1000): Promise<AggTrade[]> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/aggTrades?symbol=${symbol}&limit=${limit}`;
    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`[CVD] Failed to fetch aggTrades for ${symbol}:`, error);
    return [];
  }
}

/**
 * Fetch current price
 */
async function fetchCurrentPrice(symbol: string): Promise<number> {
  try {
    const url = `https://fapi.binance.com/fapi/v1/ticker/price?symbol=${symbol}`;
    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();
    return parseFloat(data.price);
  } catch (error) {
    console.error(`[CVD] Failed to fetch price for ${symbol}:`, error);
    return 0;
  }
}

// ============================================================================
// CVD CALCULATION
// ============================================================================

/**
 * Calculate CVD from aggregated trades
 * CVD = Cumulative (Buy Volume - Sell Volume)
 */
function calculateCVD(trades: AggTrade[]): CVDDataPoint[] {
  const dataPoints: CVDDataPoint[] = [];
  let cumulativeCVD = 0;

  for (const trade of trades) {
    const price = parseFloat(trade.p);
    const quantity = parseFloat(trade.q);
    const isSell = trade.m; // true = sell (buyer was maker)

    // Calculate volume direction
    const volume = quantity;
    const signedVolume = isSell ? -volume : volume;

    // Update cumulative CVD
    cumulativeCVD += signedVolume;

    dataPoints.push({
      timestamp: trade.T,
      price,
      cvd: cumulativeCVD,
      buyVolume: isSell ? 0 : volume,
      sellVolume: isSell ? volume : 0,
      netVolume: signedVolume
    });
  }

  return dataPoints;
}

/**
 * Analyze trades by size category
 */
function analyzeBySizeCategory(trades: AggTrade[]): SizeAnalysis {
  const analysis: SizeAnalysis = {
    retail: { buyVolume: 0, sellVolume: 0, netVolume: 0, percentage: 0 },
    institutional: { buyVolume: 0, sellVolume: 0, netVolume: 0, percentage: 0 },
    whale: { buyVolume: 0, sellVolume: 0, netVolume: 0, percentage: 0 }
  };

  let totalVolume = 0;

  for (const trade of trades) {
    const price = parseFloat(trade.p);
    const quantity = parseFloat(trade.q);
    const usdtValue = price * quantity;
    const isSell = trade.m;

    totalVolume += quantity;

    // Categorize by size
    let category: 'retail' | 'institutional' | 'whale';
    if (usdtValue < RETAIL_THRESHOLD) {
      category = 'retail';
    } else if (usdtValue < INSTITUTIONAL_THRESHOLD) {
      category = 'institutional';
    } else {
      category = 'whale';
    }

    // Update volumes
    if (isSell) {
      analysis[category].sellVolume += quantity;
      analysis[category].netVolume -= quantity;
    } else {
      analysis[category].buyVolume += quantity;
      analysis[category].netVolume += quantity;
    }
  }

  // Calculate percentages
  if (totalVolume > 0) {
    analysis.retail.percentage = ((analysis.retail.buyVolume + analysis.retail.sellVolume) / totalVolume) * 100;
    analysis.institutional.percentage = ((analysis.institutional.buyVolume + analysis.institutional.sellVolume) / totalVolume) * 100;
    analysis.whale.percentage = ((analysis.whale.buyVolume + analysis.whale.sellVolume) / totalVolume) * 100;
  }

  return analysis;
}

/**
 * Detect price-CVD divergence
 * Divergence indicates potential reversal
 */
function detectDivergence(cvdData: CVDDataPoint[]): DivergenceSignal {
  if (cvdData.length < 100) {
    return {
      type: 'NEUTRAL',
      strength: 0,
      message: 'Yetersiz veri - divergence analizi yapılamıyor'
    };
  }

  // Get recent data (last 20% of trades)
  const recentCount = Math.floor(cvdData.length * 0.2);
  const recentData = cvdData.slice(-recentCount);
  const _olderData = cvdData.slice(-recentCount * 2, -recentCount);

  // Calculate price trend
  const recentPriceChange = recentData[recentData.length - 1].price - recentData[0].price;
  const recentPriceTrend = recentPriceChange > 0 ? 'UP' : 'DOWN';

  // Calculate CVD trend
  const recentCVDChange = recentData[recentData.length - 1].cvd - recentData[0].cvd;
  const recentCVDTrend = recentCVDChange > 0 ? 'UP' : 'DOWN';

  // Detect divergence
  if (recentPriceTrend === 'UP' && recentCVDTrend === 'DOWN') {
    // Bearish divergence: Price rising but CVD falling
    const strength = Math.min(100, Math.abs(recentPriceChange / recentData[0].price) * 1000);
    return {
      type: 'BEARISH',
      strength,
      message: '🚨 SAHTE YUKSELIŞ! Fiyat yükseliyor ama kurumsal SATIYOR. Düşüş bekleniyor.'
    };
  } else if (recentPriceTrend === 'DOWN' && recentCVDTrend === 'UP') {
    // Bullish divergence: Price falling but CVD rising
    const strength = Math.min(100, Math.abs(recentPriceChange / recentData[0].price) * 1000);
    return {
      type: 'BULLISH',
      strength,
      message: '🔥 DİP ALIM FIRSATI! Fiyat düşüyor ama kurumsal ALIYOR. Yükseliş bekleniyor.'
    };
  }

  return {
    type: 'NEUTRAL',
    strength: 0,
    message: '✅ Fiyat ve CVD uyumlu. Normal piyasa hareketi.'
  };
}

/**
 * Analyze CVD for a symbol
 */
async function analyzeCVD(symbol: string): Promise<CVDAnalysis | null> {
  try {
    console.log(`[CVD] Analyzing ${symbol}...`);

    // Fetch data
    const [trades, currentPrice] = await Promise.all([
      fetchAggTrades(symbol, 1000),
      fetchCurrentPrice(symbol)
    ]);

    if (trades.length < 100) {
      console.log(`[CVD] Insufficient data for ${symbol}`);
      return null;
    }

    // Calculate CVD
    const cvdData = calculateCVD(trades);
    const currentCVD = cvdData[cvdData.length - 1].cvd;
    const startCVD = cvdData[0].cvd;
    const cvdChange24h = currentCVD - startCVD;

    // Determine CVD trend
    let cvdTrend: 'RISING' | 'FALLING' | 'NEUTRAL';
    if (cvdChange24h > 0) {
      cvdTrend = 'RISING';
    } else if (cvdChange24h < 0) {
      cvdTrend = 'FALLING';
    } else {
      cvdTrend = 'NEUTRAL';
    }

    // Calculate total volumes
    const totalBuyVolume = cvdData.reduce((sum, d) => sum + d.buyVolume, 0);
    const totalSellVolume = cvdData.reduce((sum, d) => sum + d.sellVolume, 0);
    const totalVolume = totalBuyVolume + totalSellVolume;

    const buyPressure = totalVolume > 0 ? (totalBuyVolume / totalVolume) * 100 : 50;
    const sellPressure = 100 - buyPressure;

    // Analyze by size
    const sizeAnalysis = analyzeBySizeCategory(trades);

    // Determine dominant force
    let dominantForce: 'RETAIL' | 'INSTITUTIONAL' | 'WHALE';
    const maxPercentage = Math.max(
      sizeAnalysis.retail.percentage,
      sizeAnalysis.institutional.percentage,
      sizeAnalysis.whale.percentage
    );

    if (maxPercentage === sizeAnalysis.whale.percentage) {
      dominantForce = 'WHALE';
    } else if (maxPercentage === sizeAnalysis.institutional.percentage) {
      dominantForce = 'INSTITUTIONAL';
    } else {
      dominantForce = 'RETAIL';
    }

    // Detect divergence
    const divergence = detectDivergence(cvdData);

    return {
      symbol,
      currentPrice,
      cvdData,
      currentCVD,
      cvdChange24h,
      cvdTrend,
      totalBuyVolume,
      totalSellVolume,
      buyPressure,
      sellPressure,
      sizeAnalysis,
      dominantForce,
      divergence,
      timestamp: Date.now()
    };
  } catch (error) {
    console.error(`[CVD] Error analyzing ${symbol}:`, error);
    return null;
  }
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    console.log('[CVD] Starting CVD analysis...');

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Analyze CVD
    const analysis = await analyzeCVD(symbol);

    if (!analysis) {
      return NextResponse.json({
        success: false,
        error: 'Failed to analyze CVD - insufficient data'
      }, { status: 500 });
    }

    const duration = Date.now() - startTime;

    console.log(`[CVD] Analysis completed for ${symbol} in ${duration}ms`);

    return NextResponse.json({
      success: true,
      data: analysis,
      metadata: {
        duration,
        timestamp: Date.now()
      }
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[CVD] Error:', error);

    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to analyze CVD',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    }, { status: 500 });
  }
}
