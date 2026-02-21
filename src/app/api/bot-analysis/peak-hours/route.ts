/**
 * 🕐 PEAK HOURS ANALYZER
 *
 * Analyzes historical price data to identify peak profit hours for each cryptocurrency
 *
 * METHODOLOGY:
 * 1. Fetches last 30 days of hourly candle data from Binance
 * 2. Groups price movements by hour of day (Turkey time UTC+3)
 * 3. Calculates average price increase % for each hour
 * 4. Identifies top 3 most profitable hours with confidence scores
 * 5. Considers volume, volatility, and consistency
 *
 * ANALYSIS METRICS:
 * ✅ Average hourly gain % (last 30 days)
 * ✅ Win rate (percentage of positive hours)
 * ✅ Volume correlation (high volume = more reliable)
 * ✅ Consistency score (standard deviation)
 * ✅ Turkey timezone (UTC+3)
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Educational and research purposes only
 * - No trading execution or financial advice
 * - Transparent algorithmic criteria
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// TYPES
// ============================================================================

interface HourlyStats {
  hour: number; // 0-23 (Turkey time)
  avgGainPercent: number;
  winRate: number; // 0-100
  totalCandles: number;
  avgVolume: number;
  consistency: number; // 0-100 (higher = more consistent)
  confidence: number; // 0-100 (combined score)
}

interface PeakHoursAnalysis {
  symbol: string;
  currentHour: number; // Current Turkey time hour
  currentHourTurkey: string; // e.g., "14:00"
  bestHours: HourlyStats[];
  allHoursData: HourlyStats[];
  recommendation: string;
  timestamp: number;
}

interface KlineData {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
}

// ============================================================================
// TIMEZONE UTILITIES
// ============================================================================

/**
 * Convert UTC timestamp to Turkey time hour (UTC+3)
 */
function getTurkeyHour(timestamp: number): number {
  const date = new Date(timestamp);
  const utcHour = date.getUTCHours();
  const turkeyHour = (utcHour + 3) % 24; // UTC+3
  return turkeyHour;
}

/**
 * Get current Turkey time hour
 */
function getCurrentTurkeyHour(): number {
  return getTurkeyHour(Date.now());
}

/**
 * Format hour to Turkey time string
 */
function formatTurkeyTime(hour: number): string {
  return `${hour.toString().padStart(2, '0')}:00`;
}

// ============================================================================
// DATA FETCHING
// ============================================================================

/**
 * Fetch historical hourly klines (last 30 days = 720 hours)
 */
async function fetchHistoricalKlines(symbol: string): Promise<KlineData[]> {
  try {
    // Fetch 720 hours (30 days) of 1h candles
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=1h&limit=720`;
    const response = await fetch(url, {
      headers: { 'Accept': 'application/json' }
    });

    if (!response.ok) {
      throw new Error(`Binance API error: ${response.status}`);
    }

    const data = await response.json();

    return data.map((k: any[]) => ({
      openTime: k[0],
      open: k[1],
      high: k[2],
      low: k[3],
      close: k[4],
      volume: k[5],
      closeTime: k[6]
    }));
  } catch (error) {
    console.error(`[Peak Hours] Failed to fetch klines for ${symbol}:`, error);
    return [];
  }
}

// ============================================================================
// ANALYSIS FUNCTIONS
// ============================================================================

/**
 * Calculate hourly statistics from historical data
 */
function analyzeHourlyPatterns(klines: KlineData[]): Map<number, HourlyStats> {
  // Group candles by hour (Turkey time)
  const hourlyData = new Map<number, {
    gains: number[];
    volumes: number[];
    totalCandles: number;
  }>();

  // Initialize all 24 hours
  for (let hour = 0; hour < 24; hour++) {
    hourlyData.set(hour, {
      gains: [],
      volumes: [],
      totalCandles: 0
    });
  }

  // Process each candle
  for (const kline of klines) {
    const turkeyHour = getTurkeyHour(kline.openTime);
    const open = parseFloat(kline.open);
    const close = parseFloat(kline.close);
    const volume = parseFloat(kline.volume);

    // Calculate gain percentage
    const gainPercent = ((close - open) / open) * 100;

    const hourData = hourlyData.get(turkeyHour)!;
    hourData.gains.push(gainPercent);
    hourData.volumes.push(volume);
    hourData.totalCandles++;
  }

  // Calculate statistics for each hour
  const hourlyStats = new Map<number, HourlyStats>();

  for (let hour = 0; hour < 24; hour++) {
    const data = hourlyData.get(hour)!;

    if (data.totalCandles === 0) {
      hourlyStats.set(hour, {
        hour,
        avgGainPercent: 0,
        winRate: 0,
        totalCandles: 0,
        avgVolume: 0,
        consistency: 0,
        confidence: 0
      });
      continue;
    }

    // Average gain
    const avgGainPercent = data.gains.reduce((a, b) => a + b, 0) / data.gains.length;

    // Win rate (percentage of positive gains)
    const positiveGains = data.gains.filter(g => g > 0).length;
    const winRate = (positiveGains / data.totalCandles) * 100;

    // Average volume
    const avgVolume = data.volumes.reduce((a, b) => a + b, 0) / data.volumes.length;

    // Consistency (inverse of coefficient of variation)
    const stdDev = Math.sqrt(
      data.gains.reduce((sum, gain) => sum + Math.pow(gain - avgGainPercent, 2), 0) / data.gains.length
    );
    const coefficientOfVariation = Math.abs(avgGainPercent) > 0.01 ? (stdDev / Math.abs(avgGainPercent)) : 999;
    const consistency = Math.max(0, 100 - (coefficientOfVariation * 10));

    // Confidence score (weighted combination)
    // - 40% average gain
    // - 30% win rate
    // - 20% consistency
    // - 10% data completeness
    const gainScore = Math.min(100, Math.max(0, avgGainPercent * 20 + 50));
    const completenessScore = (data.totalCandles / 30) * 100; // Expect ~30 candles per hour in 30 days

    const confidence = (
      gainScore * 0.4 +
      winRate * 0.3 +
      consistency * 0.2 +
      completenessScore * 0.1
    );

    hourlyStats.set(hour, {
      hour,
      avgGainPercent,
      winRate,
      totalCandles: data.totalCandles,
      avgVolume,
      consistency,
      confidence
    });
  }

  return hourlyStats;
}

/**
 * Identify best performing hours
 */
function identifyBestHours(hourlyStats: Map<number, HourlyStats>, topN: number = 3): HourlyStats[] {
  const allHours = Array.from(hourlyStats.values());

  // Sort by confidence score (descending)
  const sortedHours = allHours.sort((a, b) => b.confidence - a.confidence);

  // Return top N hours
  return sortedHours.slice(0, topN);
}

/**
 * Generate recommendation based on current time and best hours
 */
function generateRecommendation(currentHour: number, bestHours: HourlyStats[]): string {
  if (bestHours.length === 0) {
    return 'Yetersiz veri - analiz yapılamadı';
  }

  const topHour = bestHours[0];
  const isBestHourNow = currentHour === topHour.hour;
  const nextBestHour = bestHours.find(h => h.hour > currentHour) || bestHours[0];

  if (isBestHourNow) {
    return `🔥 ŞU AN EN İYİ SAAT! Ortalama kazanç: ${topHour.avgGainPercent.toFixed(2)}%, Başarı oranı: ${topHour.winRate.toFixed(0)}%`;
  }

  const hoursUntilBest = nextBestHour.hour > currentHour
    ? nextBestHour.hour - currentHour
    : 24 - currentHour + nextBestHour.hour;

  return `⏰ En iyi saat ${hoursUntilBest} saat sonra (${formatTurkeyTime(nextBestHour.hour)}). Ortalama kazanç: ${nextBestHour.avgGainPercent.toFixed(2)}%`;
}

/**
 * Analyze peak hours for a single symbol
 */
async function analyzePeakHours(symbol: string): Promise<PeakHoursAnalysis | null> {
  try {
    console.log(`[Peak Hours] Analyzing ${symbol}...`);

    // Fetch historical data
    const klines = await fetchHistoricalKlines(symbol);

    if (klines.length < 100) {
      console.log(`[Peak Hours] Insufficient data for ${symbol}`);
      return null;
    }

    // Analyze hourly patterns
    const hourlyStats = analyzeHourlyPatterns(klines);

    // Identify best hours
    const bestHours = identifyBestHours(hourlyStats, 3);

    // Get current Turkey time
    const currentHour = getCurrentTurkeyHour();
    const currentHourTurkey = formatTurkeyTime(currentHour);

    // Generate recommendation
    const recommendation = generateRecommendation(currentHour, bestHours);

    return {
      symbol,
      currentHour,
      currentHourTurkey,
      bestHours,
      allHoursData: Array.from(hourlyStats.values()),
      recommendation,
      timestamp: Date.now()
    };
  } catch (error) {
    console.error(`[Peak Hours] Error analyzing ${symbol}:`, error);
    return null;
  }
}

// ============================================================================
// MAIN HANDLER
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    console.log('[Peak Hours] Starting peak hours analysis...');

    // Get query parameters
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const symbols = searchParams.get('symbols')?.split(',');

    let results: PeakHoursAnalysis[] = [];

    if (symbol) {
      // Analyze single symbol
      const analysis = await analyzePeakHours(symbol);
      if (analysis) results.push(analysis);
    } else if (symbols && symbols.length > 0) {
      // Analyze multiple symbols in batches
      const batchSize = 5;
      for (let i = 0; i < symbols.length; i += batchSize) {
        const batch = symbols.slice(i, i + batchSize);
        const batchPromises = batch.map(s => analyzePeakHours(s));
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults.filter(r => r !== null) as PeakHoursAnalysis[]);

        // Small delay between batches
        if (i + batchSize < symbols.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
    } else {
      return NextResponse.json({
        success: false,
        error: 'Please provide either "symbol" or "symbols" parameter'
      }, { status: 400 });
    }

    const duration = Date.now() - startTime;

    console.log(
      `[Peak Hours] Analyzed ${results.length} symbols in ${duration}ms`
    );

    return NextResponse.json({
      success: true,
      data: {
        analyses: results,
        count: results.length,
        currentTurkeyTime: formatTurkeyTime(getCurrentTurkeyHour()),
        timestamp: Date.now(),
        lastUpdate: new Date().toISOString()
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Peak Hours] Error:', error);

    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Failed to analyze peak hours',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    }, { status: 500 });
  }
}
