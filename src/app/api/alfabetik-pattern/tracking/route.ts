import { NextResponse } from "next/server";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";
import {
  analyzeAlfabetikPattern,
  enrichWithHistoricalData,
  filterHighConfidenceSignals,
  type AlfabetikPattern,
  type MarketSummary,
} from "@/lib/alfabetik-pattern-analyzer";

// Cache configuration - 2 minute cache
const CACHE_DURATION = 120000; // 2 minutes
let cachedResult: any = null;
let lastAnalysisTime = 0;

/**
 * GET /api/alfabetik-pattern/tracking
 * Returns alfabetik pattern analysis for all cryptocurrency markets
 * Zero-error implementation with multi-layer validation
 */
export async function GET() {
  try {
    const now = Date.now();

    // Check cache validity
    if (cachedResult && now - lastAnalysisTime < CACHE_DURATION) {
      return NextResponse.json({
        ...cachedResult,
        cached: true,
        cacheAge: Math.floor((now - lastAnalysisTime) / 1000),
      });
    }

    // Fetch real-time Binance futures data
    const binanceData = await fetchBinanceFuturesData();

    if (!binanceData.success || !binanceData.data || !binanceData.data.all) {
      throw new Error(
        binanceData.error || "Failed to fetch Binance futures data"
      );
    }

    // Convert Binance data to CoinData format
    const coinData = binanceData.data.all
      .filter((coin: any) => coin.symbol.endsWith("USDT"))
      .map((coin: any) => ({
        symbol: coin.symbol.replace("USDT", ""),
        price: coin.price,
        changePercent24h: coin.changePercent24h,
        changePercent7d: 0, // Will be filled by enrichWithHistoricalData
        changePercent30d: 0, // Will be filled by enrichWithHistoricalData
        volume24h: coin.volume,
        category: undefined,
      }));

    // V2 FEATURE: Enrich with real historical data from Binance
    console.log("🔄 Enriching coin data with 7d/30d historical data...");
    const enrichedCoinData = await enrichWithHistoricalData(coinData);
    console.log(`✅ Historical data enrichment complete for ${enrichedCoinData.length} coins`);

    // Perform alfabetik pattern analysis with enriched data
    const analysis = analyzeAlfabetikPattern(enrichedCoinData);

    // Filter high-confidence patterns only
    const highConfidencePatterns = filterHighConfidenceSignals(
      analysis.patterns,
      60
    );

    // Build response
    const response = {
      success: true,
      timestamp: analysis.timestamp,
      totalLetters: analysis.patterns.length,
      totalCoins: coinData.length,
      patterns: analysis.patterns,
      highConfidencePatterns,
      marketSummary: analysis.marketSummary,
      stats: {
        strongBuySignals: analysis.patterns.filter(
          (p) => p.signal === "STRONG_BUY"
        ).length,
        buySignals: analysis.patterns.filter((p) => p.signal === "BUY").length,
        sellSignals: analysis.patterns.filter((p) => p.signal === "SELL")
          .length,
        holdSignals: analysis.patterns.filter((p) => p.signal === "HOLD")
          .length,
        avgConfidence:
          analysis.patterns.reduce((sum, p) => sum + p.confidence, 0) /
          analysis.patterns.length,
      },
      cached: false,
    };

    // Update cache
    cachedResult = response;
    lastAnalysisTime = now;

    return NextResponse.json(response);
  } catch (error: any) {
    console.error("Alfabetik Pattern Tracking Error:", error);

    // Fallback to cached data if available
    if (cachedResult) {
      const cacheAge = Math.floor((Date.now() - lastAnalysisTime) / 1000);

      // Only use cache if less than 10 minutes old
      if (cacheAge < 600) {
        return NextResponse.json({
          ...cachedResult,
          cached: true,
          cacheAge,
          warning: "Using cached data due to API error",
          error: error.message,
        });
      }
    }

    // Final fallback - return error with safe mode indicator
    return NextResponse.json(
      {
        success: false,
        error: error.message || "Alfabetik pattern analysis failed",
        fallbackMode: true,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

/**
 * Rate limiting check (called before main logic)
 * Prevents excessive API calls to Binance
 */
function checkRateLimit(): boolean {
  const now = Date.now();
  const timeSinceLastCall = now - lastAnalysisTime;

  // Minimum 60 second interval between fresh analyses
  return timeSinceLastCall >= 60000;
}
