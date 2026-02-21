import { NextResponse } from "next/server";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";
import {
  analyzeAlfabetikPattern,
  enrichWithHistoricalData,
  type AlfabetikPattern,
} from "@/lib/alfabetik-pattern-analyzer";
import {
  runBacktest,
  getLetterBacktest,
  type BacktestSummary,
  type LetterBacktestResult,
} from "@/lib/alfabetik-backtest-engine";

// Cache configuration - 5 minute cache
const CACHE_DURATION = 300000; // 5 minutes
let cachedBacktestResult: any = null;
let lastBacktestTime = 0;

/**
 * GET /api/alfabetik-pattern/backtest
 * Returns simulated backtest results for alfabetik patterns
 * Query params:
 *   - period: number of days (default: 30)
 *   - letter: specific letter to backtest (optional)
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const period = parseInt(searchParams.get("period") || "30", 10);
    const specificLetter = searchParams.get("letter")?.toUpperCase();

    const now = Date.now();

    // Check cache validity (only for full backtest, not letter-specific)
    if (
      !specificLetter &&
      cachedBacktestResult &&
      now - lastBacktestTime < CACHE_DURATION
    ) {
      return NextResponse.json({
        ...cachedBacktestResult,
        cached: true,
        cacheAge: Math.floor((now - lastBacktestTime) / 1000),
      });
    }

    console.log(
      `🔄 Running backtest for ${specificLetter || "all letters"} over ${period} days...`
    );

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

    // Enrich with real historical data from Binance
    console.log("🔄 Enriching coin data with 7d/30d historical data...");
    const enrichedCoinData = await enrichWithHistoricalData(coinData);
    console.log(
      `✅ Historical data enrichment complete for ${enrichedCoinData.length} coins`
    );

    // Perform alfabetik pattern analysis with enriched data
    const analysis = analyzeAlfabetikPattern(enrichedCoinData);

    // Run backtest on patterns
    let backtestResult: BacktestSummary | LetterBacktestResult;

    if (specificLetter) {
      // Backtest specific letter
      const pattern = analysis.patterns.find((p) => p.harf === specificLetter);

      if (!pattern) {
        return NextResponse.json(
          {
            success: false,
            error: `Letter '${specificLetter}' not found in analysis`,
          },
          { status: 404 }
        );
      }

      const letterBacktest = getLetterBacktest(pattern, period);

      backtestResult = letterBacktest;

      return NextResponse.json({
        success: true,
        timestamp: new Date().toISOString(),
        backtestType: "letter",
        letter: specificLetter,
        period: `${period}d`,
        result: backtestResult,
        pattern: {
          letter: pattern.harf,
          coinCount: pattern.coinSayisi,
          averagePerformance24h: pattern.ortalamaPerformans24h,
          averagePerformance7d: pattern.ortalamaPerformans7d,
          averagePerformance30d: pattern.ortalamaPerformans30d,
          momentum: pattern.momentum,
          reliability: pattern.güvenilirlik,
          signal: pattern.signal,
          confidence: pattern.confidence,
          topCoins: pattern.topCoins,
        },
        cached: false,
      });
    } else {
      // Full backtest for all patterns
      const fullBacktest = runBacktest(analysis.patterns, period);

      backtestResult = fullBacktest;

      const response = {
        success: true,
        timestamp: new Date().toISOString(),
        backtestType: "full",
        period: `${period}d`,
        summary: {
          totalPatterns: fullBacktest.totalPatterns,
          totalTrades: fullBacktest.totalTrades,
          totalWins: fullBacktest.totalWins,
          totalLosses: fullBacktest.totalLosses,
          overallWinRate: fullBacktest.overallWinRate,
          avgProfit: fullBacktest.avgProfit,
          totalProfitPercent: fullBacktest.totalProfitPercent,
          profitFactor: fullBacktest.profitFactor,
          bestLetter: fullBacktest.bestLetter,
          worstLetter: fullBacktest.worstLetter,
        },
        letterResults: fullBacktest.letterResults,
        topPerformers: fullBacktest.letterResults.slice(0, 5).map((l) => ({
          letter: l.letter,
          winRate: l.winRate,
          profitFactor: l.profitFactor,
          avgProfit: l.avgProfit,
          totalTrades: l.totalTrades,
        })),
        bottomPerformers: fullBacktest.letterResults
          .slice(-5)
          .reverse()
          .map((l) => ({
            letter: l.letter,
            winRate: l.winRate,
            profitFactor: l.profitFactor,
            avgProfit: l.avgProfit,
            totalTrades: l.totalTrades,
          })),
        cached: false,
      };

      // Update cache
      cachedBacktestResult = response;
      lastBacktestTime = now;

      return NextResponse.json(response);
    }
  } catch (error: any) {
    console.error("Alfabetik Pattern Backtest Error:", error);

    // Fallback to cached data if available
    if (cachedBacktestResult) {
      const cacheAge = Math.floor((Date.now() - lastBacktestTime) / 1000);

      // Only use cache if less than 15 minutes old
      if (cacheAge < 900) {
        return NextResponse.json({
          ...cachedBacktestResult,
          cached: true,
          cacheAge,
          warning: "Using cached data due to API error",
          error: error.message,
        });
      }
    }

    // Final fallback - return error
    return NextResponse.json(
      {
        success: false,
        error: error.message || "Alfabetik pattern backtest failed",
        fallbackMode: true,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
