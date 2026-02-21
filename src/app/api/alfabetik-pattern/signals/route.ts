import { NextResponse } from "next/server";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";
import {
  analyzeAlfabetikPattern,
  filterHighConfidenceSignals,
  type AlfabetikPattern,
} from "@/lib/alfabetik-pattern-analyzer";

// Cache configuration
const CACHE_DURATION = 120000; // 2 minutes
let cachedSignals: any = null;
let lastSignalGeneration = 0;

interface SignalRecommendation {
  type: "STRONG_BUY" | "BUY" | "SELL" | "HOLD";
  harf: string;
  reason: string;
  recommendedCoins: string[];
  entryPrices: { [symbol: string]: number };
  targetProfit: string;
  stopLoss: string;
  confidence: number;
  category: string;
  marketMakerActivity: boolean;
}

/**
 * GET /api/alfabetik-pattern/signals
 * Returns actionable trading signals based on alfabetik pattern analysis
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const minConfidence = Number(searchParams.get("minConfidence") || "70");
    const signalType = searchParams.get("type") || "all"; // all, buy, sell

    const now = Date.now();

    // Check cache
    if (cachedSignals && now - lastSignalGeneration < CACHE_DURATION) {
      const filtered = filterSignalsByType(cachedSignals.signals, signalType);

      return NextResponse.json({
        ...cachedSignals,
        signals: filtered,
        cached: true,
        cacheAge: Math.floor((now - lastSignalGeneration) / 1000),
      });
    }

    // Fetch Binance data
    const binanceData = await fetchBinanceFuturesData();

    if (!binanceData.success || !binanceData.data || !binanceData.data.all) {
      throw new Error(
        binanceData.error || "Failed to fetch Binance futures data"
      );
    }

    // Convert to CoinData format
    const coinData = binanceData.data.all
      .filter((coin: any) => coin.symbol.endsWith("USDT"))
      .map((coin: any) => ({
        symbol: coin.symbol.replace("USDT", ""),
        price: coin.price,
        changePercent24h: coin.changePercent24h,
        changePercent7d: 0,
        changePercent30d: 0,
        volume24h: coin.volume,
        category: undefined,
      }));

    // Create price map for quick lookup
    const priceMap: { [symbol: string]: number } = {};
    binanceData.data.all.forEach((coin: any) => {
      const cleanSymbol = coin.symbol.replace("USDT", "");
      priceMap[cleanSymbol] = coin.price;
    });

    // Analyze patterns
    const analysis = analyzeAlfabetikPattern(coinData);

    // Generate signals from high-confidence patterns
    const signals: SignalRecommendation[] = [];

    const highConfidencePatterns = filterHighConfidenceSignals(
      analysis.patterns,
      minConfidence
    );

    for (const pattern of highConfidencePatterns) {
      if (pattern.signal === "HOLD") continue; // Skip HOLD signals

      // Determine category
      const category = getDominantCategory(pattern);

      // Build entry prices
      const entryPrices: { [symbol: string]: number } = {};
      const coinsToRecommend =
        pattern.signal === "STRONG_BUY" || pattern.signal === "BUY"
          ? pattern.topCoins.slice(0, 3)
          : pattern.zayifCoins.slice(0, 2);

      coinsToRecommend.forEach((coin) => {
        const price = priceMap[coin];
        if (price) {
          entryPrices[coin] = price;
        }
      });

      // Generate reason
      const reason = generateReason(pattern, analysis.marketSummary, category);

      // Determine profit/loss targets
      const { targetProfit, stopLoss } = calculateTargets(pattern.signal);

      // Detect market maker activity
      const marketMakerActivity =
        pattern.coinSayisi >= 4 &&
        pattern.güvenilirlik > 75 &&
        Math.abs(pattern.ortalamaPerformans24h) > 6;

      signals.push({
        type: pattern.signal,
        harf: pattern.harf,
        reason,
        recommendedCoins: coinsToRecommend,
        entryPrices,
        targetProfit,
        stopLoss,
        confidence: pattern.confidence,
        category,
        marketMakerActivity,
      });
    }

    // Sort by confidence (highest first)
    signals.sort((a, b) => b.confidence - a.confidence);

    const response = {
      success: true,
      timestamp: analysis.timestamp,
      totalSignals: signals.length,
      signals,
      marketSummary: analysis.marketSummary,
      stats: {
        strongBuyCount: signals.filter((s) => s.type === "STRONG_BUY").length,
        buyCount: signals.filter((s) => s.type === "BUY").length,
        sellCount: signals.filter((s) => s.type === "SELL").length,
        avgConfidence:
          signals.length > 0
            ? signals.reduce((sum, s) => sum + s.confidence, 0) /
              signals.length
            : 0,
        marketMakerSignals: signals.filter((s) => s.marketMakerActivity).length,
      },
      cached: false,
    };

    // Update cache
    cachedSignals = response;
    lastSignalGeneration = now;

    // Filter by type if requested
    const filtered = filterSignalsByType(signals, signalType);

    return NextResponse.json({
      ...response,
      signals: filtered,
    });
  } catch (error: any) {
    console.error("Alfabetik Signal Generation Error:", error);

    // Fallback to cached data
    if (cachedSignals) {
      const cacheAge = Math.floor((Date.now() - lastSignalGeneration) / 1000);

      if (cacheAge < 600) {
        return NextResponse.json({
          ...cachedSignals,
          cached: true,
          cacheAge,
          warning: "Using cached signals due to API error",
          error: error.message,
        });
      }
    }

    return NextResponse.json(
      {
        success: false,
        error: error.message || "Signal generation failed",
        fallbackMode: true,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

/**
 * Get dominant category from pattern
 */
function getDominantCategory(pattern: AlfabetikPattern): string {
  const categories = pattern.kategoriAnaliz;
  let maxCategory = "other";
  let maxCount = categories.other;

  for (const [cat, count] of Object.entries(categories)) {
    if (count > maxCount) {
      maxCategory = cat;
      maxCount = count;
    }
  }

  return maxCategory;
}

/**
 * Generate human-readable reason for signal
 */
function generateReason(
  pattern: AlfabetikPattern,
  marketSummary: any,
  category: string
): string {
  const reasons: string[] = [];

  // Performance reason
  if (pattern.ortalamaPerformans24h > 5) {
    reasons.push(
      `Coins in letter ${pattern.harf} rose an average of ${pattern.ortalamaPerformans24h.toFixed(1)}% in 24 hours`
    );
  } else if (pattern.ortalamaPerformans24h < -5) {
    reasons.push(
      `Coins in letter ${pattern.harf} fell an average of ${Math.abs(pattern.ortalamaPerformans24h).toFixed(1)}% in 24 hours`
    );
  }

  // Trend reason
  if (pattern.momentum === "YUKARIDA") {
    reasons.push(`7-day trend positive (momentum upward)`);
  } else if (pattern.momentum === "ASAGIDA") {
    reasons.push(`7-day trend negative (momentum downward)`);
  }

  // Category reason
  if (category !== "other") {
    reasons.push(`Dominant category: ${categoryToEnglish(category)}`);
  }

  // Reliability reason
  if (pattern.güvenilirlik > 80) {
    reasons.push(
      `High reliability score (${pattern.güvenilirlik}%) - pattern consistent`
    );
  }

  // Market maker activity
  if (
    pattern.coinSayisi >= 4 &&
    pattern.güvenilirlik > 75 &&
    Math.abs(pattern.ortalamaPerformans24h) > 6
  ) {
    reasons.push(`Market maker activity detected`);
  }

  // Rotation trend
  if (marketSummary.rotationTrend) {
    reasons.push(`Market trend: ${marketSummary.rotationTrend}`);
  }

  return reasons.join(". ");
}

/**
 * Calculate profit and stop loss targets based on signal type
 */
function calculateTargets(
  signalType: string
): { targetProfit: string; stopLoss: string } {
  switch (signalType) {
    case "STRONG_BUY":
      return { targetProfit: "15-20%", stopLoss: "-8%" };
    case "BUY":
      return { targetProfit: "10-15%", stopLoss: "-6%" };
    case "SELL":
      return { targetProfit: "8-12% (short)", stopLoss: "-5%" };
    default:
      return { targetProfit: "N/A", stopLoss: "N/A" };
  }
}

/**
 * Convert category to English
 */
function categoryToEnglish(category: string): string {
  const map: { [key: string]: string } = {
    layer1: "Layer 1 Blockchain",
    layer2: "Layer 2 Solutions",
    defi: "DeFi Protocols",
    ai: "AI/Artificial Intelligence",
    gaming: "Gaming",
    other: "Other",
  };

  return map[category] || category;
}

/**
 * Filter signals by type
 */
function filterSignalsByType(
  signals: SignalRecommendation[],
  type: string
): SignalRecommendation[] {
  if (type === "all") return signals;

  const typeUpper = type.toUpperCase();

  return signals.filter((s) => {
    if (typeUpper === "BUY") {
      return s.type === "BUY" || s.type === "STRONG_BUY";
    }
    return s.type === typeUpper;
  });
}
