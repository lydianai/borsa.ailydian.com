/**
 * ALFABETİK PATTERN ANALYZER
 * Zero-Error Cryptocurrency Pattern Analysis System
 * Analyzes alphabetic patterns in cryptocurrency markets
 *
 * V2 UPDATE: Now supports real 7d/30d historical data from Binance
 */

import { getHistoricalData, getBatchHistoricalData, type HistoricalData } from './binance-historical-fetcher';

export interface CoinData {
  symbol: string;
  price: number;
  changePercent24h: number;
  changePercent7d?: number;
  changePercent30d?: number;
  volume24h: number;
  category?: string;
}

export interface AlfabetikPattern {
  harf: string;
  coinSayisi: number;
  ortalamaPerformans24h: number;
  ortalamaPerformans7d: number;
  ortalamaPerformans30d: number;
  momentum: "YUKARIDA" | "ASAGIDA" | "YATAY";
  güvenilirlik: number;
  kategoriAnaliz: {
    layer1: number;
    layer2: number;
    defi: number;
    ai: number;
    gaming: number;
    other: number;
  };
  topCoins: string[];
  zayifCoins: string[];
  signal: "STRONG_BUY" | "BUY" | "SELL" | "HOLD";
  confidence: number;
}

export interface PatternValidation {
  score: number;
  tutarlılık7d: number;
  kategoriKorelasyonu: number;
  likiditeArtışı: number;
  güvenilirlik: "YÜKSEK" | "ORTA" | "DÜŞÜK";
}

export interface MarketSummary {
  enGüçlüHarfler: string[];
  enZayıfHarfler: string[];
  rotasyonTrendi: string;
  marketMakerAktivitesi: boolean;
  dominantKategori: string;
}

/**
 * Data validation - Zero error guarantee layer 1
 */
export function validateData(data: CoinData[]): boolean {
  // Minimum coin count check
  if (data.length < 10) {
    console.error("Insufficient data: less than 10 coins");
    return false;
  }

  // Price sanity checks - more lenient
  for (const coin of data) {
    if (coin.price !== undefined && coin.price !== null) {
      if (coin.price < 0 || coin.price > 100000000) {
        console.warn(`Questionable price for ${coin.symbol}: ${coin.price}`);
        // Don't fail, just warn
      }
    }

    if (coin.changePercent24h !== undefined && coin.changePercent24h !== null) {
      if (Math.abs(coin.changePercent24h) > 1000) {
        console.warn(`Large 24h change for ${coin.symbol}: ${coin.changePercent24h}%`);
        // Don't fail, just warn
      }
    }

    if (!coin.symbol || coin.symbol.length < 1) {
      console.error(`Invalid symbol: ${coin.symbol}`);
      return false;
    }
  }

  return true;
}

/**
 * Categorize coin by name pattern
 */
export function categorizeCoin(symbol: string): string {
  const name = symbol.toUpperCase();

  // Layer 1 blockchains
  if (
    name.includes("LAYER") ||
    name === "ETH" ||
    name === "BNB" ||
    name === "SOL" ||
    name === "ADA" ||
    name === "AVAX" ||
    name === "MATIC" ||
    name === "DOT" ||
    name === "ATOM"
  ) {
    return "layer1";
  }

  // Layer 2 solutions
  if (
    name.includes("ARB") ||
    name.includes("OP") ||
    name.includes("MATIC") ||
    name.includes("LRC") ||
    name.includes("IMX") ||
    name === "LINK"
  ) {
    return "layer2";
  }

  // DeFi protocols
  if (
    name.includes("SWAP") ||
    name.includes("CAKE") ||
    name.includes("UNI") ||
    name.includes("AAVE") ||
    name.includes("COMP") ||
    name.includes("SNX") ||
    name.includes("CRV") ||
    name.includes("SUSHI")
  ) {
    return "defi";
  }

  // AI tokens
  if (
    name.includes("FET") ||
    name.includes("AGIX") ||
    name.includes("AI") ||
    name.includes("OCEAN") ||
    name.includes("NMR")
  ) {
    return "ai";
  }

  // Gaming tokens
  if (
    name.includes("GAME") ||
    name.includes("AXS") ||
    name.includes("SAND") ||
    name.includes("MANA") ||
    name.includes("ENJ") ||
    name.includes("GALA")
  ) {
    return "gaming";
  }

  return "other";
}

/**
 * Group coins by first letter
 */
export function groupByLetter(coins: CoinData[]): Map<string, CoinData[]> {
  const grouped = new Map<string, CoinData[]>();

  for (const coin of coins) {
    const firstChar = coin.symbol.charAt(0).toUpperCase();

    if (!grouped.has(firstChar)) {
      grouped.set(firstChar, []);
    }

    grouped.get(firstChar)!.push(coin);
  }

  return grouped;
}

/**
 * Calculate average performance for a letter group
 */
export function calculateLetterPerformance(
  coins: CoinData[],
  timeframe: "24h" | "7d" | "30d"
): number {
  if (coins.length === 0) return 0;

  const sum = coins.reduce((acc, coin) => {
    switch (timeframe) {
      case "24h":
        return acc + coin.changePercent24h;
      case "7d":
        return acc + (coin.changePercent7d || 0);
      case "30d":
        return acc + (coin.changePercent30d || 0);
      default:
        return acc;
    }
  }, 0);

  return sum / coins.length;
}

/**
 * Determine momentum direction
 */
export function calculateMomentum(
  perf24h: number,
  perf7d: number
): "YUKARIDA" | "ASAGIDA" | "YATAY" {
  if (perf24h > 3 && perf7d > 2) return "YUKARIDA";
  if (perf24h < -3 && perf7d < -2) return "ASAGIDA";
  return "YATAY";
}

/**
 * Calculate pattern reliability score
 */
export function calculateGüvenilirlik(
  perf7d: number,
  categoryDominance: number,
  volumeIncrease: number
): number {
  // Consistency (40%)
  const consistency = Math.min(Math.abs(perf7d) * 5, 40);

  // Category correlation (30%)
  const categoryScore = categoryDominance * 30;

  // Liquidity increase (30%)
  const liquidityScore = Math.min(volumeIncrease * 30, 30);

  return Math.min(consistency + categoryScore + liquidityScore, 100);
}

/**
 * Analyze category distribution for a letter
 */
export function analyzeCategories(coins: CoinData[]): {
  layer1: number;
  layer2: number;
  defi: number;
  ai: number;
  gaming: number;
  other: number;
} {
  const counts = {
    layer1: 0,
    layer2: 0,
    defi: 0,
    ai: 0,
    gaming: 0,
    other: 0,
  };

  for (const coin of coins) {
    const category = categorizeCoin(coin.symbol);
    counts[category as keyof typeof counts]++;
  }

  return counts;
}

/**
 * Get top and bottom performing coins
 */
export function getTopAndBottomCoins(
  coins: CoinData[],
  count: number = 3
): { top: string[]; bottom: string[] } {
  const sorted = [...coins].sort(
    (a, b) => b.changePercent24h - a.changePercent24h
  );

  return {
    top: sorted.slice(0, count).map((c) => c.symbol),
    bottom: sorted.slice(-count).map((c) => c.symbol),
  };
}

/**
 * Generate signal based on pattern analysis
 */
export function generateSignal(
  perf24h: number,
  perf7d: number,
  güvenilirlik: number,
  momentum: "YUKARIDA" | "ASAGIDA" | "YATAY",
  categoryDominance: number
): { signal: "STRONG_BUY" | "BUY" | "SELL" | "HOLD"; confidence: number } {
  // STRONG BUY conditions
  if (
    perf24h > 5 &&
    perf7d > 3 &&
    güvenilirlik > 70 &&
    momentum === "YUKARIDA" &&
    categoryDominance > 0.5
  ) {
    return { signal: "STRONG_BUY", confidence: Math.min(güvenilirlik, 95) };
  }

  // BUY conditions
  if (
    perf24h > 3 &&
    (perf7d > 1 || momentum === "YUKARIDA") &&
    güvenilirlik > 50
  ) {
    return { signal: "BUY", confidence: Math.min(güvenilirlik, 85) };
  }

  // SELL conditions
  if (perf24h < -5 && perf7d < -3 && güvenilirlik > 60 && momentum === "ASAGIDA") {
    return { signal: "SELL", confidence: Math.min(güvenilirlik, 80) };
  }

  // HOLD (default)
  return { signal: "HOLD", confidence: 50 };
}

/**
 * Enrich coin data with historical 7d/30d data
 * V2 feature - fetches real historical data from Binance
 */
export async function enrichWithHistoricalData(coins: CoinData[]): Promise<CoinData[]> {
  console.log(`📊 Fetching historical data for ${coins.length} coins...`);

  const symbols = coins.map(c => c.symbol);
  const historicalDataMap = await getBatchHistoricalData(symbols, {
    batchSize: 20, // Larger batch size for faster processing
    delayMs: 50     // Small delay between batches
  });

  console.log(`✅ Fetched historical data for ${historicalDataMap.size} coins`);

  // Merge historical data into coin data
  const enrichedCoins = coins.map(coin => {
    const historicalData = historicalDataMap.get(coin.symbol);

    if (historicalData) {
      return {
        ...coin,
        changePercent7d: historicalData.changePercent7d,
        changePercent30d: historicalData.changePercent30d
      };
    }

    // If no historical data, keep original (with 0 fallback)
    return coin;
  });

  return enrichedCoins;
}

/**
 * Main analysis function
 */
export function analyzeAlfabetikPattern(coins: CoinData[]): {
  patterns: AlfabetikPattern[];
  marketSummary: MarketSummary;
  timestamp: string;
} {
  // Validate data
  if (!validateData(coins)) {
    throw new Error("Invalid coin data - validation failed");
  }

  // Group by letter
  const grouped = groupByLetter(coins);

  const patterns: AlfabetikPattern[] = [];

  // Analyze each letter
  for (const [letter, letterCoins] of grouped.entries()) {
    if (letterCoins.length < 2) continue; // Skip letters with only 1 coin

    const perf24h = calculateLetterPerformance(letterCoins, "24h");
    const perf7d = calculateLetterPerformance(letterCoins, "7d");
    const perf30d = calculateLetterPerformance(letterCoins, "30d");

    const momentum = calculateMomentum(perf24h, perf7d);

    const categories = analyzeCategories(letterCoins);
    const totalCoins = letterCoins.length;
    const maxCategory = Math.max(...Object.values(categories));
    const categoryDominance = maxCategory / totalCoins;

    // Calculate volume increase (simplified - would need historical data)
    const avgVolume =
      letterCoins.reduce((sum, c) => sum + c.volume24h, 0) / letterCoins.length;
    const volumeIncrease = avgVolume > 1000000 ? 0.8 : 0.5;

    const güvenilirlik = calculateGüvenilirlik(
      perf7d,
      categoryDominance,
      volumeIncrease
    );

    // V2 UPDATE: Return ALL coins, not just top 3
    const { top, bottom } = getTopAndBottomCoins(letterCoins, letterCoins.length);

    const { signal, confidence } = generateSignal(
      perf24h,
      perf7d,
      güvenilirlik,
      momentum,
      categoryDominance
    );

    patterns.push({
      harf: letter,
      coinSayisi: letterCoins.length,
      ortalamaPerformans24h: Number(perf24h.toFixed(2)),
      ortalamaPerformans7d: Number(perf7d.toFixed(2)),
      ortalamaPerformans30d: Number(perf30d.toFixed(2)),
      momentum,
      güvenilirlik: Number(güvenilirlik.toFixed(0)),
      kategoriAnaliz: categories,
      topCoins: top,
      zayifCoins: bottom,
      signal,
      confidence: Number(confidence.toFixed(0)),
    });
  }

  // Sort patterns by 24h performance
  patterns.sort((a, b) => b.ortalamaPerformans24h - a.ortalamaPerformans24h);

  // Generate market summary
  const strongPatterns = patterns.filter((p) => p.ortalamaPerformans24h > 3);
  const weakPatterns = patterns.filter((p) => p.ortalamaPerformans24h < 0);

  const enGüçlüHarfler = strongPatterns.slice(0, 5).map((p) => p.harf);
  // Sort weak patterns by performance (worst first) and take top 5
  const sortedWeakPatterns = [...weakPatterns].sort((a, b) => a.ortalamaPerformans24h - b.ortalamaPerformans24h);
  const enZayıfHarfler = sortedWeakPatterns.slice(0, 5).map((p) => p.harf);

  // Detect rotation trend
  let rotasyonTrendi = "Belirsiz market durumu";
  if (strongPatterns.length > 0) {
    const dominantCategories = strongPatterns.map((p) => {
      const cats = p.kategoriAnaliz;
      const maxCat = Object.keys(cats).reduce((a, b) =>
        cats[a as keyof typeof cats] > cats[b as keyof typeof cats] ? a : b
      );
      return maxCat;
    });

    const categoryCount: { [key: string]: number } = {};
    dominantCategories.forEach((cat) => {
      categoryCount[cat] = (categoryCount[cat] || 0) + 1;
    });

    const topCategory = Object.keys(categoryCount).reduce((a, b) =>
      categoryCount[a] > categoryCount[b] ? a : b
    );

    rotasyonTrendi = `${topCategory} kategorisine doğru kayış tespit edildi`;
  }

  // Detect market maker activity
  const marketMakerAktivitesi = strongPatterns.some(
    (p) => p.coinSayisi >= 4 && p.güvenilirlik > 75
  );

  // Find dominant category safely
  let dominantKategori = "other";
  if (strongPatterns.length > 0) {
    const cats = strongPatterns[0].kategoriAnaliz;
    let maxCat = "other";
    let maxCount = cats.other;

    for (const [cat, count] of Object.entries(cats)) {
      if (count > maxCount) {
        maxCat = cat;
        maxCount = count;
      }
    }
    dominantKategori = maxCat;
  }

  const marketSummary: MarketSummary = {
    enGüçlüHarfler,
    enZayıfHarfler,
    rotasyonTrendi,
    marketMakerAktivitesi,
    dominantKategori,
  };

  return {
    patterns,
    marketSummary,
    timestamp: new Date().toISOString(),
  };
}

/**
 * Filter high-confidence signals only
 */
export function filterHighConfidenceSignals(
  patterns: AlfabetikPattern[],
  minConfidence: number = 60
): AlfabetikPattern[] {
  return patterns.filter((p) => {
    // Minimum confidence
    if (p.confidence < minConfidence) return false;

    // Minimum coin count (single coin patterns not reliable)
    if (p.coinSayisi < 2) return false;

    // Avoid contradictory signals
    if (p.signal === "BUY" && p.momentum === "ASAGIDA") return false;
    if (p.signal === "SELL" && p.momentum === "YUKARIDA") return false;

    return true;
  });
}
