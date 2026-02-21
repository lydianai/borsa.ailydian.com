/**
 * 🔍 COIN SCANNER API
 *
 * Scans 600+ coins from Binance and finds top BUY signals
 * using multi-strategy consensus analysis
 *
 * FEATURES:
 * - Parallel processing for fast scanning
 * - Multi-strategy consensus (AI, Quantum, Conservative, etc.)
 * - Diversity filter (different strategy combinations)
 * - Top N most reliable BUY signals
 *
 * USAGE:
 * GET /api/coin-scanner?limit=10&minConfidence=70
 *
 * WHITE-HAT RULES:
 * - Read-only operations
 * - Transparent signal scoring
 * - Educational purpose only
 * - Rate-limited API calls
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Rate limiting configuration
const BATCH_SIZE = 20; // Process 20 coins at a time
const BATCH_DELAY = 500; // 500ms delay between batches

interface StrategySignal {
  name: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason: string;
}

interface CoinAnalysis {
  symbol: string;
  buyStrategies: StrategySignal[];
  consensusScore: number;
  diversityScore: number;
  topReason: string;
}

/**
 * Fetch all USDT pairs from Binance
 */
async function fetchBinanceSymbols(): Promise<string[]> {
  try {
    const response = await fetch('https://api.binance.com/api/v3/exchangeInfo');
    const data = await response.json();

    const usdtPairs = data.symbols
      .filter((s: any) =>
        s.symbol.endsWith('USDT') &&
        s.status === 'TRADING' &&
        s.quoteAsset === 'USDT'
      )
      .map((s: any) => s.symbol);

    return usdtPairs;
  } catch (error) {
    console.error('[Coin Scanner] Failed to fetch Binance symbols:', error);
    return [];
  }
}

/**
 * Analyze a single coin using strategy-analysis endpoint
 */
async function analyzeCoin(symbol: string, baseUrl: string): Promise<CoinAnalysis | null> {
  try {
    const response = await fetch(`${baseUrl}/api/strategy-analysis/${symbol}`, {
      signal: AbortSignal.timeout(5000), // 5 second timeout per coin
    });

    if (!response.ok) return null;

    const data = await response.json();

    if (!data.success || !data.data?.strategies) return null;

    // Filter only BUY signals
    const buyStrategies: StrategySignal[] = data.data.strategies
      .filter((s: any) => s.signal === 'BUY')
      .map((s: any) => ({
        name: s.name,
        signal: s.signal,
        confidence: s.confidence || 0,
        reason: s.reason || '',
      }));

    if (buyStrategies.length === 0) return null;

    // Calculate consensus score (average confidence of all buy signals)
    const consensusScore = buyStrategies.reduce((sum, s) => sum + s.confidence, 0) / buyStrategies.length;

    // Calculate diversity score (number of different strategies agreeing)
    const diversityScore = buyStrategies.length;

    // Get top reason from highest confidence strategy
    const topStrategy = buyStrategies.reduce((max, s) => s.confidence > max.confidence ? s : max);

    return {
      symbol,
      buyStrategies,
      consensusScore,
      diversityScore,
      topReason: `${topStrategy.name}: ${topStrategy.reason}`,
    };
  } catch (error) {
    return null;
  }
}

/**
 * Sleep utility for batch delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Scan all coins in batches
 */
async function scanCoins(
  symbols: string[],
  baseUrl: string,
  onProgress?: (progress: number) => void
): Promise<CoinAnalysis[]> {
  const results: CoinAnalysis[] = [];
  const totalBatches = Math.ceil(symbols.length / BATCH_SIZE);

  for (let i = 0; i < symbols.length; i += BATCH_SIZE) {
    const batch = symbols.slice(i, i + BATCH_SIZE);
    const batchNumber = Math.floor(i / BATCH_SIZE) + 1;

    console.log(`[Coin Scanner] Processing batch ${batchNumber}/${totalBatches} (${batch.length} coins)`);

    // Process batch in parallel
    const batchResults = await Promise.all(
      batch.map(symbol => analyzeCoin(symbol, baseUrl))
    );

    // Filter out nulls and add to results
    const validResults = batchResults.filter((r): r is CoinAnalysis => r !== null);
    results.push(...validResults);

    // Progress callback
    if (onProgress) {
      const progress = Math.round((i + batch.length) / symbols.length * 100);
      onProgress(progress);
    }

    // Rate limiting delay between batches
    if (i + BATCH_SIZE < symbols.length) {
      await sleep(BATCH_DELAY);
    }
  }

  return results;
}

/**
 * Select diverse signals (different strategy combinations)
 */
function selectDiverseSignals(results: CoinAnalysis[], limit: number): CoinAnalysis[] {
  // Sort by consensus score first
  const sorted = [...results].sort((a, b) => b.consensusScore - a.consensusScore);

  const selected: CoinAnalysis[] = [];
  const usedStrategyCombinations = new Set<string>();

  for (const result of sorted) {
    if (selected.length >= limit) break;

    // Create a signature of the strategy combination
    const strategySignature = result.buyStrategies
      .map(s => s.name)
      .sort()
      .join(',');

    // Prefer unique strategy combinations for diversity
    if (!usedStrategyCombinations.has(strategySignature)) {
      selected.push(result);
      usedStrategyCombinations.add(strategySignature);
    }
  }

  // If we don't have enough diverse signals, fill with remaining high-confidence ones
  if (selected.length < limit) {
    for (const result of sorted) {
      if (selected.length >= limit) break;
      if (!selected.includes(result)) {
        selected.push(result);
      }
    }
  }

  return selected;
}

/**
 * GET handler
 */
export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const limit = parseInt(searchParams.get('limit') || '10', 10);
  const minConfidence = parseInt(searchParams.get('minConfidence') || '60', 10);
  const maxCoins = parseInt(searchParams.get('maxCoins') || '100', 10); // Scan max 100 coins for performance

  try {
    console.log('[Coin Scanner] Starting scan...');
    console.log(`[Coin Scanner] Parameters: limit=${limit}, minConfidence=${minConfidence}, maxCoins=${maxCoins}`);

    // Get base URL from environment or request
    const baseUrl = process.env.BASE_URL || `http://localhost:${process.env.PORT || 3000}`;

    // Fetch all Binance USDT symbols
    const allSymbols = await fetchBinanceSymbols();
    console.log(`[Coin Scanner] Found ${allSymbols.length} USDT pairs on Binance`);

    if (allSymbols.length === 0) {
      return NextResponse.json(
        { success: false, error: 'No symbols found from Binance' },
        { status: 500 }
      );
    }

    // Limit the number of coins to scan (for performance)
    const symbolsToScan = allSymbols.slice(0, maxCoins);
    console.log(`[Coin Scanner] Scanning ${symbolsToScan.length} coins...`);

    // Scan all coins
    const allResults = await scanCoins(symbolsToScan, baseUrl);
    console.log(`[Coin Scanner] Found ${allResults.length} coins with BUY signals`);

    // Filter by minimum confidence
    const filtered = allResults.filter(r => r.consensusScore >= minConfidence);
    console.log(`[Coin Scanner] ${filtered.length} coins passed confidence filter (>=${minConfidence})`);

    // Select diverse signals
    const topSignals = selectDiverseSignals(filtered, limit);
    console.log(`[Coin Scanner] Selected ${topSignals.length} diverse signals`);

    const duration = Date.now() - startTime;

    return NextResponse.json({
      success: true,
      data: {
        signals: topSignals.map(s => ({
          symbol: s.symbol,
          consensusScore: Math.round(s.consensusScore),
          strategyCount: s.diversityScore,
          strategies: s.buyStrategies.map(st => st.name).join(', '),
          topReason: s.topReason,
        })),
        metadata: {
          totalScanned: symbolsToScan.length,
          totalWithBuySignals: allResults.length,
          passedConfidenceFilter: filtered.length,
          returnedSignals: topSignals.length,
          minConfidence,
          scanDuration: duration,
        },
      },
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Coin Scanner] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Scan failed',
        duration,
      },
      { status: 500 }
    );
  }
}
