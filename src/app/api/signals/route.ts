import { NextResponse } from "next/server";
import { aiMemorySystem } from "@/lib/ai-memory-system";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";
import { batchAnalyzeWithAITaLib } from "@/lib/ai-talib-analyzer";

// 5 dakika cache - ENOMEM önleme
const SIGNALS_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
let cachedData: any = null;
let cacheTimestamp = 0;

interface TradingSignal {
  id: string;
  symbol: string;
  type: "BUY" | "SELL" | "HOLD";
  price: number;
  confidence: number;
  strength: number; // 1-10
  strategy: string;
  targets?: string[];
  timestamp: string;
  reasoning?: string;
}

// Generate fallback mock signals when database is empty
function generateMockSignals(count: number = 15): TradingSignal[] {
  const mockSymbols = [
    { symbol: 'BTCUSDT', price: 67234.50 },
    { symbol: 'ETHUSDT', price: 3521.80 },
    { symbol: 'SOLUSDT', price: 142.35 },
    { symbol: 'BNBUSDT', price: 589.20 },
    { symbol: 'XRPUSDT', price: 0.5234 },
    { symbol: 'ADAUSDT', price: 0.4123 },
    { symbol: 'DOGEUSDT', price: 0.1456 },
    { symbol: 'AVAXUSDT', price: 36.78 },
    { symbol: 'MATICUSDT', price: 0.7845 },
    { symbol: 'DOTUSDT', price: 6.234 },
    { symbol: 'LINKUSDT', price: 14.567 },
    { symbol: 'UNIUSDT', price: 8.234 },
    { symbol: 'ATOMUSDT', price: 9.876 },
    { symbol: 'LTCUSDT', price: 84.56 },
    { symbol: 'TRXUSDT', price: 0.1678 },
  ];

  const signals: TradingSignal[] = [];
  const types: Array<"BUY" | "SELL" | "HOLD"> = ["BUY", "BUY", "BUY", "SELL", "HOLD"];
  const strategies = [
    'MOMENTUM_BREAKOUT',
    'VOLUME_SURGE',
    'TECHNICAL_ANALYSIS',
    'DOWNTREND_REVERSAL',
    'RSI_DIVERGENCE'
  ];

  for (let i = 0; i < Math.min(count, mockSymbols.length); i++) {
    const mock = mockSymbols[i];
    const type = types[Math.floor(Math.random() * types.length)];
    const confidence = 50 + Math.floor(Math.random() * 35); // 50-85%
    const strength = 5 + Math.floor(Math.random() * 5); // 5-10

    signals.push({
      id: `mock-${type.toLowerCase()}-${mock.symbol}-${Date.now()}-${i}`,
      symbol: mock.symbol,
      type,
      price: mock.price,
      confidence,
      strength,
      strategy: strategies[Math.floor(Math.random() * strategies.length)],
      targets: type === 'BUY'
        ? [`${(mock.price * 1.02).toFixed(6)}`, `${(mock.price * 1.05).toFixed(6)}`]
        : type === 'SELL'
        ? [`${(mock.price * 0.98).toFixed(6)}`, `${(mock.price * 0.95).toFixed(6)}`]
        : undefined,
      timestamp: new Date().toISOString(),
      reasoning: `Mock signal for ${mock.symbol} - fallback data when database is empty`,
    });
  }

  return signals.sort((a, b) => (b.confidence * b.strength) - (a.confidence * a.strength));
}

// Generate AI-powered trading signals with real Ta-Lib technical analysis
async function generateTradingSignals(marketData: any[], limit = 100): Promise<TradingSignal[]> {
  const signals: TradingSignal[] = [];

  console.log(`[Trading Signals] Analyzing ${marketData.length} coins with Ta-Lib...`);

  // ✅ GEVŞET FİLTRELERİ: Daha fazla coin analiz et
  const candidates = marketData
    .filter((m) => m.volume24h > 50000) // 100K → 50K
    .filter((m) => Math.abs(m.changePercent24h) > 0.5 || m.volume24h > 5000000) // 1% → 0.5%, 10M → 5M
    .sort((a, b) => b.volume24h - a.volume24h)
    .slice(0, Math.min(limit, 30)); // 20 → 30 coin

  console.log(`[Trading Signals] Selected ${candidates.length} candidates for analysis`);

  // Perform Ta-Lib technical analysis on all candidates
  const technicalSignals = await batchAnalyzeWithAITaLib(
    candidates.map(m => ({ symbol: m.symbol, price: m.price })),
    3 // 3 concurrent (ENOMEM önleme - düşürüldü)
  );

  console.log(`[Trading Signals] Ta-Lib analysis completed for ${technicalSignals.length} coins`);

  // Convert technical signals to trading signals format with AI Memory integration
  for (const techSignal of technicalSignals) {
    if (techSignal.type === 'HOLD') continue; // Skip HOLD signals

    // Get AI Memory history
    const history = aiMemorySystem.getCoinHistory(techSignal.symbol);

    // Check for signal reversals
    const alert = aiMemorySystem.checkAlert(techSignal.symbol, techSignal.type);

    // Adjust confidence based on AI learning
    let finalConfidence = techSignal.confidence;
    if (alert.alert && alert.riskLevel === 'HIGH') {
      finalConfidence = Math.max(40, finalConfidence - 20);
    } else if (history && history.profitRate > 5) {
      finalConfidence = Math.min(95, finalConfidence + 5);
    }

    // Determine strategy based on pattern
    let strategy = 'TECHNICAL_ANALYSIS';
    if (techSignal.pattern.includes('uptrend') || techSignal.pattern.includes('strong')) {
      strategy = 'MOMENTUM_BREAKOUT';
    } else if (techSignal.pattern.includes('volatile')) {
      strategy = 'VOLUME_SURGE';
    } else if (techSignal.pattern.includes('downtrend')) {
      strategy = 'DOWNTREND_REVERSAL';
    }

    // Record to AI Memory
    aiMemorySystem.recordMovement(
      techSignal.symbol,
      techSignal.type,
      candidates.find(m => m.symbol === techSignal.symbol)?.price || 0,
      finalConfidence,
      strategy
    );

    // Add to signals
    signals.push({
      id: `${techSignal.type.toLowerCase()}-${techSignal.symbol}-${Date.now()}`,
      symbol: techSignal.symbol,
      type: techSignal.type,
      price: candidates.find(m => m.symbol === techSignal.symbol)?.price || 0,
      confidence: Math.round(finalConfidence),
      strength: techSignal.strength,
      strategy,
      targets: techSignal.targets,
      timestamp: new Date().toISOString(),
      reasoning: techSignal.reasoning,
    });
  }

  // Sort by confidence * strength and return top 20
  return signals.sort((a, b) => (b.confidence * b.strength) - (a.confidence * a.strength)).slice(0, 20);
}

export async function GET(request: Request) {
  try {
    // Check cache first
    const now = Date.now();
    if (cachedData && (now - cacheTimestamp) < SIGNALS_CACHE_TTL) {
      console.log('[Trading Signals] Serving from cache');
      return NextResponse.json({
        ...cachedData,
        cached: true,
        nextUpdate: Math.round((SIGNALS_CACHE_TTL - (now - cacheTimestamp)) / 1000),
      });
    }

    console.log('[Trading Signals] Starting fresh analysis...');

    // Get limit from query params (default: 20, max: 20 - ENOMEM önleme)
    const { searchParams } = new URL(request.url);
    const limit = Math.min(parseInt(searchParams.get('limit') || '20'), 20);

    // Fetch market data directly (no internal HTTP calls - prevents ECONNREFUSED)
    const marketResult = await fetchBinanceFuturesData();

    if (!marketResult.success || !marketResult.data) {
      throw new Error(marketResult.error || "Market data fetch failed");
    }

    // Generate AI trading signals with Ta-Lib analysis
    let signals = await generateTradingSignals(marketResult.data.all, limit);

    // ✅ FALLBACK: If Ta-Lib fails, use simple Binance-based signals
    if (signals.length === 0) {
      console.log('[Trading Signals] Ta-Lib analysis returned 0 signals, using Binance-based fallback...');

      // Simple momentum-based signals from real Binance data
      const candidates = marketResult.data.all
        .filter((m: any) => m.volume24h > 1000000 && Math.abs(m.changePercent24h) > 2)
        .sort((a: any, b: any) => Math.abs(b.changePercent24h) - Math.abs(a.changePercent24h))
        .slice(0, 15);

      signals = candidates.map((coin: any) => ({
        id: `${coin.changePercent24h > 0 ? 'buy' : 'sell'}-${coin.symbol}-${Date.now()}`,
        symbol: coin.symbol,
        type: coin.changePercent24h > 0 ? 'BUY' : 'SELL' as 'BUY' | 'SELL',
        price: coin.price,
        confidence: Math.min(85, 60 + Math.abs(coin.changePercent24h) * 3),
        strength: Math.min(10, 5 + Math.floor(Math.abs(coin.changePercent24h) / 2)),
        strategy: coin.changePercent24h > 0 ? 'MOMENTUM_BREAKOUT' : 'DOWNTREND_REVERSAL',
        targets: coin.changePercent24h > 0
          ? [(coin.price * 1.03).toFixed(6), (coin.price * 1.05).toFixed(6)]
          : [(coin.price * 0.97).toFixed(6), (coin.price * 0.95).toFixed(6)],
        timestamp: new Date().toISOString(),
        reasoning: `Real-time ${coin.changePercent24h > 0 ? 'bullish' : 'bearish'} momentum: ${coin.changePercent24h.toFixed(2)}% (24h)`
      }));
    }

    console.log(`[Trading Signals] Generated ${signals.length} signals (100% real-time data)`);

    // AI Learning istatistikleri
    const learningStats = aiMemorySystem.getLearningStats();
    const buyToSellTransitions = aiMemorySystem.getBuyToSellTransitions();

    const responseData = {
      success: true,
      data: {
        signals,
        totalSignals: signals.length,
        lastUpdate: new Date().toISOString(),
        aiLearning: {
          totalAnalyzed: learningStats.totalAnalyzed,
          successRate: learningStats.averageAccuracy.toFixed(2) + '%',
          buyToSellTransitions: buyToSellTransitions.length,
          topRiskyCoins: buyToSellTransitions.slice(0, 5).map(t => ({
            symbol: t.symbol,
            transitions: t.transitionCount,
            risk: t.riskScore,
          })),
        },
        marketStats: {
          totalMarkets: marketResult.data.totalMarkets,
          avgChange: (
            marketResult.data.all.reduce(
              (sum: number, m: any) => sum + m.changePercent24h,
              0,
            ) / marketResult.data.all.length
          ).toFixed(2),
          topGainer:
            marketResult.data.all.sort(
              (a: any, b: any) => b.changePercent24h - a.changePercent24h,
            )[0]?.symbol || "N/A",
          topLoser:
            marketResult.data.all.sort(
              (a: any, b: any) => a.changePercent24h - b.changePercent24h,
            )[0]?.symbol || "N/A",
        },
      },
    };

    // Save to cache
    cachedData = responseData;
    cacheTimestamp = Date.now();

    return NextResponse.json(responseData);
  } catch (error) {
    console.error("[Trading Signals API Error]:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to generate trading signals",
      },
      { status: 500 },
    );
  }
}
