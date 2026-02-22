import { NextResponse } from "next/server";
import { aiMemorySystem } from "@/lib/ai-memory-system";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";
import { batchAnalyzeWithAITaLib } from "@/lib/ai-talib-analyzer";

// AI Configuration (provider-agnostic)
const AI_API_URL = process.env.AI_API_URL || 'https://api.groq.com/openai/v1/chat/completions';
const AI_API_KEY = process.env.AI_API_KEY || process.env.GROQ_API_KEY || '';

// 5 dakika cache - ENOMEM önleme
const AI_SIGNALS_CACHE_TTL = 5 * 60 * 1000; // 5 minutes
let cachedData: any = null;
let cacheTimestamp = 0;

interface AISignal {
  id: string;
  symbol: string;
  type: "BUY" | "SELL" | "HOLD";
  price: number;
  confidence: number;
  strength: number; // 1-10
  strategy: string;
  reasoning: string;
  targets?: string[];
  timestamp: string;
  aiModel: string;
  unifiedStrategy?: {
    buyPercentage: number;
    waitPercentage: number;
    topRecommendations: string[];
    riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  };
}

// Generate fallback mock AI signals when database is empty
function generateMockAISignals(count: number = 15): AISignal[] {
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

  const signals: AISignal[] = [];
  const types: Array<"BUY" | "SELL" | "HOLD"> = ["BUY", "BUY", "BUY", "SELL", "HOLD"];
  const strategies = [
    'AI_TECHNICAL_TALIB',
    'AI_DEEP_ANALYSIS',
    'AI_MOMENTUM',
    'AI_PATTERN_RECOGNITION',
    'AI_SENTIMENT_ANALYSIS'
  ];
  const aiModels = ['Strategy-Engine-A', 'Strategy-Engine-B', 'Technical-Analysis-Pro'];

  for (let i = 0; i < Math.min(count, mockSymbols.length); i++) {
    const mock = mockSymbols[i];
    const type = types[Math.floor(Math.random() * types.length)];
    const confidence = 55 + Math.floor(Math.random() * 30); // 55-85%
    const strength = 6 + Math.floor(Math.random() * 4); // 6-10

    signals.push({
      id: `ai-mock-${mock.symbol}-${Date.now()}-${i}`,
      symbol: mock.symbol,
      type,
      price: mock.price,
      confidence,
      strength,
      strategy: strategies[Math.floor(Math.random() * strategies.length)],
      reasoning: `AI-powered analysis for ${mock.symbol} - BULLISH momentum detected with strong volume confirmation`,
      targets: type === 'BUY'
        ? [`${(mock.price * 1.03).toFixed(6)}`, `${(mock.price * 1.06).toFixed(6)}`]
        : type === 'SELL'
        ? [`${(mock.price * 0.97).toFixed(6)}`, `${(mock.price * 0.94).toFixed(6)}`]
        : undefined,
      timestamp: new Date().toISOString(),
      aiModel: aiModels[Math.floor(Math.random() * aiModels.length)],
    });
  }

  return signals.sort((a, b) => (b.confidence * b.strength) - (a.confidence * a.strength));
}

// Strategy Engine A integration for enhanced signal analysis
async function analyzeWithStrategyEngineA(
  marketData: any[],
  signals: any[],
): Promise<any[]> {
  if (!AI_API_KEY) {
    console.log("[Strategy Engine A] API key not configured, using fallback analysis");
    return signals;
  }

  try {
    // Prepare market analysis prompt for AI Model
    const topMovers = marketData
      .filter((m) => Math.abs(m.changePercent24h) > 2)
      .sort(
        (a, b) => Math.abs(b.changePercent24h) - Math.abs(a.changePercent24h),
      )
      .slice(0, 10);

    const marketPrompt = `
As an expert cryptocurrency trading analyst, analyze these market conditions and provide AI-enhanced trading signals:

TOP MARKET MOVERS (24h):
${topMovers.map((m) => `${m.symbol}: ${m.changePercent24h > 0 ? "+" : ""}${m.changePercent24h.toFixed(2)}% (Price: $${m.price.toFixed(6)}, Volume: ${(m.volume24h / 1000000).toFixed(1)}M)`).join("\n")}

CURRENT SIGNALS:
${signals
  .slice(0, 5)
  .map(
    (s) =>
      `${s.symbol}: ${s.type} (Confidence: ${s.confidence}%, Strategy: ${s.strategy})`,
  )
  .join("\n")}

Provide enhanced analysis for the top 5 opportunities. Consider:
1. Momentum and volume patterns
2. Risk/reward ratios
3. Market sentiment
4. Technical indicators context

Respond with JSON format:
{
  "enhanced_signals": [
    {
      "symbol": "SYMBOL",
      "type": "BUY/SELL/HOLD",
      "confidence": 85,
      "strength": 8,
      "reasoning": "Detailed analysis of why this signal is generated",
      "targets": ["target1", "target2"]
    }
  ]
}
`;

    const aiModel = process.env.STRATEGY_AI_MODEL || "llama-3.3-70b-versatile";

    const response = await fetch(
      AI_API_URL,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${AI_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: aiModel,
          messages: [
            {
              role: "system",
              content:
                "Sen uzman bir kripto para trading analistisin. Teknik analiz, piyasa duyarlılığı ve risk yönetimi konusunda derin bilgiye sahipsin. Kısa, veriye dayalı trading sinyalleri ve net gerekçeler sun.",
            },
            {
              role: "user",
              content: marketPrompt,
            },
          ],
          temperature: 0.3,
          max_tokens: 1000,
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`AI API error: ${response.status}`);
    }

    const result = await response.json();
    let content = result.choices[0].message.content;

    // Extract JSON from response (handle text before/after JSON)
    // Remove markdown code blocks if present
    content = content.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();

    // Try to find JSON object in content
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      console.log('[AI Analysis] No JSON found in response, using fallback');
      return signals;
    }

    const aiAnalysis = JSON.parse(jsonMatch[0]);

    return aiAnalysis.enhanced_signals || signals;
  } catch (error) {
    console.error("[AI Analysis Error]:", error);
    return signals; // Fallback to original signals
  }
}

// Strategy Engine C - Unified Strategy Aggregator (18 Strategies)
// DISABLED: To prevent ECONNREFUSED errors, we skip unified signals enrichment
// The AI signals work independently without needing the unified strategy data
async function fetchUnifiedSignals(): Promise<Map<string, any>> {
  try {
    console.log('[Strategy Engine C] Unified signals integration disabled (prevents ECONNREFUSED)');
    // Return empty map - AI signals work without unified strategy enrichment
    return new Map();
  } catch (error) {
    console.error('[Strategy Engine C Error]:', error);
    return new Map();
  }
}

// Strategy Engine B integration - REAL Ta-Lib Technical Analysis
async function analyzeWithTechnicalAI(marketData: any[]): Promise<any[]> {
  try {
    console.log(`[AI Technical Analysis] Analyzing ${marketData.length} coins with Ta-Lib...`);

    // Filter viable coins (sufficient volume and data quality)
    const viableCoins = marketData
      .filter((m) => m.volume24h > 50000) // Min $50K volume for reliable data
      .filter((m) => m.symbol && m.price > 0)
      .slice(0, 20); // ENOMEM önleme: max 20 coin

    console.log(`[AI Technical Analysis] Selected ${viableCoins.length} viable coins`);

    // Batch analyze with real Ta-Lib indicators (3 concurrent - ENOMEM önleme)
    const technicalSignals = await batchAnalyzeWithAITaLib(
      viableCoins.map(m => ({ symbol: m.symbol, price: m.price })),
      3 // ENOMEM önleme - düşürüldü
    );

    console.log(`[AI Technical Analysis] Completed analysis for ${technicalSignals.length} coins`);

    // Enhance each signal with AI Memory and historical data
    const enhancedSignals = technicalSignals.map((techSignal) => {
      // Get AI Memory history
      const history = aiMemorySystem.getCoinHistory(techSignal.symbol);

      // Check for signal reversals (BUY → SELL transitions)
      const alert = aiMemorySystem.checkAlert(techSignal.symbol, techSignal.type);

      // Adjust confidence based on AI learning
      let finalConfidence = techSignal.confidence;
      if (alert.alert && alert.riskLevel === 'HIGH') {
        finalConfidence = Math.max(40, finalConfidence - 20);
      } else if (history && history.profitRate > 5) {
        // Boost confidence if historically profitable pattern
        finalConfidence = Math.min(95, finalConfidence + 5);
      }

      // Record movement in AI Memory
      aiMemorySystem.recordMovement(
        techSignal.symbol,
        techSignal.type,
        viableCoins.find(m => m.symbol === techSignal.symbol)?.price || 0,
        finalConfidence,
        'AI_TECHNICAL_TALIB'
      );

      // Build comprehensive reasoning
      const reasoning = [
        techSignal.reasoning,
        `Pattern: ${techSignal.pattern}`,
        `Risk Score: ${techSignal.riskScore}/100`,
        alert.alert ? `⚠️ ${alert.message}` : '',
      ].filter(Boolean).join(' | ');

      return {
        symbol: techSignal.symbol,
        type: techSignal.type,
        confidence: Math.round(finalConfidence),
        strength: techSignal.strength,
        reasoning,
        targets: techSignal.targets,
        alert: alert.alert ? alert.message : undefined,
        riskLevel: alert.riskLevel,
        pattern: techSignal.pattern,
        riskScore: techSignal.riskScore,
        technicalAnalysis: techSignal.technicalAnalysis,
        historicalData: history ? {
          buyToSellCount: history.buyToSellCount,
          profitRate: history.profitRate.toFixed(2) + '%',
        } : undefined,
      };
    });

    // Filter out HOLD signals for cleaner results
    const activeSignals = enhancedSignals.filter(s => s.type !== 'HOLD');

    console.log(`[AI Technical Analysis] Generated ${activeSignals.length} active signals (${enhancedSignals.filter(s => s.type === 'BUY').length} BUY, ${enhancedSignals.filter(s => s.type === 'SELL').length} SELL)`);

    return activeSignals;
  } catch (error) {
    console.error("[AI Technical Analysis Error]:", error);
    return [];
  }
}

export async function GET() {
  try {
    // Check cache first
    const now = Date.now();
    if (cachedData && (now - cacheTimestamp) < AI_SIGNALS_CACHE_TTL) {
      console.log('[AI Signals] Serving from cache');
      return NextResponse.json({
        ...cachedData,
        cached: true,
        nextUpdate: Math.round((AI_SIGNALS_CACHE_TTL - (now - cacheTimestamp)) / 1000),
      });
    }

    console.log('[AI Signals] Starting fresh analysis...');

    // Fetch market data and unified strategy signals in parallel (no internal HTTP calls - prevents ECONNREFUSED)
    const [marketResult, unifiedSignalsMap] = await Promise.all([
      fetchBinanceFuturesData(),
      fetchUnifiedSignals(),
    ]);

    if (!marketResult.success || !marketResult.data) {
      throw new Error(marketResult.error || "Market data fetch failed");
    }

    const marketData = marketResult.data.all;

    // Generate base signals with real Ta-Lib technical analysis
    const baseSignals = await analyzeWithTechnicalAI(marketData);

    // Enhance with Strategy Engine A
    const enhancedSignals = await analyzeWithStrategyEngineA(marketData, baseSignals);

    // ✅ REMOVED FALLBACK MOCK DATA: No more old hardcoded prices
    // Instead of showing fake signals with outdated prices, we show real status
    // This ensures ALL prices are REAL-TIME from Binance/Bybit/CoinGecko
    let finalSignals: AISignal[];
    if (enhancedSignals.length === 0) {
      console.log('[AI Signals] No signals generated (100% real-time data requirement)');
      finalSignals = [];
    } else {
      // Format final signals with metadata and unified strategy data
      finalSignals = enhancedSignals.map((signal, index) => {
      const unifiedData = unifiedSignalsMap.get(signal.symbol);

      return {
        id: `ai-${signal.symbol}-${Date.now()}-${index}`,
        symbol: signal.symbol,
        type: signal.type,
        price:
          marketData.find((m: any) => m.symbol === signal.symbol)?.price || 0,
        confidence: signal.confidence,
        strength: signal.strength,
        strategy: signal.reasoning.includes("BULLISH") || signal.reasoning.includes("BEARISH")
          ? "AI_TECHNICAL_TALIB"
          : "AI_DEEP_ANALYSIS",
        reasoning: signal.reasoning,
        targets: signal.targets,
        timestamp: new Date().toISOString(),
        aiModel: signal.reasoning.includes("BULLISH") || signal.reasoning.includes("BEARISH")
          ? "Technical-Analysis-Pro"
          : "Strategy-Engine-A",
        unifiedStrategy: unifiedData ? {
          buyPercentage: unifiedData.buyPercentage,
          waitPercentage: unifiedData.waitPercentage,
          topRecommendations: unifiedData.topRecommendations,
          riskLevel: unifiedData.riskLevel,
        } : undefined,
      };
    });

      // Sort by confidence and strength
      finalSignals.sort(
        (a, b) => b.confidence * b.strength - a.confidence * a.strength,
      );
    }

    // AI Learning istatistikleri
    const learningStats = aiMemorySystem.getLearningStats();
    const buyToSellTransitions = aiMemorySystem.getBuyToSellTransitions();

    // Count signals with unified strategy data
    const signalsWithUnified = finalSignals.filter(s => s.unifiedStrategy).length;

    const responseData = {
      success: true,
      data: {
        signals: finalSignals.slice(0, 100), // Top 100 AI signals (çok daha fazla!)
        totalSignals: finalSignals.length,
        lastUpdate: new Date().toISOString(),
        aiModels: ["Strategy-Engine-A", "Strategy-Engine-B", "Strategy-Engine-C"],
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
        unifiedStrategy: {
          available: unifiedSignalsMap.size > 0,
          totalSignals: unifiedSignalsMap.size,
          enrichedSignals: signalsWithUnified,
          enrichmentRate: ((signalsWithUnified / finalSignals.length) * 100).toFixed(1) + '%',
        },
        marketStats: {
          totalMarkets: marketData.length,
          avgChange: (
            marketData.reduce(
              (sum: number, m: any) => sum + m.changePercent24h,
              0,
            ) / marketData.length
          ).toFixed(2),
          analyzedMarkets: marketData.filter(
            (m: any) => Math.abs(m.changePercent24h) > 1.5,
          ).length,
          technicalAnalyzed: finalSignals.filter((s) =>
            s.aiModel.includes("Technical"),
          ).length,
          aiEnhanced: finalSignals.filter((s) => s.aiModel.includes("Engine"))
            .length,
          unifiedProcessed: signalsWithUnified,
        },
      },
    };

    // Save to cache
    cachedData = responseData;
    cacheTimestamp = Date.now();

    return NextResponse.json(responseData);
  } catch (error) {
    console.error("[AI Signals API Error]:", error);
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to generate AI signals",
      },
      { status: 500 },
    );
  }
}
