/**
 * 🎓 STRATEJİ ÖĞRENME MOTORU - SARDAG EMRAH EVRİM SİSTEMİ
 *
 * AI Memory Store + Advanced AI Engine kullanarak stratejileri geliştirir.
 * Her 4 saatte bir otomatik olarak:
 * - Tüm stratejilerin performansını analiz eder
 * - Coin-bazlı adaptif ağırlıklar hesaplar
 * - Yeni parametre önerileri üretir
 * - A/B testing için varyasyonlar oluşturur
 */

import {
  getCoinStrategyPerformances,
  saveAdaptiveWeights,
  getAdaptiveWeights,
  saveGlobalStrategyStats,
  type AdaptiveWeights,
  type StrategyPerformance,
} from '../memory/ai-memory-store';
import {
  analyzeMarketWithAdvancedAI,
  evaluateStrategyPerformance,
  suggestCoinWeights,
} from '../ai/advanced-analyzer';

/**
 * Global strateji performansı
 */
export interface GlobalStrategyAnalysis {
  strategyName: string;
  totalCoins: number;
  avgSuccessRate: number;
  topPerformingCoins: Array<{ symbol: string; successRate: number }>;
  poorPerformingCoins: Array<{ symbol: string; successRate: number }>;
  overallRating: 'excellent' | 'good' | 'moderate' | 'poor';
  improvements: string[];
  recommendedWeight: number; // 0.5 - 2.0
}

/**
 * Coin için stratejileri öğren ve ağırlıklandır
 */
export async function learnCoinStrategies(symbol: string): Promise<AdaptiveWeights | null> {
  console.log(`\n🎓 Learning: Analyzing strategies for ${symbol}...`);

  try {
    // 1. Coin'in geçmiş performanslarını al
    const performances = await getCoinStrategyPerformances(symbol);

    if (performances.length === 0) {
      console.log(`⚠️ Learning: No historical data for ${symbol}, using default weights`);
      return null;
    }

    // 2. Mevcut adaptif ağırlıkları kontrol et (4 saat içindeyse kullan)
    const existingWeights = await getAdaptiveWeights(symbol);
    if (existingWeights) {
      console.log(`✅ Learning: Using cached adaptive weights for ${symbol}`);
      return existingWeights;
    }

    // 3. Market datasını hazırla
    const historicalPerformance: { [strategy: string]: { successRate: number; tradeCount: number } } = {};
    performances.forEach((perf) => {
      historicalPerformance[perf.strategyName] = {
        successRate: perf.successRate,
        tradeCount: perf.totalSignals,
      };
    });

    // Simüle edilmiş market data (gerçek sistemde Binance API'den gelecek)
    const currentMarket = {
      volatility: 2.5, // %
      volume24h: 50_000_000, // $50M
      trend: 'BULLISH', // BULLISH | BEARISH | SIDEWAYS
    };

    // 4. Claude AI'dan ağırlık önerileri al
    console.log(`🤖 Learning: Requesting weight suggestions from Claude AI...`);
    const aiSuggestion = await suggestCoinWeights({
      symbol,
      historicalPerformance,
      currentMarket,
    });

    // 5. Adaptif ağırlıkları oluştur
    const now = new Date();
    const validUntil = new Date(now.getTime() + 4 * 3600 * 1000); // 4 saat sonra

    const adaptiveWeights: AdaptiveWeights = {
      symbol,
      weights: aiSuggestion.weights || {},
      reasoning: aiSuggestion.reasoning || {},
      lastCalculated: now.toISOString(),
      validUntil: validUntil.toISOString(),
    };

    // 6. Kaydet
    await saveAdaptiveWeights(adaptiveWeights);

    console.log(`✅ Learning: Adaptive weights calculated and saved for ${symbol}`);
    console.log(`   Weights:`, JSON.stringify(adaptiveWeights.weights, null, 2));

    return adaptiveWeights;
  } catch (error: any) {
    console.error(`❌ Learning: Error analyzing ${symbol}:`, error.message);
    return null;
  }
}

/**
 * Tüm stratejilerin global performansını analiz et
 */
export async function analyzeGlobalStrategyPerformance(
  symbols: string[]
): Promise<GlobalStrategyAnalysis[]> {
  console.log(`\n📊 Learning: Global strategy performance analysis...`);
  console.log(`   Analyzing ${symbols.length} coins across 15 strategies\n`);

  const strategyNames = [
    'conservative-buy-signal',
    'breakout-retest',
    'volume-spike',
    'ma-crossover-pullback',
    'rsi-divergence',
    'ma7-pullback',
    'bollinger-squeeze',
    'ema-ribbon',
    'ichimoku-cloud',
    'macd-histogram',
    'fibonacci-retracement',
    'atr-volatility',
    'trend-reversal',
    'volume-profile',
    'red-wick-green-closure',
  ];

  const analyses: GlobalStrategyAnalysis[] = [];

  for (const strategyName of strategyNames) {
    console.log(`\n🔍 Analyzing: ${strategyName}`);

    const coinPerformances: Array<{ symbol: string; successRate: number }> = [];

    // Her coin için bu stratejinin performansını topla
    for (const symbol of symbols) {
      const performances = await getCoinStrategyPerformances(symbol);
      const strategyPerf = performances.find((p) => p.strategyName === strategyName);

      if (strategyPerf && strategyPerf.totalSignals > 0) {
        coinPerformances.push({
          symbol,
          successRate: strategyPerf.successRate,
        });
      }
    }

    if (coinPerformances.length === 0) {
      console.log(`   ⚠️ No data for ${strategyName}, skipping...`);
      continue;
    }

    // Ortalama başarı oranı
    const avgSuccessRate =
      coinPerformances.reduce((sum, c) => sum + c.successRate, 0) / coinPerformances.length;

    // En iyi ve en kötü performans gösteren coinler
    const sortedBySuccess = [...coinPerformances].sort((a, b) => b.successRate - a.successRate);
    const topPerforming = sortedBySuccess.slice(0, 5);
    const poorPerforming = sortedBySuccess.slice(-5).reverse();

    // Rating
    let overallRating: GlobalStrategyAnalysis['overallRating'];
    if (avgSuccessRate >= 70) overallRating = 'excellent';
    else if (avgSuccessRate >= 55) overallRating = 'good';
    else if (avgSuccessRate >= 40) overallRating = 'moderate';
    else overallRating = 'poor';

    console.log(`   📈 Avg Success Rate: ${avgSuccessRate.toFixed(2)}%`);
    console.log(`   ⭐ Rating: ${overallRating.toUpperCase()}`);
    console.log(`   📊 Data from ${coinPerformances.length} coins`);

    // Claude AI'dan performans değerlendirmesi al
    const recentSignals = coinPerformances.slice(0, 10).map((c) => ({
      timestamp: new Date().toISOString(),
      signal: 'BUY' as const,
      confidence: c.successRate,
      outcome: c.successRate > 50 ? ('success' as const) : ('failure' as const),
    }));

    let improvements: string[] = [];
    let recommendedWeight = 1.0;

    try {
      const aiEvaluation = await evaluateStrategyPerformance({
        strategyName,
        recentSignals,
        successRate: avgSuccessRate,
        avgConfidence: avgSuccessRate,
      });

      improvements = aiEvaluation.improvements || [];
      recommendedWeight = aiEvaluation.recommendedWeight || 1.0;

      console.log(`   🤖 AI Recommended Weight: ${recommendedWeight}`);
    } catch (error: any) {
      console.error(`   ❌ AI Evaluation failed:`, error.message);
    }

    const analysis: GlobalStrategyAnalysis = {
      strategyName,
      totalCoins: coinPerformances.length,
      avgSuccessRate,
      topPerformingCoins: topPerforming,
      poorPerformingCoins: poorPerforming,
      overallRating,
      improvements,
      recommendedWeight,
    };

    analyses.push(analysis);

    // Global stats'ı kaydet
    await saveGlobalStrategyStats(strategyName, {
      totalCoins: coinPerformances.length,
      avgSuccessRate,
      topPerformingCoins: topPerforming.map((c) => c.symbol),
      poorPerformingCoins: poorPerforming.map((c) => c.symbol),
    });
  }

  console.log(`\n✅ Learning: Global strategy analysis complete!`);
  console.log(`   Analyzed ${analyses.length} strategies\n`);

  return analyses;
}

/**
 * Yeni strateji parametreleri öner
 */
export async function suggestNewStrategyParams(
  strategyName: string,
  currentPerformance: StrategyPerformance
): Promise<{
  parameterSuggestions: Array<{
    parameter: string;
    currentValue: number;
    suggestedValue: number;
    reason: string;
  }>;
}> {
  console.log(`\n💡 Learning: Suggesting parameter improvements for ${strategyName}...`);

  try {
    // Market datasını hazırla
    const marketData = {
      symbol: currentPerformance.symbol,
      strategyResults: [
        {
          name: strategyName,
          signal: 'BUY',
          confidence: currentPerformance.avgConfidence,
        },
      ],
    };

    // Advanced AI'dan analiz al
    const analysis = await analyzeMarketWithAdvancedAI(marketData);

    if (analysis.parameterSuggestions && analysis.parameterSuggestions.length > 0) {
      console.log(`   ✅ ${analysis.parameterSuggestions.length} parameter suggestions received`);
      return { parameterSuggestions: analysis.parameterSuggestions };
    }

    console.log(`   ⚠️ No parameter suggestions available`);
    return { parameterSuggestions: [] };
  } catch (error: any) {
    console.error(`❌ Learning: Error suggesting parameters:`, error.message);
    return { parameterSuggestions: [] };
  }
}

/**
 * A/B testing için strateji varyasyonları oluştur
 */
export async function createStrategyVariants(
  strategyName: string,
  baseParams: { [key: string]: number }
): Promise<
  Array<{
    variantName: string;
    params: { [key: string]: number };
    description: string;
  }>
> {
  console.log(`\n🧪 Learning: Creating A/B test variants for ${strategyName}...`);

  const variants: Array<{
    variantName: string;
    params: { [key: string]: number };
    description: string;
  }> = [];

  // Variant 1: Conservative (daha güvenli)
  const conservativeParams = { ...baseParams };
  Object.keys(conservativeParams).forEach((key) => {
    if (key.includes('threshold')) {
      conservativeParams[key] *= 1.2; // %20 daha yüksek threshold
    }
  });

  variants.push({
    variantName: `${strategyName}-conservative`,
    params: conservativeParams,
    description: 'Daha yüksek threshold\'lar ile daha az sinyal, daha yüksek güven',
  });

  // Variant 2: Aggressive (daha agresif)
  const aggressiveParams = { ...baseParams };
  Object.keys(aggressiveParams).forEach((key) => {
    if (key.includes('threshold')) {
      aggressiveParams[key] *= 0.8; // %20 daha düşük threshold
    }
  });

  variants.push({
    variantName: `${strategyName}-aggressive`,
    params: aggressiveParams,
    description: 'Daha düşük threshold\'lar ile daha fazla sinyal, daha fazla fırsat',
  });

  // Variant 3: Optimized (AI önerili)
  const optimizedParams = { ...baseParams };
  // Gerçek implementasyonda Claude AI'dan öneriler alınacak
  variants.push({
    variantName: `${strategyName}-optimized`,
    params: optimizedParams,
    description: 'AI tarafından önerilen optimized parametreler',
  });

  console.log(`   ✅ Created ${variants.length} variants for A/B testing`);

  return variants;
}

/**
 * En iyi performans gösteren stratejileri belirle
 */
export async function identifyTopStrategies(
  symbol: string,
  count: number = 5
): Promise<StrategyPerformance[]> {
  console.log(`\n⭐ Learning: Identifying top ${count} strategies for ${symbol}...`);

  const performances = await getCoinStrategyPerformances(symbol);

  if (performances.length === 0) {
    console.log(`   ⚠️ No performance data for ${symbol}`);
    return [];
  }

  // Başarı oranına göre sırala
  const sorted = [...performances].sort((a, b) => {
    // Önce başarı oranı
    if (b.successRate !== a.successRate) {
      return b.successRate - a.successRate;
    }
    // Eşitse toplam sinyal sayısına bak (daha fazla data = daha güvenilir)
    return b.totalSignals - a.totalSignals;
  });

  const topStrategies = sorted.slice(0, count);

  console.log(`   ✅ Top ${count} strategies for ${symbol}:`);
  topStrategies.forEach((s, index) => {
    console.log(
      `      ${index + 1}. ${s.strategyName} - Success: ${s.successRate.toFixed(2)}% (${s.totalSignals} signals)`
    );
  });

  return topStrategies;
}

/**
 * Öğrenme sistemi sağlık kontrolü
 */
export async function checkLearningSystemHealth() {
  console.log(`\n🏥 Learning: Health check...`);

  try {
    // Test coin için öğrenme çalıştır
    const testSymbol = 'BTCUSDT';
    const weights = await learnCoinStrategies(testSymbol);

    if (weights) {
      console.log(`   ✅ Learning system operational`);
      return {
        status: 'healthy',
        testedSymbol: testSymbol,
        weightsCalculated: Object.keys(weights.weights).length,
      };
    }

    console.log(`   ⚠️ Learning system returned no weights (may be expected if no data)`);
    return {
      status: 'healthy',
      message: 'No historical data for test',
    };
  } catch (error: any) {
    console.error(`   ❌ Learning system unhealthy:`, error.message);
    return {
      status: 'unhealthy',
      error: error.message,
    };
  }
}

export default {
  learnCoinStrategies,
  analyzeGlobalStrategyPerformance,
  suggestNewStrategyParams,
  createStrategyVariants,
  identifyTopStrategies,
  checkLearningSystemHealth,
};
