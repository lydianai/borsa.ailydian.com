/**
 * 🌟 NIRVANA DASHBOARD API
 * Unified overview of all completed trading strategies
 *
 * Aggregates data from multiple signal generation strategies
 * with IP protection to hide proprietary implementation details
 */

import { NextResponse } from 'next/server';
import { sanitizeAPIResponse } from '@/lib/ip-protection';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface StrategyOverview {
  name: string;
  status: 'active' | 'completed' | 'pending';
  totalSignals: number;
  buySignals: number;
  sellSignals: number;
  waitSignals: number;
  avgConfidence: number;
  topOpportunity?: {
    symbol: string;
    signal: string;
    confidence: number;
  };
  error?: string;
}

interface NirvanaOverview {
  totalStrategies: number;
  activeStrategies: number;
  totalSignals: number;
  highConfidenceSignals: number;
  marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  sentimentScore: number; // -100 to +100
  strategies: StrategyOverview[];
  topOpportunities: Array<{
    symbol: string;
    strategy: string;
    signal: string;
    confidence: number;
    reason: string;
  }>;
  timestamp: string;
}

/**
 * Fetch strategy data with error handling and timeout
 */
async function fetchStrategy(endpoint: string, timeoutMs: number = 3000): Promise<any> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const baseUrl = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error: any) {
    clearTimeout(timeoutId);
    console.error(`[Nirvana] Error fetching ${endpoint}:`, error.message);
    return { success: false, error: error.message };
  }
}

export async function GET() {
  try {
    console.log('[Nirvana] Aggregating all strategy data...');

    // Fetch all strategies in parallel (including Python microservices)
    const [
      tradingSignals,
      aiSignals,
      conservativeSignals,
      breakoutSignals,
      marketCorrelation,
      btcEthAnalysis,
      omnipotentFutures,
      unifiedSignals,
      pythonAIModels,
      pythonSignalGenerator,
      pythonTALib,
    ] = await Promise.all([
      fetchStrategy('/api/signals?limit=600'),
      fetchStrategy('/api/ai-signals'),
      fetchStrategy('/api/conservative-signals'),
      fetchStrategy('/api/breakout-retest'), // FIXED: was /api/breakout-signals
      fetchStrategy('/api/market-correlation?limit=600'),
      fetchStrategy('/api/btc-eth-analysis'),
      fetchStrategy('/api/omnipotent-futures?limit=600'),
      fetchStrategy('/api/unified-signals?limit=600&minBuyPercentage=50'),
      // Python Microservices (health checks)
      fetchStrategy('/api/python-services/ai-models/health'),
      fetchStrategy('/api/python-services/signal-generator/health'),
      fetchStrategy('/api/python-services/talib-service/health'),
    ]);

    const strategies: StrategyOverview[] = [];
    let totalSignals = 0;
    let highConfidenceSignals = 0;
    let bullishScore = 0;
    let bearishScore = 0;

    // 1. Trading Signals
    if (tradingSignals.success) {
      const signals = tradingSignals.data?.signals || [];
      const buyCount = signals.filter((s: any) => s.type === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.type === 'SELL').length;
      const avgConf = signals.length > 0
        ? signals.reduce((sum: number, s: any) => sum + (s.confidence || 0), 0) / signals.length
        : 0;

      strategies.push({
        name: 'Trading Signals',
        status: 'completed',
        totalSignals: signals.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: signals.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: signals[0] ? {
          symbol: signals[0].symbol,
          signal: signals[0].type,
          confidence: signals[0].confidence,
        } : undefined,
      });

      totalSignals += signals.length;
      bullishScore += buyCount;
      bearishScore += sellCount;
      highConfidenceSignals += signals.filter((s: any) => s.confidence >= 75).length;
    } else {
      strategies.push({
        name: 'Trading Signals',
        status: 'active',
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 0,
        error: tradingSignals.error,
      });
    }

    // 2. AI Signals
    if (aiSignals.success) {
      const signals = aiSignals.data?.signals || [];
      const buyCount = signals.filter((s: any) => s.type === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.type === 'SELL').length;
      const avgConf = signals.length > 0
        ? signals.reduce((sum: number, s: any) => sum + (s.confidence || 0), 0) / signals.length
        : 0;

      strategies.push({
        name: 'AI Signals (Groq)',
        status: 'completed',
        totalSignals: signals.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: signals.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: signals[0] ? {
          symbol: signals[0].symbol,
          signal: signals[0].type,
          confidence: signals[0].confidence,
        } : undefined,
      });

      totalSignals += signals.length;
      bullishScore += buyCount;
      bearishScore += sellCount;
      highConfidenceSignals += signals.filter((s: any) => s.confidence >= 75).length;
    } else {
      strategies.push({
        name: 'AI Signals (Groq)',
        status: 'active',
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 0,
        error: aiSignals.error,
      });
    }

    // 3. Conservative Signals
    if (conservativeSignals.success) {
      const signals = conservativeSignals.data?.signals || [];
      const buyCount = signals.filter((s: any) => s.signal === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.signal === 'SELL').length;
      const avgConf = signals.length > 0
        ? signals.reduce((sum: number, s: any) => sum + (s.confidence || 0), 0) / signals.length
        : 0;

      strategies.push({
        name: 'Conservative Signals',
        status: 'completed',
        totalSignals: signals.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: signals.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: signals[0] ? {
          symbol: signals[0].symbol,
          signal: signals[0].signal,
          confidence: signals[0].confidence,
        } : undefined,
      });

      totalSignals += signals.length;
      bullishScore += buyCount;
      bearishScore += sellCount;
    }

    // 4. Breakout-Retest
    if (breakoutSignals.success) {
      const signals = breakoutSignals.data?.signals || [];
      const buyCount = signals.filter((s: any) => s.signal === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.signal === 'SELL').length;
      const avgConf = signals.length > 0
        ? signals.reduce((sum: number, s: any) => sum + (s.confidence || 0), 0) / signals.length
        : 0;

      strategies.push({
        name: 'Breakout-Retest',
        status: 'completed',
        totalSignals: signals.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: signals.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: signals[0] ? {
          symbol: signals[0].symbol,
          signal: signals[0].signal,
          confidence: signals[0].confidence,
        } : undefined,
      });

      totalSignals += signals.length;
      bullishScore += buyCount;
      bearishScore += sellCount;
      highConfidenceSignals += signals.filter((s: any) => s.confidence >= 75).length;
    }

    // 5. Market Correlation
    if (marketCorrelation.success) {
      const correlations = marketCorrelation.data?.correlations || [];
      const buyCount = correlations.filter((c: any) => c.signal === 'BUY').length;
      const sellCount = correlations.filter((c: any) => c.signal === 'SELL').length;
      const avgConf = correlations.length > 0
        ? correlations.reduce((sum: number, c: any) => sum + (c.confidence || 0), 0) / correlations.length
        : 0;

      strategies.push({
        name: 'Market Correlation',
        status: 'completed',
        totalSignals: correlations.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: correlations.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: correlations[0] ? {
          symbol: correlations[0].symbol,
          signal: correlations[0].signal,
          confidence: correlations[0].confidence,
        } : undefined,
      });

      totalSignals += correlations.length;
      bullishScore += buyCount;
      bearishScore += sellCount;
    }

    // 6. BTC-ETH Analysis (info only, not signals)
    if (btcEthAnalysis.success) {
      const data = btcEthAnalysis.data;
      const isBullish = data.trend === 'Rising';
      const isBearish = data.trend === 'Falling';

      strategies.push({
        name: 'BTC-ETH Analysis',
        status: 'completed',
        totalSignals: 1,
        buySignals: isBullish ? 1 : 0,
        sellSignals: isBearish ? 1 : 0,
        waitSignals: (!isBullish && !isBearish) ? 1 : 0,
        avgConfidence: Math.round((data.correlation30d || 0) * 100),
      });

      if (isBullish) bullishScore += 1;
      if (isBearish) bearishScore += 1;
    }

    // 7. Omnipotent Futures (Wyckoff)
    if (omnipotentFutures.success) {
      const overview = omnipotentFutures.data?.marketOverview;
      const buyCount = overview?.signals?.BUY || 0;
      const sellCount = overview?.signals?.SELL || 0;
      const waitCount = overview?.signals?.WAIT || 0;

      strategies.push({
        name: 'Omnipotent Futures (Wyckoff)',
        status: 'completed',
        totalSignals: buyCount + sellCount + waitCount,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: waitCount,
        avgConfidence: overview?.highConfidenceSignals || 0,
      });

      totalSignals += (buyCount + sellCount + waitCount);
      bullishScore += buyCount;
      bearishScore += sellCount;
      highConfidenceSignals += (overview?.highConfidenceSignals || 0);
    }

    // 8. Unified Signals
    if (unifiedSignals.success) {
      const signals = unifiedSignals.data?.signals || [];
      const buyCount = signals.filter((s: any) => s.overallSignal === 'BUY').length;
      const sellCount = signals.filter((s: any) => s.overallSignal === 'SELL').length;
      const avgConf = parseFloat(unifiedSignals.data?.stats?.avgConfidence || '0');

      strategies.push({
        name: 'Unified Signals (Consensus)',
        status: 'completed',
        totalSignals: signals.length,
        buySignals: buyCount,
        sellSignals: sellCount,
        waitSignals: signals.length - buyCount - sellCount,
        avgConfidence: Math.round(avgConf),
        topOpportunity: signals[0] ? {
          symbol: signals[0].symbol,
          signal: signals[0].overallSignal,
          confidence: signals[0].overallConfidence,
        } : undefined,
      });

      bullishScore += buyCount;
      bearishScore += sellCount;
    }

    // 9. Python AI Models (ML/AI Ensemble)
    if (pythonAIModels.status === 'healthy') {
      strategies.push({
        name: 'Python AI Ensemble',
        status: 'completed',
        totalSignals: 0, // Models loaded but not yet generating signals
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: pythonAIModels.models_loaded || 0,
        topOpportunity: {
          symbol: 'SYSTEM',
          signal: 'READY',
          confidence: 100,
        },
      });
    } else {
      strategies.push({
        name: 'Python AI Ensemble',
        status: 'pending',
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 0,
        error: 'Service offline',
      });
    }

    // 10. Python Signal Generator
    if (pythonSignalGenerator.status === 'healthy') {
      strategies.push({
        name: 'Signal Generator (Python)',
        status: 'completed',
        totalSignals: 0, // Ready but not yet generating signals
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 100,
        topOpportunity: {
          symbol: 'SYSTEM',
          signal: 'READY',
          confidence: 100,
        },
      });
    } else {
      strategies.push({
        name: 'Signal Generator (Python)',
        status: 'pending',
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 0,
        error: 'Service offline',
      });
    }

    // 11. Python TA-Lib Professional (158 indicators)
    if (pythonTALib.status === 'healthy' && pythonTALib.talib_available) {
      strategies.push({
        name: 'TA-Lib Professional (158 indicators)',
        status: 'completed',
        totalSignals: pythonTALib.total_indicators || 158,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 100,
        topOpportunity: {
          symbol: `v${pythonTALib.talib_version || '0.6.7'}`,
          signal: 'READY',
          confidence: 100,
        },
      });
    } else {
      strategies.push({
        name: 'TA-Lib Professional',
        status: 'pending',
        totalSignals: 0,
        buySignals: 0,
        sellSignals: 0,
        waitSignals: 0,
        avgConfidence: 0,
        error: 'Service offline',
      });
    }

    // Calculate market sentiment
    const sentimentScore = ((bullishScore - bearishScore) / Math.max(bullishScore + bearishScore, 1)) * 100;
    let marketSentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
    if (sentimentScore > 20) marketSentiment = 'BULLISH';
    else if (sentimentScore < -20) marketSentiment = 'BEARISH';

    // Aggregate top opportunities
    const topOpportunities: any[] = [];

    // Add from each strategy
    if (unifiedSignals.success) {
      const signals = unifiedSignals.data?.signals || [];
      signals.slice(0, 3).forEach((s: any) => {
        if (s.overallSignal !== 'WAIT') {
          topOpportunities.push({
            symbol: s.symbol,
            strategy: 'Unified (5 strategies)',
            signal: s.overallSignal,
            confidence: s.overallConfidence,
            reason: s.recommendation,
          });
        }
      });
    }

    const nirvanaData: NirvanaOverview = {
      totalStrategies: strategies.length,
      activeStrategies: strategies.filter(s => s.status === 'completed').length,
      totalSignals,
      highConfidenceSignals,
      marketSentiment,
      sentimentScore: Math.round(sentimentScore),
      strategies,
      topOpportunities: topOpportunities.slice(0, 5),
      timestamp: new Date().toISOString(),
    };

    console.log('[Nirvana] Overview complete');
    console.log(`- Total Strategies: ${nirvanaData.totalStrategies}`);
    console.log(`- Market Sentiment: ${marketSentiment} (${Math.round(sentimentScore)})`);
    console.log(`- Total Signals: ${totalSignals}`);

    // Apply IP protection - obfuscate model names and technical indicators
    return NextResponse.json(sanitizeAPIResponse({
      success: true,
      data: nirvanaData,
    }));

  } catch (error: any) {
    console.error('[Nirvana] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Nirvana dashboard aggregation failed',
      },
      { status: 500 }
    );
  }
}
