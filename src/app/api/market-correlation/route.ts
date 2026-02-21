/**
 * 🌐 MARKET CORRELATION API
 * Real correlation analysis with BTC, funding rates, liquidation risk
 *
 * ✅ FALLBACK SYSTEM: Binance → Bybit → CoinGecko
 */

import { NextResponse } from 'next/server';
import { batchAnalyzeCorrelations } from '@/lib/market-correlation-analyzer';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface FuturesCoin {
  symbol: string;
  lastPrice: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
}

interface CorrelationData {
  symbol: string;
  price: number;
  change24h: number;
  omnipotentScore: number;
  marketPhase: string;
  trend: string;
  volumeProfile: string;
  fundingBias: string;
  liquidationRisk: number;
  volatility: number;
  btcCorrelation: number;
  signal: string;
  confidence: number;
}

export async function GET(request: Request) {
  try {
    // Get limit from query params (default: 100, max: 600)
    const { searchParams } = new URL(request.url);
    const limit = Math.min(parseInt(searchParams.get('limit') || '100'), 600);

    console.log(`[Market Correlation] Fetching market data with fallback system for top ${limit} coins...`);

    // ✅ FALLBACK SYSTEM: Use shared data fetcher with Binance → Bybit → CoinGecko
    const marketDataResult = await fetchBinanceFuturesData();

    if (!marketDataResult.success || !marketDataResult.data || !marketDataResult.data.all) {
      console.error('[Market Correlation] Market data fetch failed:', marketDataResult.error);
      throw new Error(marketDataResult.error || 'Piyasa verisi alınamadı - tüm kaynaklar başarısız');
    }

    const allMarkets = marketDataResult.data.all;
    console.log(`[Market Correlation] ✅ Fetched ${allMarkets.length} markets from fallback system`);

    // Convert to Futures format for compatibility
    const allTickers: FuturesCoin[] = allMarkets.map(m => ({
      symbol: m.symbol,
      lastPrice: m.price.toString(),
      priceChange: m.change24h.toString(),
      priceChangePercent: m.changePercent24h.toString(),
      volume: m.volume24h.toString(),
      quoteVolume: m.volume24h.toString(),
      highPrice: m.high24h.toString(),
      lowPrice: m.low24h.toString(),
    }));

    // Filter for USDT perpetuals only
    const usdtTickers = allTickers.filter((ticker) =>
      ticker.symbol.endsWith('USDT') && !ticker.symbol.includes('_')
    );

    console.log(`[Market Correlation] Found ${usdtTickers.length} USDT perpetual contracts`);

    // Analyze top N coins by volume (configurable limit)
    const topCoins = usdtTickers
      .sort((a, b) => parseFloat(b.quoteVolume.toString()) - parseFloat(a.quoteVolume.toString()))
      .slice(0, limit);

    console.log(`[Market Correlation] Analyzing top ${topCoins.length} coins with real correlation data...`);

    // Extract symbols for batch analysis
    const symbols = topCoins.map(coin => coin.symbol);

    // Run real correlation analysis on all coins
    const correlationMetrics = await batchAnalyzeCorrelations(symbols, 10);

    console.log(`[Market Correlation] Correlation analysis complete for ${correlationMetrics.size} coins`);

    // Build correlation data with real metrics
    const correlationData: CorrelationData[] = topCoins.map((coin) => {
      const metrics = correlationMetrics.get(coin.symbol);
      const changePercent = parseFloat(coin.priceChangePercent);

      // Calculate omnipotent score based on multiple factors
      let omnipotentScore = 50; // Base score

      if (metrics) {
        // BTC correlation strength (0-20 points)
        omnipotentScore += Math.abs(metrics.btcCorrelation) * 20;

        // Volume profile (0-15 points)
        const volumePoints = metrics.volumeProfile === 'HIGH' ? 15 : metrics.volumeProfile === 'MEDIUM' ? 8 : 0;
        omnipotentScore += volumePoints;

        // Low liquidation risk = higher score (0-15 points)
        omnipotentScore += (100 - metrics.liquidationRisk) * 0.15;

        // Price momentum (0-20 points)
        omnipotentScore += Math.min(20, Math.abs(changePercent) * 2);

        // Funding balance (0-10 points)
        const fundingPoints = metrics.fundingBias === 'BALANCED' ? 10 : 5;
        omnipotentScore += fundingPoints;
      }

      omnipotentScore = Math.min(100, Math.round(omnipotentScore));

      // Determine market phase based on correlation and momentum
      let marketPhase = 'UNKNOWN';
      if (metrics) {
        if (changePercent > 2 && metrics.btcCorrelation > 0.5) {
          marketPhase = 'MARKUP'; // Rising with BTC
        } else if (changePercent < -2 && metrics.btcCorrelation > 0.5) {
          marketPhase = 'MARKDOWN'; // Falling with BTC
        } else if (Math.abs(changePercent) < 1 && metrics.volatility < 2) {
          marketPhase = 'ACCUMULATION'; // Low volatility, sideways
        } else if (Math.abs(changePercent) < 1 && metrics.volatility >= 2) {
          marketPhase = 'DISTRIBUTION'; // High volatility, sideways
        }
      }

      // Determine trend
      const trend = changePercent > 1 ? 'BULLISH' : changePercent < -1 ? 'BEARISH' : 'SIDEWAYS';

      // Generate signal based on all factors
      let signal = 'NEUTRAL';
      let confidence = 50;

      if (metrics && omnipotentScore > 70) {
        if (changePercent > 2 && metrics.liquidationRisk < 50 && metrics.volumeProfile !== 'LOW') {
          signal = 'BUY';
          confidence = Math.min(90, 60 + (omnipotentScore - 70));
        } else if (changePercent < -2 && metrics.liquidationRisk < 50) {
          signal = 'SELL';
          confidence = Math.min(90, 60 + (omnipotentScore - 70));
        }
      }

      return {
        symbol: coin.symbol,
        price: parseFloat(coin.lastPrice),
        change24h: changePercent,
        omnipotentScore,
        marketPhase,
        trend,
        volumeProfile: metrics?.volumeProfile || 'MEDIUM',
        fundingBias: metrics?.fundingBias || 'BALANCED',
        liquidationRisk: metrics?.liquidationRisk || 50,
        volatility: metrics?.volatility || 0,
        btcCorrelation: metrics?.btcCorrelation || 0,
        signal,
        confidence,
      };
    });

    // Sort by Omnipotent Score (highest first)
    correlationData.sort((a, b) => b.omnipotentScore - a.omnipotentScore);

    // Market overview stats
    const marketOverview = {
      totalCoins: correlationData.length,
      avgOmnipotentScore: Math.round(
        correlationData.reduce((sum, c) => sum + c.omnipotentScore, 0) / correlationData.length
      ),
      bullishCount: correlationData.filter((c) => c.signal === 'BUY').length,
      bearishCount: correlationData.filter((c) => c.signal === 'SELL').length,
      neutralCount: correlationData.filter((c) => c.signal === 'NEUTRAL' || c.signal === 'WAIT').length,
      avgVolatility: (
        correlationData.reduce((sum, c) => sum + c.volatility, 0) / correlationData.length
      ).toFixed(2),
      highConfidenceSignals: correlationData.filter((c) => c.confidence >= 75).length,
      marketPhaseDistribution: {
        ACCUMULATION: correlationData.filter((c) => c.marketPhase === 'ACCUMULATION').length,
        MARKUP: correlationData.filter((c) => c.marketPhase === 'MARKUP').length,
        DISTRIBUTION: correlationData.filter((c) => c.marketPhase === 'DISTRIBUTION').length,
        MARKDOWN: correlationData.filter((c) => c.marketPhase === 'MARKDOWN').length,
      },
    };

    console.log('[Market Correlation] Analysis complete');

    return NextResponse.json({
      success: true,
      data: {
        correlations: correlationData,
        marketOverview,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error('[Market Correlation] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Market correlation analysis failed',
      },
      { status: 500 }
    );
  }
}
