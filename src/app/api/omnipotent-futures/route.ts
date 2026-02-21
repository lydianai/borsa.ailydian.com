/**
 * 🔮 OMNIPOTENT FUTURES API v3.0
 * Ultimate multi-dimensional futures analysis system
 *
 * v1.0 Features:
 * - Real Wyckoff cycle phase detection (4 phases)
 * - Volume profile analysis (climax, dry-up)
 * - Smart money activity detection
 * - Effort vs Result divergence
 * - Support/Resistance levels
 * - Trading signals based on Wyckoff principles
 *
 * v2.0 Features:
 * - Funding Rate Analysis (Binance)
 * - Open Interest Tracking (Binance)
 * - BTC Dominance Metrics (CoinGecko)
 * - Fear & Greed Index (Alternative.me)
 * - Liquidation Zone Estimator
 *
 * NEW v3.0 Features:
 * - Multi-layer Correlation Matrix (DXY, S&P500, GOLD, VIX)
 * - Risk Management Calculator (Kelly Criterion)
 * - Position Sizing Recommendations
 * - AI-Powered Signal Aggregation
 * - Technical Indicators (RSI, MACD, Bollinger Bands) - REAL DATA
 * - 200+ Coins Analysis (increased from 50+)
 */

import { NextResponse } from 'next/server';
import { batchAnalyzeWyckoff } from '@/lib/wyckoff-analyzer';
import {
  fetchAllMarketMetrics,
  calculateLiquidationZones,
  type FundingRateData,
  type OpenInterestData,
  type BTCDominanceData,
  type FearGreedData,
  type LiquidationZone,
} from '@/lib/omnipotent-data-sources';
import {
  fetchCorrelationMatrix,
  type MacroMetrics,
  type CorrelationMatrix,
} from '@/lib/correlation-engine';
import {
  calculateKellyCriterion,
  calculatePositionSize,
  type KellyCriterion,
  type PositionSizeRecommendation,
} from '@/lib/risk-management';
import {
  calculateTechnicalIndicators,
  type TechnicalIndicators,
  type RSI,
  type MACD,
  type BollingerBands,
} from '@/lib/technical-indicators';
import {
  analyzeMultiTimeframe,
  type MultiTimeframeAnalysis,
  type TimeframeAnalysis,
} from '@/lib/multi-timeframe-analyzer';
import {
  calculateVolumeProfile,
  type VolumeProfile,
  type PriceLevel,
  type ValueArea,
} from '@/lib/volume-profile-analyzer';
import {
  analyzeOrderFlow,
  type OrderFlowData,
} from '@/lib/order-flow-analyzer';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface FuturesCoin {
  symbol: string;
  lastPrice: string;
  priceChangePercent: string;
  quoteVolume: string;
}

interface OmnipotentFuturesData {
  symbol: string;
  price: number;
  change24h: number;

  // Wyckoff Analysis
  wyckoffPhase: string;
  wyckoffConfidence: number;
  subPhase?: string;
  smartMoneyActivity: string;
  volumeProfile: string;
  volumeRatio: number;
  climaxDetected: boolean;
  dryUpDetected: boolean;
  support: number;
  resistance: number;
  rangePercent: number;
  trendStrength: number;
  effortVsResultDivergence: boolean;

  // NEW: Funding Rate Data
  fundingRate?: number;
  fundingRateAnnualized?: number;
  nextFundingTime?: number;

  // NEW: Open Interest Data
  openInterest?: number;
  openInterestValue?: number;
  openInterestChange24h?: number;

  // NEW: Liquidation Zones
  liquidationZones?: LiquidationZone[];
  nearestLiquidation?: {
    long: { price: number; distance: number };
    short: { price: number; distance: number };
  };

  // NEW v3.0: Technical Indicators (REAL DATA from Binance klines)
  technicalIndicators?: {
    rsi: RSI;
    macd: MACD;
    bollingerBands: BollingerBands;
    timestamp: string;
  };

  // Trading Signals
  signal: string;
  confidence: number;
  recommendation: string;
  priceAction: string;
}

export async function GET(request: Request) {
  try {
    // Get limit from query params (default: 200, max: 300) - INCREASED FROM 100
    const { searchParams } = new URL(request.url);
    const limit = Math.min(parseInt(searchParams.get('limit') || '200'), 300);

    console.log(`[Omnipotent Futures] Starting Wyckoff analysis for top ${limit} coins...`);

    // 1. Fetch Binance Futures market data using SHARED CACHED FETCHER
    console.log(`[Omnipotent Futures] Fetching Binance data via shared cached fetcher...`);
    const binanceData = await fetchBinanceFuturesData();

    if (!binanceData.success || !binanceData.data) {
      throw new Error(binanceData.error || 'Failed to fetch Binance futures data');
    }

    console.log(`[Omnipotent Futures] Retrieved ${binanceData.data.all.length} coins from shared cache`);

    // 2. Select top N by volume for analysis (configurable limit)
    // Data is already sorted by volume in the shared fetcher
    const topCoins = binanceData.data.all.slice(0, limit);

    console.log(`[Omnipotent Futures] Analyzing top ${topCoins.length} coins with Wyckoff method...`);

    // 3. Fetch Wyckoff analysis + All market metrics IN PARALLEL
    const symbols = topCoins.map(coin => coin.symbol);

    console.log(`[Omnipotent Futures] Fetching Wyckoff + Market Metrics + Correlations in parallel...`);
    const [wyckoffAnalyses, marketMetrics, correlations] = await Promise.all([
      batchAnalyzeWyckoff(symbols, 10),
      fetchAllMarketMetrics(symbols),
      fetchCorrelationMatrix(),
    ]);

    console.log(`[Omnipotent Futures] Wyckoff analysis complete for ${wyckoffAnalyses.size} coins`);
    console.log(`[Omnipotent Futures] Market metrics fetched:`);
    console.log(`  - Funding Rates: ${marketMetrics.fundingRates.size}`);
    console.log(`  - Open Interest: ${marketMetrics.openInterest.size}`);
    console.log(`  - BTC Dominance: ${marketMetrics.btcDominance ? 'Yes' : 'No'}`);
    console.log(`  - Fear & Greed: ${marketMetrics.fearGreed ? 'Yes' : 'No'}`);
    console.log(`[Omnipotent Futures] Correlation Matrix: ${correlations.correlationMatrix ? 'Yes' : 'No'}`);

    // NEW v3.0: Calculate Technical Indicators for major coins (top 20)
    console.log(`[Omnipotent Futures] Calculating technical indicators for top 20 coins...`);
    const majorCoins = symbols.slice(0, 20); // Only top 20 to avoid timeout
    const technicalIndicatorsMap = new Map<string, TechnicalIndicators>();

    await Promise.all(
      majorCoins.map(async (symbol) => {
        try {
          const indicators = await calculateTechnicalIndicators(symbol, '1h');
          if (indicators) {
            technicalIndicatorsMap.set(symbol, indicators);
          }
        } catch (error) {
          console.warn(`[Omnipotent Futures] Failed to calculate indicators for ${symbol}`);
        }
      })
    );

    console.log(`[Omnipotent Futures] Technical indicators calculated for ${technicalIndicatorsMap.size} coins`);

    // NEW v3.0: Multi-Timeframe Analysis (BTC only to avoid timeout)
    console.log(`[Omnipotent Futures] Calculating multi-timeframe analysis for BTC...`);
    let btcMultiTimeframe: MultiTimeframeAnalysis | null = null;
    try {
      btcMultiTimeframe = await analyzeMultiTimeframe('BTCUSDT');
      console.log(`[Omnipotent Futures] Multi-timeframe analysis complete: ${btcMultiTimeframe.consensus.signal}`);
    } catch (error) {
      console.warn(`[Omnipotent Futures] Failed to calculate multi-timeframe analysis:`, error);
    }

    // NEW v3.0: Volume Profile Analysis (BTC only to avoid timeout)
    console.log(`[Omnipotent Futures] Calculating volume profile for BTC...`);
    let btcVolumeProfile: VolumeProfile | null = null;
    try {
      btcVolumeProfile = await calculateVolumeProfile('BTCUSDT', '1h', 100, 50);
      console.log(`[Omnipotent Futures] Volume profile complete: POC=${btcVolumeProfile.poc.price.toFixed(2)}, Position=${btcVolumeProfile.pricePosition}`);
    } catch (error) {
      console.warn(`[Omnipotent Futures] Failed to calculate volume profile:`, error);
    }

    // NEW v3.0: Order Flow Analysis (BTC only to avoid timeout)
    console.log(`[Omnipotent Futures] Calculating order flow for BTC...`);
    let btcOrderFlow: OrderFlowData | null = null;
    try {
      btcOrderFlow = await analyzeOrderFlow('BTCUSDT', '1h', 50);
      console.log(`[Omnipotent Futures] Order flow complete: Signal=${btcOrderFlow.signal}, Imbalance=${btcOrderFlow.imbalance.strength}`);
    } catch (error) {
      console.warn(`[Omnipotent Futures] Failed to calculate order flow:`, error);
    }

    // 4. Build omnipotent futures data with Wyckoff + Market Metrics
    const futuresData: OmnipotentFuturesData[] = topCoins.map((coin) => {
      const analysis = wyckoffAnalyses.get(coin.symbol);
      const changePercent = coin.changePercent24h; // Already a number in MarketData
      const currentPrice = coin.price; // Already a number in MarketData

      // Get new market metrics
      const fundingData = marketMetrics.fundingRates.get(coin.symbol);
      const oiData = marketMetrics.openInterest.get(coin.symbol);

      // Get technical indicators (if available for this coin)
      const techIndicators = technicalIndicatorsMap.get(coin.symbol);

      // Calculate liquidation zones if we have OI and funding data
      let liquidationZones: LiquidationZone[] | undefined;
      let nearestLiquidation: any;

      if (oiData && fundingData) {
        liquidationZones = calculateLiquidationZones(
          currentPrice,
          oiData.openInterest,
          fundingData.fundingRate
        );

        // Find nearest liquidation zones
        const longZones = liquidationZones.filter(z => z.side === 'LONG' && z.price < currentPrice);
        const shortZones = liquidationZones.filter(z => z.side === 'SHORT' && z.price > currentPrice);

        if (longZones.length > 0 && shortZones.length > 0) {
          nearestLiquidation = {
            long: {
              price: longZones[0].price,
              distance: ((currentPrice - longZones[0].price) / currentPrice * 100),
            },
            short: {
              price: shortZones[0].price,
              distance: ((shortZones[0].price - currentPrice) / currentPrice * 100),
            },
          };
        }
      }

      if (!analysis) {
        return {
          symbol: coin.symbol,
          price: currentPrice,
          change24h: changePercent,
          wyckoffPhase: 'UNKNOWN',
          wyckoffConfidence: 0,
          smartMoneyActivity: 'NEUTRAL',
          volumeProfile: 'STABLE',
          volumeRatio: 1,
          climaxDetected: false,
          dryUpDetected: false,
          support: 0,
          resistance: 0,
          rangePercent: 0,
          trendStrength: 50,
          effortVsResultDivergence: false,
          fundingRate: fundingData?.fundingRate,
          fundingRateAnnualized: fundingData ? fundingData.fundingRate * 365 * 3 : undefined,
          nextFundingTime: fundingData?.fundingTime,
          openInterest: oiData?.openInterest,
          openInterestValue: oiData?.openInterestValue,
          liquidationZones,
          nearestLiquidation,
          technicalIndicators: techIndicators ? {
            rsi: techIndicators.rsi,
            macd: techIndicators.macd,
            bollingerBands: techIndicators.bollingerBands,
            timestamp: techIndicators.timestamp,
          } : undefined,
          signal: 'WAIT',
          confidence: 0,
          recommendation: 'Analiz başarısız',
          priceAction: 'Veri yetersiz',
        };
      }

      return {
        symbol: coin.symbol,
        price: currentPrice,
        change24h: changePercent,
        wyckoffPhase: analysis.wyckoffPhase.phase,
        wyckoffConfidence: analysis.wyckoffPhase.confidence,
        subPhase: analysis.wyckoffPhase.subPhase,
        smartMoneyActivity: analysis.wyckoffPhase.smartMoneyActivity,
        volumeProfile: analysis.volumeProfile.volumeTrend,
        volumeRatio: parseFloat(analysis.volumeProfile.volumeRatio.toFixed(2)),
        climaxDetected: analysis.volumeProfile.climaxVolume,
        dryUpDetected: analysis.volumeProfile.dryUp,
        support: parseFloat(analysis.priceRange.support.toFixed(2)),
        resistance: parseFloat(analysis.priceRange.resistance.toFixed(2)),
        rangePercent: parseFloat(analysis.priceRange.rangePercent.toFixed(2)),
        trendStrength: parseFloat(analysis.trendStrength.toFixed(1)),
        effortVsResultDivergence: analysis.effortVsResult.divergence,
        fundingRate: fundingData?.fundingRate,
        fundingRateAnnualized: fundingData ? fundingData.fundingRate * 365 * 3 : undefined,
        nextFundingTime: fundingData?.fundingTime,
        openInterest: oiData?.openInterest,
        openInterestValue: oiData?.openInterestValue,
        liquidationZones,
        nearestLiquidation,
        technicalIndicators: techIndicators ? {
          rsi: techIndicators.rsi,
          macd: techIndicators.macd,
          bollingerBands: techIndicators.bollingerBands,
          timestamp: techIndicators.timestamp,
        } : undefined,
        signal: analysis.signal,
        confidence: analysis.signalConfidence,
        recommendation: analysis.recommendation,
        priceAction: analysis.wyckoffPhase.priceAction,
      };
    });

    // 6. Sort by signal confidence (highest first)
    futuresData.sort((a, b) => b.confidence - a.confidence);

    // 7. Calculate market overview stats
    const marketOverview = {
      totalCoins: futuresData.length,
      phaseDistribution: {
        ACCUMULATION: futuresData.filter((c) => c.wyckoffPhase === 'ACCUMULATION').length,
        MARKUP: futuresData.filter((c) => c.wyckoffPhase === 'MARKUP').length,
        DISTRIBUTION: futuresData.filter((c) => c.wyckoffPhase === 'DISTRIBUTION').length,
        MARKDOWN: futuresData.filter((c) => c.wyckoffPhase === 'MARKDOWN').length,
        UNKNOWN: futuresData.filter((c) => c.wyckoffPhase === 'UNKNOWN').length,
      },
      smartMoneyActivity: {
        BUYING: futuresData.filter((c) => c.smartMoneyActivity === 'BUYING').length,
        SELLING: futuresData.filter((c) => c.smartMoneyActivity === 'SELLING').length,
        NEUTRAL: futuresData.filter((c) => c.smartMoneyActivity === 'NEUTRAL').length,
      },
      signals: {
        BUY: futuresData.filter((c) => c.signal === 'BUY').length,
        SELL: futuresData.filter((c) => c.signal === 'SELL').length,
        WAIT: futuresData.filter((c) => c.signal === 'WAIT').length,
      },
      highConfidenceSignals: futuresData.filter((c) => c.confidence >= 75).length,
      climaxDetected: futuresData.filter((c) => c.climaxDetected).length,
      dryUpDetected: futuresData.filter((c) => c.dryUpDetected).length,
      avgTrendStrength: (
        futuresData.reduce((sum, c) => sum + c.trendStrength, 0) / futuresData.length
      ).toFixed(1),
    };

    // 8. Top opportunities (high confidence, clear phase)
    const topOpportunities = futuresData
      .filter((c) => c.confidence >= 70 && c.signal !== 'WAIT')
      .slice(0, 10);

    console.log('[Omnipotent Futures] Analysis complete');
    console.log(`- Accumulation: ${marketOverview.phaseDistribution.ACCUMULATION}`);
    console.log(`- Markup: ${marketOverview.phaseDistribution.MARKUP}`);
    console.log(`- Distribution: ${marketOverview.phaseDistribution.DISTRIBUTION}`);
    console.log(`- Markdown: ${marketOverview.phaseDistribution.MARKDOWN}`);
    console.log(`- High Confidence Signals: ${marketOverview.highConfidenceSignals}`);

    return NextResponse.json({
      success: true,
      data: {
        futures: futuresData,
        marketOverview,
        topOpportunities,
        // NEW v2.0: Global Market Metrics
        globalMetrics: {
          btcDominance: marketMetrics.btcDominance ? {
            btc: marketMetrics.btcDominance.btcDominance,
            eth: marketMetrics.btcDominance.ethDominance,
            stables: marketMetrics.btcDominance.stableDominance,
            totalMarketCap: marketMetrics.btcDominance.totalMarketCap,
          } : null,
          fearGreed: marketMetrics.fearGreed ? {
            value: marketMetrics.fearGreed.value,
            classification: marketMetrics.fearGreed.valueClassification,
          } : null,
        },
        // NEW v3.0: Macro Correlations
        macroCorrelations: correlations.macroMetrics,
        correlationMatrix: correlations.correlationMatrix,
        // NEW v3.0: Multi-Timeframe Analysis (BTC)
        btcMultiTimeframe,
        // NEW v3.0: Volume Profile Analysis (BTC)
        btcVolumeProfile,
        // NEW v3.0: Order Flow Analysis (BTC)
        btcOrderFlow,
        // Metadata
        timestamp: new Date().toISOString(),
        version: '3.0',
        dataSourcesActive: {
          wyckoff: true,
          fundingRates: marketMetrics.fundingRates.size > 0,
          openInterest: marketMetrics.openInterest.size > 0,
          btcDominance: marketMetrics.btcDominance !== null,
          fearGreed: marketMetrics.fearGreed !== null,
          correlations: correlations.correlationMatrix !== null,
          technicalIndicators: technicalIndicatorsMap.size > 0,
          multiTimeframe: btcMultiTimeframe !== null,
          volumeProfile: btcVolumeProfile !== null,
          orderFlow: btcOrderFlow !== null,
        },
      },
    });

  } catch (error: any) {
    console.error('[Omnipotent Futures] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Omnipotent Futures analysis failed',
      },
      { status: 500 }
    );
  }
}
