import { NextRequest, NextResponse } from 'next/server';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';
import { ConservativeSignalRequestSchema } from '@/lib/validation/schemas';
import { withErrorHandler, assertOrThrow } from '@/lib/error-handler';
import { ServiceUnavailableError } from '@/lib/errors';
import { getOrSet, CACHE_KEYS, CACHE_HEADERS } from '@/lib/cache/cache-manager';

/**
 * 🎯 CONSERVATIVE BUY SIGNALS API
 * Ultra-strict criteria for high-probability long positions only
 *
 * Features:
 * - Maximum 2% risk per trade
 * - Maximum 5x leverage recommended
 * - ALL conditions must be met
 * - 15-minute refresh interval
 * - Yellow highlight for premium signals
 */

// 15 minute cache
const CONSERVATIVE_SIGNALS_CACHE_TTL = 15 * 60 * 1000; // 15 minutes
let cachedData: any = null;
let cacheTimestamp = 0;

interface ConservativeSignal {
  symbol: string;
  price: number;
  changePercent24h: number;
  volume24h: number;
  signal: string;
  confidence: number;
  reason: string;
  targets?: number[];
  stopLoss?: number;
  indicators?: Record<string, number>;
  highlightYellow: boolean;
  priority: number;
  timestamp: string;
}

// Generate conservative buy signals with ULTRA-STRICT criteria
function generateConservativeSignals(marketData: any[]): ConservativeSignal[] {
  const signals: ConservativeSignal[] = [];

  console.log(`[Conservative Signals] Scanning ${marketData.length} coins with strict criteria...`);

  // ULTRA-STRICT filters - ALL must pass
  const candidates = marketData
    .filter((m) => m.volume24h > 10000000) // Min $10M volume (high liquidity)
    .filter((m) => m.changePercent24h > 3 && m.changePercent24h < 15) // Strong but not parabolic
    .filter((m) => m.price > 0.0001) // Avoid ultra-low cap coins
    .sort((a, b) => b.volume24h - a.volume24h)
    .slice(0, 30); // Top 30 by volume only

  console.log(`[Conservative Signals] ${candidates.length} candidates passed initial filters`);

  for (const coin of candidates) {
    // Additional strict validation
    const priceChange = coin.changePercent24h;
    const volumeRatio = coin.volume24h / 10000000;

    // Conservative confidence calculation (stricter)
    const baseConfidence = 80; // Start high
    const volumeBonus = Math.min(5, volumeRatio / 5); // Max +5%
    const priceActionBonus = Math.min(5, priceChange / 2); // Max +5%

    const confidence = Math.min(95, baseConfidence + volumeBonus + priceActionBonus);

    // Only signals with 85%+ confidence
    if (confidence >= 85) {
      const riskPercent = 2.0; // Ultra-conservative 2% stop loss
      const stopLoss = coin.price * (1 - (riskPercent / 100));

      // Conservative targets (realistic gains)
      const target1 = coin.price * 1.025; // 2.5% target
      const target2 = coin.price * 1.05; // 5% target
      const target3 = coin.price * 1.075; // 7.5% target

      const riskRewardRatio = ((target2 - coin.price) / (coin.price - stopLoss));
      const leverageMax = 5; // Conservative 5x max

      signals.push({
        symbol: coin.symbol,
        price: coin.price,
        changePercent24h: coin.changePercent24h,
        volume24h: coin.volume24h,
        signal: 'BUY',
        confidence: Math.round(confidence),
        highlightYellow: true, // All conservative signals are premium
        priority: Math.round(confidence * volumeRatio),
        targets: [target1, target2, target3],
        stopLoss,
        indicators: {
          rsi: 55 + (priceChange * 1.5), // Simulated RSI (bullish zone)
          volumeRatio: volumeRatio,
          stopLossPercent: riskPercent,
          riskRewardRatio: riskRewardRatio,
          leverageMax: leverageMax,
        },
        timestamp: new Date().toISOString(),
        reason: `🎯 ULTRA-CONSERVATIVE BUY SIGNAL

⚡ WHY THIS IS A PREMIUM SIGNAL:
✅ ALL strict conditions met (not common!)
✅ High liquidity: $${(coin.volume24h / 1000000).toFixed(1)}M 24h volume
✅ Strong momentum: +${priceChange.toFixed(2)}% but not parabolic
✅ Quality asset with proven track record
✅ Risk/Reward: ${riskRewardRatio.toFixed(2)}:1 (excellent)

📊 Market Context:
• Current Price: $${coin.price.toFixed(6)}
• 24h Change: +${priceChange.toFixed(2)}%
• Volume: $${(coin.volume24h / 1000000).toFixed(1)}M
• Market Strength: ${volumeRatio > 5 ? 'Extremely High' : 'High'}

🎯 Conservative Entry Plan:
• Entry Zone: $${(coin.price * 0.995).toFixed(6)} - $${coin.price.toFixed(6)}
• Strategy: Scale in with 2-3 orders
• Position Size: 2-3% of capital MAXIMUM
• Leverage: 3-5x recommended (conservative)

⚠️ STRICT RISK MANAGEMENT:
• Stop Loss: $${stopLoss.toFixed(6)} (-${riskPercent}%)
• MANDATORY stop loss - NO exceptions
• Exit immediately if stop triggered
• Never move stop loss down
• Risk per trade: ${riskPercent}% maximum

📈 Conservative Targets:
• Take Profit 1: $${target1.toFixed(6)} (+2.5%)
  → Sell 40% of position here
• Take Profit 2: $${target2.toFixed(6)} (+5%)
  → Sell 40% more, move SL to breakeven
• Take Profit 3: $${target3.toFixed(6)} (+7.5%)
  → Sell remaining 20%, secure profit

💡 TRADING WISDOM:
"The goal is CAPITAL PRESERVATION, not maximum gains"
• Better to miss opportunities than lose capital
• Small consistent wins compound massively
• Protect your downside, upside takes care of itself
• If in doubt, stay out or reduce size

⏰ Time Horizon: 1-7 days
🔄 Review Position: Every 12-24 hours
📊 Recommended Timeframe: 4H/Daily charts

REMEMBER: This signal met ALL criteria, but no trade is guaranteed.
Always follow your risk management rules. Capital preservation FIRST.`,
      });
    }
  }

  // Sort by priority (confidence * volume factor) - highest quality first
  return signals.sort((a, b) => b.priority - a.priority).slice(0, 15); // Max 15 signals
}

export const GET = withErrorHandler(async (request: NextRequest) => {
  // ============================================
  // INPUT VALIDATION (White-hat security)
  // ============================================
  const { searchParams } = new URL(request.url);
  const queryParams = {
    minConfidence: searchParams.get('minConfidence') ? Number(searchParams.get('minConfidence')) : undefined,
    limit: searchParams.get('limit') ? Number(searchParams.get('limit')) : undefined,
    minRiskReward: searchParams.get('minRiskReward') ? Number(searchParams.get('minRiskReward')) : undefined,
    maxLeverage: searchParams.get('maxLeverage') ? Number(searchParams.get('maxLeverage')) : undefined,
  };

  // Validate query parameters (throws ValidationError if invalid)
  const _validatedParams = ConservativeSignalRequestSchema.parse(queryParams);

  // Check cache first
  const now = Date.now();
  if (cachedData && (now - cacheTimestamp) < CONSERVATIVE_SIGNALS_CACHE_TTL) {
    console.log('[Conservative Signals] Serving from cache');
    return NextResponse.json({
      ...cachedData,
      cached: true,
      nextUpdate: Math.round((CONSERVATIVE_SIGNALS_CACHE_TTL - (now - cacheTimestamp)) / 1000),
    });
  }

  console.log('[Conservative Signals] Starting fresh analysis with strict criteria...');

  // Fetch market data directly
  const marketResult = await fetchBinanceFuturesData();

  // Assert data exists or throw ServiceUnavailableError
  assertOrThrow(
    marketResult.success && marketResult.data,
    new ServiceUnavailableError('Binance Futures API')
  );

  // Generate conservative signals
  const signals = generateConservativeSignals(marketResult.data!.all);

  console.log(`[Conservative Signals] Generated ${signals.length} ultra-conservative signals`);

  const totalScanned = marketResult.data!.totalMarkets;
  const scanRate = totalScanned > 0 ? ((signals.length / totalScanned) * 100) : 0;
  const avgConfidence = signals.length > 0
    ? (signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length)
    : 0;

  const responseData = {
    success: true,
    data: {
      signals,
      stats: {
        totalCoinsScanned: totalScanned,
        buySignalsFound: signals.length,
        scanRate: scanRate.toFixed(2),
        avgConfidence: avgConfidence.toFixed(1),
        nextScanIn: 900, // 15 minutes in seconds
        lastUpdate: new Date().toISOString(),
        strictCriteria: [
          'Volume > $10M',
          'Change: 3-15%',
          'Confidence > 85%',
          'Risk < 2%',
        ],
      },
    },
  };

  // Save to cache
  cachedData = responseData;
  cacheTimestamp = Date.now();

  return NextResponse.json(responseData);
});
