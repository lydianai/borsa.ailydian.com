/**
 * LyTrade AI ASSISTANT API
 * Unified AI endpoint combining:
 * - Provider-agnostic AI (any OpenAI-compatible API)
 * - All Trading Strategies
 * - Real-time Market Data
 *
 * Configure via environment variables:
 * - AI_API_KEY or GROQ_API_KEY: Your API key
 * - AI_API_URL: API endpoint (default: Groq)
 * - AI_CHAT_MODEL: Model name (default: llama-3.3-70b-versatile)
 */

import { NextRequest, NextResponse } from 'next/server';

// AI Configuration
const AI_API_URL = process.env.AI_API_URL || 'https://api.groq.com/openai/v1/chat/completions';
const AI_CHAT_MODEL = process.env.AI_CHAT_MODEL || 'llama-3.3-70b-versatile';
const AI_API_KEY = process.env.AI_API_KEY || process.env.GROQ_API_KEY || '';

export const dynamic = 'force-dynamic';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

// Helper function to fetch with timeout
// CRITICAL FIX: NO CACHE for real-time price data
async function fetchWithTimeout(url: string, timeoutMs: number = 5000): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store', // ✅ ALWAYS fetch fresh real-time data (no cache!)
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
      },
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(`Fetch timeout after ${timeoutMs}ms for ${url}`);
    }
    throw error;
  }
}

// Get current market overview with ALL coins
async function getMarketOverview(baseUrl: string) {
  try {
    const response = await fetchWithTimeout(`${baseUrl}/api/binance/futures`, 5000);
    const result = await response.json();

    if (result.success) {
      const top10 = result.data.topVolume.slice(0, 10);
      const topGainers = result.data.topGainers.slice(0, 5);
      const allCoins = result.data.all || [];

      return {
        totalMarkets: result.data.totalMarkets,
        allCoins, // 600+ coin datasını döndür
        top10Volume: top10.map((c: any) => ({
          symbol: c.symbol,
          price: c.price,
          change: c.changePercent24h,
          volume: c.volume24h,
        })),
        topGainers: topGainers.map((c: any) => ({
          symbol: c.symbol,
          price: c.price,
          change: c.changePercent24h,
        })),
      };
    }
  } catch (error) {
    console.error('[AI Assistant] Market data error:', error);
  }
  return null;
}

// Get ALL strategies for a symbol (11 strategy APIs combined - UNIFIED INTELLIGENCE)
// OPTIMIZED: All API calls run in PARALLEL for maximum speed
async function getAllStrategies(symbol: string, baseUrl: string) {
  try {
    const fullSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
    console.log(`[AI Assistant - Unified Intelligence] Fetching ALL strategies for ${fullSymbol} in PARALLEL...`);

    const strategies: any[] = [];
    let totalBuy = 0;
    let totalSell = 0;
    let totalWait = 0;

    // PARALLEL EXECUTION: Fetch all strategies at once for speed
    const [
      signalsData,
      aiSignalsData,
      conservativeData,
      quantumData,
      correlationData,
      breakoutData,
      unifiedData,
      wyckoffData,
      talibData,
      btcEthData,
      tradMarketsData,
    ] = await Promise.allSettled([
      fetchWithTimeout(`${baseUrl}/api/signals?limit=600`, 4000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/ai-signals`, 6000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/conservative-signals`, 4000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/quantum-signals`, 4000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/market-correlation?limit=600`, 5000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/breakout-retest`, 4000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/unified-signals?minBuyPercentage=50&limit=600`, 6000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/omnipotent-futures?limit=600`, 5000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/talib-analysis/${fullSymbol}`, 5000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/btc-eth-analysis`, 4000).then(r => r.json()),
      fetchWithTimeout(`${baseUrl}/api/traditional-markets`, 4000).then(r => r.json()),
    ]);

    // 1. Manual Signals
    if (signalsData.status === 'fulfilled' && signalsData.value?.success) {
      const signal = signalsData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        strategies.push({ name: 'Ta-Lib AI Signals', signal: signal.type, confidence: signal.confidence, description: signal.strategy, weight: 1 });
        if (signal.type === 'BUY') totalBuy++;
        else if (signal.type === 'SELL') totalSell++;
        else totalWait++;
      }
    }

    // 2. AI Signals
    if (aiSignalsData.status === 'fulfilled' && aiSignalsData.value?.success) {
      const signal = aiSignalsData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        strategies.push({ name: 'AI Enhanced', signal: signal.type, confidence: signal.confidence, description: signal.reasoning, weight: 1 });
        if (signal.type === 'BUY') totalBuy++;
        else if (signal.type === 'SELL') totalSell++;
        else totalWait++;
      }
    }

    // 3. Conservative Signals
    if (conservativeData.status === 'fulfilled' && conservativeData.value?.success) {
      const signal = conservativeData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        strategies.push({ name: 'Conservative Buy', signal: signal.signal, confidence: signal.confidence, description: signal.reason, weight: 1 });
        if (signal.signal === 'BUY') totalBuy++;
        else if (signal.signal === 'SELL') totalSell++;
        else totalWait++;
      }
    }

    // 4. Quantum Signals (2x weight)
    if (quantumData.status === 'fulfilled' && quantumData.value?.success) {
      const signal = quantumData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        strategies.push({ name: 'Quantum Portfolio', signal: signal.type, confidence: signal.confidence, description: signal.reasoning, weight: 2 });
        if (signal.type === 'BUY') totalBuy += 2;
        else if (signal.type === 'SELL') totalSell += 2;
        else totalWait += 2;
      }
    }

    // 5. Market Correlation
    if (correlationData.status === 'fulfilled' && correlationData.value?.success) {
      const correlation = correlationData.value.data.correlations.find((c: any) => c.symbol === fullSymbol);
      if (correlation) {
        strategies.push({
          name: 'Market Correlation',
          signal: correlation.signal,
          confidence: correlation.confidence,
          description: `${correlation.marketPhase}, BTC Corr: ${(correlation.btcCorrelation * 100).toFixed(0)}%, Omnipotent: ${correlation.omnipotentScore}`,
          weight: 1
        });
        if (correlation.signal === 'BUY') totalBuy++;
        else if (correlation.signal === 'SELL') totalSell++;
        else totalWait++;
      }
    }

    // 6. Breakout-Retest
    if (breakoutData.status === 'fulfilled' && breakoutData.value?.success) {
      const signal = breakoutData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        strategies.push({ name: 'Breakout-Retest', signal: signal.signal, confidence: signal.confidence, description: signal.reason, weight: 1 });
        if (signal.signal === 'BUY') totalBuy++;
        else if (signal.signal === 'SELL') totalSell++;
        else totalWait++;
      }
    }

    // 7. Unified 18+ Strategies (3x weight - most comprehensive)
    if (unifiedData.status === 'fulfilled' && unifiedData.value?.success) {
      const signal = unifiedData.value.data.signals.find((s: any) => s.symbol === fullSymbol);
      if (signal) {
        const unifiedSignal = signal.buyPercentage >= 60 ? 'BUY' : signal.buyPercentage <= 40 ? 'SELL' : 'WAIT';
        strategies.push({
          name: 'Unified 18+ Strategies',
          signal: unifiedSignal,
          confidence: signal.overallConfidence,
          description: `${signal.buyPercentage}% BUY consensus, ${signal.activeStrategies} strategies`,
          weight: 3
        });
        if (unifiedSignal === 'BUY') totalBuy += 3;
        else if (unifiedSignal === 'SELL') totalSell += 3;
        else totalWait += 3;
      }
    }

    // 8. Wyckoff Method (2x weight)
    if (wyckoffData.status === 'fulfilled' && wyckoffData.value?.success) {
      const wyckoff = wyckoffData.value.data.futures.find((f: any) => f.symbol === fullSymbol);
      if (wyckoff) {
        strategies.push({
          name: 'Wyckoff Method',
          signal: wyckoff.signal,
          confidence: wyckoff.confidence,
          description: `${wyckoff.wyckoffPhase}${wyckoff.subPhase ? ' - ' + wyckoff.subPhase : ''}, Smart Money: ${wyckoff.smartMoneyActivity}`,
          weight: 2
        });
        if (wyckoff.signal === 'BUY') totalBuy += 2;
        else if (wyckoff.signal === 'SELL') totalSell += 2;
        else totalWait += 2;
      }
    }

    // 9. Ta-Lib 158 Indicators (2x weight)
    if (talibData.status === 'fulfilled' && talibData.value?.success && talibData.value.data) {
      const talib = talibData.value.data;
      strategies.push({
        name: 'Ta-Lib 158 Indicators',
        signal: talib.signal,
        confidence: talib.confidence,
        description: `${talib.pattern}, Strength: ${talib.strength}/10`,
        weight: 2
      });
      if (talib.signal === 'BUY') totalBuy += 2;
      else if (talib.signal === 'SELL') totalSell += 2;
      else totalWait += 2;
    }

    // 10. BTC-ETH Market Leaders
    if (btcEthData.status === 'fulfilled' && btcEthData.value?.success) {
      const sentiment = btcEthData.value.data.analysis.overallSentiment;
      const sentimentSignal = sentiment === 'BULLISH' ? 'BUY' : sentiment === 'BEARISH' ? 'SELL' : 'WAIT';
      strategies.push({
        name: 'BTC-ETH Market Leaders',
        signal: sentimentSignal,
        confidence: btcEthData.value.data.analysis.confidence,
        description: `Market: ${sentiment}, BTC: ${btcEthData.value.data.btc.trend}, ETH: ${btcEthData.value.data.eth.trend}`,
        weight: 1
      });
      if (sentimentSignal === 'BUY') totalBuy++;
      else if (sentimentSignal === 'SELL') totalSell++;
      else totalWait++;
    }

    // 11. Traditional Markets
    if (tradMarketsData.status === 'fulfilled' && tradMarketsData.value?.success) {
      const riskOn = tradMarketsData.value.data.analysis.marketSentiment === 'RISK_ON';
      const tradMarketSignal = riskOn ? 'BUY' : 'SELL';
      strategies.push({
        name: 'Traditional Markets',
        signal: tradMarketSignal,
        confidence: tradMarketsData.value.data.analysis.cryptoImpact === 'POSITIVE' ? 75 : tradMarketsData.value.data.analysis.cryptoImpact === 'NEGATIVE' ? 25 : 50,
        description: `${tradMarketsData.value.data.analysis.marketSentiment}, Crypto Impact: ${tradMarketsData.value.data.analysis.cryptoImpact}`,
        weight: 1
      });
      if (tradMarketSignal === 'BUY') totalBuy++;
      else if (tradMarketSignal === 'SELL') totalSell++;
      else totalWait++;
    }

    console.log(`[AI Assistant] Collected ${strategies.length} strategy analyses for ${fullSymbol}`);
    console.log(`[AI Assistant] Vote Count - BUY: ${totalBuy}, SELL: ${totalSell}, WAIT: ${totalWait}`);

    return {
      strategies,
      totalBuy,
      totalSell,
      totalWait,
      totalStrategies: strategies.length,
    };
  } catch (error) {
    console.error('[AI Assistant] Get all strategies error:', error);
    return { strategies: [], totalBuy: 0, totalSell: 0, totalWait: 0, totalStrategies: 0 };
  }
}

// OPTION A: Multi-Timeframe Analysis (5 timeframes - 5m, 15m, 1h, 4h, 1d)
async function getMultiTimeframeAnalysis(symbol: string, baseUrl: string) {
  try {
    const fullSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
    const timeframes = [
      { interval: '5m', name: '5 dakika', weight: 1 },
      { interval: '15m', name: '15 dakika', weight: 2 },
      { interval: '1h', name: '1 saat', weight: 3 },
      { interval: '4h', name: '4 saat', weight: 4 },
      { interval: '1d', name: '1 gün', weight: 5 },
    ];

    console.log(`[Multi-Timeframe] Analyzing ${fullSymbol} across 5 timeframes...`);

    // Fetch klines for all timeframes in PARALLEL
    const klinePromises = timeframes.map(tf =>
      fetchWithTimeout(`${baseUrl}/api/binance/klines/${fullSymbol}?interval=${tf.interval}&limit=50`, 4000)
        .then(r => r.json())
        .catch(err => {
          console.error(`[Multi-Timeframe] Error fetching ${tf.interval}:`, err.message);
          return { success: false };
        })
    );

    const klineResults = await Promise.allSettled(klinePromises);

    // Analyze each timeframe for trend
    const trends = timeframes.map((tf, i) => {
      if (klineResults[i].status === 'rejected' || !klineResults[i].value?.success) {
        return {
          timeframe: tf.interval,
          name: tf.name,
          trend: 'UNKNOWN',
          strength: 0,
          weight: tf.weight,
          emoji: '⚪',
        };
      }

      const klines = klineResults[i].value.data;
      if (!klines || klines.length < 20) {
        return {
          timeframe: tf.interval,
          name: tf.name,
          trend: 'UNKNOWN',
          strength: 0,
          weight: tf.weight,
          emoji: '⚪',
        };
      }

      // Simple trend calculation: Compare recent prices with older prices
      const recentPrices = klines.slice(-10).map((k: any) => parseFloat(k.close));
      const olderPrices = klines.slice(0, 10).map((k: any) => parseFloat(k.close));

      const recentAvg = recentPrices.reduce((a: number, b: number) => a + b, 0) / recentPrices.length;
      const olderAvg = olderPrices.reduce((a: number, b: number) => a + b, 0) / olderPrices.length;

      const priceChange = ((recentAvg - olderAvg) / olderAvg) * 100;

      // Calculate strength (0-100)
      const strength = Math.min(Math.abs(priceChange) * 10, 100);

      // Determine trend
      let trend = 'NEUTRAL';
      let emoji = '🟡';

      if (priceChange > 1.5) {
        trend = 'BULLISH';
        emoji = strength > 70 ? '🟢' : '🟢';
      } else if (priceChange < -1.5) {
        trend = 'BEARISH';
        emoji = strength > 70 ? '🔴' : '🔴';
      }

      return {
        timeframe: tf.interval,
        name: tf.name,
        trend,
        strength: Math.round(strength),
        priceChange: priceChange.toFixed(2),
        weight: tf.weight,
        emoji,
      };
    });

    // Calculate weighted alignment score
    const bullishTrends = trends.filter(t => t.trend === 'BULLISH');
    const bearishTrends = trends.filter(t => t.trend === 'BEARISH');
    const neutralTrends = trends.filter(t => t.trend === 'NEUTRAL');

    // Weighted votes (max 15: 1+2+3+4+5)
    const bullishWeight = bullishTrends.reduce((sum, t) => sum + t.weight, 0);
    const bearishWeight = bearishTrends.reduce((sum, t) => sum + t.weight, 0);
    const neutralWeight = neutralTrends.reduce((sum, t) => sum + t.weight, 0);
    const totalWeight = 15; // 1+2+3+4+5

    // Determine overall trend based on weighted majority
    let overallTrend = 'NEUTRAL';
    let alignmentScore = 50;

    if (bullishWeight > bearishWeight && bullishWeight > neutralWeight) {
      overallTrend = 'BULLISH';
      alignmentScore = Math.round((bullishWeight / totalWeight) * 100);
    } else if (bearishWeight > bullishWeight && bearishWeight > neutralWeight) {
      overallTrend = 'BEARISH';
      alignmentScore = Math.round((bearishWeight / totalWeight) * 100);
    } else {
      alignmentScore = Math.round((neutralWeight / totalWeight) * 100);
    }

    console.log(`[Multi-Timeframe] ${fullSymbol} Overall: ${overallTrend} (${alignmentScore}% alignment)`);
    console.log(`[Multi-Timeframe] Weighted votes - BULLISH: ${bullishWeight}, BEARISH: ${bearishWeight}, NEUTRAL: ${neutralWeight}`);

    return {
      trends,
      alignmentScore,
      overallTrend,
      bullishWeight,
      bearishWeight,
      neutralWeight,
      totalWeight,
    };
  } catch (error) {
    console.error('[Multi-Timeframe] Error:', error);
    return null;
  }
}

// Get unified analysis (ALL Strategies + Coin Data)
async function getUnifiedAnalysis(symbol: string, baseUrl: string, marketData: any) {
  try {
    const fullSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;

    // Find coin in market data
    const coinData = marketData.allCoins.find((c: any) =>
      c.symbol === symbol || c.symbol === fullSymbol || c.symbol === symbol + 'USDT'
    );

    if (!coinData) {
      console.log(`[AI Assistant] Coin ${fullSymbol} not found in market data`);
      return null;
    }

    // Get all strategies
    const strategyData = await getAllStrategies(symbol, baseUrl);

    // OPTION A: Get multi-timeframe analysis (5 timeframes in parallel)
    const multiTimeframe = await getMultiTimeframeAnalysis(symbol, baseUrl);

    // Calculate final decision
    const total = strategyData.totalBuy + strategyData.totalSell + strategyData.totalWait;
    let finalDecision = 'WAIT';
    let confidence = 50;

    if (total > 0) {
      if (strategyData.totalBuy > strategyData.totalSell && strategyData.totalBuy > strategyData.totalWait) {
        finalDecision = 'BUY';
        confidence = Math.round((strategyData.totalBuy / total) * 100);
      } else if (strategyData.totalSell > strategyData.totalBuy && strategyData.totalSell > strategyData.totalWait) {
        finalDecision = 'SELL';
        confidence = Math.round((strategyData.totalSell / total) * 100);
      } else {
        confidence = Math.round((strategyData.totalWait / total) * 100);
      }
    }

    // Calculate overall score
    const score = strategyData.strategies.length > 0
      ? Math.round(strategyData.strategies.reduce((sum, s) => sum + s.confidence, 0) / strategyData.strategies.length)
      : 50;

    return {
      symbol: fullSymbol,
      price: coinData.price,
      change: coinData.changePercent24h,
      volume: coinData.volume24h,
      recommendation: finalDecision,
      confidence,
      score,
      buySignals: strategyData.totalBuy,
      sellSignals: strategyData.totalSell,
      waitSignals: strategyData.totalWait,
      strategies: strategyData.strategies,
      totalStrategies: strategyData.totalStrategies,
      multiTimeframe, // OPTION A: Multi-Timeframe Analysis
    };
  } catch (error) {
    console.error('[AI Assistant] Unified analysis error:', error);
  }
  return null;
}

// Extract symbol from user message (supports Turkish queries)
// DYNAMIC: Accepts any coin from market data
function extractSymbol(message: string, availableCoins: string[]): string | null {
  // Remove Turkish suffixes and clean the message
  const cleanMessage = message
    .toUpperCase()
    .replace(/ALINIR\s*MI?/gi, '')
    .replace(/ALSAM\s*MI?/gi, '')
    .replace(/ALIRIM\s*MI?/gi, '')
    .replace(/ALMALIYIM\s*MI?/gi, '')
    .replace(/SATILIR\s*MI?/gi, '')
    .replace(/SATSAM\s*MI?/gi, '')
    .replace(/BEKLEMELI\s*MIYIM?/gi, '')
    .replace(/HAKKINDA/gi, '')
    .replace(/NE\s*YAPMALIYIM/gi, '')
    .trim();

  // Try to match any coin from available coins (600+)
  // availableCoins now contains full symbols like "BTCUSDT"
  // CRITICAL FIX: Use word boundaries to avoid false matches (e.g., "dolar" matching "AR")
  for (const coin of availableCoins) {
    // Match both "BTC" and "BTCUSDT" formats
    const baseSymbol = coin.replace('USDT', '').toUpperCase();
    const fullSymbol = coin.toUpperCase();

    // Use regex with word boundaries to match ONLY whole words
    const baseRegex = new RegExp(`\\b${baseSymbol}\\b`, 'i');
    const fullRegex = new RegExp(`\\b${fullSymbol}\\b`, 'i');

    if (baseRegex.test(cleanMessage) || fullRegex.test(cleanMessage)) {
      return baseSymbol; // Return base symbol (e.g., "BTC")
    }
  }

  return null;
}

// Format detailed analysis with reasoning (UPDATED FOR 11 UNIFIED STRATEGIES + MULTI-TIMEFRAME - WEIGHTED CONSENSUS)
function formatDetailedAnalysis(analysis: any): string {
  const { symbol, price, change, volume, recommendation, confidence, score, buySignals, sellSignals, waitSignals, strategies, totalStrategies, multiTimeframe } = analysis;

  // Calculate total weighted votes
  const totalWeightedVotes = buySignals + sellSignals + (waitSignals || 0);

  // Decision section with unified intelligence branding
  let response = `╔═══════════════════════════════════╗\n`;
  response += `║  🧠 UNIFIED INTELLIGENCE DECISION ║\n`;
  response += `╚═══════════════════════════════════╝\n\n`;
  response += `🎯 KARAR: ${recommendation === 'BUY' ? '✅ AL' : recommendation === 'SELL' ? '❌ SAT' : '⏸️ BEKLE'}\n`;
  response += `🔥 Güvenilirlik: %${confidence}\n`;
  response += `⚖️ Ağırlıklı Oy: ${recommendation === 'BUY' ? buySignals : recommendation === 'SELL' ? sellSignals : waitSignals}/${totalWeightedVotes} (15 maks)\n`;
  response += `═══════════════════════════════════\n\n`;

  // Current stats
  response += `📊 ${symbol} - GÜNCEL DURUM:\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `💰 Fiyat: $${price.toLocaleString()}\n`;
  response += `📈 24s Değişim: ${change >= 0 ? '+' : ''}${change.toFixed(2)}%\n`;
  response += `📊 24s Hacim: $${(volume / 1_000_000).toFixed(2)}M\n`;
  response += `⭐ Genel Skor: ${score}/100\n`;
  response += `🔬 Analiz Edilen Strateji: ${totalStrategies}/11 unified strategy\n\n`;

  // OPTION A: Multi-Timeframe Analysis Section
  if (multiTimeframe) {
    response += `⏰ MULTI-TIMEFRAME TREND ANALYSIS:\n`;
    response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
    response += `📊 5 Zaman Dilimi Eş Zamanlı Analiz\n`;
    response += `🎯 Genel Trend: ${multiTimeframe.overallTrend} (%${multiTimeframe.alignmentScore} hizalanma)\n`;
    response += `⚖️ Ağırlıklı Oy: ${multiTimeframe.overallTrend === 'BULLISH' ? multiTimeframe.bullishWeight : multiTimeframe.overallTrend === 'BEARISH' ? multiTimeframe.bearishWeight : multiTimeframe.neutralWeight}/${multiTimeframe.totalWeight}\n\n`;

    multiTimeframe.trends.forEach((t: any) => {
      response += `${t.emoji} ${t.name.padEnd(12)} | ${t.trend.padEnd(8)} | Güç: ${t.strength}% | Değişim: ${t.priceChange > 0 ? '+' : ''}${t.priceChange}%\n`;
    });

    response += `\n💡 Trend Hizalanması:\n`;
    response += `   🟢 Yükseliş: ${multiTimeframe.bullishWeight} ağırlıklı oy\n`;
    response += `   🔴 Düşüş: ${multiTimeframe.bearishWeight} ağırlıklı oy\n`;
    response += `   🟡 Nötr: ${multiTimeframe.neutralWeight} ağırlıklı oy\n\n`;
  }

  // Weighted signal summary
  response += `🔔 AĞIRLIKLI SİNYAL ÖZETİ:\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `✅ AL Sinyalleri: ${buySignals} ağırlıklı oy\n`;
  response += `❌ SAT Sinyalleri: ${sellSignals} ağırlıklı oy\n`;
  response += `⏸️ BEKLE Sinyalleri: ${waitSignals || 0} ağırlıklı oy\n`;
  response += `📊 Toplam: ${totalWeightedVotes}/15 weighted votes\n`;
  response += `💪 Konsensüs Gücü: ${buySignals > sellSignals ? '🟢 GÜÇLÜ AL' : sellSignals > buySignals ? '🔴 GÜÇLÜ SAT' : '🟡 NÖTR'}\n\n`;

  // Reasoning section with enhanced explanations
  response += `💡 NEDEN ${recommendation === 'BUY' ? 'ALMALIYIM' : recommendation === 'SELL' ? 'SATMALIYIM' : 'BEKLEMELİYİM'}?\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;

  if (recommendation === 'BUY') {
    response += `Bu karara 11 unified strategy ile varmamızın sebepleri:\n\n`;
    response += `1️⃣ Ağırlıklı Çoğunluk AL Sinyali: ${buySignals}/${totalWeightedVotes} weighted vote AL diyor\n`;
    response += `2️⃣ Genel Skor Yüksek: ${score}/100 puan güçlü bir alım fırsatı gösteriyor\n`;
    response += `3️⃣ Multi-Strategy Consensus: ${strategies.filter((s: any) => s.signal === 'BUY').length} farklı strateji AL onayı verdi\n`;
    if (change > 0) {
      response += `4️⃣ Pozitif Momentum: %${change.toFixed(2)} yükseliş trendi başlamış olabilir\n`;
    }
  } else if (recommendation === 'SELL') {
    response += `Bu karara 11 unified strategy ile varmamızın sebepleri:\n\n`;
    response += `1️⃣ Ağırlıklı Çoğunluk SAT Sinyali: ${sellSignals}/${totalWeightedVotes} weighted vote SAT diyor\n`;
    response += `2️⃣ Genel Skor Düşük: ${score}/100 puan risk gösteriyor\n`;
    response += `3️⃣ Multi-Strategy Warning: ${strategies.filter((s: any) => s.signal === 'SELL').length} farklı strateji SAT uyarısı verdi\n`;
    if (change < 0) {
      response += `4️⃣ Negatif Momentum: %${change.toFixed(2)} düşüş trendi devam edebilir\n`;
    }
  } else {
    response += `Bu karara 11 unified strategy ile varmamızın sebepleri:\n\n`;
    response += `1️⃣ Kararsız Weighted Votes: AL (${buySignals}) ve SAT (${sellSignals}) dengede\n`;
    response += `2️⃣ Orta Skor: ${score}/100 puan net bir yön göstermiyor\n`;
    response += `3️⃣ Mixed Signals: Stratejiler arasında konsensüs yok\n`;
    response += `4️⃣ Daha Net Sinyal Bekleyin: Piyasa yönünü netleştirene kadar beklemek güvenli\n`;
  }

  response += `\n\n📋 11 UNIFIED STRATEGY - DETAYLI ANALİZ:\n`;
  response += `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n`;

  // Categorize strategies by signal
  const buyStrats = strategies.filter((s: any) => s.signal === 'BUY');
  const sellStrats = strategies.filter((s: any) => s.signal === 'SELL');
  const waitStrats = strategies.filter((s: any) => s.signal === 'WAIT');

  if (buyStrats.length > 0) {
    response += `✅ AL SİNYALİ VEREN STRATEJİLER (${buyStrats.length}):\n\n`;
    buyStrats.forEach((s: any, i: number) => {
      const weight = s.weight || 1;
      const weightEmoji = weight === 3 ? '🔥🔥🔥' : weight === 2 ? '🔥🔥' : '🔥';
      response += `${i + 1}. ${s.name} ${weightEmoji} (${weight}x ağırlık)\n`;
      response += `   • Güven: %${s.confidence}\n`;
      if (s.description) response += `   • ${s.description}\n`;
      response += `\n`;
    });
  }

  if (sellStrats.length > 0) {
    response += `❌ SAT SİNYALİ VEREN STRATEJİLER (${sellStrats.length}):\n\n`;
    sellStrats.forEach((s: any, i: number) => {
      const weight = s.weight || 1;
      const weightEmoji = weight === 3 ? '🔥🔥🔥' : weight === 2 ? '🔥🔥' : '🔥';
      response += `${i + 1}. ${s.name} ${weightEmoji} (${weight}x ağırlık)\n`;
      response += `   • Güven: %${s.confidence}\n`;
      if (s.description) response += `   • ${s.description}\n`;
      response += `\n`;
    });
  }

  if (waitStrats.length > 0) {
    response += `⏸️ BEKLE SİNYALİ VEREN STRATEJİLER (${waitStrats.length}):\n\n`;
    waitStrats.forEach((s: any, i: number) => {
      const weight = s.weight || 1;
      const weightEmoji = weight === 3 ? '🔥🔥🔥' : weight === 2 ? '🔥🔥' : '🔥';
      response += `${i + 1}. ${s.name} ${weightEmoji} (${weight}x ağırlık)\n`;
      response += `   • Güven: %${s.confidence}\n`;
      if (s.description) response += `   • ${s.description}\n`;
      response += `\n`;
    });
  }

  // Footer with unified intelligence branding
  response += `\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n`;
  response += `⚠️ BU BİR YATIRIM TAVSİYESİ DEĞİLDİR\n`;
  response += `Kendi araştırmanızı yapın ve riskinizi yönetin.\n`;
  response += `🧠 LyTrade UNIFIED INTELLIGENCE - 11 Strategy Consensus\n`;
  response += `🔥 Powered by: Ta-Lib 158 + Wyckoff + Quantum + 18+ Unified Strategies\n`;

  return response;
}

export async function POST(request: NextRequest) {
  try {
    const { message, history } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json(
        { success: false, error: 'Invalid message' },
        { status: 400 }
      );
    }

    // Get base URL from request
    const url = new URL(request.url);
    const baseUrl = `${url.protocol}//${url.host}`;

    // ALWAYS fetch market overview to get 600+ coins
    const market = await getMarketOverview(baseUrl);
    if (!market) {
      return NextResponse.json(
        { success: false, error: 'Failed to fetch market data' },
        { status: 500 }
      );
    }

    // Extract available coin symbols for dynamic matching
    const availableCoins = market.allCoins.map((c: any) => c.symbol);

    // Check if AI API is configured
    if (!AI_API_KEY) {
      // Fallback response without AI enhancement - use detailed formatter
      const symbol = extractSymbol(message, availableCoins);

      if (symbol) {
        const analysis = await getUnifiedAnalysis(symbol, baseUrl, market);
        if (analysis) {
          return NextResponse.json({
            success: true,
            response: formatDetailedAnalysis(analysis),
          });
        }
      }

      if (market) {
        return NextResponse.json({
          success: true,
          response: `Piyasa özeti:\n\n` +
            `📈 Toplam ${market.totalMarkets} coin takip ediliyor\n\n` +
            `En yüksek hacimli 5 coin:\n` +
            market.top10Volume.slice(0, 5).map((c: any, i: number) =>
              `${i + 1}. ${c.symbol}: $${c.price.toLocaleString()} (${c.change >= 0 ? '+' : ''}${c.change.toFixed(2)}%)`
            ).join('\n') + '\n\n' +
            `En çok yükselenler:\n` +
            market.topGainers.map((c: any, i: number) =>
              `${i + 1}. ${c.symbol}: +${c.change.toFixed(2)}%`
            ).join('\n'),
        });
      }

      return NextResponse.json({
        success: true,
        response: 'Gelişmiş AI API yapılandırılmadı. Temel piyasa verilerine erişim için API anahtarını ekleyin.',
      });
    }

    // Build context for AI with 600+ coins awareness
    let contextData = '';

    // Extract symbol if mentioned
    const symbol = extractSymbol(message, availableCoins);
    console.log(`[AI Assistant] Extracted symbol: ${symbol}`);

    if (symbol) {
      const analysis = await getUnifiedAnalysis(symbol, baseUrl, market);
      console.log(`[AI Assistant] Analysis for ${symbol}:`, analysis ? `Price: $${analysis.price}, Change: ${analysis.change}%` : 'NULL');

      if (analysis) {
        contextData += `\n\n${symbol} GERÇEK ZAMANLI VERİLER (UNIFIED INTELLIGENCE):\n`;
        contextData += `Fiyat: $${analysis.price}\n`;
        contextData += `24h Değişim: ${analysis.change}%\n`;
        contextData += `24h Hacim: $${(analysis.volume / 1_000_000).toFixed(2)}M\n`;
        contextData += `\nUNIFIED KARAR:\n`;
        contextData += `Öneri: ${analysis.recommendation}\n`;
        contextData += `Güvenilirlik: %${analysis.confidence}\n`;
        contextData += `Genel Skor: ${analysis.score}/100\n`;
        contextData += `\nAĞIRLIKLI OY DAĞILIMI (Weighted Consensus):\n`;
        contextData += `✅ AL: ${analysis.buySignals} weighted votes\n`;
        contextData += `❌ SAT: ${analysis.sellSignals} weighted votes\n`;
        contextData += `⏸️ BEKLE: ${analysis.waitSignals || 0} weighted votes\n`;
        contextData += `📊 Toplam: ${analysis.buySignals + analysis.sellSignals + (analysis.waitSignals || 0)}/15 votes\n`;

        contextData += `\n11 UNIFIED STRATEGY SONUÇLARI (${analysis.totalStrategies} aktif strateji):\n`;
        analysis.strategies.forEach((s: any, i: number) => {
          const weight = s.weight || 1;
          const weightLabel = weight === 3 ? '[3x]' : weight === 2 ? '[2x]' : '[1x]';
          contextData += `${i + 1}. ${s.name} ${weightLabel}: ${s.signal} (Güven: ${s.confidence}%)`;
          if (s.description) contextData += ` - ${s.description}`;
          contextData += `\n`;
        });
      } else {
        contextData += `\n\n${symbol} hakkında detaylı bilgi bulunamadı. Bu coin Binance Futures'da olmayabilir veya çok düşük hacimli olabilir.\n`;
      }
    } else {
      // General market overview with ALL 600+ coins
      contextData += `\n\nGEÇERLİ PİYASA BİLGİLERİ:\n`;
      contextData += `Toplam Market: ${market.totalMarkets} coin (TÜM Binance Futures USDT çiftleri)\n`;
      contextData += `\nMevcut tüm coinler: ${availableCoins.slice(0, 50).join(', ')}... ve ${market.totalMarkets - 50} coin daha\n`;
      contextData += `\nEn yüksek hacimli coinler:\n`;
      market.top10Volume.slice(0, 5).forEach((c: any, i: number) => {
        contextData += `${i + 1}. ${c.symbol}: $${c.price} (${c.change >= 0 ? '+' : ''}${c.change}%)\n`;
      });
      contextData += `\nEn çok yükselenler:\n`;
      market.topGainers.forEach((c: any, i: number) => {
        contextData += `${i + 1}. ${c.symbol}: +${c.change}%\n`;
      });
    }

    // Build conversation history
    const conversationHistory: any[] = history
      ? history.map((msg: Message) => ({
          role: msg.role,
          content: msg.content,
        }))
      : [];

    // System prompt with 600+ coins and 11 unified strategies + MULTI-TIMEFRAME awareness
    const systemPrompt = `Sen LyTrade Trading Scanner'ın UNIFIED INTELLIGENCE AI asistanısın.

🚨 KRİTİK KURAL - MUTLAKA UYULMASI GEREKEN:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ ASLA, KESİNLİKLE kendi bilgin veya eğitim verinden fiyat bilgisi UYDURMA!
⚠️ SADECE ve SADECE aşağıdaki "GERÇEK ZAMANLI PİYASA VERİLERİ" bölümündeki fiyatları kullan!
⚠️ Eğer context'te fiyat yoksa, "Bu coin için güncel fiyat verisi bulunamadı" de!
⚠️ $28,000, $30,000 gibi eski BTC fiyatları ASLA KULLANMA - 2025'te BTC ~$109,000!

🎯 SİSTEM YETENEKLERİN (FULL POWER):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Market Coverage:
- ${market.totalMarkets} Binance Futures USDT perpetual contracts (BTC, ETH, TRB, LEVER, ve TÜM diğerleri)
- Gerçek zamanlı fiyat, hacim, değişim, liquidation verileri (context'te verilmiştir)

⏰ MULTI-TIMEFRAME ANALYSIS (OPTION A - Newly Activated):
  - 5 Zaman Dilimi Eş Zamanlı Analiz: 5m, 15m, 1h, 4h, 1d
  - Ağırlıklı Trend Hizalanması (1+2+3+4+5 = 15 weighted votes)
  - Trend Gücü ve Yön Analizi (BULLISH/BEARISH/NEUTRAL)
  - Zaman dilimi ağırlıklandırması: 1d (5x), 4h (4x), 1h (3x), 15m (2x), 5m (1x)
  - Paralel veri toplama ile hızlı analiz

🧠 11 UNIFIED STRATEGY ENGINES (Weighted Consensus):
  1. Ta-Lib AI Signals (1x) - 158 technical indicators with AI
  2. AI Enhanced Signals (1x) - Deep Learning momentum
  3. Conservative Buy Signal (1x) - Ultra-strict criteria
  4. Quantum Portfolio (2x) - Quantum optimization ⚡
  5. Market Correlation (1x) - BTC correlation & omnipotent score
  6. Breakout-Retest (1x) - Pattern recognition
  7. 🔥 Unified 18+ Strategies (3x) - MEGA consensus aggregator
  8. 🔥 Wyckoff Method (2x) - Smart money tracking (4 phases)
  9. 🔥 Ta-Lib 158 Indicators (2x) - Full technical analysis suite
  10. BTC-ETH Market Leaders (1x) - Dominant pair sentiment
  11. Traditional Markets (1x) - Global risk-on/risk-off

⚖️ Total Voting Weight: 15 weighted votes per coin (strategy consensus)
⏰ Total MTF Weight: 15 weighted votes per coin (timeframe alignment)
🎯 Decision Method: Weighted majority consensus across both dimensions
🧮 Confidence Calculation: Weighted average of all strategies + timeframe confirmation

GÖREV:
- Kullanıcıya herhangi bir coin hakkında ÇOK DETAYLI bilgi ver
- 11 stratejinin ÇOK BOYUTLU analizini birleştir
- AL/SAT/BEKLE kararlarının DERİN NEDENLERINI açıkla
- Her stratejinin ne dediğini ve neden önemli olduğunu anlat
- Teknik analiz, Wyckoff fazları, smart money, BTC korelasyonu, global risk gibi TÜM faktörleri değerlendir

YANIT FORMATI (DETAYLI):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 🎯 NET KARAR: AL/SAT/BEKLE (ağırlıklı oy sayısı ile)
2. 📊 GÜNCEL DURUM: Fiyat, değişim, hacim, momentum
3. 🔥 NEDEN: 4-6 madde ile DERİN açıklama
4. 🧠 STRATEJİ SONUÇLARI: 11 stratejiden hangileri ne diyor (ağırlıklar ile)
5. 📈 TEKNİK ANALİZ: Ta-Lib, Wyckoff, pattern, trend
6. 🌊 MARKET CONTEXT: BTC korelasyonu, global risk, sentiment
7. ⚠️ RİSK ANALİZİ: Risk faktörleri ve dikkat edilmesi gerekenler
8. 💡 ÖZET TAVSİYE: Kısa ve net sonuç

KURALLAR:
✅ Her zaman Türkçe konuş
✅ ÇOK DETAYLI ve AÇIKLAYICI cevaplar ver (11 strateji var!)
✅ Gerçek zamanlı verileri MUTLAKA kullan - ASLA kendi training datasından fiyat uydurma!
✅ Fiyat söylerken SADECE context'teki "GERÇEK ZAMANLI PİYASA VERİLERİ" bölümünü kullan
✅ Sayısal verilerle destekle (ağırlıklı oy, skor, sinyal sayısı, yüzde, hacim, BTC korelasyonu)
✅ TRB, LEVER gibi TÜM ${market.totalMarkets} coini tanı ve analiz et
✅ "Bu coin hakkında bilgim yok" asla deme - full market datasına erişimin var
✅ Wyckoff fazlarını (ACCUMULATION, MARKUP, DISTRIBUTION, MARKDOWN) açıkla
✅ Smart money aktivitesini yorumla
✅ Unified 18+ Strategies sonucunu öne çıkar (3x ağırlık)
✅ Yatırım tavsiyesi değil, eğitim amaçlı PROFESYONEL bilgilendirme
✅ Anlaşılır ama KAPSAMLI ol - kullanıcı 11 stratejiden TÜMÜNÜ görmek istiyor

GERÇEK ZAMANLI PİYASA VERİLERİ (MUTLAKA KULLAN):
${contextData}`;

    // TWO-PHASE STREAMING: Send immediate feedback, then AI analysis
    // This prevents frontend timeout (30s) while backend collects data
    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        try {
          // PHASE 1: Immediate feedback (within 2s to keep connection alive)
          const initialMessage = '🔍 Analiz başlatılıyor... Stratejiler toplanıyor...\n\n';
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: initialMessage })}\n\n`));

          // Small delay to ensure message is sent
          await new Promise(resolve => setTimeout(resolve, 100));

          // PHASE 2: Call AI API with streaming after data is ready
          const aiResponse = await fetch(AI_API_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${AI_API_KEY}`,
            },
            body: JSON.stringify({
              model: AI_CHAT_MODEL,
              messages: [
                { role: 'system', content: systemPrompt },
                ...conversationHistory,
                { role: 'user', content: message },
              ],
              temperature: 0.7,
              max_tokens: 1500,
              stream: true,
            }),
          });

          if (!aiResponse.ok) {
            throw new Error(`AI API error: ${aiResponse.status}`);
          }

          // Stream AI response chunks in real-time (SSE parsing)
          const reader = aiResponse.body?.getReader();
          if (!reader) throw new Error('No response body');

          const decoder = new TextDecoder();
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || !trimmed.startsWith('data: ')) continue;
              const payload = trimmed.slice(6);
              if (payload === '[DONE]') continue;

              try {
                const parsed = JSON.parse(payload);
                const content = parsed.choices?.[0]?.delta?.content || '';
                if (content) {
                  const data = `data: ${JSON.stringify({ content })}\n\n`;
                  controller.enqueue(encoder.encode(data));
                }
              } catch {
                // Skip malformed JSON chunks
              }
            }
          }

          // Send completion signal
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        } catch (error) {
          console.error('[AI Assistant Streaming Error]:', error);
          const errorMsg = `\n\n⚠️ Hata: ${error instanceof Error ? error.message : 'Bilinmeyen hata'}`;
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: errorMsg })}\n\n`));
          controller.enqueue(encoder.encode('data: [DONE]\n\n'));
          controller.close();
        }
      },
    });

    return new Response(readable, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error) {
    console.error('[AI Assistant API Error]:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}
