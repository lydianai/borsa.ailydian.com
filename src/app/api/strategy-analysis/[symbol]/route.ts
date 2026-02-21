/**
 * STRATEGY ANALYSIS API
 *
 * Aggregates all trading strategies for a specific symbol
 * Used by Decision Engine to make trading decisions
 *
 * ✅ FALLBACK SYSTEM: Binance → Bybit → CoinGecko
 */

import { NextRequest, NextResponse } from 'next/server';
import { fetchBinanceFuturesData } from '@/lib/binance-data-fetcher';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface Signal {
  symbol: string;
  type: string;
  confidence: number;
  reason: string;
  entryPrice?: number;
  stopLoss?: number;
  targets?: any;
}

interface Strategy {
  name: string;
  signal: string;
  confidence: number;
  reason: string;
  score?: number;
}

/**
 * Fetch signals from an endpoint with error handling
 * Uses dynamic base URL for production compatibility
 */
async function fetchSignals(endpoint: string, request: NextRequest): Promise<any> {
  try {
    // Get base URL from environment or request headers
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL ||
                    (request.headers.get('host') ? `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('host')}` : 'http://localhost:3000');

    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    if (!response.ok) {
      console.warn(`[Strategy Analysis] ${endpoint} returned ${response.status}`);
      return { success: false, error: `HTTP ${response.status}` };
    }

    return await response.json();
  } catch (error: any) {
    console.warn(`[Strategy Analysis] Error fetching ${endpoint}:`, error.message);
    return { success: false, error: error.message };
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol: symbolParam } = await params;
    const symbol = symbolParam || 'BTCUSDT';

    console.log(`[Strategy Analysis] Fetching strategies for ${symbol}...`);

    // ✅ FALLBACK SYSTEM: Fetch price data with Binance → Bybit → CoinGecko fallback
    console.log(`[Strategy Analysis] Fetching price data with fallback system...`);
    const priceData = await fetchBinanceFuturesData();

    if (!priceData.success || !priceData.data || !priceData.data.all) {
      console.error('[Strategy Analysis] Price data fetch failed:', priceData.error);
      throw new Error(priceData.error || 'Fiyat verisi alınamadı - tüm kaynaklar başarısız');
    }

    console.log(`[Strategy Analysis] ✅ Price data fetched successfully (${priceData.data.all.length} coins)`);

    // Match using exact symbol (BTCUSDT, ETHUSDT, etc.)
    const coinData = priceData.data.all.find((c: any) => c.symbol === symbol);

    if (!coinData) {
      console.error(`[Strategy Analysis] Coin not found: ${symbol}`);
      console.error(`[Strategy Analysis] Available symbols sample:`, priceData.data.all.slice(0, 5).map((c: any) => c.symbol));
      throw new Error('Coin bulunamadı veya Binance API hatası');
    }

    const currentPrice = Number(coinData.price);
    const changePercent24h = Number(coinData.change24h) || 0;

    // Fetch signals from all sources in parallel
    const [
      tradingSignals,
      aiSignals,
      conservativeSignals,
      quantumSignals,
      breakoutSignals,
      mfiSignals,
    ] = await Promise.all([
      fetchSignals(`/api/signals?limit=200`, request),
      fetchSignals(`/api/ai-signals`, request),
      fetchSignals(`/api/conservative-signals`, request),
      fetchSignals(`/api/quantum-signals`, request),
      fetchSignals(`/api/breakout-retest`, request),
      fetchSignals(`/api/mfi-monitor?symbol=${symbol}`, request),
    ]);

    // Aggregate strategies for this symbol
    const strategies: Strategy[] = [];

    // Process Trading Signals (Technical Analysis)
    if (tradingSignals.success && tradingSignals.data && tradingSignals.data.signals) {
      const signals = tradingSignals.data.signals.filter((s: Signal) => s.symbol === symbol);
      if (signals.length > 0) {
        const signal = signals[0]; // Take the first one
        strategies.push({
          name: 'Teknik Analiz Sinyalleri',
          signal: (signal.type || 'NEUTRAL').toUpperCase(),
          confidence: signal.confidence || 0.5,
          reason: signal.reason || 'Gelişmiş teknik gösterge analizi',
        });
      }
    }

    // Process AI Signals
    if (aiSignals.success && aiSignals.data && aiSignals.data.signals) {
      const signals = aiSignals.data.signals.filter((s: Signal) => s.symbol === symbol);
      if (signals.length > 0) {
        const signal = signals[0];
        strategies.push({
          name: 'Yapay Zeka Sinyalleri',
          signal: (signal.type || 'NEUTRAL').toUpperCase(),
          confidence: signal.confidence || 0.5,
          reason: signal.reason || 'Yapay zeka destekli gelişmiş piyasa analizi',
        });
      }
    }

    // Process Conservative Signals
    if (conservativeSignals.success && conservativeSignals.data && conservativeSignals.data.signals) {
      const signals = conservativeSignals.data.signals.filter((s: Signal) => s.symbol === symbol);
      if (signals.length > 0) {
        const signal = signals[0];
        strategies.push({
          name: 'Conservative Signals',
          signal: (signal.type || 'NEUTRAL').toUpperCase(),
          confidence: (signal.confidence || 50) / 100,
          reason: signal.reason || 'Conservative multi-indicator strategy',
        });
      }
    }

    // Process Quantum Signals
    if (quantumSignals.success && quantumSignals.data && quantumSignals.data.signals) {
      const signals = quantumSignals.data.signals.filter((s: Signal) => s.symbol === symbol);
      if (signals.length > 0) {
        const signal = signals[0];
        strategies.push({
          name: 'Quantum Signals',
          signal: (signal.type || 'NEUTRAL').toUpperCase(),
          confidence: (signal.confidence || 50) / 100,
          reason: signal.reason || 'Advanced quantum analysis',
        });
      }
    }

    // Process Breakout-Retest Signals
    if (breakoutSignals.success && breakoutSignals.data && breakoutSignals.data.signals) {
      const signals = breakoutSignals.data.signals.filter((s: Signal) => s.symbol === symbol);
      if (signals.length > 0) {
        const signal = signals[0];
        strategies.push({
          name: 'Breakout-Retest',
          signal: (signal.type || 'NEUTRAL').toUpperCase(),
          confidence: (signal.confidence || 50) / 100,
          reason: signal.reason || '3-phase breakout pattern',
        });
      }
    }

    // Process MFI (Money Flow Index) Signals
    if (mfiSignals.success && mfiSignals.data && mfiSignals.data.overall) {
      const mfiData = mfiSignals.data;
      const overall = mfiData.overall;

      // Convert MFI signal to standard strategy format
      let mfiSignal = 'HOLD';
      let mfiConfidence = 0.5;
      let mfiReason = overall.message || 'MFI analysis';

      // Map MFI signals to confidence levels
      if (overall.signal === 'STRONG_BUY') {
        mfiSignal = 'BUY';
        mfiConfidence = 0.9; // High confidence
        mfiReason = `All timeframes oversold (Avg MFI: ${overall.avg_mfi})`;
      } else if (overall.signal === 'BUY') {
        mfiSignal = 'BUY';
        mfiConfidence = 0.75;
        mfiReason = `${overall.oversold_count}/${overall.total_timeframes} timeframes oversold (Avg MFI: ${overall.avg_mfi})`;
      } else if (overall.signal === 'WATCH') {
        mfiSignal = 'HOLD';
        mfiConfidence = 0.6;
        mfiReason = `${overall.oversold_count}/${overall.total_timeframes} timeframes oversold (Avg MFI: ${overall.avg_mfi})`;
      } else if (overall.signal === 'SELL') {
        mfiSignal = 'SELL';
        mfiConfidence = 0.7;
        mfiReason = `Overbought detected (Avg MFI: ${overall.avg_mfi})`;
      }

      // Add extra confidence for critical alerts
      if (mfiData.critical_alerts && mfiData.critical_alerts.length > 0) {
        const criticalAlert = mfiData.critical_alerts[0];
        if (criticalAlert.sudden_drop && criticalAlert.sudden_drop.detected) {
          mfiConfidence = Math.min(0.95, mfiConfidence + 0.1);
          mfiReason += ` - CRITICAL: Sudden drop of ${criticalAlert.sudden_drop.drop_percentage}% detected on ${criticalAlert.timeframe}`;
        }
      }

      strategies.push({
        name: 'MFI (Money Flow Index)',
        signal: mfiSignal,
        confidence: mfiConfidence,
        reason: mfiReason,
      });
    }

    // If no strategies found, add a neutral strategy
    if (strategies.length === 0) {
      strategies.push({
        name: 'Default',
        signal: 'HOLD',
        confidence: 0.5,
        reason: 'Yeterli strateji sinyali bulunamadı',
      });
    }

    console.log(`[Strategy Analysis] Found ${strategies.length} strategies for ${symbol}`);

    // Calculate signal counts
    const buyCount = strategies.filter(s => s.signal === 'BUY').length;
    const sellCount = strategies.filter(s => s.signal === 'SELL').length;
    const waitCount = strategies.filter(s => s.signal === 'HOLD' || s.signal === 'WAIT').length;
    const neutralCount = strategies.filter(s => s.signal === 'NEUTRAL').length;

    // Calculate overall score (0-100) based on confidence-weighted signals
    let totalWeightedScore = 0;
    let totalWeight = 0;
    strategies.forEach(s => {
      const weight = s.confidence;
      let signalScore = 50; // HOLD/NEUTRAL = 50
      if (s.signal === 'BUY') signalScore = 100;
      if (s.signal === 'SELL') signalScore = 0;
      totalWeightedScore += signalScore * weight;
      totalWeight += weight;
    });
    const overallScore = totalWeight > 0 ? Math.round(totalWeightedScore / totalWeight) : 50;

    // Determine overall recommendation
    let recommendation = 'WAIT';
    if (overallScore >= 65) recommendation = 'BUY';
    else if (overallScore <= 35) recommendation = 'SELL';

    // Generate comprehensive AI analysis text
    const aiAnalysis = `
📊 **Kapsamlı Piyasa Analizi: ${symbol.replace('USDT', '')}/USDT**

**Mevcut Durum:**
• Fiyat: $${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 6 })}
• 24 Saatlik Değişim: ${changePercent24h >= 0 ? '+' : ''}${changePercent24h.toFixed(2)}%
• Genel Skor: ${overallScore}/100
• Öneri: ${recommendation === 'BUY' ? 'AL' : recommendation === 'SELL' ? 'SAT' : 'BEKLE'}

**Strateji Dağılımı:**
• AL Sinyalleri: ${buyCount}
• SAT Sinyalleri: ${sellCount}
• BEKLE Sinyalleri: ${waitCount}
• NÖTR Sinyalleri: ${neutralCount}

**Analiz Özeti:**
${strategies.length > 0 ? strategies.map((s, i) =>
  `${i + 1}. **${s.name}**: ${s.signal} (Güven: ${s.confidence}%)\n   ${s.reason || 'Detaylı analiz mevcut değil.'}`
).join('\n\n') : 'Henüz strateji sinyali oluşturulmadı.'}

**Risk Uyarısı:**
Bu analiz otomatik olarak oluşturulmuştur ve yalnızca bilgilendirme amaçlıdır. Yatırım kararlarınızı vermeden önce kendi araştırmanızı yapın ve profesyonel finansal danışmanlık alın. Geçmiş performans gelecekteki sonuçların garantisi değildir.

**Son Güncelleme:** ${new Date().toLocaleString('tr-TR')}
    `.trim();

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        price: currentPrice,
        changePercent24h,
        strategies,
        buyCount,
        sellCount,
        waitCount,
        neutralCount,
        overallScore,
        recommendation,
        aiAnalysis,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error('[Strategy Analysis] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Strategy analysis failed',
      },
      { status: 404 }
    );
  }
}
