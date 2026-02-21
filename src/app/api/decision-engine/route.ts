import { NextRequest, NextResponse } from 'next/server';

/**
 * Decision Engine API
 * Tüm stratejileri analiz edip KARAR verir:
 * - Alım/Satım Kararı
 * - Giriş Fiyatı
 * - Hedef Fiyatlar (TP1, TP2, TP3)
 * - Stop Loss
 * - Risk/Reward Ratio
 * - Gerekçeler
 */

interface StrategySignal {
  name: string;
  signal: string;
  confidence: number;
  reason: string;
  score?: number;
}

interface DecisionResult {
  success: boolean;
  data?: {
    symbol: string;
    currentPrice: number;
    decision: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
    confidence: number;

    // Fiyat Seviyeleri
    entryPrice: number;
    stopLoss: number;
    targets: {
      tp1: number;
      tp2: number;
      tp3: number;
    };

    // Risk/Reward
    riskRewardRatio: number;
    potentialGain: number;
    potentialLoss: number;

    // Analiz
    buySignalsCount: number;
    sellSignalsCount: number;
    totalStrategies: number;
    strongestSignals: Array<{
      name: string;
      signal: string;
      confidence: number;
      reason: string;
    }>;

    // Gerekçeler
    reasons: string[];
    summary: string;

    timestamp: number;
  };
  error?: string;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol') || 'BTCUSDT';

    // Get dynamic base URL for production compatibility
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL ||
                    (request.headers.get('host') ? `${request.headers.get('x-forwarded-proto') || 'http'}://${request.headers.get('host')}` : 'http://localhost:3000');

    console.log(`[Decision Engine] Using base URL: ${baseUrl}`);

    // 1. Strategy Analysis API'den tüm sinyalleri çek
    const strategyResponse = await fetch(
      `${baseUrl}/api/strategy-analysis/${symbol}`,
      {
        cache: 'no-store',
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    if (!strategyResponse.ok) {
      const errorText = await strategyResponse.text();
      console.error(`[Decision Engine] Strategy API failed: ${strategyResponse.status} - ${errorText}`);
      throw new Error(`Strategy analysis API failed: ${strategyResponse.status} - ${errorText.substring(0, 100)}`);
    }

    const strategyData = await strategyResponse.json();

    if (!strategyData.success || !strategyData.data) {
      console.error('[Decision Engine] Invalid strategy data:', strategyData);
      throw new Error(`Invalid strategy data: ${JSON.stringify(strategyData).substring(0, 100)}`);
    }

    const { strategies, price, changePercent24h } = strategyData.data;
    const currentPrice = price;

    // 2. Fetch MFI data for additional confidence scoring
    let mfiData: any = null;
    let mfiBoost = 0;
    let mfiReasons: string[] = [];

    try {
      const mfiResponse = await fetch(
        `${baseUrl}/api/mfi-monitor?symbol=${symbol}`,
        {
          cache: 'no-store',
          headers: { 'Content-Type': 'application/json' }
        }
      );

      if (mfiResponse.ok) {
        const mfiResult = await mfiResponse.json();
        if (mfiResult.success && mfiResult.data) {
          mfiData = mfiResult.data;

          // Apply MFI confidence boost based on oversold conditions
          const overall = mfiData.overall;

          if (overall.signal === 'STRONG_BUY') {
            // All timeframes oversold - maximum boost
            mfiBoost = 0.15;
            mfiReasons.push(`🔥 MFI CRITICAL: Tüm ${overall.total_timeframes} zaman dilimi aşırı satım bölgesinde (Ort MFI: ${overall.avg_mfi})`);
          } else if (overall.signal === 'BUY') {
            // Majority oversold - strong boost
            mfiBoost = 0.12;
            mfiReasons.push(`💎 MFI: ${overall.oversold_count}/${overall.total_timeframes} zaman dilimi aşırı satım (Ort MFI: ${overall.avg_mfi})`);
          } else if (overall.signal === 'WATCH') {
            // Some oversold - moderate boost
            mfiBoost = 0.08;
            mfiReasons.push(`⚠️ MFI: ${overall.oversold_count}/${overall.total_timeframes} zaman dilimi aşırı satım eşiğinde (Ort MFI: ${overall.avg_mfi})`);
          }

          // Extra boost for sudden drops (critical alerts)
          if (mfiData.critical_alerts && mfiData.critical_alerts.length > 0) {
            const criticalAlert = mfiData.critical_alerts[0];
            if (criticalAlert.sudden_drop && criticalAlert.sudden_drop.detected) {
              mfiBoost += 0.05; // Additional +5% for sudden drops
              mfiReasons.push(`⚡ CRITICAL: ${criticalAlert.timeframe} zaman diliminde %${criticalAlert.sudden_drop.drop_percentage} ani düşüş tespit edildi!`);
            }
          }

          console.log(`[Decision Engine] MFI Boost for ${symbol}: +${(mfiBoost * 100).toFixed(1)}%`);
        }
      }
    } catch (error) {
      console.warn('[Decision Engine] MFI fetch failed, continuing without MFI boost:', error);
    }

    // 3. Sinyalleri analiz et
    const buySignals = strategies.filter((s: StrategySignal) =>
      s.signal === 'buy' || s.signal === 'BUY' || s.signal === 'STRONG_BUY'
    );
    const sellSignals = strategies.filter((s: StrategySignal) =>
      s.signal === 'sell' || s.signal === 'SELL' || s.signal === 'STRONG_SELL'
    );
    const _holdSignals = strategies.filter((s: StrategySignal) =>
      s.signal === 'hold' || s.signal === 'HOLD' || s.signal === 'NEUTRAL'
    );

    // 4. Confidence skorlarını hesapla
    let avgBuyConfidence = buySignals.length > 0
      ? buySignals.reduce((sum: number, s: StrategySignal) => sum + s.confidence, 0) / buySignals.length
      : 0;

    let avgSellConfidence = sellSignals.length > 0
      ? sellSignals.reduce((sum: number, s: StrategySignal) => sum + s.confidence, 0) / sellSignals.length
      : 0;

    // Apply MFI boost to buy confidence
    if (mfiBoost > 0) {
      avgBuyConfidence = Math.min(0.95, avgBuyConfidence + mfiBoost);
      console.log(`[Decision Engine] Buy confidence boosted: ${avgBuyConfidence.toFixed(2)} (boost: +${(mfiBoost * 100).toFixed(1)}%)`);
    }

    // 4. Karar Ver
    let decision: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
    let overallConfidence: number;

    const buyRatio = buySignals.length / strategies.length;
    const sellRatio = sellSignals.length / strategies.length;

    if (buyRatio >= 0.7 && avgBuyConfidence >= 0.7) {
      decision = 'STRONG_BUY';
      overallConfidence = avgBuyConfidence;
    } else if (buyRatio >= 0.5 && avgBuyConfidence >= 0.6) {
      decision = 'BUY';
      overallConfidence = avgBuyConfidence;
    } else if (sellRatio >= 0.7 && avgSellConfidence >= 0.7) {
      decision = 'STRONG_SELL';
      overallConfidence = avgSellConfidence;
    } else if (sellRatio >= 0.5 && avgSellConfidence >= 0.6) {
      decision = 'SELL';
      overallConfidence = avgSellConfidence;
    } else {
      decision = 'HOLD';
      overallConfidence = 0.5;
    }

    // 5. Fiyat Seviyeleri Hesapla (Technical Analysis)
    let entryPrice = currentPrice;
    let stopLoss = 0;
    let tp1 = 0;
    let tp2 = 0;
    let tp3 = 0;

    if (decision === 'STRONG_BUY' || decision === 'BUY') {
      // LONG pozisyon
      stopLoss = currentPrice * 0.97; // %3 stop loss
      tp1 = currentPrice * 1.02; // %2 kar
      tp2 = currentPrice * 1.05; // %5 kar
      tp3 = currentPrice * 1.10; // %10 kar
    } else if (decision === 'STRONG_SELL' || decision === 'SELL') {
      // SHORT pozisyon
      stopLoss = currentPrice * 1.03; // %3 stop loss
      tp1 = currentPrice * 0.98; // %2 kar
      tp2 = currentPrice * 0.95; // %5 kar
      tp3 = currentPrice * 0.90; // %10 kar
    } else {
      // HOLD - no position
      stopLoss = currentPrice * 0.95;
      tp1 = currentPrice * 1.05;
      tp2 = currentPrice * 1.10;
      tp3 = currentPrice * 1.15;
    }

    // 6. Risk/Reward Hesapla
    const potentialLoss = Math.abs(currentPrice - stopLoss);
    const potentialGain = Math.abs(tp3 - currentPrice);
    const riskRewardRatio = potentialLoss > 0 ? potentialGain / potentialLoss : 0;

    // 7. En güçlü sinyalleri seç (top 3)
    const strongestSignals = [...strategies]
      .sort((a: StrategySignal, b: StrategySignal) => b.confidence - a.confidence)
      .slice(0, 3)
      .map((s: StrategySignal) => ({
        name: s.name,
        signal: s.signal,
        confidence: s.confidence,
        reason: s.reason,
      }));

    // 8. Gerekçeleri oluştur
    const reasons: string[] = [];

    // Add MFI reasons first (high priority)
    if (mfiReasons.length > 0) {
      reasons.push(...mfiReasons);
    }

    reasons.push(`${buySignals.length} ALIŞ sinyali, ${sellSignals.length} SATIŞ sinyali tespit edildi`);

    if (decision === 'STRONG_BUY') {
      reasons.push('Stratejilerin %70+ güçlü ALIŞ sinyali veriyor');
      reasons.push(`Ortalama güven: %${(avgBuyConfidence * 100).toFixed(0)}`);
    } else if (decision === 'BUY') {
      reasons.push('Stratejilerin çoğunluğu ALIŞ sinyali veriyor');
      reasons.push(`Ortalama güven: %${(avgBuyConfidence * 100).toFixed(0)}`);
    } else if (decision === 'STRONG_SELL') {
      reasons.push('Stratejilerin %70+ güçlü SATIŞ sinyali veriyor');
      reasons.push(`Ortalama güven: %${(avgSellConfidence * 100).toFixed(0)}`);
    } else if (decision === 'SELL') {
      reasons.push('Stratejilerin çoğunluğu SATIŞ sinyali veriyor');
      reasons.push(`Ortalama güven: %${(avgSellConfidence * 100).toFixed(0)}`);
    } else {
      reasons.push('Sinyaller karışık - BEKLENİYOR');
      reasons.push('Net bir trend oluşmadı');
    }

    if (changePercent24h > 5) {
      reasons.push(`Son 24 saatte %${changePercent24h.toFixed(2)} yükseliş`);
    } else if (changePercent24h < -5) {
      reasons.push(`Son 24 saatte %${Math.abs(changePercent24h).toFixed(2)} düşüş`);
    }

    reasons.push(`Risk/Reward Oranı: ${riskRewardRatio.toFixed(2)}`);

    // 9. Özet Oluştur
    let summary = '';
    if (decision === 'STRONG_BUY') {
      summary = `${symbol} için GÜÇLÜ ALIŞ önerisi. Giriş: $${entryPrice.toFixed(2)}, Hedef: $${tp3.toFixed(2)}, Stop: $${stopLoss.toFixed(2)}`;
    } else if (decision === 'BUY') {
      summary = `${symbol} için ALIŞ önerisi. Giriş: $${entryPrice.toFixed(2)}, Hedef: $${tp2.toFixed(2)}, Stop: $${stopLoss.toFixed(2)}`;
    } else if (decision === 'STRONG_SELL') {
      summary = `${symbol} için GÜÇLÜ SATIŞ önerisi. Giriş: $${entryPrice.toFixed(2)}, Hedef: $${tp3.toFixed(2)}, Stop: $${stopLoss.toFixed(2)}`;
    } else if (decision === 'SELL') {
      summary = `${symbol} için SATIŞ önerisi. Giriş: $${entryPrice.toFixed(2)}, Hedef: $${tp2.toFixed(2)}, Stop: $${stopLoss.toFixed(2)}`;
    } else {
      summary = `${symbol} için BEKLE önerisi. Net bir sinyal yok, pozisyon almak için erken.`;
    }

    const result: DecisionResult = {
      success: true,
      data: {
        symbol,
        currentPrice,
        decision,
        confidence: overallConfidence,
        entryPrice,
        stopLoss,
        targets: {
          tp1,
          tp2,
          tp3,
        },
        riskRewardRatio,
        potentialGain,
        potentialLoss,
        buySignalsCount: buySignals.length,
        sellSignalsCount: sellSignals.length,
        totalStrategies: strategies.length,
        strongestSignals,
        reasons,
        summary,
        timestamp: Date.now(),
      },
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error('Decision Engine Error:', error);
    return NextResponse.json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
