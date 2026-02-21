/**
 * 📅 TELEGRAM SIGNAL SCHEDULER
 * Belirli zaman aralıklarında API'leri çağırıp Telegram'a bildirim gönderir
 *
 * Zaman Dilimleri:
 * - 1 saatlik: Her saat başı
 * - 4 saatlik: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
 * - Günlük: UTC 00:00 (Türkiye 03:00)
 * - Haftalık: Pazartesi UTC 00:00
 *
 * Features:
 * - Nirvana Dashboard özeti (günlük)
 * - Omnipotent Futures signals (4 saatlik)
 * - BTC-ETH Analysis (günlük)
 * - Market Correlation signals (1 saatlik - yüksek confidence)
 * - Crypto News (anlık + günlük özet)
 */

import {
  notifyNirvanaOverview,
  notifyOmnipotentFuturesSignal,
  notifyBTCETHAnalysis,
  notifyMarketCorrelationDetail,
  notifyCryptoNews,
} from '../telegram 2/unified-notification-bridge';

// Base URL (production veya local)
const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

// ============================================================================
// 1️⃣ NIRVANA DASHBOARD - GÜNLÜK ÖZET (UTC 00:00)
// ============================================================================

export async function sendNirvanaDaily() {
  try {
    console.log('[Scheduler] Nirvana dashboard günlük özet gönderiliyor...');

    const response = await fetch(`${BASE_URL}/api/nirvana`);
    const data = await response.json();

    if (!data.success || !data.data) {
      console.error('[Scheduler] Nirvana API hatası:', data.error);
      return;
    }

    await notifyNirvanaOverview({
      totalStrategies: data.data.totalStrategies,
      activeStrategies: data.data.activeStrategies,
      totalSignals: data.data.totalSignals,
      marketSentiment: data.data.marketSentiment,
      sentimentScore: data.data.sentimentScore,
      topOpportunities: data.data.topOpportunities || [],
    });

    console.log('[Scheduler] ✅ Nirvana günlük özet gönderildi');
  } catch (error: any) {
    console.error('[Scheduler] Nirvana hatası:', error.message);
  }
}

// ============================================================================
// 2️⃣ OMNIPOTENT FUTURES - 4 SAATLİK SINYALLER
// ============================================================================

export async function sendOmnipotentFuturesSignals() {
  try {
    console.log('[Scheduler] Omnipotent Futures 4 saatlik sinyaller gönderiliyor...');

    const response = await fetch(`${BASE_URL}/api/omnipotent-futures?limit=600`);
    const data = await response.json();

    if (!data.success || !data.data?.futures) {
      console.error('[Scheduler] Omnipotent Futures API hatası:', data.error);
      return;
    }

    // Yüksek confidence (>= 75) sinyalleri filtrele
    const highConfidenceSignals = data.data.futures.filter(
      (f: any) => f.confidence >= 75 && f.signal !== 'WAIT'
    );

    console.log(
      `[Scheduler] ${highConfidenceSignals.length}/${data.data.futures.length} yüksek güven sinyali bulundu`
    );

    // En yüksek 5 sinyali gönder (spam önleme)
    for (const signal of highConfidenceSignals.slice(0, 5)) {
      await notifyOmnipotentFuturesSignal({
        symbol: signal.symbol,
        price: signal.price,
        wyckoffPhase: signal.wyckoffPhase,
        signal: signal.signal,
        confidence: signal.confidence,
        omnipotentScore: signal.omnipotentScore,
        volumeProfile: signal.volumeProfile,
        reason: signal.reasoning,
      });

      // Rate limiting (500ms bekle)
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    console.log('[Scheduler] ✅ Omnipotent Futures sinyalleri gönderildi');
  } catch (error: any) {
    console.error('[Scheduler] Omnipotent Futures hatası:', error.message);
  }
}

// ============================================================================
// 3️⃣ BTC-ETH ANALYSIS - GÜNLÜK KORELASYON (UTC 00:00)
// ============================================================================

export async function sendBTCETHDaily() {
  try {
    console.log('[Scheduler] BTC-ETH günlük korelasyon gönderiliyor...');

    const response = await fetch(`${BASE_URL}/api/btc-eth-analysis`);
    const data = await response.json();

    if (!data.success || !data.data) {
      console.error('[Scheduler] BTC-ETH API hatası:', data.error);
      return;
    }

    await notifyBTCETHAnalysis({
      correlation30d: data.data.correlation30d,
      trend: data.data.trend,
      signal: data.data.signal,
      divergenceStrength: data.data.divergenceStrength,
    });

    console.log('[Scheduler] ✅ BTC-ETH günlük korelasyon gönderildi');
  } catch (error: any) {
    console.error('[Scheduler] BTC-ETH hatası:', error.message);
  }
}

// ============================================================================
// 4️⃣ MARKET CORRELATION - SAATLİK YÜKSEK SİNYALLER
// ============================================================================

export async function sendMarketCorrelationSignals() {
  try {
    console.log('[Scheduler] Market Correlation saatlik sinyaller gönderiliyor...');

    const response = await fetch(`${BASE_URL}/api/market-correlation?limit=600`);
    const data = await response.json();

    if (!data.success || !data.data?.correlations) {
      console.error('[Scheduler] Market Correlation API hatası:', data.error);
      return;
    }

    // Yüksek confidence (>= 80) ve yüksek omnipotent score (>= 85) sinyalleri filtrele
    const highQualitySignals = data.data.correlations.filter(
      (c: any) =>
        c.confidence >= 80 && c.omnipotentScore >= 85 && c.signal !== 'WAIT'
    );

    console.log(
      `[Scheduler] ${highQualitySignals.length}/${data.data.correlations.length} yüksek kalite sinyal bulundu`
    );

    // En yüksek 3 sinyali gönder (spam önleme)
    for (const signal of highQualitySignals.slice(0, 3)) {
      await notifyMarketCorrelationDetail({
        symbol: signal.symbol,
        price: signal.price,
        btcCorrelation: signal.btcCorrelation,
        omnipotentScore: signal.omnipotentScore,
        marketPhase: signal.marketPhase,
        trend: signal.trend,
        signal: signal.signal,
        confidence: signal.confidence,
        fundingBias: signal.fundingBias,
        liquidationRisk: signal.liquidationRisk,
      });

      // Rate limiting (500ms bekle)
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    console.log('[Scheduler] ✅ Market Correlation sinyalleri gönderildi');
  } catch (error: any) {
    console.error('[Scheduler] Market Correlation hatası:', error.message);
  }
}

// ============================================================================
// 5️⃣ CRYPTO NEWS - ANLIK + GÜNLÜK ÖZET
// ============================================================================

export async function sendCryptoNews() {
  try {
    console.log('[Scheduler] Crypto News kontrol ediliyor...');

    const response = await fetch(`${BASE_URL}/api/crypto-news?refresh=true`);
    const data = await response.json();

    if (!data.success || !data.data || data.data.length === 0) {
      console.log('[Scheduler] Yeni haber bulunamadı');
      return;
    }

    console.log(`[Scheduler] ${data.data.length} yeni haber bulundu`);

    // Yüksek impact (>= 8) haberleri gönder
    const importantNews = data.data.filter((n: any) => n.impactScore >= 8);

    for (const news of importantNews) {
      await notifyCryptoNews({
        title: news.title,
        titleTR: news.titleTR,
        descriptionTR: news.descriptionTR,
        url: news.url,
        impactScore: news.impactScore,
        category: news.category,
        sentiment: news.sentiment,
        tags: news.tags,
      });

      // Rate limiting (1s bekle)
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    console.log('[Scheduler] ✅ Crypto News gönderildi');
  } catch (error: any) {
    console.error('[Scheduler] Crypto News hatası:', error.message);
  }
}

// ============================================================================
// 6️⃣ QUANTUM LADDER - SAATLİK ALIM SİNYALLERİ (KARAR MERKEZİ)
// ============================================================================

export async function sendQuantumLadderHourly() {
  try {
    console.log('[Scheduler] Quantum Ladder saatlik alım sinyalleri gönderiliyor...');

    // 1. BTC ve ETH için analiz (her zaman gönder - kritik coinler)
    const priorityCoins = ['BTCUSDT', 'ETHUSDT'];

    for (const symbol of priorityCoins) {
      try {
        const response = await fetch(`${BASE_URL}/api/decision-engine?symbol=${symbol}`);
        const data = await response.json();

        if (!data.success || !data.data) continue;

        const d = data.data;
        const baseSymbol = symbol.replace('USDT', '');

        // Sadece ALIM sinyalleri gönder
        if (d.decision && (d.decision.includes('BUY') || d.decision.includes('STRONG'))) {
          const decisionTR = d.decision === 'STRONG_BUY' ? 'GÜÇLÜ ALIM' :
                            d.decision === 'BUY' ? 'ALIM' : d.decision;

          let message = `🎯 <b>QUANTUM LADDER - KARAR MERKEZİ</b>\n\n`;
          message += `📊 <b>${baseSymbol}/USDT</b>\n`;
          message += `💰 Anlık Fiyat: <b>$${d.currentPrice.toFixed(2)}</b>\n\n`;

          message += `🎯 <b>KARAR: ${decisionTR}</b>\n`;
          message += `📈 Güven Skoru: <b>${(d.confidence * 100).toFixed(0)}%</b>\n`;
          message += `━━━━━━━━━━━━━━━━━\n\n`;

          if (d.strongestSignals && d.strongestSignals.length > 0) {
            message += `🔥 <b>EN GÜÇLÜ STRATEJİLER</b>\n`;
            d.strongestSignals.slice(0, 3).forEach((s: any, idx: number) => {
              message += `${idx + 1}. ${s.name} (${(s.confidence * 100).toFixed(0)}%)\n`;
              message += `   └ ${s.reason}\n`;
            });
            message += `\n`;
          }

          message += `💡 <b>İŞLEM BİLGİLERİ</b>\n`;
          message += `🎯 Giriş: $${d.entryPrice.toFixed(2)}\n`;
          message += `🛑 Stop-Loss: $${d.stopLoss.toFixed(2)}\n`;
          message += `🎯 Hedef 1: $${d.targets.tp1.toFixed(2)}\n`;
          message += `🎯 Hedef 2: $${d.targets.tp2.toFixed(2)}\n`;
          message += `🎯 Hedef 3: $${d.targets.tp3.toFixed(2)}\n`;
          message += `⚖️ Risk/Reward: ${d.riskRewardRatio.toFixed(2)}\n\n`;

          if (d.reasons && d.reasons.length > 0) {
            message += `📋 <b>NEDEN ALIM?</b>\n`;
            d.reasons.slice(0, 4).forEach((r: string) => {
              message += `• ${r}\n`;
            });
            message += `\n`;
          }

          message += `⏰ ${new Date().toLocaleString('tr-TR')}\n`;
          message += `━━━━━━━━━━━━━━━━━\n`;
          message += `⚖️ <i>Sadece eğitim amaçlıdır. Yatırım tavsiyesi değildir.</i>`;

          // Send via Telegram
          await fetch(`${BASE_URL}/api/telegram/live`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
          });

          console.log(`[Scheduler] ✅ ${symbol} quantum ladder sinyali gönderildi (${decisionTR})`);

          // Rate limiting
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
      } catch (symbolError: any) {
        console.error(`[Scheduler] ${symbol} hatası:`, symbolError.message);
      }
    }

    // 2. Diğer yüksek confidence ALIM sinyalleri (top 5)
    const buySignalsResponse = await fetch(`${BASE_URL}/api/buy-signals-coins`);
    const buySignalsData = await buySignalsResponse.json();

    if (buySignalsData.success && buySignalsData.data?.coins) {
      // Filter yüksek confidence (>= 80) ve BTC/ETH olmayan coinler
      const otherCoins = buySignalsData.data.coins
        .filter((c: any) =>
          c.confidence >= 80 &&
          !priorityCoins.includes(c.symbol)
        )
        .slice(0, 5);

      console.log(`[Scheduler] ${otherCoins.length} ek yüksek güven alım sinyali bulundu`);

      for (const coin of otherCoins) {
        try {
          const response = await fetch(`${BASE_URL}/api/decision-engine?symbol=${coin.symbol}`);
          const data = await response.json();

          if (!data.success || !data.data) continue;

          const d = data.data;
          const baseSymbol = coin.symbol.replace('USDT', '');

          if (d.decision && d.decision.includes('BUY')) {
            const decisionTR = d.decision === 'STRONG_BUY' ? 'GÜÇLÜ ALIM' : 'ALIM';

            let message = `🎯 <b>QUANTUM LADDER - FIRSATLAR</b>\n\n`;
            message += `📊 <b>${baseSymbol}/USDT</b>\n`;
            message += `💰 Anlık Fiyat: <b>$${d.currentPrice.toFixed(6)}</b>\n\n`;

            message += `🎯 <b>KARAR: ${decisionTR}</b>\n`;
            message += `📈 Güven: <b>${(d.confidence * 100).toFixed(0)}%</b>\n`;
            message += `━━━━━━━━━━━━━━━━━\n\n`;

            message += `💡 <b>İŞLEM</b>\n`;
            message += `🎯 Giriş: $${d.entryPrice.toFixed(6)}\n`;
            message += `🛑 Stop: $${d.stopLoss.toFixed(6)}\n`;
            message += `🎯 Hedef: $${d.targets.tp1.toFixed(6)}\n`;
            message += `⚖️ R/R: ${d.riskRewardRatio.toFixed(2)}\n\n`;

            if (d.strongestSignals && d.strongestSignals.length > 0) {
              message += `🔥 <b>TOP STRATEJI</b>\n`;
              const top = d.strongestSignals[0];
              message += `${top.name} (${(top.confidence * 100).toFixed(0)}%)\n`;
              message += `${top.reason}\n\n`;
            }

            message += `⏰ ${new Date().toLocaleString('tr-TR')}\n`;
            message += `⚖️ <i>Eğitim amaçlıdır. Yatırım tavsiyesi değildir.</i>`;

            await fetch(`${BASE_URL}/api/telegram/live`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ message }),
            });

            console.log(`[Scheduler] ✅ ${coin.symbol} ek sinyal gönderildi`);

            // Rate limiting
            await new Promise((resolve) => setTimeout(resolve, 1000));
          }
        } catch (symbolError: any) {
          console.error(`[Scheduler] ${coin.symbol} hatası:`, symbolError.message);
        }
      }
    }

    console.log('[Scheduler] ✅ Quantum Ladder saatlik sinyaller tamamlandı');
  } catch (error: any) {
    console.error('[Scheduler] Quantum Ladder hatası:', error.message);
  }
}

// ============================================================================
// SCHEDULER MAIN FUNCTIONS
// ============================================================================

/**
 * 1 Saatlik Scheduler (Her saat başı)
 */
export async function runHourlyScheduler() {
  console.log('\n🕐 === 1 SAATLİK SCHEDULER BAŞLADI ===');
  await sendQuantumLadderHourly(); // 🎯 KARAR MERKEZİ - Quantum Ladder Alım Sinyalleri
  await sendMarketCorrelationSignals();
  console.log('✅ === 1 SAATLİK SCHEDULER TAMAMLANDI ===\n');
}

/**
 * 4 Saatlik Scheduler (00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
 */
export async function run4HourlyScheduler() {
  console.log('\n🕓 === 4 SAATLİK SCHEDULER BAŞLADI ===');
  await sendOmnipotentFuturesSignals();
  await sendCryptoNews();
  console.log('✅ === 4 SAATLİK SCHEDULER TAMAMLANDI ===\n');
}

/**
 * Günlük Scheduler (UTC 00:00 - Türkiye 03:00)
 */
export async function runDailyScheduler() {
  console.log('\n📅 === GÜNLÜK SCHEDULER BAŞLADI ===');
  await sendNirvanaDaily();
  await sendBTCETHDaily();
  await sendCryptoNews(); // Günlük özet
  console.log('✅ === GÜNLÜK SCHEDULER TAMAMLANDI ===\n');
}

/**
 * Haftalık Scheduler (Pazartesi UTC 00:00)
 */
export async function runWeeklyScheduler() {
  console.log('\n📆 === HAFTALIK SCHEDULER BAŞLADI ===');
  await sendNirvanaDaily(); // Haftalık özet
  console.log('✅ === HAFTALIK SCHEDULER TAMAMLANDI ===\n');
}

// Manual test fonksiyonu
export async function testAllSchedulers() {
  console.log('🧪 === TÜM SCHEDULER TEST BAŞLADI ===\n');
  await runHourlyScheduler();
  await run4HourlyScheduler();
  await runDailyScheduler();
  console.log('✅ === TÜM SCHEDULER TEST TAMAMLANDI ===\n');
}
