/**
 * 🐋 ONCHAIN WHALE ALERTS - TELEGRAM NOTIFICATIONS
 * Balina birikimi (accumulation) bildirimlerini Telegram'a gönderir
 *
 * Özellikler:
 * - Sadece "accumulation" (birikim) sinyalleri (BUY fırsatları)
 * - Minimum %70 confidence
 * - Exchange'den çıkan büyük miktarlar (whale accumulation)
 * - Spam prevention (10 dakika)
 */

import { NextResponse } from 'next/server';
import { notifyNewSignal } from '@/lib/telegram/notifications';

// Spam prevention: Son gönderilen whale alert'leri tutuyoruz
const lastSentWhaleAlerts = new Map<string, number>();
const WHALE_SPAM_PREVENTION_MINUTES = 60; // 🐋 60 dakika - whale movements daha nadir

interface WhaleAlert {
  symbol: string;
  activity: 'accumulation' | 'distribution';
  confidence: number;
  riskScore: number;
  exchangeNetflow: number;
  summary: string;
}

/**
 * Format whale alert message for Telegram (özel format - profesyonel kripto bilgisi)
 */
function formatWhaleAlert(whale: WhaleAlert): string {
  const netflowAmount = (Math.abs(whale.exchangeNetflow) / 1000000).toFixed(2);
  const isNegativeFlow = whale.exchangeNetflow < 0; // Negatif = Exchange'den ÇIKIŞ

  // Asset-specific açıklamalar (profesyonel kripto bilgisi)
  let interpretation = '';
  let actionRecommendation = '';

  if (whale.activity === 'accumulation' && isNegativeFlow) {
    // Exchange'den ÇIKIŞ = Balinalar cold wallet'a taşıyor = LONG-TERM HOLD
    if (whale.symbol === 'BTC' || whale.symbol === 'ETH') {
      interpretation = '🟢 <b>BULLISH Sinyal:</b> Balinalar cold wallet\'a taşıyor (long-term hold)';
      actionRecommendation = '💡 <b>Yorum:</b> Büyük yatırımcılar satmayı düşünmüyor. Fiyat artışı beklenebilir.';
    } else if (whale.symbol === 'USDT' || whale.symbol === 'USDC') {
      interpretation = '🟢 <b>BULLISH Sinyal:</b> Stablecoin\'ler exchange\'den çıkıyor';
      actionRecommendation = '💡 <b>Yorum:</b> Balinalar nakit tutmak yerine DeFi veya kripto almak için hazırlanıyor.';
    } else {
      interpretation = '🟢 <b>BULLISH Sinyal:</b> Exchange rezervi azalıyor (satış baskısı düşük)';
      actionRecommendation = '💡 <b>Yorum:</b> Arz azaldığında talep artarsa fiyat yükselir.';
    }
  } else if (whale.activity === 'distribution') {
    // Exchange'e GİRİŞ = Balinalar satmaya hazırlanıyor = BEARISH
    if (whale.symbol === 'BTC' || whale.symbol === 'ETH') {
      interpretation = '🔴 <b>BEARISH Sinyal:</b> Balinalar exchange\'e transfer yapıyor (satış hazırlığı)';
      actionRecommendation = '⚠️ <b>Yorum:</b> Büyük satışlar olabilir. Kısa vadeli düşüş riski var.';
    } else if (whale.symbol === 'USDT' || whale.symbol === 'USDC') {
      interpretation = '🔴 <b>BEARISH Sinyal:</b> Stablecoin\'ler exchange\'e geliyor';
      actionRecommendation = '⚠️ <b>Yorum:</b> Balinalar kripto satıp nakit tutmaya geçiyor (piyasadan çıkış).';
    } else {
      interpretation = '🔴 <b>BEARISH Sinyal:</b> Exchange rezervi artıyor (satış baskısı yüksek)';
      actionRecommendation = '⚠️ <b>Yorum:</b> Arz artarsa ve talep aynı kalırsa fiyat düşer.';
    }
  }

  let message = `🐋 <b>ONCHAIN BALİNA ALARMI</b>\n\n`;
  message += `<b>Varlık:</b> ${whale.symbol}\n`;
  message += `<b>Aktivite:</b> ${whale.activity === 'accumulation' ? '🟢 BİRİKİM' : '🔴 DAĞITIM'}\n`;
  message += `<b>Güven:</b> ${whale.confidence}%\n\n`;

  message += `📊 <b>Blockchain Verisi:</b>\n`;
  message += `Exchange Netflow: ${isNegativeFlow ? '➖' : '➕'} ${netflowAmount}M\n`;
  message += `${isNegativeFlow ? '📤 Exchange\'den ÇIKTI' : '📥 Exchange\'e GİRDİ'}\n`;
  message += `Risk Skoru: ${whale.riskScore.toFixed(1)}/10\n\n`;

  message += interpretation + '\n\n';
  message += actionRecommendation + '\n\n';

  message += `📋 <b>Detay:</b>\n${whale.summary}\n\n`;

  message += `🕐 ${new Date().toLocaleString('tr-TR', { timeZone: 'Europe/Istanbul' })}\n`;
  message += `📡 Kaynak: Onchain Analiz`;

  return message;
}

/**
 * Whale alert'leri Telegram'a gönder (direkt bot API ile - özel format)
 */
async function sendWhaleAlertsToTelegram(whaleAlerts: WhaleAlert[]): Promise<{
  sent: number;
  failed: number;
  errors: string[];
}> {
  let totalSent = 0;
  let totalFailed = 0;
  const allErrors: string[] = [];

  // ✨ DIREKT TELEGRAM API KULLAN (özel whale alert formatı için)
  const { bot } = await import('@/lib/telegram/bot');
  const chatIds = process.env.TELEGRAM_ALLOWED_CHAT_IDS
    ? process.env.TELEGRAM_ALLOWED_CHAT_IDS.split(',').map((id) => parseInt(id.trim(), 10))
    : [];

  if (chatIds.length === 0) {
    console.warn('[Whale Alerts] No chat IDs configured');
    return { sent: 0, failed: 0, errors: ['No recipients configured'] };
  }

  for (const whale of whaleAlerts) {
    try {
      const message = formatWhaleAlert(whale);

      for (const chatId of chatIds) {
        try {
          await bot.api.sendMessage(chatId, message, { parse_mode: 'HTML' });
          totalSent++;
        } catch (error: any) {
          totalFailed++;
          allErrors.push(`Chat ${chatId}: ${error.message}`);
          console.error(`[Whale Alerts] Failed to send to ${chatId}:`, error.message);
        }
      }

      // Spam prevention kaydı
      if (totalSent > 0) {
        lastSentWhaleAlerts.set(whale.symbol, Date.now());
      }

      console.log(`[Whale Alerts] 🐋 ${whale.symbol} accumulation - Sent: ${totalSent}`);

    } catch (error: any) {
      totalFailed++;
      allErrors.push(`${whale.symbol}: ${error.message}`);
      console.error(`[Whale Alerts] Error sending ${whale.symbol}:`, error);
    }

    // Rate limiting: 1 second delay between whale alerts
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  return { sent: totalSent, failed: totalFailed, errors: allErrors };
}

/**
 * GET: Sistem durumu
 */
export async function GET() {
  return NextResponse.json({
    status: 'active',
    description: 'Onchain Whale Alerts - Large accumulation notifications',
    features: [
      'Whale accumulation detection (exchange outflow)',
      'Minimum 70% confidence',
      'Only accumulation (BUY opportunities)',
      '10-minute spam prevention',
      'Separate from trading signals'
    ],
    usage: {
      manual: 'POST /api/telegram/whale-alerts',
      automated: 'Can be added to cron (optional)',
      checkData: 'GET /api/onchain/whale-alerts'
    },
    whiteHatRules: [
      '✅ Blockchain data only (public ledger)',
      '✅ No market manipulation',
      '✅ Educational purposes',
      '✅ Transparent algorithms'
    ],
    lastRun: lastSentWhaleAlerts.size > 0 ?
      Array.from(lastSentWhaleAlerts.entries()).map(([symbol, timestamp]) => ({
        symbol,
        time: new Date(timestamp).toISOString()
      })) : 'No whale alerts sent yet'
  });
}

/**
 * POST: 🐋 Whale Alert'leri Telegram'a gönder
 */
export async function POST() {
  const startTime = Date.now();

  try {
    console.log('[Whale Alerts] 🐋 Starting whale accumulation detection...');

    // 1. Onchain whale alert verilerini al
    const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/onchain/whale-alerts`, {
      next: { revalidate: 0 }
    });

    if (!response.ok) {
      throw new Error(`Onchain API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success || !data.data?.whaleActivity) {
      return NextResponse.json({
        success: false,
        message: '❌ No whale activity data available',
        stats: {
          totalWhaleActivity: 0,
          filtered: 0,
          sent: 0
        }
      });
    }

    // 2. Sadece accumulation (birikim) sinyalleri al
    const whaleActivity: WhaleAlert[] = data.data.whaleActivity;

    const now = Date.now();
    let filtered = whaleActivity.filter(w => {
      // Sadece accumulation (BUY fırsatları)
      if (w.activity !== 'accumulation') return false;

      // Minimum %70 confidence
      if (w.confidence < 70) return false;

      // Spam prevention: Son 10 dakikada gönderilmemiş
      const lastSent = lastSentWhaleAlerts.get(w.symbol) || 0;
      const minutesAgo = (now - lastSent) / 1000 / 60;
      if (minutesAgo < WHALE_SPAM_PREVENTION_MINUTES) return false;

      return true;
    });

    // 3. Confidence'a göre sırala
    filtered.sort((a, b) => b.confidence - a.confidence);

    console.log(`[Whale Alerts] 🐋 Filtered: ${whaleActivity.length} → ${filtered.length} accumulation signals`);

    if (filtered.length === 0) {
      return NextResponse.json({
        success: true,
        message: 'No high-confidence whale accumulation to send',
        stats: {
          totalWhaleActivity: whaleActivity.length,
          filtered: 0,
          sent: 0,
          reason: 'All filtered out (low confidence or spam-prevented or distribution signals)'
        }
      });
    }

    // 4. 📲 Telegram'a gönder
    const result = await sendWhaleAlertsToTelegram(filtered);

    const duration = Date.now() - startTime;

    return NextResponse.json({
      success: true,
      message: `✅ Whale Alerts completed in ${duration}ms`,
      stats: {
        totalWhaleActivity: whaleActivity.length,
        filtered: filtered.length,
        sent: result.sent,
        failed: result.failed,
        duration: `${duration}ms`
      },
      whaleAlerts: filtered.map(w => ({
        symbol: w.symbol,
        activity: w.activity,
        confidence: w.confidence,
        exchangeNetflow: `${(Math.abs(w.exchangeNetflow) / 1000000).toFixed(2)}M (outflow)`
      })),
      errors: result.errors.length > 0 ? result.errors : undefined
    });

  } catch (error: any) {
    console.error('[Whale Alerts] Fatal error:', error);
    return NextResponse.json({
      success: false,
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    }, { status: 500 });
  }
}
