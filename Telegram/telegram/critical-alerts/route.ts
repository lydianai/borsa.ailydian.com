/**
 * 🚨 TELEGRAM CRITICAL ALERTS ROUTE
 *
 * Anlık kritik piyasa değişikliklerini Telegram'a bildirir
 *
 * Kullanım:
 * - Cron job tarafından her 5 dakikada bir çağrılır
 * - Manuel tetikleme için: POST /api/telegram/critical-alerts
 */

import { NextResponse } from 'next/server';
import {
  analyzeCriticalChanges,
  detectPriceMovement,
  detectVolumeSpike,
  detectCorrelationBreak
} from '@/lib/critical-change-detector';

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

/**
 * Telegram mesaj gönder
 */
async function sendTelegramMessage(message: string): Promise<boolean> {
  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
    console.warn('[Critical Alerts] Telegram credentials not found');
    return false;
  }

  try {
    const response = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text: message,
          parse_mode: 'HTML'
        })
      }
    );

    if (!response.ok) {
      const error = await response.text();
      console.error('[Critical Alerts] Telegram API error:', error);
      return false;
    }

    return true;
  } catch (error: any) {
    console.error('[Critical Alerts] Telegram send error:', error.message);
    return false;
  }
}

/**
 * Alert mesajını formatla
 */
function formatAlertMessage(alert: any): string {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  let message = `🚨 <b>${alert.title}</b>\n\n`;

  switch (alert.type) {
    case 'PRICE_SPIKE':
      const isUp = alert.data.direction === 'YÜKSELİŞ';
      const emoji = isUp ? '🟢' : '🔴';
      const actionEmoji = isUp ? '📈' : '📉';
      const signal = isUp ? 'AL' : 'SAT';

      message += `${emoji} <b>${alert.data.symbol}</b> ${actionEmoji}\n`;
      message += `• Fiyat: $${alert.data.price.toLocaleString('en-US')}\n`;
      message += `• Değişim: ${alert.data.change > 0 ? '+' : ''}${alert.data.change.toFixed(2)}%\n`;
      message += `• Zaman: ${alert.data.timeframe}\n`;
      message += `• Durum: ${alert.priority === 'CRITICAL' ? '⚡ KRİTİK' : '🟡 YÜKSEK'}\n\n`;

      if (isUp) {
        message += `${emoji} <b>SİNYAL: ${signal}</b> (LONG)\n`;
        message += `💡 <b>Aksiyon:</b> Güçlü yükseliş, alım pozisyonu değerlendir\n`;
      } else {
        message += `${emoji} <b>SİNYAL: ${signal}</b> (SHORT)\n`;
        message += `💡 <b>Aksiyon:</b> Keskin düşüş, stop-loss kontrol et veya short\n`;
      }
      break;

    case 'VOLUME_SPIKE':
      const isCriticalVolume = alert.data.multiplier >= 5;
      const volumeEmoji = isCriticalVolume ? '⚡' : '🟡';

      message += `${volumeEmoji} 📊 <b>${alert.data.symbol}</b>\n`;
      message += `• Normal Volume: $${(alert.data.normalVolume / 1_000_000).toFixed(1)}M\n`;
      message += `• Şu an: $${(alert.data.currentVolume / 1_000_000).toFixed(1)}M\n`;
      message += `• Artış: ${alert.data.multiplier.toFixed(1)}x (${((alert.data.multiplier - 1) * 100).toFixed(0)}%)\n`;
      message += `• Durum: ${isCriticalVolume ? '⚡ KRİTİK' : '🟡 YÜKSEK'}\n\n`;

      if (isCriticalVolume) {
        message += `🟢 <b>SİNYAL: AL/SAT POTANSİYEL</b>\n`;
        message += `💡 <b>Aksiyon:</b> Güçlü hareket bekleniyor! Fiyat yönünü bekle ve gir!\n`;
      } else {
        message += `🟡 <b>DİKKAT: Whale Aktivitesi</b>\n`;
        message += `💡 <b>Aksiyon:</b> Büyük oyuncular aktif, dikkatli izle\n`;
      }
      break;

    case 'CORRELATION_BREAK':
      const corrEmoji = alert.data.direction === 'GÜÇLENDI' ? '📈' : '📉';
      message += `${corrEmoji} <b>${alert.data.symbol1} ↔ ${alert.data.symbol2}</b>\n`;
      message += `• Önceki: ${alert.data.previousCorr.toFixed(2)}\n`;
      message += `• Şu an: ${alert.data.currentCorr.toFixed(2)}\n`;
      message += `• Değişim: ${alert.data.direction} %${alert.data.changePercent.toFixed(0)}\n\n`;

      if (alert.data.direction === 'ZAYIFLADI') {
        message += `💡 <b>Aksiyon:</b> Divergence (ayrışma) başladı! Bağımsız harekete hazır ol\n`;
      } else {
        message += `💡 <b>Aksiyon:</b> Birlikte hareket ediyorlar\n`;
      }
      break;

    case 'WHALE_MOVEMENT':
      const isOutflow = alert.data.type === 'exchange_outflow';
      const isInflow = alert.data.type === 'exchange_inflow';
      const whaleEmoji = isOutflow ? '🟢' : isInflow ? '🔴' : '🟡';
      const whaleSignal = isOutflow ? 'AL' : isInflow ? 'SAT' : 'İZLE';

      message += `${whaleEmoji} 🐋 <b>${alert.data.symbol}</b>\n`;
      message += `• Miktar: $${alert.data.amountMillion.toFixed(1)}M\n`;
      message += `• Tip: ${alert.data.type === 'single_transfer' ? 'Büyük Transfer' : alert.data.type === 'exchange_inflow' ? 'Exchange Girişi 🔴' : 'Exchange Çıkışı 🟢'}\n`;
      message += `• Durum: ${alert.priority === 'CRITICAL' ? '⚡ KRİTİK' : '🟡 YÜKSEK'}\n\n`;

      if (isOutflow) {
        message += `${whaleEmoji} <b>SİNYAL: ${whaleSignal}</b> (BULLISH)\n`;
        message += `💡 <b>Aksiyon:</b> Balinalar cold wallet'a taşıyor - Hodl sinyali!\n`;
      } else if (isInflow) {
        message += `${whaleEmoji} <b>SİNYAL: ${whaleSignal}</b> (BEARISH)\n`;
        message += `💡 <b>Aksiyon:</b> Balinalar satış hazırlığında - Dikkatli ol!\n`;
      } else {
        message += `🟡 <b>DİKKAT: Büyük Transfer</b>\n`;
        message += `💡 <b>Aksiyon:</b> Whale hareketi - Gelişmeleri takip et\n`;
      }
      break;
  }

  message += `\n⏰ ${timeStr}`;

  return message;
}

/**
 * POST /api/telegram/critical-alerts
 */
export async function POST() {
  try {
    console.log('[Critical Alerts] Starting critical analysis...');

    // 1. Market verilerini çek (Binance Futures)
    const marketResponse = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/binance/futures`, {
      next: { revalidate: 0 }
    });

    if (!marketResponse.ok) {
      throw new Error('Failed to fetch market data');
    }

    const marketData = await marketResponse.json();

    if (!marketData.success || !marketData.data?.all) {
      throw new Error('Invalid market data format');
    }

    const markets = marketData.data.all;

    // 2. Her coin için 7-gün ortalama volume hesapla (basit örnek)
    // Production'da bu verileri cache'den veya DB'den almalısın
    const historicalVolumes = new Map<string, number>();
    markets.forEach((m: any) => {
      // Basitleştirme: Mevcut volume'ün %80'i normal kabul ediliyor
      historicalVolumes.set(m.symbol, m.volume24h * 0.8);
    });

    // 3. Kritik değişiklikleri tespit et
    const marketDataForAnalysis = markets.map((m: any) => ({
      symbol: m.symbol,
      price: m.price,
      volume24h: m.volume24h,
      change1h: m.change1h,
      change4h: m.change4h,
      change24h: m.changePercent24h
    }));

    const criticalAlerts = analyzeCriticalChanges(marketDataForAnalysis, historicalVolumes);

    console.log(`[Critical Alerts] Found ${criticalAlerts.length} critical alerts`);

    // 4. Her alert için Telegram bildirimi gönder (maksimum 5)
    const sentAlerts = [];
    const maxAlerts = 5; // Spam önleme için limit

    for (const alert of criticalAlerts.slice(0, maxAlerts)) {
      const message = formatAlertMessage(alert);
      const sent = await sendTelegramMessage(message);

      if (sent) {
        sentAlerts.push(alert.symbol);
        console.log(`[Critical Alerts] ✅ Alert sent: ${alert.symbol} (${alert.type})`);
      }

      // Rate limit: Her mesaj arasında 1 saniye bekle
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    return NextResponse.json({
      success: true,
      message: `✅ Critical alerts sent: ${sentAlerts.length}/${criticalAlerts.length}`,
      data: {
        totalAlerts: criticalAlerts.length,
        sentAlerts: sentAlerts.length,
        alerts: sentAlerts
      }
    });

  } catch (error: any) {
    console.error('[Critical Alerts] Error:', error);

    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}

/**
 * GET /api/telegram/critical-alerts (test endpoint)
 */
export async function GET() {
  return NextResponse.json({
    success: true,
    message: 'Critical Alerts API - Use POST to trigger analysis',
    info: {
      method: 'POST',
      description: 'Analyzes market for critical changes and sends Telegram alerts',
      triggers: [
        'Price: >5% (1h), >10% (4h), >15% (24h)',
        'Volume: 2x normal (high), 5x normal (critical)',
        'Correlation: >20% change',
        'Whale: $10M+ single, $50M+ exchange flow'
      ],
      spamPrevention: '30 minutes cooldown per symbol'
    }
  });
}
