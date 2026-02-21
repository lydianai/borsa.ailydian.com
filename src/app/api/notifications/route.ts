/**
 * NOTIFICATIONS API
 * Real-time notification streaming endpoint
 * Beyaz Şapka: Educational purposes only
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Helper to generate notifications based on system events
async function generateSystemNotifications() {
  const notifications: Array<{
    type: string;
    priority: string;
    title: string;
    message: string;
    source: string;
    data: any;
    actionUrl: string;
  }> = [];

  try {
    // Check for high-confidence signals
    const signalsRes = await fetch('http://localhost:3000/api/quantum-pro/signals?minConfidence=0.85&limit=5');
    const signalsData = await signalsRes.json();

    if (signalsData.success && signalsData.data.signals.length > 0) {
      signalsData.data.signals.slice(0, 3).forEach((signal: any) => {
        notifications.push({
          type: 'SIGNAL',
          priority: signal.confidence > 0.9 ? 'HIGH' : 'MEDIUM',
          title: signal.signal + ' Sinyali: ' + signal.symbol,
          message: signal.confidence + '% güven ile ' + signal.signal + ' sinyali',
          source: 'Quantum Pro',
          data: signal,
          actionUrl: '/quantum-pro'
        });
      });
    }
  } catch (err) {
    console.error('[Notifications] Quantum signals error:', err);
  }

  try {
    // Check for whale alerts
    const whaleRes = await fetch('http://localhost:3000/api/onchain/whale-alerts');
    const whaleData = await whaleRes.json();

    if (whaleData.success && whaleData.data.length > 0) {
      const recentWhales = whaleData.data.slice(0, 2);
      recentWhales.forEach((whale: any) => {
        if (whale.riskLevel === 'HIGH') {
          notifications.push({
            type: 'WHALE',
            priority: 'HIGH',
            title: 'Whale Alert: ' + whale.symbol,
            message: 'Büyük ' + (whale.netFlow > 0 ? 'alım' : 'satış') + ' hareketi tespit edildi',
            source: 'Whale Alerts',
            data: whale,
            actionUrl: '/bot-analysis'
          });
        }
      });
    }
  } catch (err) {
    console.error('[Notifications] Whale alerts error:', err);
  }

  try {
    // Check for bot status changes
    const botsRes = await fetch('http://localhost:3000/api/quantum-pro/bots');
    const botsData = await botsRes.json();

    if (botsData.success && botsData.data.summary) {
      const { errorBots, activeBots } = botsData.data.summary;

      if (errorBots > 0) {
        notifications.push({
          type: 'BOT',
          priority: 'MEDIUM',
          title: 'Bot Uyarısı',
          message: errorBots + ' bot hata durumunda',
          source: 'Quantum Pro Bots',
          data: { errorBots, activeBots },
          actionUrl: '/quantum-pro?tab=botlar'
        });
      }

      if (activeBots >= 10) {
        notifications.push({
          type: 'BOT',
          priority: 'LOW',
          title: 'Bot Sistemi Aktif',
          message: activeBots + ' bot başarıyla çalışıyor',
          source: 'Quantum Pro Bots',
          data: { errorBots, activeBots },
          actionUrl: '/quantum-pro?tab=botlar'
        });
      }
    }
  } catch (err) {
    console.error('[Notifications] Bots error:', err);
  }

  try {
    // Check for news
    const newsRes = await fetch('http://localhost:3000/api/crypto-news');
    const newsData = await newsRes.json();

    if (newsData.success && newsData.data.length > 0) {
      const topNews = newsData.data.slice(0, 2);
      topNews.forEach((news: any) => {
        if (news.sentiment === 'POSITIVE' || news.sentiment === 'NEGATIVE') {
          notifications.push({
            type: 'NEWS',
            priority: news.sentiment === 'NEGATIVE' ? 'MEDIUM' : 'LOW',
            title: news.title || 'Önemli Haber',
            message: news.description || (news.body && news.body.substring(0, 100)),
            source: 'Crypto News',
            data: news,
            actionUrl: '/market-commentary'
          });
        }
      });
    }
  } catch (err) {
    console.error('[Notifications] News error:', err);
  }

  try {
    // Check for price movements
    const binanceRes = await fetch('http://localhost:3000/api/binance/futures');
    const binanceData = await binanceRes.json();

    if (binanceData.success && binanceData.data.all) {
      const bigMovers = binanceData.data.all
        .filter((coin: any) => Math.abs(coin.changePercent24h) > 10)
        .slice(0, 3);

      bigMovers.forEach((coin: any) => {
        notifications.push({
          type: 'PRICE',
          priority: Math.abs(coin.changePercent24h) > 15 ? 'HIGH' : 'MEDIUM',
          title: 'Büyük Hareket: ' + coin.symbol,
          message: (coin.changePercent24h > 0 ? '+' : '') + coin.changePercent24h.toFixed(2) + '% değişim',
          source: 'Binance Futures',
          data: coin,
          actionUrl: '/'
        });
      });
    }
  } catch (err) {
    console.error('[Notifications] Price movements error:', err);
  }

  return notifications;
}

export async function GET(_request: NextRequest) {
  try {
    const notifications = await generateSystemNotifications();

    return NextResponse.json({
      success: true,
      data: notifications,
      timestamp: new Date().toISOString()
    });
  } catch (error: any) {
    console.error('[Notifications API] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message
      },
      { status: 500 }
    );
  }
}
