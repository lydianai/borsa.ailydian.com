import { NextResponse } from 'next/server';

/**
 * 🔔 SMART ALERTS API
 * Fiyat ve strateji tabanlı akıllı uyarılar sistemi
 */

export type AlertType = 'PRICE_ABOVE' | 'PRICE_BELOW' | 'PRICE_CHANGE' | 'STRATEGY_SIGNAL' | 'RSI' | 'VOLUME';
export type AlertStatus = 'ACTIVE' | 'TRIGGERED' | 'PAUSED' | 'EXPIRED';

export interface Alert {
  id: string;
  symbol: string;
  type: AlertType;
  status: AlertStatus;

  // Fiyat uyarıları için
  targetPrice?: number;
  changePercent?: number;

  // Strateji uyarıları için
  strategyId?: string;
  strategyName?: string;
  signalType?: 'AL' | 'SAT';
  minConfidence?: number;

  // RSI uyarıları için
  rsiThreshold?: number;
  rsiCondition?: 'ABOVE' | 'BELOW';

  // Volume uyarıları için
  volumeMultiplier?: number;

  // Metadata
  name: string;
  description?: string;
  created: string;
  lastChecked?: string;
  triggered?: string;
  expiresAt?: string;

  // Bildirim
  notifyEmail?: boolean;
  notifyPush?: boolean;
  notifySound?: boolean;
}

// In-memory storage (gerçek uygulamada database kullanılmalı)
let alerts: Alert[] = [
  {
    id: 'alert-1',
    symbol: 'BTC',
    type: 'PRICE_ABOVE',
    status: 'ACTIVE',
    targetPrice: 100000,
    name: 'BTC 100K Hedefi',
    description: 'Bitcoin 100,000$ seviyesine ulaştığında bildir',
    created: new Date().toISOString(),
    notifyEmail: true,
    notifyPush: true,
    notifySound: true,
  },
  {
    id: 'alert-2',
    symbol: 'ETH',
    type: 'STRATEGY_SIGNAL',
    status: 'ACTIVE',
    strategyId: 'default-balanced',
    strategyName: 'Dengeli Strateji',
    signalType: 'AL',
    minConfidence: 75,
    name: 'ETH Güçlü Alım Sinyali',
    description: 'Dengeli strateji %75+ güvenle AL sinyali verdiğinde bildir',
    created: new Date().toISOString(),
    notifyEmail: false,
    notifyPush: true,
    notifySound: true,
  },
  {
    id: 'alert-3',
    symbol: 'SOL',
    type: 'RSI',
    status: 'ACTIVE',
    rsiThreshold: 30,
    rsiCondition: 'BELOW',
    name: 'SOL Aşırı Satış',
    description: 'Solana RSI 30\'un altına düştüğünde bildir',
    created: new Date().toISOString(),
    notifyPush: true,
    notifySound: true,
  },
  {
    id: 'alert-4',
    symbol: 'BNB',
    type: 'PRICE_CHANGE',
    status: 'TRIGGERED',
    changePercent: 10,
    name: 'BNB Büyük Hareket',
    description: '24 saatte %10+ değişim',
    created: new Date(Date.now() - 86400000).toISOString(),
    triggered: new Date().toISOString(),
    notifyPush: true,
  },
];

/**
 * GET /api/talib/alerts
 * Tüm uyarıları döndürür
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const status = searchParams.get('status') as AlertStatus | null;

    let filteredAlerts = alerts;

    // Symbol filtresi
    if (symbol) {
      filteredAlerts = filteredAlerts.filter(
        (alert) => alert.symbol.toUpperCase() === symbol.toUpperCase()
      );
    }

    // Status filtresi
    if (status) {
      filteredAlerts = filteredAlerts.filter((alert) => alert.status === status);
    }

    // İstatistikler
    const stats = {
      total: alerts.length,
      active: alerts.filter((a) => a.status === 'ACTIVE').length,
      triggered: alerts.filter((a) => a.status === 'TRIGGERED').length,
      paused: alerts.filter((a) => a.status === 'PAUSED').length,
    };

    return NextResponse.json({
      success: true,
      alerts: filteredAlerts,
      stats,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Alerts] GET error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Uyarılar yüklenemedi',
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/talib/alerts
 * Yeni uyarı oluşturur
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const {
      symbol,
      type,
      name,
      description,
      targetPrice,
      changePercent,
      strategyId,
      strategyName,
      signalType,
      minConfidence,
      rsiThreshold,
      rsiCondition,
      volumeMultiplier,
      expiresAt,
      notifyEmail,
      notifyPush,
      notifySound,
    } = body;

    // Validasyon
    if (!symbol || symbol.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Coin sembolü gerekli' },
        { status: 400 }
      );
    }

    if (!type) {
      return NextResponse.json(
        { success: false, error: 'Uyarı tipi gerekli' },
        { status: 400 }
      );
    }

    if (!name || name.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Uyarı adı gerekli' },
        { status: 400 }
      );
    }

    // Tip-spesifik validasyon
    if (type === 'PRICE_ABOVE' || type === 'PRICE_BELOW') {
      if (!targetPrice || targetPrice <= 0) {
        return NextResponse.json(
          { success: false, error: 'Hedef fiyat gerekli' },
          { status: 400 }
        );
      }
    }

    if (type === 'PRICE_CHANGE') {
      if (!changePercent || changePercent <= 0) {
        return NextResponse.json(
          { success: false, error: 'Değişim yüzdesi gerekli' },
          { status: 400 }
        );
      }
    }

    if (type === 'STRATEGY_SIGNAL') {
      if (!strategyId) {
        return NextResponse.json(
          { success: false, error: 'Strateji seçilmeli' },
          { status: 400 }
        );
      }
    }

    if (type === 'RSI') {
      if (!rsiThreshold || !rsiCondition) {
        return NextResponse.json(
          { success: false, error: 'RSI eşik değeri ve koşul gerekli' },
          { status: 400 }
        );
      }
    }

    // Yeni uyarı oluştur
    const newAlert: Alert = {
      id: `alert-${Date.now()}`,
      symbol: symbol.trim().toUpperCase(),
      type,
      status: 'ACTIVE',
      name: name.trim(),
      description: description?.trim(),
      created: new Date().toISOString(),
      targetPrice,
      changePercent,
      strategyId,
      strategyName,
      signalType,
      minConfidence,
      rsiThreshold,
      rsiCondition,
      volumeMultiplier,
      expiresAt,
      notifyEmail,
      notifyPush,
      notifySound,
    };

    alerts.push(newAlert);

    return NextResponse.json({
      success: true,
      alert: newAlert,
      message: 'Uyarı başarıyla oluşturuldu',
    });
  } catch (error: any) {
    console.error('[Alerts] POST error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Uyarı oluşturulamadı',
      },
      { status: 500 }
    );
  }
}

/**
 * PUT /api/talib/alerts
 * Uyarıyı günceller (status değişiklikleri için)
 */
export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const { id, status } = body;

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Uyarı ID gerekli' },
        { status: 400 }
      );
    }

    const alertIndex = alerts.findIndex((a) => a.id === id);

    if (alertIndex === -1) {
      return NextResponse.json(
        { success: false, error: 'Uyarı bulunamadı' },
        { status: 404 }
      );
    }

    // Güncelle
    if (status) {
      alerts[alertIndex].status = status;
    }

    alerts[alertIndex].lastChecked = new Date().toISOString();

    return NextResponse.json({
      success: true,
      alert: alerts[alertIndex],
      message: 'Uyarı güncellendi',
    });
  } catch (error: any) {
    console.error('[Alerts] PUT error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Uyarı güncellenemedi',
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/talib/alerts?id=xxx
 * Uyarıyı siler
 */
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Uyarı ID gerekli' },
        { status: 400 }
      );
    }

    const alertIndex = alerts.findIndex((a) => a.id === id);

    if (alertIndex === -1) {
      return NextResponse.json(
        { success: false, error: 'Uyarı bulunamadı' },
        { status: 404 }
      );
    }

    // Sil
    const deleted = alerts.splice(alertIndex, 1)[0];

    return NextResponse.json({
      success: true,
      deleted,
      message: 'Uyarı silindi',
    });
  } catch (error: any) {
    console.error('[Alerts] DELETE error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Uyarı silinemedi',
      },
      { status: 500 }
    );
  }
}

/**
 * Background alert checker (gerçek zamanlı uyarı kontrolü)
 * Gerçek uygulamada bu bir cron job veya WebSocket ile çalışmalı
 */
export async function checkAlerts(currentPrices: Record<string, number>) {
  const triggeredAlerts: Alert[] = [];

  for (const alert of alerts) {
    if (alert.status !== 'ACTIVE') continue;

    const currentPrice = currentPrices[alert.symbol];
    if (!currentPrice) continue;

    let shouldTrigger = false;

    switch (alert.type) {
      case 'PRICE_ABOVE':
        shouldTrigger = alert.targetPrice ? currentPrice >= alert.targetPrice : false;
        break;

      case 'PRICE_BELOW':
        shouldTrigger = alert.targetPrice ? currentPrice <= alert.targetPrice : false;
        break;

      // Diğer tip kontrolleri buraya eklenebilir
    }

    if (shouldTrigger) {
      alert.status = 'TRIGGERED';
      alert.triggered = new Date().toISOString();
      triggeredAlerts.push(alert);
    }
  }

  return triggeredAlerts;
}
