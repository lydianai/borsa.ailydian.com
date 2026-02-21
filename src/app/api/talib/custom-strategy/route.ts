import { NextResponse } from 'next/server';

/**
 * 🎯 CUSTOM STRATEGY BUILDER API
 * Kullanıcının özel stratejiler oluşturmasını sağlar
 */

export interface StrategyIndicator {
  name: string;
  enabled: boolean;
  weight: number; // 0-100
  params?: Record<string, number>;
}

export interface CustomStrategy {
  id: string;
  name: string;
  description: string;
  indicators: StrategyIndicator[];
  created: string;
  lastUsed?: string;
  performance?: {
    winRate: number;
    totalTrades: number;
    avgProfit: number;
  };
}

// In-memory storage (gerçek uygulamada database kullanılmalı)
let customStrategies: CustomStrategy[] = [
  {
    id: 'default-balanced',
    name: 'Dengeli Strateji',
    description: 'RSI, MACD ve Bollinger Bands kombinasyonu',
    indicators: [
      { name: 'RSI', enabled: true, weight: 30, params: { period: 14, oversold: 30, overbought: 70 } },
      { name: 'MACD', enabled: true, weight: 25, params: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 } },
      { name: 'Bollinger Bands', enabled: true, weight: 20, params: { period: 20, stdDev: 2 } },
      { name: 'Moving Average', enabled: true, weight: 15, params: { shortPeriod: 50, longPeriod: 200 } },
      { name: 'Volume', enabled: true, weight: 10, params: { multiplier: 1.5 } },
    ],
    created: new Date().toISOString(),
    performance: {
      winRate: 68,
      totalTrades: 45,
      avgProfit: 3.2,
    },
  },
  {
    id: 'aggressive-momentum',
    name: 'Agresif Momentum',
    description: 'Yüksek momentum odaklı strateji',
    indicators: [
      { name: 'RSI', enabled: true, weight: 20, params: { period: 14, oversold: 35, overbought: 65 } },
      { name: 'MACD', enabled: true, weight: 35, params: { fastPeriod: 8, slowPeriod: 21, signalPeriod: 5 } },
      { name: 'Momentum', enabled: true, weight: 30, params: { period: 10 } },
      { name: 'Volume', enabled: true, weight: 15, params: { multiplier: 2 } },
    ],
    created: new Date().toISOString(),
    performance: {
      winRate: 72,
      totalTrades: 38,
      avgProfit: 4.5,
    },
  },
  {
    id: 'conservative-trend',
    name: 'Muhafazakar Trend',
    description: 'Düşük riskli, trend takip stratejisi',
    indicators: [
      { name: 'Moving Average', enabled: true, weight: 40, params: { shortPeriod: 50, longPeriod: 200 } },
      { name: 'RSI', enabled: true, weight: 25, params: { period: 14, oversold: 25, overbought: 75 } },
      { name: 'Bollinger Bands', enabled: true, weight: 20, params: { period: 20, stdDev: 2.5 } },
      { name: 'Support/Resistance', enabled: true, weight: 15, params: { lookback: 50 } },
    ],
    created: new Date().toISOString(),
    performance: {
      winRate: 75,
      totalTrades: 52,
      avgProfit: 2.8,
    },
  },
];

// Mevcut göstergeler listesi
const availableIndicators: StrategyIndicator[] = [
  { name: 'RSI', enabled: false, weight: 20, params: { period: 14, oversold: 30, overbought: 70 } },
  { name: 'MACD', enabled: false, weight: 20, params: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 } },
  { name: 'Bollinger Bands', enabled: false, weight: 15, params: { period: 20, stdDev: 2 } },
  { name: 'Moving Average', enabled: false, weight: 15, params: { shortPeriod: 50, longPeriod: 200 } },
  { name: 'Volume', enabled: false, weight: 10, params: { multiplier: 1.5 } },
  { name: 'Momentum', enabled: false, weight: 10, params: { period: 10 } },
  { name: 'Support/Resistance', enabled: false, weight: 10, params: { lookback: 50 } },
  { name: 'Trend Analysis', enabled: false, weight: 10, params: { sensitivity: 0.7 } },
  { name: 'Volatility', enabled: false, weight: 10, params: { period: 20 } },
  { name: 'Fibonacci', enabled: false, weight: 10, params: { levels: 5 } },
];

/**
 * GET /api/talib/custom-strategy
 * Tüm özel stratejileri ve mevcut göstergeleri döndürür
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');

    if (action === 'indicators') {
      // Sadece mevcut göstergeleri döndür
      return NextResponse.json({
        success: true,
        indicators: availableIndicators,
      });
    }

    // Tüm stratejileri döndür
    return NextResponse.json({
      success: true,
      strategies: customStrategies,
      availableIndicators,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Custom Strategy] GET error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Stratejiler yüklenemedi',
      },
      { status: 500 }
    );
  }
}

/**
 * POST /api/talib/custom-strategy
 * Yeni özel strateji oluşturur
 */
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name, description, indicators } = body;

    // Validasyon
    if (!name || name.trim().length === 0) {
      return NextResponse.json(
        { success: false, error: 'Strateji adı gerekli' },
        { status: 400 }
      );
    }

    if (!indicators || indicators.length === 0) {
      return NextResponse.json(
        { success: false, error: 'En az bir gösterge seçilmeli' },
        { status: 400 }
      );
    }

    // Toplam ağırlık kontrolü
    const totalWeight = indicators
      .filter((ind: StrategyIndicator) => ind.enabled)
      .reduce((sum: number, ind: StrategyIndicator) => sum + ind.weight, 0);

    if (totalWeight !== 100) {
      return NextResponse.json(
        {
          success: false,
          error: `Toplam ağırlık 100 olmalı (şu an: ${totalWeight})`,
        },
        { status: 400 }
      );
    }

    // Yeni strateji oluştur
    const newStrategy: CustomStrategy = {
      id: `custom-${Date.now()}`,
      name: name.trim(),
      description: description?.trim() || '',
      indicators,
      created: new Date().toISOString(),
    };

    customStrategies.push(newStrategy);

    return NextResponse.json({
      success: true,
      strategy: newStrategy,
      message: 'Strateji başarıyla oluşturuldu',
    });
  } catch (error: any) {
    console.error('[Custom Strategy] POST error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Strateji oluşturulamadı',
      },
      { status: 500 }
    );
  }
}

/**
 * PUT /api/talib/custom-strategy
 * Mevcut stratejiyi günceller
 */
export async function PUT(request: Request) {
  try {
    const body = await request.json();
    const { id, name, description, indicators } = body;

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Strateji ID gerekli' },
        { status: 400 }
      );
    }

    const strategyIndex = customStrategies.findIndex((s) => s.id === id);

    if (strategyIndex === -1) {
      return NextResponse.json(
        { success: false, error: 'Strateji bulunamadı' },
        { status: 404 }
      );
    }

    // Güncelle
    customStrategies[strategyIndex] = {
      ...customStrategies[strategyIndex],
      name: name || customStrategies[strategyIndex].name,
      description: description || customStrategies[strategyIndex].description,
      indicators: indicators || customStrategies[strategyIndex].indicators,
      lastUsed: new Date().toISOString(),
    };

    return NextResponse.json({
      success: true,
      strategy: customStrategies[strategyIndex],
      message: 'Strateji güncellendi',
    });
  } catch (error: any) {
    console.error('[Custom Strategy] PUT error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Strateji güncellenemedi',
      },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/talib/custom-strategy?id=xxx
 * Stratejiyi siler
 */
export async function DELETE(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json(
        { success: false, error: 'Strateji ID gerekli' },
        { status: 400 }
      );
    }

    const strategyIndex = customStrategies.findIndex((s) => s.id === id);

    if (strategyIndex === -1) {
      return NextResponse.json(
        { success: false, error: 'Strateji bulunamadı' },
        { status: 404 }
      );
    }

    // Sil
    const deleted = customStrategies.splice(strategyIndex, 1)[0];

    return NextResponse.json({
      success: true,
      deleted,
      message: 'Strateji silindi',
    });
  } catch (error: any) {
    console.error('[Custom Strategy] DELETE error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Strateji silinemedi',
      },
      { status: 500 }
    );
  }
}
