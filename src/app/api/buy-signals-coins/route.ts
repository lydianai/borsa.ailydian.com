/**
 * BUY SIGNALS COINS API
 *
 * Tüm strateji API'lerden BUY sinyali veren coinleri toplar
 * Quantum Ladder sayfası için coin rotation
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface Signal {
  symbol: string;
  type: string;
  confidence: number;
}

/**
 * Fetch signals from an endpoint
 */
async function fetchSignals(endpoint: string): Promise<Signal[]> {
  try {
    const response = await fetch(`http://localhost:3000${endpoint}`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-store',
    });

    if (!response.ok) {
      console.warn(`[BUY Signals] ${endpoint} returned ${response.status}`);
      return [];
    }

    const data = await response.json();

    if (data.success && data.data && data.data.signals) {
      return data.data.signals;
    }

    return [];
  } catch (error: any) {
    console.warn(`[BUY Signals] Error fetching ${endpoint}:`, error.message);
    return [];
  }
}

export async function GET(_request: NextRequest) {
  try {
    console.log('[BUY Signals] Fetching all BUY signals...');

    // Binance'den mevcut coinleri çek ve paralel olarak stratejilerden sinyalleri al
    const [
      binanceData,
      aiSignals,
      quantumSignals,
      conservativeSignals,
      breakoutSignals,
      tradingSignals,
    ] = await Promise.all([
      fetch('http://localhost:3000/api/binance/futures', { cache: 'no-store' }).then(r => r.json()),
      fetchSignals('/api/ai-signals'),
      fetchSignals('/api/quantum-signals'),
      fetchSignals('/api/conservative-signals'),
      fetchSignals('/api/breakout-retest'),
      fetchSignals('/api/signals?limit=200'),
    ]);

    // Binance'de mevcut coinleri set'e koy (hızlı lookup için)
    const availableCoins = new Set<string>();
    if (binanceData.success && binanceData.data && binanceData.data.all) {
      binanceData.data.all.forEach((coin: any) => {
        availableCoins.add(coin.symbol);
      });
    }
    console.log(`[BUY Signals] Found ${availableCoins.size} available coins on Binance`);

    // Tüm sinyalleri birleştir
    const allSignals = [
      ...aiSignals,
      ...quantumSignals,
      ...conservativeSignals,
      ...breakoutSignals,
      ...tradingSignals,
    ];

    // Sadece BUY sinyallerini filtrele
    const buySignals = allSignals.filter((signal: Signal) => {
      const type = signal.type?.toLowerCase();
      return type === 'buy' || type === 'strong_buy';
    });

    // Unique coinleri topla ve confidence'a göre sırala
    const coinMap = new Map<string, { symbol: string; confidence: number; count: number }>();

    buySignals.forEach((signal: Signal) => {
      // USDT suffix'ini kontrol et ve ekle
      let symbol = signal.symbol;
      if (!symbol.endsWith('USDT')) {
        symbol = symbol + 'USDT';
      }

      // Sadece Binance'de mevcut olan coinleri ekle
      if (!availableCoins.has(symbol)) {
        console.log(`[BUY Signals] Skipping ${symbol} - not available on Binance`);
        return;
      }

      const existing = coinMap.get(symbol);
      if (existing) {
        // Coin zaten var - confidence ortalaması al ve count artır
        existing.confidence = (existing.confidence + signal.confidence) / 2;
        existing.count += 1;
      } else {
        // Yeni coin ekle
        coinMap.set(symbol, {
          symbol: symbol,
          confidence: signal.confidence,
          count: 1,
        });
      }
    });

    // Map'i array'e çevir ve confidence'a göre sırala
    const buyCoins = Array.from(coinMap.values())
      .sort((a, b) => b.confidence - a.confidence);

    console.log(`[BUY Signals] Found ${buyCoins.length} coins with BUY signals`);

    return NextResponse.json({
      success: true,
      data: {
        coins: buyCoins,
        totalCoins: buyCoins.length,
        timestamp: Date.now(),
      },
    });
  } catch (error: any) {
    console.error('[BUY Signals] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch BUY signals',
      },
      { status: 500 }
    );
  }
}
