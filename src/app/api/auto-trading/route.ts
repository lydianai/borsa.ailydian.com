/**
 * AUTO TRADING API
 * Otomatik trading motorunu kontrol eden API endpoint
 * TÜM BINANCE FUTURES USDT-M PAİRLERİ (200+ coin)
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAutoTradingEngine, TradingConfig } from '@/services-45backend/AutoTradingEngine';

export const dynamic = 'force-dynamic';

// Binance Futures USDT-M tüm çiftleri al
async function getAllBinanceFuturesPairs(): Promise<string[]> {
  try {
    const response = await fetch('https://fapi.binance.com/fapi/v1/exchangeInfo');

    if (!response.ok) {
      throw new Error(`Binance Futures API error: ${response.status}`);
    }

    const data = await response.json();

    // Sadece USDT-M perpetual futures'ları al (aktif olanlar)
    const usdtPairs = data.symbols
      .filter((s: any) =>
        s.symbol.endsWith('USDT') &&
        s.contractType === 'PERPETUAL' &&
        s.status === 'TRADING'
      )
      .map((s: any) => s.symbol.replace('USDT', '')); // BTCUSDT -> BTC

    console.log(`✅ Loaded ${usdtPairs.length} Binance Futures USDT-M pairs`);
    return usdtPairs;
  } catch (error) {
    console.error('❌ Error loading Binance Futures pairs, using fallback:', error);
    // Fallback to top coins
    return [
      'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'DOGE', 'ADA', 'TRX',
      'AVAX', 'SHIB', 'DOT', 'LINK', 'MATIC', 'BCH', 'UNI', 'LTC',
    ];
  }
}

// Default trading configuration (will be populated with ALL futures pairs)
let defaultConfig: TradingConfig | null = null;

async function getDefaultConfig(): Promise<TradingConfig> {
  if (!defaultConfig) {
    const tradingPairs = await getAllBinanceFuturesPairs();

    defaultConfig = {
      enabled: false,
      mode: 'paper', // paper trading ile başla
      maxPositionSize: 100, // $100 per position
      maxDailyLoss: 2, // 2% max daily loss
      maxLeverage: 3, // 3x leverage
      tradingPairs: tradingPairs, // ✨ TÜM BINANCE FUTURES USDT-M COINS
      aiBotsEnabled: {
        quantumPro: true,
        masterOrchestrator: true,
        attentionTransformer: true,
        gradientBoosting: true,
        reinforcementLearning: true,
        tensorflowOptimizer: true,
      },
      riskManagement: {
        stopLossPercent: 2, // 2% stop loss
        takeProfitPercent: 5, // 5% take profit
        trailingStopPercent: 1, // 1% trailing stop
        maxConcurrentTrades: 5,
      },
    };

    console.log(`🎯 Auto Trading Config initialized with ${tradingPairs.length} pairs`);
  }

  return defaultConfig;
}

// GET - Motor durumunu al
export async function GET(_request: NextRequest) {
  try {
    const config = await getDefaultConfig();
    const engine = getAutoTradingEngine(config);
    const status = engine.getStatus();

    return NextResponse.json({
      success: true,
      data: status,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}

// POST - Motor kontrolü (start/stop/config)
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, config } = body;

    const defaultConf = await getDefaultConfig();
    const engine = getAutoTradingEngine(defaultConf);

    switch (action) {
      case 'start':
        await engine.start();
        return NextResponse.json({
          success: true,
          message: 'Auto trading engine started',
          status: engine.getStatus(),
        });

      case 'stop':
        await engine.stop();
        return NextResponse.json({
          success: true,
          message: 'Auto trading engine stopped',
          status: engine.getStatus(),
        });

      case 'updateConfig':
        if (config) {
          engine.updateConfig(config);
          return NextResponse.json({
            success: true,
            message: 'Configuration updated',
            status: engine.getStatus(),
          });
        }
        break;

      case 'getStatus':
        return NextResponse.json({
          success: true,
          status: engine.getStatus(),
        });

      default:
        return NextResponse.json(
          {
            success: false,
            error: `Unknown action: ${action}`,
          },
          { status: 400 }
        );
    }

    return NextResponse.json(
      {
        success: false,
        error: 'Invalid request',
      },
      { status: 400 }
    );
  } catch (error: any) {
    console.error('Auto Trading API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
