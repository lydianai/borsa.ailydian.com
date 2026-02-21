/**
 * AUTO TRADING API
 * Otomatik trading motorunu kontrol eden API endpoint
 * NOW WITH TOP 100 COINS SUPPORT (Binance + CoinGecko)
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAutoTradingEngine, TradingConfig } from '@/services/AutoTradingEngine';
import { getMarketDataService } from '@/services/MarketDataService';

export const dynamic = 'force-dynamic';

// Get Top 100 trading pairs dynamically
async function getTop100TradingPairs(): Promise<string[]> {
  try {
    const marketService = getMarketDataService();
    const pairs = await marketService.getTradingPairs();
    console.log(`✅ Loaded ${pairs.length} trading pairs from Top 100`);
    return pairs;
  } catch (error) {
    console.error('❌ Error loading top 100 pairs, using fallback:', error);
    // Fallback to top 20
    return [
      'BTC', 'ETH', 'USDT', 'BNB', 'SOL', 'USDC', 'XRP', 'DOGE', 'ADA', 'TRX',
      'AVAX', 'SHIB', 'DOT', 'LINK', 'MATIC', 'BCH', 'UNI', 'LTC', 'ICP', 'NEAR',
    ];
  }
}

// Default trading configuration (will be populated with top 100)
let defaultConfig: TradingConfig | null = null;

async function getDefaultConfig(): Promise<TradingConfig> {
  if (!defaultConfig) {
    const tradingPairs = await getTop100TradingPairs();

    defaultConfig = {
      enabled: false,
      mode: 'paper', // paper trading ile başla
      maxPositionSize: 100, // $100 per position
      maxDailyLoss: 2, // 2% max daily loss
      maxLeverage: 3, // 3x leverage
      tradingPairs: tradingPairs, // ✨ TOP 100 COINS
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
export async function GET(request: NextRequest) {
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
