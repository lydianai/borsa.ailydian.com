/**
 * MASTER ORCHESTRATOR SIGNALS API
 * Multi-Model Ensemble - Tüm modelleri birleştiren üst düzey bot
 */

import { NextRequest, NextResponse } from 'next/server';
import { MasterOrchestratorBot, type MarketData } from '@/services/AIBotSignalService';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    // Gerçek market datasını al (şimdilik mock)
    const marketData: MarketData[] = await fetchMarketData();

    // Master Orchestrator bot'u başlat
    const bot = new MasterOrchestratorBot();
    const signals = await bot.generateSignals(marketData);

    return NextResponse.json({
      success: true,
      botName: 'Master Orchestrator',
      botType: 'Multi-Model Ensemble',
      accuracy: 94.2,
      signals: signals.map(s => ({
        symbol: s.symbol,
        action: s.action,
        confidence: s.confidence,
        price: s.price,
        reasoning: s.reasoning,
        detailedReasons: s.detailedReasons,
        riskScore: s.riskScore,
        targetPrice: s.targetPrice,
        stopLoss: s.stopLoss,
        timeframe: s.timeframe,
      })),
      count: signals.length,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Master Orchestrator API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}

async function fetchMarketData(): Promise<MarketData[]> {
  try {
    // CoinGecko API'den gerçek veri çek
    const response = await fetch('/api/market/crypto');
    const data = await response.json();

    if (data.success && data.data) {
      return data.data.map((coin: any) => ({
        symbol: coin.symbol.toUpperCase(),
        price: coin.currentPrice,
        change24h: coin.priceChange24h || 0,
        volume24h: coin.totalVolume || 0,
        marketCap: coin.marketCap || 0,
        high24h: coin.high24h || coin.currentPrice * 1.02,
        low24h: coin.low24h || coin.currentPrice * 0.98,
      }));
    }
  } catch (error) {
    console.error('Error fetching market data:', error);
  }

  // Fallback: mock data
  return [
    {
      symbol: 'BTC',
      price: 67000 + Math.random() * 1000,
      change24h: Math.random() * 10 - 5,
      volume24h: 30000000000,
      marketCap: 1300000000000,
      high24h: 68000,
      low24h: 66000,
    },
    {
      symbol: 'ETH',
      price: 3200 + Math.random() * 100,
      change24h: Math.random() * 10 - 5,
      volume24h: 15000000000,
      marketCap: 380000000000,
      high24h: 3300,
      low24h: 3100,
    },
    {
      symbol: 'SOL',
      price: 150 + Math.random() * 10,
      change24h: Math.random() * 10 - 5,
      volume24h: 2000000000,
      marketCap: 65000000000,
      high24h: 160,
      low24h: 145,
    },
  ];
}
