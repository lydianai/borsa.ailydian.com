/**
 * TRADING SIGNALS GENERATOR API
 * Generate BUY/SELL/HOLD signals with confidence scores
 * GET /api/trading/signals?symbol=BTC
 * GET /api/trading/signals?top=10 (for Top N coins)
 */

import { NextRequest, NextResponse } from 'next/server';
import { masterIntegrationService } from '@/services/integration/MasterIntegrationService';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const top = searchParams.get('top');

    // Single symbol signal
    if (symbol) {
      console.log(`🎯 Generating signal for ${symbol}`);

      const signal = await masterIntegrationService.generateTradingSignal(symbol);

      if (!signal) {
        return NextResponse.json(
          { success: false, error: `Failed to generate signal for ${symbol}` },
          { status: 404 }
        );
      }

      return NextResponse.json({
        success: true,
        symbol,
        signal,
        timestamp: Date.now(),
      });
    }

    // Top N coins signals
    if (top) {
      const limit = Math.min(parseInt(top), 50);
      console.log(`🎯 Generating signals for Top ${limit} coins`);

      const startTime = Date.now();

      // Get Top 100 comprehensive data
      const comprehensiveData = await masterIntegrationService.getTop100ComprehensiveData('1h', limit);

      // Generate signals for each
      const signalsPromises = comprehensiveData.map(async (data) => {
        try {
          const signal = await masterIntegrationService.generateTradingSignal(data.coin.symbol);
          return {
            symbol: data.coin.symbol,
            name: data.coin.name,
            price: data.coin.price,
            signal,
          };
        } catch (error) {
          return null;
        }
      });

      const signals = (await Promise.all(signalsPromises)).filter(s => s !== null);

      const processingTime = Date.now() - startTime;

      console.log(`✅ Generated ${signals.length} signals in ${processingTime}ms`);

      // Sort by confidence (highest first)
      signals.sort((a, b) => (b?.signal?.confidence || 0) - (a?.signal?.confidence || 0));

      return NextResponse.json({
        success: true,
        limit,
        count: signals.length,
        signals,
        processingTime,
        timestamp: Date.now(),
      });
    }

    return NextResponse.json(
      { success: false, error: 'Either symbol or top parameter is required' },
      { status: 400 }
    );
  } catch (error: any) {
    console.error('❌ Error in signals API:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

// POST - Generate signal for a specific symbol with AI ensemble
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, timeframe = '1h' } = body;

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol is required' },
        { status: 400 }
      );
    }

    console.log(`🎯 POST: Generating ensemble signal for ${symbol} [${timeframe}]`);

    // Generate comprehensive trading signal with all 6 bots
    const signal = await masterIntegrationService.generateTradingSignal(symbol, timeframe);

    if (!signal) {
      return NextResponse.json(
        { success: false, error: `Failed to generate signal for ${symbol}` },
        { status: 404 }
      );
    }

    // Get current price data
    const priceData = await fetch(`https://api.binance.com/api/v3/ticker/24hr?symbol=${symbol}USDT`)
      .then(res => res.json())
      .catch(() => null);

    // Mock bot signals (in production, this would come from actual AI models)
    const botSignals = [
      {
        bot_id: 'quantum-pro',
        bot_name: 'Quantum Pro AI',
        signal: signal.action,
        confidence: signal.confidence,
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.97 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.08 : 0,
        risk_reward_ratio: 2.67,
        reasoning: `Quantum Pro AI analysis: ${signal.reasoning || 'Strong technical indicators'}`,
      },
      {
        bot_id: 'master-orchestrator',
        bot_name: 'Master AI Orchestrator',
        signal: signal.action,
        confidence: Math.max(0.5, signal.confidence - 0.05),
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.96 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.10 : 0,
        risk_reward_ratio: 2.75,
        reasoning: 'Multi-model ensemble consensus',
      },
      {
        bot_id: 'attention-transformer',
        bot_name: 'Attention Transformer',
        signal: signal.action,
        confidence: Math.max(0.5, signal.confidence - 0.08),
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.975 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.075 : 0,
        risk_reward_ratio: 3.0,
        reasoning: 'Deep learning pattern recognition',
      },
      {
        bot_id: 'gradient-boosting',
        bot_name: 'Gradient Boosting Engine',
        signal: signal.action,
        confidence: Math.max(0.5, signal.confidence - 0.1),
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.98 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.06 : 0,
        risk_reward_ratio: 2.0,
        reasoning: 'XGBoost feature importance analysis',
      },
      {
        bot_id: 'reinforcement-learning',
        bot_name: 'Reinforcement Learning Agent',
        signal: Math.random() > 0.5 ? signal.action : 'HOLD',
        confidence: Math.max(0.5, signal.confidence - 0.15),
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.97 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.09 : 0,
        risk_reward_ratio: 3.0,
        reasoning: 'Q-learning optimal policy',
      },
      {
        bot_id: 'tensorflow-optimizer',
        bot_name: 'TensorFlow Optimizer',
        signal: signal.action,
        confidence: Math.max(0.5, signal.confidence - 0.06),
        entry_price: priceData ? parseFloat(priceData.lastPrice) : 0,
        stop_loss: priceData ? parseFloat(priceData.lastPrice) * 0.97 : 0,
        take_profit: priceData ? parseFloat(priceData.lastPrice) * 1.085 : 0,
        risk_reward_ratio: 2.83,
        reasoning: 'Neural network prediction',
      },
    ];

    return NextResponse.json({
      success: true,
      symbol,
      timeframe,
      ensemble_signal: signal.action,
      ensemble_confidence: signal.confidence,
      signals: botSignals,
      timestamp: Date.now(),
    });

  } catch (error: any) {
    console.error('❌ Error in POST signals API:', error.message);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}
