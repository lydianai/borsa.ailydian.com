import { NextRequest, NextResponse } from 'next/server';
import { UnifiedTradingBot } from '@/services/UnifiedTradingBot';

let botInstance: UnifiedTradingBot;

function getBotInstance() {
  if (!botInstance) {
    botInstance = UnifiedTradingBot.getInstance();
  }
  return botInstance;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');
    const minConfidence = parseFloat(searchParams.get('minConfidence') || '0.7');

    const bot = getBotInstance();

    switch (action) {
      case 'status':
        return NextResponse.json({
          success: true,
          data: bot.getStatus()
        });

      case 'metrics':
        return NextResponse.json({
          success: true,
          metrics: bot.getMetrics()
        });

      case 'signals':
        const signals = bot.getSignals(minConfidence);
        return NextResponse.json({
          success: true,
          count: signals.length,
          signals: signals,
          timestamp: Date.now()
        });

      default:
        return NextResponse.json({
          success: true,
          data: bot.getStatus(),
          signals: bot.getSignals(minConfidence)
        });
    }

  } catch (error: any) {
    console.error('Unified Bot API Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Internal server error'
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    const bot = getBotInstance();

    switch (action) {
      case 'pause':
        bot.pause();
        return NextResponse.json({
          success: true,
          message: 'Bot paused',
          status: bot.getMetrics()
        });

      case 'resume':
        bot.resume();
        return NextResponse.json({
          success: true,
          message: 'Bot resumed',
          status: bot.getMetrics()
        });

      default:
        return NextResponse.json({
          success: false,
          error: 'Invalid action'
        }, { status: 400 });
    }

  } catch (error: any) {
    console.error('Unified Bot Control Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Internal server error'
    }, { status: 500 });
  }
}
