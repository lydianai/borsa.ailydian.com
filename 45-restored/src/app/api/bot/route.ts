/**
 * TRADING BOT API
 * Manage automated trading bots (PAPER TRADING ONLY)
 */

import { NextRequest, NextResponse } from 'next/server';
import { getTradingBotEngine } from '@/services/bot/TradingBotEngine';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const botEngine = getTradingBotEngine();

/**
 * GET - List all bots and their status
 */
export async function GET() {
  try {
    const bots = botEngine.getBots();
    const positions = botEngine.getPositions();

    return NextResponse.json({
      success: true,
      bots: bots.map(bot => ({
        ...bot,
        stats: botEngine.getStats(bot.id),
      })),
      positions,
      summary: {
        totalBots: bots.length,
        activeBots: bots.filter(b => b.enabled).length,
        openPositions: positions.filter(p => p.status === 'open').length,
        totalPositions: positions.length,
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

/**
 * POST - Create a new bot
 */
export async function POST(request: NextRequest) {
  try {
    const config = await request.json();

    // Create bot
    const bot = botEngine.createBot(config);

    return NextResponse.json({
      success: true,
      bot,
      message: 'Bot created successfully (PAPER TRADING MODE)',
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 400 }
    );
  }
}

/**
 * PUT - Start/Stop bot engine
 */
export async function PUT(request: NextRequest) {
  try {
    const { action } = await request.json();

    if (action === 'start') {
      botEngine.start();
      return NextResponse.json({
        success: true,
        message: 'Bot engine started (PAPER TRADING)',
      });
    } else if (action === 'stop') {
      botEngine.stop();
      return NextResponse.json({
        success: true,
        message: 'Bot engine stopped',
      });
    } else {
      return NextResponse.json(
        { success: false, error: 'Invalid action. Use "start" or "stop"' },
        { status: 400 }
      );
    }
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
