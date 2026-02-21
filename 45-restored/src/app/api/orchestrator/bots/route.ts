import { NextRequest, NextResponse } from 'next/server';
import { getOrchestrator } from '@/services/orchestrator/UnifiedRobotOrchestrator';

export async function GET(request: NextRequest) {
  try {
    const orchestrator = getOrchestrator();
    const bots = orchestrator.getAllBots();

    return NextResponse.json({
      success: true,
      bots: bots.map(bot => ({
        id: bot.id,
        name: bot.name,
        type: bot.type,
        status: bot.status,
        healthScore: bot.healthScore,
        lastHeartbeat: bot.lastHeartbeat,
        performance: bot.performance
      })),
      total: bots.length,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] List bots error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
