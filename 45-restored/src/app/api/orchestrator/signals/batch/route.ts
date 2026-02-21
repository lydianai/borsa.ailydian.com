import { NextRequest, NextResponse } from 'next/server';
import { getOrchestrator } from '@/services/orchestrator/UnifiedRobotOrchestrator';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbols } = body;

    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Symbols array is required' },
        { status: 400 }
      );
    }

    const orchestrator = getOrchestrator();
    const results = [];

    for (const symbol of symbols) {
      try {
        const consensus = await orchestrator.generateConsensusSignal(symbol);
        if (consensus) {
          results.push(consensus);
        }
      } catch (error) {
        console.error(`[API] Failed to generate signal for ${symbol}:`, error);
      }
    }

    return NextResponse.json({
      success: true,
      signals: results,
      total: results.length,
      requested: symbols.length,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Batch signals error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
