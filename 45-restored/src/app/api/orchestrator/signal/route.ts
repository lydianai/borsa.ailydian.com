import { NextRequest, NextResponse } from 'next/server';
import { getOrchestrator } from '@/services/orchestrator/UnifiedRobotOrchestrator';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol } = body;

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol is required' },
        { status: 400 }
      );
    }

    const orchestrator = getOrchestrator();
    const consensus = await orchestrator.generateConsensusSignal(symbol);

    if (!consensus) {
      return NextResponse.json(
        { success: false, error: 'Failed to generate consensus signal' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      consensus,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Generate signal error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
