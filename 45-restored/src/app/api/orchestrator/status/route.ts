import { NextRequest, NextResponse } from 'next/server';
import { getOrchestrator } from '@/services/orchestrator/UnifiedRobotOrchestrator';

export async function GET(request: NextRequest) {
  try {
    const orchestrator = getOrchestrator();
    const status = orchestrator.getStatus();

    return NextResponse.json({
      success: true,
      orchestrator: status,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Orchestrator status error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
