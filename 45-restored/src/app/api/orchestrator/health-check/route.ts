import { NextRequest, NextResponse } from 'next/server';
import { getOrchestrator } from '@/services/orchestrator/UnifiedRobotOrchestrator';

export async function POST(request: NextRequest) {
  try {
    const orchestrator = getOrchestrator();
    const results = await orchestrator.performHealthChecks();

    const summary = {
      total: results.length,
      healthy: results.filter(r => r.status === 'healthy').length,
      degraded: results.filter(r => r.status === 'degraded').length,
      unhealthy: results.filter(r => r.status === 'unhealthy').length
    };

    return NextResponse.json({
      success: true,
      summary,
      results,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Health check error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
