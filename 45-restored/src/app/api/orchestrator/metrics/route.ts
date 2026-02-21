import { NextRequest, NextResponse } from 'next/server';
import { getPerformanceMonitor } from '@/services/orchestrator/PerformanceMonitor';

export async function GET(request: NextRequest) {
  try {
    const monitor = getPerformanceMonitor();
    const report = monitor.getReport();

    return NextResponse.json({
      success: true,
      report,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Performance report error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
