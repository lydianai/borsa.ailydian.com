/**
 * Performance Monitoring API
 *
 * White-hat compliance: Provides performance metrics for system health
 */

import { NextRequest, NextResponse } from 'next/server';
import performanceMonitor from '@/lib/monitoring/performance';

export const dynamic = 'force-dynamic';

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const period = parseInt(searchParams.get('period') || '3600000'); // Default 1 hour
    const format = searchParams.get('format') as 'json' | 'prometheus' || 'json';

    if (format === 'prometheus') {
      const metrics = performanceMonitor.exportMetrics('prometheus');
      return new NextResponse(metrics, {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
        },
      });
    }

    const report = performanceMonitor.getReport(period);

    return NextResponse.json({
      success: true,
      data: report,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('Performance monitoring error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to get performance metrics',
      },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { name, value, unit, metadata } = body;

    if (!name || value === undefined) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: name, value',
        },
        { status: 400 }
      );
    }

    performanceMonitor.record(name, value, unit || 'ms', metadata);

    return NextResponse.json({
      success: true,
      message: 'Metric recorded',
    });
  } catch (error) {
    console.error('Performance recording error:', error);
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to record metric',
      },
      { status: 500 }
    );
  }
}
