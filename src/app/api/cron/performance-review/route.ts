/**
 * 🔵 VERCEL CRON JOB - Strategy Performance Review
 * Schedule: Every 4 hours (at minute 0)
 * Cron: `0 * /4 * * *` (remove space)
 *
 * Vercel tarafından her 4 saatte bir çağrılır.
 * Strateji performansını analiz eder ve ağırlıkları günceller.
 *
 * Security: CRON_SECRET ile korunur
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAutonomousQueue } from '@/lib/queue/autonomous-queue';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
export const maxDuration = 60;

export async function GET(request: NextRequest) {
  try {
    // Güvenlik kontrolü
    const authHeader = request.headers.get('authorization');
    const cronSecret = process.env.CRON_SECRET;

    if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
      console.error('❌ Unauthorized cron request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] Vercel Cron: Strategy Performance Review triggered`);

    const autonomousQueue = getAutonomousQueue();
    const job = await autonomousQueue.add('strategy-performance-review', {
      analyzeLastHours: 24,
      updateWeights: true, // Strateji ağırlıklarını güncelle
      generateNewParams: true, // Yeni parametreler öner
      triggeredBy: 'vercel-cron',
      timestamp,
    });

    console.log(`✅ Job enqueued: ${job.id}`);

    return NextResponse.json({
      success: true,
      jobId: job.id,
      type: 'strategy-performance-review',
      timestamp,
    });
  } catch (error: any) {
    console.error('❌ Performance Review cron error:', error.message);

    return NextResponse.json(
      {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
