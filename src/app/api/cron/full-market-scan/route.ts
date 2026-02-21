/**
 * 🟢 VERCEL CRON JOB - Full-Market Scan
 * Schedule: Every hour (at minute 0)
 * Cron: `0 * * * *`
 *
 * Vercel tarafından her 1 saatte bir çağrılır.
 * Tüm 600+ coin'i tarar + AI öğrenme aktif.
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
    console.log(`\n⏰ [${timestamp}] Vercel Cron: Full-Market Scan triggered`);

    const autonomousQueue = getAutonomousQueue();
    const job = await autonomousQueue.add('full-market-scan', {
      timeframe: '1h',
      minVolume: 1_000_000, // $1M+
      coinCount: 600,
      aiEnhanced: true,
      strategyLearning: true, // AI öğrenme aktif
      triggeredBy: 'vercel-cron',
      timestamp,
    });

    console.log(`✅ Job enqueued: ${job.id}`);

    return NextResponse.json({
      success: true,
      jobId: job.id,
      type: 'full-market-scan',
      timestamp,
    });
  } catch (error: any) {
    console.error('❌ Full-Market Scan cron error:', error.message);

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
