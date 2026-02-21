/**
 * 🟠 VERCEL CRON JOB - Medium-Volatility Scan
 * Schedule: Every 15 minutes
 * Cron: `* /15 * * * *` (remove space)
 *
 * Vercel tarafından her 15 dakikada bir çağrılır.
 * $5M+ hacimli 100 coin'i tarar.
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
    console.log(`\n⏰ [${timestamp}] Vercel Cron: Medium-Volatility Scan triggered`);

    const autonomousQueue = getAutonomousQueue();
    const job = await autonomousQueue.add('medium-volatility-scan', {
      timeframe: '15m',
      minVolume: 5_000_000, // $5M+
      coinCount: 100,
      triggeredBy: 'vercel-cron',
      timestamp,
    });

    console.log(`✅ Job enqueued: ${job.id}`);

    return NextResponse.json({
      success: true,
      jobId: job.id,
      type: 'medium-volatility-scan',
      timestamp,
    });
  } catch (error: any) {
    console.error('❌ Medium-Volatility Scan cron error:', error.message);

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
