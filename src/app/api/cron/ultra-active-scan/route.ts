/**
 * 🔴 VERCEL CRON JOB - Ultra-Active Scan
 * Schedule: Every 5 minutes
 * Cron: `* /5 * * * *` (remove space)
 *
 * Vercel tarafından her 5 dakikada bir çağrılır.
 * $10M+ hacimli 50 coin'i tarar.
 *
 * Security: CRON_SECRET ile korunur
 */

import { NextRequest, NextResponse } from 'next/server';
import { getAutonomousQueue } from '@/lib/queue/autonomous-queue';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';
export const maxDuration = 60; // Vercel Hobby: max 60 seconds

export async function GET(request: NextRequest) {
  try {
    // 1. Güvenlik: CRON_SECRET kontrolü
    const authHeader = request.headers.get('authorization');
    const cronSecret = process.env.CRON_SECRET;

    if (cronSecret && authHeader !== `Bearer ${cronSecret}`) {
      console.error('❌ Unauthorized cron request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const timestamp = new Date().toISOString();
    console.log(`\n⏰ [${timestamp}] Vercel Cron: Ultra-Active Scan triggered`);

    // 2. BullMQ kuyruğuna job ekle
    const autonomousQueue = getAutonomousQueue();
    const job = await autonomousQueue.add('ultra-active-scan', {
      timeframe: '5m',
      minVolume: 10_000_000, // $10M+
      coinCount: 50,
      aiEnhanced: true,
      triggeredBy: 'vercel-cron',
      timestamp,
    });

    console.log(`✅ Job enqueued: ${job.id}`);

    return NextResponse.json({
      success: true,
      jobId: job.id,
      type: 'ultra-active-scan',
      timestamp,
    });
  } catch (error: any) {
    console.error('❌ Ultra-Active Scan cron error:', error.message);

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
