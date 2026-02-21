/**
 * 🏥 OTONOM SİSTEM SAĞLIK KONTROLÜ API
 *
 * Tüm otonom sistemin sağlığını kontrol eder:
 * - Redis bağlantısı
 * - BullMQ queue durumu
 * - Advanced AI Engine bağlantısı
 * - Learning System
 * - Memory Store
 * - Cron job'lar
 */

import { NextResponse } from 'next/server';
import { checkRedisHealth } from '@/lib/queue/redis-client';
import { checkQueueHealth } from '@/lib/queue/autonomous-queue';
import { checkAdvancedAIHealth } from '@/lib/ai/advanced-analyzer';
import { checkLearningSystemHealth } from '@/lib/learning/strategy-learning-engine';
import { checkMemoryHealth } from '@/lib/memory/ai-memory-store';
import { getAutonomousScannerStatus } from '@/lib/cron/autonomous-scanner';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET() {
  try {
    const startTime = Date.now();

    // Paralel health check'ler
    const [
      redisHealth,
      queueHealth,
      claudeHealth,
      learningHealth,
      memoryHealth,
      cronStatus,
    ] = await Promise.allSettled([
      checkRedisHealth(),
      checkQueueHealth(),
      checkAdvancedAIHealth(),
      checkLearningSystemHealth(),
      checkMemoryHealth(),
      Promise.resolve(getAutonomousScannerStatus()),
    ]);

    const responseTime = Date.now() - startTime;

    // Sonuçları parse et
    const results = {
      timestamp: new Date().toISOString(),
      responseTime: `${responseTime}ms`,
      overall: 'healthy' as 'healthy' | 'degraded' | 'unhealthy',
      services: {
        redis: redisHealth.status === 'fulfilled' ? redisHealth.value : { status: 'unhealthy', error: 'Failed to check' },
        queue: queueHealth.status === 'fulfilled' ? queueHealth.value : { status: 'unhealthy', error: 'Failed to check' },
        claudeAI: claudeHealth.status === 'fulfilled' ? claudeHealth.value : { status: 'unhealthy', error: 'Failed to check' },
        learningSystem: learningHealth.status === 'fulfilled' ? learningHealth.value : { status: 'unhealthy', error: 'Failed to check' },
        memoryStore: memoryHealth.status === 'fulfilled' ? memoryHealth.value : { status: 'unhealthy', error: 'Failed to check' },
        cronJobs: cronStatus.status === 'fulfilled' ? { status: 'healthy', ...cronStatus.value } : { status: 'unhealthy', error: 'Failed to check' },
      },
    };

    // Overall health hesapla
    const services = Object.values(results.services);
    const unhealthyCount = services.filter((s: any) => s.status === 'unhealthy').length;

    if (unhealthyCount === 0) {
      results.overall = 'healthy';
    } else if (unhealthyCount <= 2) {
      results.overall = 'degraded';
    } else {
      results.overall = 'unhealthy';
    }

    const statusCode = results.overall === 'healthy' ? 200 : results.overall === 'degraded' ? 207 : 503;

    return NextResponse.json(results, { status: statusCode });
  } catch (error: any) {
    console.error('❌ Health check error:', error.message);

    return NextResponse.json(
      {
        timestamp: new Date().toISOString(),
        overall: 'unhealthy',
        error: error.message,
      },
      { status: 503 }
    );
  }
}
