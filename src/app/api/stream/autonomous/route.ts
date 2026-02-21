/**
 * 📡 OTONOM SİSTEM REAL-TIME STREAM - Server-Sent Events (SSE)
 *
 * Sürekli olarak otonom sistemin durumunu stream eder:
 * - Her 10 saniyede health check
 * - Job durumları (queue status)
 * - Learning system updates
 * - Yeni sinyal bildirimleri
 *
 * Frontend bu stream'i dinleyerek real-time dashboard güncellemesi yapar.
 */

import { NextRequest } from 'next/server';
import { checkRedisHealth } from '@/lib/queue/redis-client';
import { checkQueueHealth } from '@/lib/queue/autonomous-queue';
import { getAutonomousScannerStatus } from '@/lib/cron/autonomous-scanner';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

// SSE stream helper
function createSSEStream() {
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      // SSE başlangıç mesajı
      const sendEvent = (event: string, data: any) => {
        const message = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
        controller.enqueue(encoder.encode(message));
      };

      // İlk bağlantı mesajı
      sendEvent('connected', {
        timestamp: new Date().toISOString(),
        message: 'Otonom Sistem Stream Aktif',
      });

      // Ana stream loop
      const intervalId = setInterval(async () => {
        try {
          // 1. Health check
          const [redisHealth, queueHealth, cronStatus] = await Promise.allSettled([
            checkRedisHealth(),
            checkQueueHealth(),
            Promise.resolve(getAutonomousScannerStatus()),
          ]);

          const healthData = {
            timestamp: new Date().toISOString(),
            redis: redisHealth.status === 'fulfilled' ? redisHealth.value : { status: 'unhealthy' },
            queue: queueHealth.status === 'fulfilled' ? queueHealth.value : { status: 'unhealthy' },
            cron: cronStatus.status === 'fulfilled' ? cronStatus.value : { status: 'unhealthy' },
          };

          sendEvent('health', healthData);

          // 2. Queue durumu detaylı
          if (queueHealth.status === 'fulfilled') {
            sendEvent('queue_status', {
              timestamp: new Date().toISOString(),
              ...queueHealth.value,
            });
          }

          // 3. Heartbeat
          sendEvent('heartbeat', {
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
          });
        } catch (error: any) {
          sendEvent('error', {
            timestamp: new Date().toISOString(),
            message: error.message,
          });
        }
      }, 10000); // Her 10 saniye

      // Cleanup - client disconnect olduğunda
      const cleanup = () => {
        clearInterval(intervalId);
        controller.close();
      };

      // Stream kapandığında cleanup
      return cleanup;
    },
  });

  return stream;
}

export async function GET(_request: NextRequest) {
  const stream = createSSEStream();

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no', // Nginx buffering'i kapat
    },
  });
}
