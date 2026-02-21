/**
 * DATA SERVICE HEALTH CHECK ENDPOINT
 * GET /api/data-service/health
 *
 * Returns health status of:
 * - WebSocket connection
 * - Circuit breakers
 * - Connection statistics
 */

import { NextResponse } from 'next/server';
import binanceWebSocketService from '@/lib/data-service/binance-websocket';
import circuitBreakerManager from '@/lib/resilience/circuit-breaker';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    // Get WebSocket stats
    const wsStats = binanceWebSocketService.getStats();
    const wsHealthy = binanceWebSocketService.isHealthy();

    // Get circuit breaker health
    const circuitHealth = circuitBreakerManager.getHealth();
    const circuitStats = circuitBreakerManager.getAllStats();

    // Calculate uptime
    const uptime = wsStats.connectedSince
      ? Math.floor((Date.now() - wsStats.connectedSince) / 1000)
      : 0;

    // Determine overall health
    const healthy = wsHealthy && circuitHealth.healthy;

    const response = {
      timestamp: new Date().toISOString(),
      healthy,
      services: {
        websocket: {
          healthy: wsHealthy,
          connected: wsStats.connected,
          uptime: uptime > 0 ? `${uptime}s` : 'N/A',
          subscriptions: wsStats.subscriptions.length,
          messagesReceived: wsStats.messagesReceived,
          lastMessage: wsStats.lastMessageTime
            ? new Date(wsStats.lastMessageTime).toISOString()
            : null,
          reconnectAttempts: wsStats.reconnectAttempts,
        },
        circuitBreakers: {
          healthy: circuitHealth.healthy,
          breakers: circuitHealth.breakers,
          stats: circuitStats,
        },
      },
    };

    const statusCode = healthy ? 200 : 503;

    return NextResponse.json(response, { status: statusCode });
  } catch (error: any) {
    console.error('[DataService Health] Error:', error);
    return NextResponse.json(
      {
        timestamp: new Date().toISOString(),
        healthy: false,
        error: error.message,
      },
      { status: 500 }
    );
  }
}
