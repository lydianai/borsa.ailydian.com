/**
 * SYSTEM STATUS API
 * Health check for all microservices
 */

import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

interface ServiceStatus {
  name: string;
  url: string;
  status: 'healthy' | 'unhealthy' | 'unknown';
  responseTime?: number;
  details?: any;
}

async function checkService(name: string, url: string): Promise<ServiceStatus> {
  const startTime = Date.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    const response = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store'
    });

    clearTimeout(timeoutId);
    const responseTime = Date.now() - startTime;

    if (response.ok) {
      const data = await response.json();
      return {
        name,
        url,
        status: 'healthy',
        responseTime,
        details: data,
      };
    } else {
      return {
        name,
        url,
        status: 'unhealthy',
        responseTime,
        details: { error: `HTTP ${response.status}` },
      };
    }
  } catch (error: any) {
    const responseTime = Date.now() - startTime;
    return {
      name,
      url,
      status: 'unhealthy',
      responseTime,
      details: { error: error.message },
    };
  }
}

export async function GET() {
  try {
    const services = [
      { name: 'AI Models (Python)', url: 'http://localhost:5003/health' },
      { name: 'Signal Generator (Python)', url: 'http://localhost:5004/health' },
      { name: 'TA-Lib Service (Python)', url: 'http://localhost:5005/health' },
      { name: 'Binance API', url: 'http://localhost:3000/api/binance/price?symbol=BTCUSDT' },
      { name: 'Market Data API', url: 'http://localhost:3000/api/market/crypto' },
    ];

    // Check all services in parallel
    const results = await Promise.all(
      services.map(s => checkService(s.name, s.url))
    );

    const healthyCount = results.filter(r => r.status === 'healthy').length;
    const totalCount = results.length;
    const systemHealth = healthyCount === totalCount ? 'healthy' :
                        healthyCount > 0 ? 'degraded' : 'critical';

    return NextResponse.json({
      success: true,
      system: {
        status: systemHealth,
        healthy: healthyCount,
        total: totalCount,
        uptime: process.uptime(),
        timestamp: Date.now(),
      },
      services: results,
    });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
