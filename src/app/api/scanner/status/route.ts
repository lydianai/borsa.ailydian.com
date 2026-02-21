/**
 * GET /api/scanner/status
 * Returns continuous scanner status and statistics
 */

import { NextRequest, NextResponse } from 'next/server';
import continuousScannerService from '@/lib/scanner/continuous-scanner';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(_request: NextRequest) {
  try {
    const stats = continuousScannerService.getStats();
    const config = continuousScannerService.getConfig();
    const healthy = continuousScannerService.isHealthy();

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      healthy,
      scanner: {
        status: stats.isRunning ? 'running' : 'stopped',
        stats: {
          totalScansTriggered: stats.totalScansTriggered,
          totalSymbolsScanned: stats.totalSymbolsScanned,
          totalBatchesProcessed: stats.totalBatchesProcessed,
          lastScanTime: stats.lastScanTime,
          nextScanTime: stats.nextScanTime,
          currentBatch: stats.currentBatch,
          totalBatches: stats.totalBatches,
          errors: stats.errors,
        },
        circuitBreaker: {
          state: stats.circuitBreakerState,
          healthy: stats.circuitBreakerState === 'CLOSED',
        },
        config: {
          scanIntervalMs: config.scanIntervalMs,
          scanIntervalMinutes: config.scanIntervalMs / 60000,
          batchSize: config.batchSize,
          batchDelayMs: config.batchDelayMs,
          priorityMode: config.priorityMode,
          strategiesCount: config.strategies.length,
        },
      },
    });
  } catch (error: any) {
    console.error('[API] Scanner status error:', error);
    return NextResponse.json(
      { error: 'Failed to get scanner status', message: error.message },
      { status: 500 }
    );
  }
}
