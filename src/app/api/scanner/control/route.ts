/**
 * POST /api/scanner/control
 * Control continuous scanner (start, stop, trigger)
 *
 * Security:
 * - Requires INTERNAL_SERVICE_TOKEN for authentication
 * - White-hat logging: All control actions logged
 */

import { NextRequest, NextResponse } from 'next/server';
import continuousScannerService from '@/lib/scanner/continuous-scanner';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * Validate internal service token
 */
function validateServiceToken(request: NextRequest): boolean {
  const token = request.headers.get('x-service-token');
  const expectedToken = process.env.INTERNAL_SERVICE_TOKEN;

  if (!expectedToken) {
    console.warn('[API] INTERNAL_SERVICE_TOKEN not set in environment');
    return false;
  }

  return token === expectedToken;
}

export async function POST(request: NextRequest) {
  try {
    // 1. Validate authentication
    if (!validateServiceToken(request)) {
      console.warn('[API] Scanner control: Unauthorized request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 2. Parse request body
    const body = await request.json();
    const { action } = body;

    if (!action || typeof action !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid action field' },
        { status: 400 }
      );
    }

    // 3. Execute action
    console.log(`[API] Scanner control: Action "${action}" requested`);

    switch (action) {
      case 'start':
        continuousScannerService.start();
        console.log('[API] ✅ Scanner started');
        return NextResponse.json({
          success: true,
          action: 'start',
          message: 'Continuous scanner started',
          stats: continuousScannerService.getStats(),
        });

      case 'stop':
        continuousScannerService.stop();
        console.log('[API] ✅ Scanner stopped');
        return NextResponse.json({
          success: true,
          action: 'stop',
          message: 'Continuous scanner stopped',
          stats: continuousScannerService.getStats(),
        });

      case 'trigger':
        // Trigger immediate scan (requires private method access - using restart hack)
        const wasRunning = continuousScannerService.getStats().isRunning;

        if (wasRunning) {
          continuousScannerService.stop();
          await new Promise((resolve) => setTimeout(resolve, 100));
        }

        continuousScannerService.start();

        console.log('[API] ✅ Scanner triggered (restarted)');
        return NextResponse.json({
          success: true,
          action: 'trigger',
          message: 'Scan triggered immediately',
          stats: continuousScannerService.getStats(),
        });

      case 'reset':
        continuousScannerService.resetStats();
        console.log('[API] ✅ Scanner stats reset');
        return NextResponse.json({
          success: true,
          action: 'reset',
          message: 'Scanner statistics reset',
          stats: continuousScannerService.getStats(),
        });

      default:
        return NextResponse.json(
          { error: `Unknown action: ${action}. Allowed: start, stop, trigger, reset` },
          { status: 400 }
        );
    }
  } catch (error: any) {
    console.error('[API] Scanner control error:', error);
    return NextResponse.json(
      { error: 'Scanner control failed', message: error.message },
      { status: 500 }
    );
  }
}
