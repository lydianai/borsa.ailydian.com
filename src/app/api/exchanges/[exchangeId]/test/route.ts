/**
 * Exchange Connection Test
 *
 * White-hat compliance: Test exchange API connection
 * POST: Retest exchange connection and permissions
 */

import { NextRequest, NextResponse } from 'next/server';
import { requirePayment } from '@/lib/auth/helpers';
import { retestExchangeConnection } from '@/lib/exchanges/manager';

/**
 * POST /api/exchanges/[exchangeId]/test
 * Test exchange connection and validate permissions
 */
export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ exchangeId: string }> }
) {
  try {
    const user = await requirePayment();
    const { exchangeId } = await params;

    const result = await retestExchangeConnection(exchangeId, user.id);

    if (!result) {
      return NextResponse.json(
        { error: 'Exchange not found or credentials invalid' },
        { status: 404 }
      );
    }

    // Return comprehensive test results
    return NextResponse.json({
      success: result.success,
      exchange: result.exchange,
      canRead: result.canRead,
      canTrade: result.canTrade,
      canWithdraw: result.canWithdraw,
      error: result.error,
      warnings: result.warnings,
      message: result.success
        ? 'Connection test successful'
        : 'Connection test failed',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Connection test failed' },
      { status: 500 }
    );
  }
}
