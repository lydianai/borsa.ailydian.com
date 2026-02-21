/**
 * GET /api/push/stats
 * Get push notification statistics
 *
 * Returns device token statistics and Firebase status
 */

import { NextRequest, NextResponse } from 'next/server';
import deviceTokenManager from '@/lib/push/device-token-manager';
import { isFirebaseAvailable } from '@/lib/push/firebase-admin';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(_request: NextRequest) {
  try {
    // Get device token stats
    const stats = deviceTokenManager.getStats();

    // Check Firebase availability
    const firebaseAvailable = isFirebaseAvailable();

    return NextResponse.json({
      timestamp: new Date().toISOString(),
      firebase: {
        available: firebaseAvailable,
        status: firebaseAvailable ? 'connected' : 'not configured',
      },
      devices: {
        totalTokens: stats.totalTokens,
        totalUsers: stats.totalUsers,
        platformBreakdown: stats.platformBreakdown,
      },
    });
  } catch (error: any) {
    console.error('[API] Push stats error:', error);

    return NextResponse.json(
      { error: 'Failed to get push stats', message: error.message },
      { status: 500 }
    );
  }
}
