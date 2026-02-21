/**
 * GET /api/security/audit
 * Get audit logs and security statistics
 *
 * Security:
 * - Requires INTERNAL_SERVICE_TOKEN
 * - Returns sanitized audit events
 */

import { NextRequest, NextResponse } from 'next/server';
import auditLogger, { AuditEventType, AuditSeverity } from '@/lib/security/audit-logger';
import {
  globalRateLimiter,
  strictRateLimiter,
  authRateLimiter,
  scannerRateLimiter,
} from '@/lib/security/rate-limiter';

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

export async function GET(request: NextRequest) {
  try {
    // 1. Validate authentication
    if (!validateServiceToken(request)) {
      console.warn('[API] Security audit: Unauthorized request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 2. Get query parameters
    const { searchParams } = new URL(request.url);
    const type = searchParams.get('type') as AuditEventType | null;
    const severity = searchParams.get('severity') as AuditSeverity | null;
    const limit = parseInt(searchParams.get('limit') || '100', 10);

    // 3. Query audit logs
    const events = auditLogger.query({
      type: type || undefined,
      severity: severity || undefined,
      limit: Math.min(limit, 1000), // Max 1000 events
    });

    // 4. Get statistics
    const auditStats = auditLogger.getStats();

    // 5. Get rate limiter statistics
    const rateLimiterStats = {
      global: globalRateLimiter.getStats(),
      strict: strictRateLimiter.getStats(),
      auth: authRateLimiter.getStats(),
      scanner: scannerRateLimiter.getStats(),
    };

    // 6. Return response
    return NextResponse.json({
      timestamp: new Date().toISOString(),
      audit: {
        events,
        stats: auditStats,
      },
      rateLimiters: rateLimiterStats,
    });
  } catch (error: any) {
    console.error('[API] Security audit error:', error);

    return NextResponse.json(
      { error: 'Failed to get audit logs', message: error.message },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/security/audit
 * Clear audit logs (for testing only)
 */
export async function DELETE(request: NextRequest) {
  try {
    // 1. Validate authentication
    if (!validateServiceToken(request)) {
      console.warn('[API] Security audit clear: Unauthorized request');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // 2. Only allow in development
    if (process.env.NODE_ENV === 'production') {
      return NextResponse.json(
        { error: 'Cannot clear audit logs in production' },
        { status: 403 }
      );
    }

    // 3. Clear audit logs
    auditLogger.clear();

    return NextResponse.json({
      success: true,
      message: 'Audit logs cleared',
    });
  } catch (error: any) {
    console.error('[API] Security audit clear error:', error);

    return NextResponse.json(
      { error: 'Failed to clear audit logs', message: error.message },
      { status: 500 }
    );
  }
}
