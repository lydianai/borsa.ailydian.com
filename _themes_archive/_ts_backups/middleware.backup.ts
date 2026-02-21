/**
 * 🛡️ NEXT.JS GLOBAL MIDDLEWARE
 * Security-first middleware for all requests
 *
 * Features:
 * - Security headers (CSP, HSTS, etc.)
 * - Rate limiting
 * - Audit logging
 * - CORS handling
 * - Bot detection
 * - Origin validation
 * - White-hat compliance
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  applySecurityHeaders,
  applyCORSHeaders,
  handlePreflight,
  validateOrigin,
  isBot,
} from './middleware/security-headers';
import {
  globalRateLimiter,
  strictRateLimiter,
  authRateLimiter,
  getClientIP,
  checkRateLimit,
} from './lib/security/rate-limiter';
import auditLogger, {
  AuditEventType,
  AuditSeverity,
  logRateLimitExceeded,
  logUnauthorizedAccess,
  logAPIRequest,
} from './lib/security/audit-logger';
import { personalAuthMiddleware } from './lib/security/personal-auth';

// ============================================================================
// CONFIGURATION
// ============================================================================

// Paths that require strict rate limiting
const STRICT_PATHS = [
  '/api/queue/enqueue',
  '/api/scanner/control',
  '/api/push/send',
];

// Paths that require auth rate limiting
const AUTH_PATHS = ['/api/auth/login', '/api/auth/register', '/api/auth/reset'];

// Public paths (no auth required)
const PUBLIC_PATHS = [
  '/',
  '/api/health',
  '/api/scanner/status',
  '/api/queue/metrics',
  '/api/push/stats',
];

// Paths that should skip middleware processing
const SKIP_PATHS = [
  '/_next',
  '/static',
  '/favicon.ico',
  '/robots.txt',
  '/sitemap.xml',
];

// ============================================================================
// MIDDLEWARE
// ============================================================================

export async function middleware(request: NextRequest) {
  const startTime = Date.now();
  const { pathname } = request.nextUrl;
  const method = request.method;

  // Skip processing for static assets
  if (SKIP_PATHS.some((path) => pathname.startsWith(path))) {
    return NextResponse.next();
  }

  try {
    // 1. Personal authentication (IP whitelist + secret token)
    // TEMPORARILY DISABLED for deployment access
    // Can be re-enabled by changing false to true below
    if (false && process.env.PERSONAL_AUTH_DISABLED !== 'true') {
      const authResponse = personalAuthMiddleware(request);
      if (authResponse) {
        return applySecurityHeaders(authResponse); // Access denied
      }
    }

    // 2. Handle CORS preflight
    const preflightResponse = handlePreflight(request);
    if (preflightResponse) {
      return applySecurityHeaders(preflightResponse);
    }

    // 3. Get client IP
    const clientIP = getClientIP(request);

    // 4. Bot detection
    if (isBot(request) && !PUBLIC_PATHS.some((path) => pathname.startsWith(path))) {
      auditLogger.log(
        AuditEventType.SUSPICIOUS_ACTIVITY,
        `Bot detected: ${pathname}`,
        {
          severity: AuditSeverity.WARNING,
          ipAddress: clientIP,
          endpoint: pathname,
          method,
          metadata: { userAgent: request.headers.get('user-agent') },
        }
      );

      // Allow bots for public paths, block for others
      return new NextResponse('Forbidden', { status: 403 });
    }

    // 5. Origin validation (CSRF protection)
    if (method !== 'GET' && method !== 'HEAD') {
      if (!validateOrigin(request)) {
        logUnauthorizedAccess(pathname, clientIP, 'Invalid origin');

        return new NextResponse('Forbidden', { status: 403 });
      }
    }

    // 6. Rate limiting
    let rateLimitResult;

    if (STRICT_PATHS.some((path) => pathname.startsWith(path))) {
      // Strict rate limiting
      rateLimitResult = checkRateLimit(strictRateLimiter, clientIP);
    } else if (AUTH_PATHS.some((path) => pathname.startsWith(path))) {
      // Auth rate limiting
      rateLimitResult = checkRateLimit(authRateLimiter, clientIP);
    } else if (pathname.startsWith('/api/')) {
      // Global API rate limiting
      rateLimitResult = checkRateLimit(globalRateLimiter, clientIP);
    }

    // Handle rate limit exceeded
    if (rateLimitResult && !rateLimitResult.allowed) {
      logRateLimitExceeded(clientIP, pathname, clientIP);

      const response = new NextResponse('Too Many Requests', {
        status: 429,
        headers: rateLimitResult.headers,
      });

      return applySecurityHeaders(response);
    }

    // 7. Process request
    const response = NextResponse.next();

    // 8. Apply security headers
    applySecurityHeaders(response);

    // 9. Apply CORS headers
    applyCORSHeaders(response, request);

    // 10. Apply rate limit headers
    if (rateLimitResult) {
      for (const [key, value] of rateLimitResult.headers.entries()) {
        response.headers.set(key, value);
      }
    }

    // 11. Log request
    const duration = Date.now() - startTime;

    // Don't log static assets or health checks to avoid spam
    if (!pathname.startsWith('/_next') && pathname !== '/api/health') {
      logAPIRequest(method, pathname, 200, duration, {
        ipAddress: clientIP,
        userAgent: request.headers.get('user-agent') || undefined,
      });
    }

    return response;
  } catch (error: any) {
    // Log error
    console.error('[Middleware] Error:', error);

    auditLogger.log(AuditEventType.API_ERROR, `Middleware error: ${error.message}`, {
      severity: AuditSeverity.ERROR,
      endpoint: pathname,
      method,
      metadata: { error: error.message },
    });

    // Return error response
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}

// ============================================================================
// MIDDLEWARE CONFIG
// ============================================================================

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
