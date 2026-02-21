/**
 * 🛡️ SECURITY HEADERS MIDDLEWARE
 * Comprehensive security headers for production deployment
 *
 * Features:
 * - Content Security Policy (CSP)
 * - HSTS (HTTP Strict Transport Security)
 * - X-Frame-Options (Clickjacking protection)
 * - X-Content-Type-Options (MIME sniffing protection)
 * - X-XSS-Protection
 * - Referrer-Policy
 * - Permissions-Policy
 * - CORS configuration
 * - White-hat compliance: Industry-standard security headers
 */

import { NextRequest, NextResponse } from 'next/server';

// ============================================================================
// CSP CONFIGURATION
// ============================================================================

/**
 * Content Security Policy directives
 * Customize based on your external resources
 */
export const CSP_DIRECTIVES = {
  'default-src': ["'self'"],
  'script-src': [
    "'self'",
    "'unsafe-inline'", // For inline scripts (minimize in production)
    "'unsafe-eval'", // For eval() (required by some libraries)
    'https://www.gstatic.com', // Firebase
    'https://apis.google.com', // Google APIs
  ],
  'style-src': [
    "'self'",
    "'unsafe-inline'", // For inline styles
    'https://fonts.googleapis.com',
  ],
  'font-src': ["'self'", 'https://fonts.gstatic.com', 'data:'],
  'img-src': [
    "'self'",
    'data:',
    'https:', // Allow all HTTPS images (for crypto icons, etc.)
    'blob:',
  ],
  'connect-src': [
    "'self'",
    'https://fapi.binance.com', // Binance Futures API
    'https://fstream.binance.com', // Binance WebSocket
    'wss://fstream.binance.com',
    'https://fcm.googleapis.com', // Firebase Cloud Messaging (fallback)
    'https://*.firebaseapp.com',
    'https://*.googleapis.com',
    'https://*.google.com', // Firebase push endpoints
  ],
  'worker-src': ["'self'", 'blob:'], // Service Worker + Web Workers
  'manifest-src': ["'self'"], // PWA manifest
  'frame-src': ["'none'"], // No iframes allowed
  'object-src': ["'none'"], // No Flash, Java applets, etc.
  'base-uri': ["'self'"],
  'form-action': ["'self'"],
  'frame-ancestors': ["'none'"], // Prevent embedding in iframes
  'upgrade-insecure-requests': [], // Upgrade HTTP to HTTPS
};

/**
 * Build CSP header value from directives
 */
function buildCSP(): string {
  return Object.entries(CSP_DIRECTIVES)
    .map(([directive, sources]) => {
      if (sources.length === 0) {
        return directive;
      }
      return `${directive} ${sources.join(' ')}`;
    })
    .join('; ');
}

// ============================================================================
// SECURITY HEADERS
// ============================================================================

/**
 * Get all security headers
 */
export function getSecurityHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    // Content Security Policy
    'Content-Security-Policy': buildCSP(),

    // HSTS: Force HTTPS for 1 year
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains; preload',

    // X-Frame-Options: Prevent clickjacking
    'X-Frame-Options': 'DENY',

    // X-Content-Type-Options: Prevent MIME sniffing
    'X-Content-Type-Options': 'nosniff',

    // X-XSS-Protection: Legacy XSS protection
    'X-XSS-Protection': '1; mode=block',

    // Referrer-Policy: Control referrer information
    'Referrer-Policy': 'strict-origin-when-cross-origin',

    // Permissions-Policy: Control browser features
    'Permissions-Policy': [
      'camera=()',
      'microphone=()',
      'geolocation=()',
      'interest-cohort=()', // Block FLoC tracking
      'payment=(self)',
      'usb=()',
    ].join(', '),

    // X-DNS-Prefetch-Control: Disable DNS prefetching
    'X-DNS-Prefetch-Control': 'off',

    // X-Download-Options: Prevent IE downloads
    'X-Download-Options': 'noopen',

    // X-Permitted-Cross-Domain-Policies: Adobe products security
    'X-Permitted-Cross-Domain-Policies': 'none',
  };

  // Development-only headers
  if (process.env.NODE_ENV === 'development') {
    // Relax CSP for development
    headers['Content-Security-Policy-Report-Only'] = headers['Content-Security-Policy'];
    delete headers['Content-Security-Policy'];
  }

  return headers;
}

/**
 * Apply security headers to response
 */
export function applySecurityHeaders(response: NextResponse): NextResponse {
  const headers = getSecurityHeaders();

  for (const [key, value] of Object.entries(headers)) {
    response.headers.set(key, value);
  }

  return response;
}

/**
 * Apply CORS headers
 * White-hat security: Strict origin validation
 */
export function applyCORSHeaders(
  response: NextResponse,
  request: NextRequest
): NextResponse {
  const origin = request.headers.get('origin');

  // Use CORS_ALLOWED_ORIGINS (not ALLOWED_ORIGINS) - standardized naming
  const allowedOrigins = (
    process.env.CORS_ALLOWED_ORIGINS ||
    process.env.ALLOWED_ORIGINS || // Fallback for backwards compatibility
    'http://localhost:3000'
  ).split(',').map(o => o.trim());

  // Production: Strict origin validation
  if (process.env.NODE_ENV === 'production') {
    // Only allow whitelisted origins in production
    if (origin && allowedOrigins.includes(origin)) {
      response.headers.set('Access-Control-Allow-Origin', origin);
      response.headers.set('Access-Control-Allow-Credentials', 'true');
    }
  } else {
    // Development: Allow localhost variations
    if (origin) {
      const isLocalhost =
        origin.startsWith('http://localhost:') ||
        origin.startsWith('http://127.0.0.1:') ||
        allowedOrigins.includes(origin);

      if (isLocalhost) {
        response.headers.set('Access-Control-Allow-Origin', origin);
        response.headers.set('Access-Control-Allow-Credentials', 'true');
      }
    }
  }

  // Allowed methods (REST + WebSocket upgrade)
  response.headers.set(
    'Access-Control-Allow-Methods',
    'GET, POST, PUT, DELETE, OPTIONS, PATCH'
  );

  // Allowed headers (include push notification headers)
  response.headers.set(
    'Access-Control-Allow-Headers',
    'Content-Type, Authorization, X-Service-Token, X-Requested-With, X-VAPID-Public-Key'
  );

  // Expose headers for client access
  response.headers.set(
    'Access-Control-Expose-Headers',
    'X-RateLimit-Remaining, X-RateLimit-Reset'
  );

  // Max age for preflight cache (24 hours)
  response.headers.set('Access-Control-Max-Age', '86400');

  return response;
}

/**
 * Handle OPTIONS preflight requests
 */
export function handlePreflight(request: NextRequest): NextResponse | null {
  if (request.method === 'OPTIONS') {
    const response = new NextResponse(null, { status: 204 });
    return applyCORSHeaders(response, request);
  }

  return null;
}

/**
 * Validate request origin (CSRF protection)
 */
export function validateOrigin(request: NextRequest): boolean {
  const origin = request.headers.get('origin');
  const host = request.headers.get('host');

  // Skip validation for same-origin requests
  if (!origin) {
    return true;
  }

  const allowedOrigins = process.env.CORS_ALLOWED_ORIGINS?.split(',') || [];

  // Check if origin is allowed
  if (allowedOrigins.includes(origin)) {
    return true;
  }

  // Check if origin matches host (same-origin)
  try {
    const originUrl = new URL(origin);
    if (originUrl.host === host) {
      return true;
    }
  } catch {
    // Invalid origin URL
    return false;
  }

  return false;
}

/**
 * Check if request is from a bot/crawler
 */
export function isBot(request: NextRequest): boolean {
  const userAgent = request.headers.get('user-agent') || '';

  const botPatterns = [
    'bot',
    'crawler',
    'spider',
    'scraper',
    'curl',
    'wget',
    'python',
    'java',
    'go-http-client',
  ];

  return botPatterns.some((pattern) =>
    userAgent.toLowerCase().includes(pattern)
  );
}

/**
 * Generate nonce for CSP inline scripts
 */
export function generateNonce(): string {
  return Buffer.from(crypto.getRandomValues(new Uint8Array(16))).toString('base64');
}
