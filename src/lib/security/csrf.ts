/**
 * CSRF (Cross-Site Request Forgery) Protection
 *
 * White-hat security implementation following OWASP guidelines
 * https://owasp.org/www-community/attacks/csrf
 *
 * Edge Runtime Compatible - uses Web Crypto API
 */

const CSRF_TOKEN_LENGTH = 32;
const CSRF_COOKIE_NAME = 'csrf-token';
const CSRF_HEADER_NAME = 'x-csrf-token';

/**
 * Generate a cryptographically secure CSRF token
 * Edge Runtime compatible - uses Web Crypto API instead of Node.js crypto
 */
export function generateCSRFToken(): string {
  // Use Web Crypto API (available in Edge Runtime)
  const array = new Uint8Array(CSRF_TOKEN_LENGTH);
  crypto.getRandomValues(array);

  // Convert to base64url format
  return btoa(String.fromCharCode(...array))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

/**
 * Validate CSRF token from cookie and header
 */
export function validateCSRFToken(cookieToken: string | undefined, headerToken: string | undefined): boolean {
  // Both must exist
  if (!cookieToken || !headerToken) {
    return false;
  }

  // Constant-time comparison to prevent timing attacks
  return timingSafeEqual(cookieToken, headerToken);
}

/**
 * Timing-safe string comparison
 * Prevents timing attacks by ensuring comparison takes same time regardless of where strings differ
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }

  return result === 0;
}

/**
 * CSRF cookie configuration
 */
export const CSRF_COOKIE_OPTIONS = {
  name: CSRF_COOKIE_NAME,
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'strict' as const,
  path: '/',
  maxAge: 60 * 60 * 24, // 24 hours
};

/**
 * CSRF header name for client-side usage
 */
export const CSRF_HEADER = CSRF_HEADER_NAME;
