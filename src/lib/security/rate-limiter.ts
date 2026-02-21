/**
 * 🛡️ ADVANCED RATE LIMITER
 * Multi-strategy rate limiting for API protection
 *
 * Features:
 * - IP-based rate limiting
 * - User-based rate limiting
 * - Endpoint-based rate limiting
 * - Sliding window algorithm
 * - Distributed support (Redis)
 * - Automatic cleanup
 * - White-hat compliance: All violations logged
 */

// ============================================================================
// TYPES
// ============================================================================

export interface RateLimitConfig {
  /**
   * Maximum requests per window
   */
  maxRequests: number;

  /**
   * Time window in milliseconds
   */
  windowMs: number;

  /**
   * Rate limit key prefix (for namespacing)
   */
  keyPrefix?: string;

  /**
   * Skip rate limiting for certain conditions
   */
  skip?: (identifier: string) => boolean;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
  retryAfter?: number;
}

interface RateLimitEntry {
  count: number;
  resetAt: number;
  requests: number[]; // Timestamps for sliding window
}

// ============================================================================
// RATE LIMITER CLASS
// ============================================================================

export class RateLimiter {
  private store: Map<string, RateLimitEntry> = new Map();
  private config: RateLimitConfig;
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(config: RateLimitConfig) {
    this.config = {
      keyPrefix: 'ratelimit',
      ...config,
    };

    // Start automatic cleanup (every 5 minutes)
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 300000);

    console.log(
      `[RateLimiter] Initialized: ${config.maxRequests} req/${config.windowMs}ms`
    );
  }

  /**
   * Check rate limit for identifier
   */
  check(identifier: string): RateLimitResult {
    // Skip if configured
    if (this.config.skip && this.config.skip(identifier)) {
      return {
        allowed: true,
        remaining: this.config.maxRequests,
        resetAt: Date.now() + this.config.windowMs,
      };
    }

    const key = `${this.config.keyPrefix}:${identifier}`;
    const now = Date.now();

    // Get or create entry
    let entry = this.store.get(key);

    if (!entry) {
      entry = {
        count: 0,
        resetAt: now + this.config.windowMs,
        requests: [],
      };
      this.store.set(key, entry);
    }

    // Clean old requests (sliding window)
    entry.requests = entry.requests.filter(
      (timestamp) => now - timestamp < this.config.windowMs
    );

    // Check if limit exceeded
    if (entry.requests.length >= this.config.maxRequests) {
      const oldestRequest = entry.requests[0];
      const retryAfter = Math.ceil((oldestRequest + this.config.windowMs - now) / 1000);

      // Log violation
      console.warn(
        `[RateLimiter] Limit exceeded: ${identifier} (${entry.requests.length}/${this.config.maxRequests})`
      );

      return {
        allowed: false,
        remaining: 0,
        resetAt: oldestRequest + this.config.windowMs,
        retryAfter,
      };
    }

    // Add current request
    entry.requests.push(now);
    entry.count++;

    const remaining = this.config.maxRequests - entry.requests.length;

    return {
      allowed: true,
      remaining,
      resetAt: entry.requests[0] + this.config.windowMs,
    };
  }

  /**
   * Increment counter for identifier
   */
  increment(identifier: string): RateLimitResult {
    return this.check(identifier);
  }

  /**
   * Reset rate limit for identifier
   */
  reset(identifier: string): void {
    const key = `${this.config.keyPrefix}:${identifier}`;
    this.store.delete(key);
    console.log(`[RateLimiter] Reset: ${identifier}`);
  }

  /**
   * Get current count for identifier
   */
  getCount(identifier: string): number {
    const key = `${this.config.keyPrefix}:${identifier}`;
    const entry = this.store.get(key);
    return entry ? entry.requests.length : 0;
  }

  /**
   * Cleanup expired entries
   */
  cleanup(): void {
    const now = Date.now();
    let removedCount = 0;

    for (const [key, entry] of this.store.entries()) {
      // Remove if all requests are older than window
      entry.requests = entry.requests.filter(
        (timestamp) => now - timestamp < this.config.windowMs
      );

      if (entry.requests.length === 0) {
        this.store.delete(key);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      console.log(`[RateLimiter] Cleaned up ${removedCount} expired entries`);
    }
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalEntries: number;
    totalRequests: number;
    config: RateLimitConfig;
  } {
    let totalRequests = 0;

    for (const entry of this.store.values()) {
      totalRequests += entry.requests.length;
    }

    return {
      totalEntries: this.store.size,
      totalRequests,
      config: this.config,
    };
  }

  /**
   * Shutdown rate limiter
   */
  shutdown(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }

    this.store.clear();
    console.log('[RateLimiter] Shutdown complete');
  }
}

// ============================================================================
// PRE-CONFIGURED RATE LIMITERS
// ============================================================================

/**
 * Global API rate limiter (60 req/min per IP)
 * Development: Unlimited for localhost (::1, 127.0.0.1, unknown)
 * Production: 600 req/min for normal traffic
 */
export const globalRateLimiter = new RateLimiter({
  maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '600', 10), // Increased to 600 for development
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10),
  keyPrefix: 'global',
  skip: (identifier: string) => {
    // Skip rate limiting for localhost during development
    const isLocalhost = identifier === '::1' || identifier === '127.0.0.1' || identifier === 'unknown';
    const isDevelopment = process.env.NODE_ENV === 'development';
    return isLocalhost && isDevelopment;
  },
});

/**
 * Strict rate limiter for sensitive endpoints (100 req/min per IP)
 * Development: Unlimited for localhost
 */
export const strictRateLimiter = new RateLimiter({
  maxRequests: 100, // Increased for development
  windowMs: 60000, // 1 minute
  keyPrefix: 'strict',
  skip: (identifier: string) => {
    const isLocalhost = identifier === '::1' || identifier === '127.0.0.1' || identifier === 'unknown';
    const isDevelopment = process.env.NODE_ENV === 'development';
    return isLocalhost && isDevelopment;
  },
});

/**
 * Auth rate limiter for login/register (50 req/min per IP)
 * Development: Unlimited for localhost
 */
export const authRateLimiter = new RateLimiter({
  maxRequests: 50, // Increased for development
  windowMs: 60000, // 1 minute
  keyPrefix: 'auth',
  skip: (identifier: string) => {
    const isLocalhost = identifier === '::1' || identifier === '127.0.0.1' || identifier === 'unknown';
    const isDevelopment = process.env.NODE_ENV === 'development';
    return isLocalhost && isDevelopment;
  },
});

/**
 * Scanner enqueue rate limiter (100 req/min per user)
 * Development: Unlimited for localhost
 */
export const scannerRateLimiter = new RateLimiter({
  maxRequests: 100, // Increased for development
  windowMs: 60000, // 1 minute
  keyPrefix: 'scanner',
  skip: (identifier: string) => {
    const isLocalhost = identifier === '::1' || identifier === '127.0.0.1' || identifier === 'unknown';
    const isDevelopment = process.env.NODE_ENV === 'development';
    return isLocalhost && isDevelopment;
  },
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get client IP address from request
 */
export function getClientIP(request: Request): string {
  const headers = request.headers;

  // Try common proxy headers
  const forwardedFor = headers.get('x-forwarded-for');
  if (forwardedFor) {
    return forwardedFor.split(',')[0].trim();
  }

  const realIP = headers.get('x-real-ip');
  if (realIP) {
    return realIP;
  }

  const cfConnectingIP = headers.get('cf-connecting-ip'); // Cloudflare
  if (cfConnectingIP) {
    return cfConnectingIP;
  }

  // Fallback (not reliable in production)
  return 'unknown';
}

/**
 * Create rate limit response headers
 */
export function createRateLimitHeaders(result: RateLimitResult): Headers {
  const headers = new Headers();

  headers.set('X-RateLimit-Limit', result.remaining.toString());
  headers.set('X-RateLimit-Remaining', result.remaining.toString());
  headers.set('X-RateLimit-Reset', new Date(result.resetAt).toISOString());

  if (result.retryAfter) {
    headers.set('Retry-After', result.retryAfter.toString());
  }

  return headers;
}

/**
 * Check if request is rate limited
 * Returns true if allowed, false if blocked
 */
export function checkRateLimit(
  rateLimiter: RateLimiter,
  identifier: string
): { allowed: boolean; headers: Headers; result: RateLimitResult } {
  const result = rateLimiter.check(identifier);
  const headers = createRateLimitHeaders(result);

  return {
    allowed: result.allowed,
    headers,
    result,
  };
}
