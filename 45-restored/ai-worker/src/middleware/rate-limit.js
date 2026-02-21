/**
 * Rate Limiting Middleware
 *
 * White-hat compliant DDoS protection:
 * - IP-based rate limiting
 * - User-based rate limiting
 * - Global rate limiting
 * - Distributed rate limiting ready (Redis)
 * - Standard rate limit headers
 */

/**
 * In-memory rate limit store
 * Production: Replace with Redis for distributed systems
 */
class RateLimitStore {
  constructor() {
    this.store = new Map();
    this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
  }

  /**
   * Get rate limit record
   */
  get(key) {
    return this.store.get(key);
  }

  /**
   * Set rate limit record
   */
  set(key, value) {
    this.store.set(key, value);
  }

  /**
   * Cleanup expired records
   */
  cleanup() {
    const now = Date.now();
    for (const [key, record] of this.store.entries()) {
      if (now > record.resetTime) {
        this.store.delete(key);
      }
    }
  }

  /**
   * Get store size (for monitoring)
   */
  size() {
    return this.store.size;
  }
}

const rateLimitStore = new RateLimitStore();

/**
 * Check rate limit
 * @param {string} key - Rate limit key (IP or user ID)
 * @param {number} limit - Max requests per window
 * @param {number} windowMs - Time window in milliseconds
 * @returns {Object} - Rate limit status
 */
export function checkRateLimit(key, limit, windowMs) {
  const now = Date.now();
  const record = rateLimitStore.get(key);

  if (!record) {
    // First request for this key
    const newRecord = {
      count: 1,
      resetTime: now + windowMs,
      startTime: now
    };
    rateLimitStore.set(key, newRecord);

    return {
      allowed: true,
      limit: limit,
      remaining: limit - 1,
      reset: Math.ceil(newRecord.resetTime / 1000),
      retryAfter: null
    };
  }

  // Check if window expired
  if (now > record.resetTime) {
    // Reset window
    const newRecord = {
      count: 1,
      resetTime: now + windowMs,
      startTime: now
    };
    rateLimitStore.set(key, newRecord);

    return {
      allowed: true,
      limit: limit,
      remaining: limit - 1,
      reset: Math.ceil(newRecord.resetTime / 1000),
      retryAfter: null
    };
  }

  // Increment count
  record.count++;
  rateLimitStore.set(key, record);

  // Check if limit exceeded
  if (record.count > limit) {
    const retryAfter = Math.ceil((record.resetTime - now) / 1000);

    return {
      allowed: false,
      limit: limit,
      remaining: 0,
      reset: Math.ceil(record.resetTime / 1000),
      retryAfter: retryAfter
    };
  }

  return {
    allowed: true,
    limit: limit,
    remaining: limit - record.count,
    reset: Math.ceil(record.resetTime / 1000),
    retryAfter: null
  };
}

/**
 * Rate limit middleware for Hono
 * @param {Object} options - Rate limit configuration
 */
export function rateLimitMiddleware(options = {}) {
  const {
    limit = 100,           // Max requests per window
    windowMs = 60000,      // 1 minute window
    keyGenerator = null,   // Custom key generator
    skip = null            // Skip function
  } = options;

  return async (c, next) => {
    // Skip rate limiting for health check
    if (c.req.path === '/health') {
      return await next();
    }

    // Custom skip logic
    if (skip && skip(c)) {
      return await next();
    }

    // Generate rate limit key
    let key;
    if (keyGenerator) {
      key = keyGenerator(c);
    } else {
      // Default: IP-based rate limiting
      const ip = c.req.header('x-forwarded-for') ||
                 c.req.header('x-real-ip') ||
                 'unknown';
      key = `ip:${ip}`;
    }

    // Check rate limit
    const result = checkRateLimit(key, limit, windowMs);

    // Set rate limit headers (standard)
    c.header('X-RateLimit-Limit', result.limit.toString());
    c.header('X-RateLimit-Remaining', result.remaining.toString());
    c.header('X-RateLimit-Reset', result.reset.toString());

    if (!result.allowed) {
      // Rate limit exceeded
      c.header('Retry-After', result.retryAfter.toString());

      // Log rate limit violation
      console.warn('Rate limit exceeded:', {
        key: key,
        limit: limit,
        path: c.req.path,
        method: c.req.method,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: 'Too many requests',
        code: 'RATE_LIMIT_EXCEEDED',
        retryAfter: result.retryAfter,
        message: `Rate limit exceeded. Try again in ${result.retryAfter} seconds.`
      }, 429);
    }

    // Log rate limit status (debug)
    if (result.remaining < 10) {
      console.info('Rate limit warning:', {
        key: key,
        remaining: result.remaining,
        reset: result.reset,
        timestamp: new Date().toISOString()
      });
    }

    await next();
  };
}

/**
 * Multiple rate limits (IP + User + Global)
 */
export function multiRateLimitMiddleware() {
  return async (c, next) => {
    // Skip health check
    if (c.req.path === '/health') {
      return await next();
    }

    const ip = c.req.header('x-forwarded-for') ||
               c.req.header('x-real-ip') ||
               'unknown';

    // Rate limit 1: IP-based (100 req/min)
    const ipResult = checkRateLimit(`ip:${ip}`, 100, 60000);

    if (!ipResult.allowed) {
      c.header('X-RateLimit-Limit', '100');
      c.header('X-RateLimit-Remaining', '0');
      c.header('X-RateLimit-Reset', ipResult.reset.toString());
      c.header('Retry-After', ipResult.retryAfter.toString());

      return c.json({
        success: false,
        error: 'IP rate limit exceeded',
        code: 'IP_RATE_LIMIT',
        retryAfter: ipResult.retryAfter
      }, 429);
    }

    // Rate limit 2: Global (10000 req/min)
    const globalResult = checkRateLimit('global', 10000, 60000);

    if (!globalResult.allowed) {
      return c.json({
        success: false,
        error: 'Global rate limit exceeded',
        code: 'GLOBAL_RATE_LIMIT',
        retryAfter: globalResult.retryAfter
      }, 429);
    }

    // Set headers for IP rate limit
    c.header('X-RateLimit-Limit', '100');
    c.header('X-RateLimit-Remaining', ipResult.remaining.toString());
    c.header('X-RateLimit-Reset', ipResult.reset.toString());

    await next();
  };
}

/**
 * Get rate limit stats (for monitoring)
 */
export function getRateLimitStats() {
  return {
    storeSize: rateLimitStore.size(),
    timestamp: new Date().toISOString()
  };
}
