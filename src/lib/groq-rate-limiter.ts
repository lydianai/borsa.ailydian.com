/**
 * Groq API Rate Limiter
 *
 * Groq Free Tier Limits:
 * - 30 requests per minute (RPM)
 * - 14,400 requests per day (RPD)
 * - 7,000 tokens per minute (TPM)
 *
 * Sliding Window Algorithm Implementation
 */

import { Redis } from '@upstash/redis'

// Redis key prefix for rate limiting
const RATE_LIMIT_PREFIX = 'ai_rl'

interface RateLimitResult {
  success: boolean
  limit: number
  remaining: number
  reset: number
  retryAfter?: number
}

interface RateLimitConfig {
  rpm: number      // Requests per minute
  rpd: number      // Requests per day
  tpm: number      // Tokens per minute
  identifier: string // IP or user ID
}

export class GroqRateLimiter {
  private redis: Redis | null = null
  private fallbackStore: Map<string, { count: number; timestamp: number; tokens: number }> = new Map()

  // Groq API limits
  private static readonly LIMITS = {
    RPM: 30,      // Free tier: 30 requests/min
    RPD: 14400,   // Free tier: 14,400 requests/day
    TPM: 7000,    // Free tier: 7,000 tokens/min
  }

  constructor() {
    // Initialize Redis if available
    if (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN) {
      this.redis = new Redis({
        url: process.env.UPSTASH_REDIS_REST_URL,
        token: process.env.UPSTASH_REDIS_REST_TOKEN,
      })
    }
  }

  /**
   * Check rate limit using sliding window algorithm
   */
  async checkRateLimit(
    identifier: string,
    tokens: number = 0
  ): Promise<RateLimitResult> {
    const now = Date.now()

    // Check minute limit
    const minuteResult = await this.checkWindow(
      identifier,
      'minute',
      GroqRateLimiter.LIMITS.RPM,
      60 * 1000, // 60 seconds
      now
    )

    if (!minuteResult.success) {
      return minuteResult
    }

    // Check daily limit
    const dayResult = await this.checkWindow(
      identifier,
      'day',
      GroqRateLimiter.LIMITS.RPD,
      24 * 60 * 60 * 1000, // 24 hours
      now
    )

    if (!dayResult.success) {
      return dayResult
    }

    // Check token limit if tokens provided
    if (tokens > 0) {
      const tokenResult = await this.checkTokenLimit(identifier, tokens, now)
      if (!tokenResult.success) {
        return tokenResult
      }
    }

    return minuteResult
  }

  /**
   * Sliding window algorithm implementation
   */
  private async checkWindow(
    identifier: string,
    window: 'minute' | 'day',
    limit: number,
    windowMs: number,
    now: number
  ): Promise<RateLimitResult> {
    const key = `${RATE_LIMIT_PREFIX}:${identifier}:${window}`

    if (this.redis) {
      return this.checkWindowRedis(key, limit, windowMs, now)
    } else {
      return this.checkWindowMemory(key, limit, windowMs, now)
    }
  }

  /**
   * Redis-based sliding window
   */
  private async checkWindowRedis(
    key: string,
    limit: number,
    windowMs: number,
    now: number
  ): Promise<RateLimitResult> {
    const windowStart = now - windowMs

    try {
      // Remove old entries
      await this.redis!.zremrangebyscore(key, 0, windowStart)

      // Count requests in current window
      const count = await this.redis!.zcount(key, windowStart, now)

      const remaining = Math.max(0, limit - count)
      const reset = now + windowMs

      if (count >= limit) {
        // Get oldest request timestamp for retry-after
        const oldest = await this.redis!.zrange(key, 0, 0, { withScores: true })
        const retryAfter = oldest.length > 0
          ? Math.ceil((Number(oldest[1]) + windowMs - now) / 1000)
          : Math.ceil(windowMs / 1000)

        return {
          success: false,
          limit,
          remaining: 0,
          reset,
          retryAfter
        }
      }

      // Add current request
      await this.redis!.zadd(key, { score: now, member: `${now}-${Math.random()}` })

      // Set expiry
      await this.redis!.expire(key, Math.ceil(windowMs / 1000))

      return {
        success: true,
        limit,
        remaining: remaining - 1,
        reset
      }
    } catch (error) {
      console.error('[GroqRateLimiter] Redis error:', error)
      // Fallback to memory
      return this.checkWindowMemory(key, limit, windowMs, now)
    }
  }

  /**
   * Memory-based fallback (for development)
   */
  private checkWindowMemory(
    key: string,
    limit: number,
    windowMs: number,
    now: number
  ): RateLimitResult {
    const stored = this.fallbackStore.get(key)
    const windowStart = now - windowMs

    // Clean old entries
    if (stored && stored.timestamp < windowStart) {
      this.fallbackStore.delete(key)
    }

    const current = this.fallbackStore.get(key) || { count: 0, timestamp: now, tokens: 0 }
    const reset = now + windowMs

    if (current.count >= limit) {
      const retryAfter = Math.ceil((current.timestamp + windowMs - now) / 1000)
      return {
        success: false,
        limit,
        remaining: 0,
        reset,
        retryAfter
      }
    }

    // Increment count
    current.count++
    current.timestamp = now
    this.fallbackStore.set(key, current)

    return {
      success: true,
      limit,
      remaining: limit - current.count,
      reset
    }
  }

  /**
   * Check token limit (TPM - Tokens Per Minute)
   */
  private async checkTokenLimit(
    identifier: string,
    tokens: number,
    now: number
  ): Promise<RateLimitResult> {
    const key = `${RATE_LIMIT_PREFIX}:${identifier}:tokens`
    const windowMs = 60 * 1000 // 1 minute
    const windowStart = now - windowMs

    if (this.redis) {
      try {
        // Remove old token counts
        await this.redis.zremrangebyscore(key, 0, windowStart)

        // Sum tokens in current window
        const entries = await this.redis.zrange(key, 0, -1, { withScores: true })
        let totalTokens = 0
        for (let i = 0; i < entries.length; i += 2) {
          const tokenCount = parseInt(entries[i] as string, 10)
          totalTokens += tokenCount
        }

        const remaining = Math.max(0, GroqRateLimiter.LIMITS.TPM - totalTokens)

        if (totalTokens + tokens > GroqRateLimiter.LIMITS.TPM) {
          return {
            success: false,
            limit: GroqRateLimiter.LIMITS.TPM,
            remaining: 0,
            reset: now + windowMs,
            retryAfter: Math.ceil(windowMs / 1000)
          }
        }

        // Add current token count
        await this.redis.zadd(key, { score: now, member: `${tokens}` })
        await this.redis.expire(key, 60)

        return {
          success: true,
          limit: GroqRateLimiter.LIMITS.TPM,
          remaining: remaining - tokens,
          reset: now + windowMs
        }
      } catch (error) {
        console.error('[GroqRateLimiter] Token limit check error:', error)
      }
    }

    // Memory fallback for tokens
    const stored = this.fallbackStore.get(key)
    if (!stored || stored.timestamp < windowStart) {
      this.fallbackStore.set(key, { count: tokens, timestamp: now, tokens })
      return {
        success: true,
        limit: GroqRateLimiter.LIMITS.TPM,
        remaining: GroqRateLimiter.LIMITS.TPM - tokens,
        reset: now + windowMs
      }
    }

    if (stored.tokens + tokens > GroqRateLimiter.LIMITS.TPM) {
      return {
        success: false,
        limit: GroqRateLimiter.LIMITS.TPM,
        remaining: 0,
        reset: now + windowMs,
        retryAfter: Math.ceil((stored.timestamp + windowMs - now) / 1000)
      }
    }

    stored.tokens += tokens
    this.fallbackStore.set(key, stored)

    return {
      success: true,
      limit: GroqRateLimiter.LIMITS.TPM,
      remaining: GroqRateLimiter.LIMITS.TPM - stored.tokens,
      reset: now + windowMs
    }
  }

  /**
   * Get current usage statistics
   */
  async getUsageStats(identifier: string): Promise<{
    minuteUsage: number
    dayUsage: number
    tokenUsage: number
  }> {
    const now = Date.now()

    const minuteKey = `${RATE_LIMIT_PREFIX}:${identifier}:minute`
    const dayKey = `${RATE_LIMIT_PREFIX}:${identifier}:day`
    const tokenKey = `${RATE_LIMIT_PREFIX}:${identifier}:tokens`

    if (this.redis) {
      try {
        const [minuteCount, dayCount, tokenEntries] = await Promise.all([
          this.redis.zcount(minuteKey, now - 60000, now),
          this.redis.zcount(dayKey, now - 86400000, now),
          this.redis.zrange(tokenKey, 0, -1)
        ])

        const tokenUsage = tokenEntries.reduce((sum, entry) => {
          const tokens = parseInt(entry as string, 10)
          return sum + (isNaN(tokens) ? 0 : tokens)
        }, 0)

        return {
          minuteUsage: minuteCount,
          dayUsage: dayCount,
          tokenUsage
        }
      } catch (error) {
        console.error('[GroqRateLimiter] Stats error:', error)
      }
    }

    // Memory fallback
    const minute = this.fallbackStore.get(minuteKey)
    const day = this.fallbackStore.get(dayKey)
    const tokens = this.fallbackStore.get(tokenKey)

    return {
      minuteUsage: minute?.count || 0,
      dayUsage: day?.count || 0,
      tokenUsage: tokens?.tokens || 0
    }
  }

  /**
   * Reset rate limit for identifier (admin only)
   */
  async resetLimit(identifier: string): Promise<void> {
    if (this.redis) {
      const keys = [
        `${RATE_LIMIT_PREFIX}:${identifier}:minute`,
        `${RATE_LIMIT_PREFIX}:${identifier}:day`,
        `${RATE_LIMIT_PREFIX}:${identifier}:tokens`
      ]

      await Promise.all(keys.map(key => this.redis!.del(key)))
    } else {
      this.fallbackStore.clear()
    }
  }
}

// Singleton instance
export const groqRateLimiter = new GroqRateLimiter()

/**
 * Express-style middleware for rate limiting
 */
export async function withGroqRateLimit(
  identifier: string,
  tokens: number = 0
): Promise<{ allowed: boolean; headers: Record<string, string> }> {
  const result = await groqRateLimiter.checkRateLimit(identifier, tokens)

  const headers: Record<string, string> = {
    'X-RateLimit-Limit': String(result.limit),
    'X-RateLimit-Remaining': String(result.remaining),
    'X-RateLimit-Reset': String(result.reset),
  }

  if (!result.success && result.retryAfter) {
    headers['Retry-After'] = String(result.retryAfter)
  }

  return {
    allowed: result.success,
    headers
  }
}
