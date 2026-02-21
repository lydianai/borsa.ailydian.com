/**
 * Cache Manager with Redis Support
 *
 * White-hat compliance: Proper caching improves performance and reduces
 * load on external APIs
 */

import { LRUCache } from 'lru-cache';

/**
 * In-memory cache for development (when Redis is not available)
 */
const memoryCache = new LRUCache<string, any>({
  max: 500, // Maximum number of items
  maxSize: 50 * 1024 * 1024, // 50MB max size
  sizeCalculation: (value) => {
    return JSON.stringify(value).length;
  },
  ttl: 1000 * 60 * 15, // 15 minutes default TTL
});

/**
 * Redis client (lazy loaded)
 */
let redisClient: any = null;

async function getRedisClient() {
  if (redisClient) return redisClient;

  // Check if Redis is configured
  if (!process.env.UPSTASH_REDIS_REST_URL || !process.env.UPSTASH_REDIS_REST_TOKEN) {
    console.warn('[Cache] Redis not configured. Using in-memory cache.');
    return null;
  }

  try {
    const { Redis } = await import('ioredis');

    // For Upstash Redis REST API
    if (process.env.UPSTASH_REDIS_REST_URL.includes('upstash')) {
      const upstashModule = await import('@upstash/redis' as any);
      const UpstashRedis = (upstashModule as any).Redis;
      redisClient = new UpstashRedis({
        url: process.env.UPSTASH_REDIS_REST_URL,
        token: process.env.UPSTASH_REDIS_REST_TOKEN,
      });
    } else {
      // Standard Redis connection
      redisClient = new Redis(process.env.UPSTASH_REDIS_REST_URL);
    }

    return redisClient;
  } catch (error) {
    console.error('[Cache] Failed to connect to Redis:', error);
    return null;
  }
}

/**
 * Cache configuration
 */
export const CACHE_KEYS = {
  MARKET_DATA: 'market:data',
  SIGNALS: 'signals:all',
  SIGNAL_BY_SYMBOL: (symbol: string) => `signals:${symbol}`,
  AI_SIGNALS: 'ai:signals',
  NEWS: 'news:latest',
  WHALE_ALERTS: 'whale:alerts',
  DECISION_ENGINE: (symbol: string) => `decision:${symbol}`,
} as const;

export const CACHE_TTL = {
  MARKET_DATA: 60, // 1 minute
  SIGNALS: 300, // 5 minutes
  AI_SIGNALS: 900, // 15 minutes
  NEWS: 600, // 10 minutes
  WHALE_ALERTS: 300, // 5 minutes
  DECISION_ENGINE: 120, // 2 minutes
} as const;

/**
 * Get value from cache
 */
export async function getCached<T>(key: string): Promise<T | null> {
  try {
    const redis = await getRedisClient();

    if (redis) {
      // Use Redis
      const value = await redis.get(key);
      return value ? (JSON.parse(value) as T) : null;
    } else {
      // Fallback to memory cache
      return (memoryCache.get(key) as T) || null;
    }
  } catch (error) {
    console.error(`[Cache] Error getting key ${key}:`, error);
    return null;
  }
}

/**
 * Set value in cache
 */
export async function setCached<T>(
  key: string,
  value: T,
  ttlSeconds: number = 300
): Promise<void> {
  try {
    const redis = await getRedisClient();

    if (redis) {
      // Use Redis with TTL
      await redis.setex(key, ttlSeconds, JSON.stringify(value));
    } else {
      // Fallback to memory cache
      memoryCache.set(key, value, { ttl: ttlSeconds * 1000 });
    }
  } catch (error) {
    console.error(`[Cache] Error setting key ${key}:`, error);
  }
}

/**
 * Delete from cache
 */
export async function deleteCached(key: string): Promise<void> {
  try {
    const redis = await getRedisClient();

    if (redis) {
      await redis.del(key);
    } else {
      memoryCache.delete(key);
    }
  } catch (error) {
    console.error(`[Cache] Error deleting key ${key}:`, error);
  }
}

/**
 * Clear all cache (pattern match for Redis)
 */
export async function clearCachePattern(pattern: string): Promise<void> {
  try {
    const redis = await getRedisClient();

    if (redis) {
      const keys = await redis.keys(pattern);
      if (keys.length > 0) {
        await redis.del(...keys);
      }
    } else {
      // For memory cache, clear all (can't pattern match easily)
      memoryCache.clear();
    }
  } catch (error) {
    console.error(`[Cache] Error clearing pattern ${pattern}:`, error);
  }
}

/**
 * Get or set pattern (cache-aside)
 */
export async function getOrSet<T>(
  key: string,
  fetchFn: () => Promise<T>,
  ttlSeconds: number = 300
): Promise<T> {
  // Try to get from cache first
  const cached = await getCached<T>(key);
  if (cached !== null) {
    return cached;
  }

  // Cache miss - fetch fresh data
  const fresh = await fetchFn();

  // Store in cache
  await setCached(key, fresh, ttlSeconds);

  return fresh;
}

/**
 * HTTP Cache Headers Helper
 */
export function getCacheHeaders(
  maxAge: number,
  options: {
    private?: boolean;
    mustRevalidate?: boolean;
    staleWhileRevalidate?: number;
  } = {}
): Record<string, string> {
  const cacheControl = [
    options.private ? 'private' : 'public',
    `max-age=${maxAge}`,
    options.mustRevalidate && 'must-revalidate',
    options.staleWhileRevalidate && `stale-while-revalidate=${options.staleWhileRevalidate}`,
  ]
    .filter(Boolean)
    .join(', ');

  return {
    'Cache-Control': cacheControl,
    'CDN-Cache-Control': `max-age=${maxAge}`,
    'Vercel-CDN-Cache-Control': `max-age=${maxAge}`,
  };
}

/**
 * Common cache header presets
 */
export const CACHE_HEADERS = {
  // No cache (for sensitive data)
  NO_CACHE: {
    'Cache-Control': 'private, no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0',
  },

  // Short cache (1 minute)
  SHORT: getCacheHeaders(60, { staleWhileRevalidate: 30 }),

  // Medium cache (5 minutes)
  MEDIUM: getCacheHeaders(300, { staleWhileRevalidate: 60 }),

  // Long cache (15 minutes)
  LONG: getCacheHeaders(900, { staleWhileRevalidate: 300 }),

  // Static assets (1 hour)
  STATIC: getCacheHeaders(3600, { staleWhileRevalidate: 1800 }),
} as const;
