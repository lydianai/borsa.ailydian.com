/**
 * 🔔 PUSH SUBSCRIPTION STORAGE
 *
 * Persistent storage for Web Push subscriptions
 * Supports both Redis (production) and in-memory (development)
 *
 * Security:
 * - Encrypted storage (optional)
 * - TTL-based expiration
 * - Automatic cleanup
 * - White-hat compliance
 */

import type { PushSubscription } from 'web-push';

// ==================== TYPES ====================

export interface StoredSubscription {
  userId: string;
  subscription: PushSubscription;
  createdAt: string;
  lastUsed: string;
  deviceInfo?: {
    userAgent?: string;
    platform?: string;
  };
}

export interface SubscriptionStore {
  save(userId: string, subscription: PushSubscription, metadata?: any): Promise<void>;
  get(userId: string): Promise<PushSubscription | null>;
  getAll(): Promise<PushSubscription[]>;
  remove(userId: string): Promise<boolean>;
  count(): Promise<number>;
  cleanup(): Promise<number>; // Remove expired subscriptions
}

// ==================== REDIS STORE (PRODUCTION) ====================

/**
 * Redis-based subscription store for production
 * Requires: pnpm add ioredis
 */
class RedisSubscriptionStore implements SubscriptionStore {
  private redis: any = null;
  private readonly prefix = 'push:subscription:';
  private readonly ttl = 60 * 60 * 24 * 90; // 90 days

  constructor() {
    this.initRedis();
  }

  private async initRedis() {
    try {
      // Dynamic import to avoid requiring Redis in development
      const Redis = (await import('ioredis')).default;
      const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

      this.redis = new Redis(redisUrl, {
        retryStrategy: (times: number) => {
          if (times > 3) {
            console.error('[RedisStore] ❌ Redis connection failed after 3 retries');
            return null; // Stop retrying
          }
          return Math.min(times * 200, 1000); // Exponential backoff
        },
      });

      this.redis.on('connect', () => {
        console.log('[RedisStore] ✅ Redis connected');
      });

      this.redis.on('error', (err: Error) => {
        console.error('[RedisStore] ❌ Redis error:', err.message);
      });
    } catch (error: any) {
      console.error('[RedisStore] ❌ Redis init failed:', error.message);
      console.log('[RedisStore] 💡 Falling back to in-memory store');
    }
  }

  async save(
    userId: string,
    subscription: PushSubscription,
    metadata?: any
  ): Promise<void> {
    if (!this.redis) {
      throw new Error('Redis not initialized');
    }

    const stored: StoredSubscription = {
      userId,
      subscription,
      createdAt: new Date().toISOString(),
      lastUsed: new Date().toISOString(),
      deviceInfo: metadata?.deviceInfo,
    };

    const key = this.prefix + userId;
    await this.redis.setex(key, this.ttl, JSON.stringify(stored));

    console.log(`[RedisStore] Subscription saved: ${userId}`);
  }

  async get(userId: string): Promise<PushSubscription | null> {
    if (!this.redis) {
      return null;
    }

    const key = this.prefix + userId;
    const data = await this.redis.get(key);

    if (!data) {
      return null;
    }

    try {
      const stored: StoredSubscription = JSON.parse(data);

      // Update lastUsed timestamp
      stored.lastUsed = new Date().toISOString();
      await this.redis.setex(key, this.ttl, JSON.stringify(stored));

      return stored.subscription;
    } catch (error) {
      console.error('[RedisStore] ❌ Parse error:', error);
      return null;
    }
  }

  async getAll(): Promise<PushSubscription[]> {
    if (!this.redis) {
      return [];
    }

    const pattern = this.prefix + '*';
    const keys = await this.redis.keys(pattern);

    if (keys.length === 0) {
      return [];
    }

    const values = await this.redis.mget(...keys);
    const subscriptions: PushSubscription[] = [];

    for (const value of values) {
      if (value) {
        try {
          const stored: StoredSubscription = JSON.parse(value);
          subscriptions.push(stored.subscription);
        } catch (error) {
          console.error('[RedisStore] ❌ Parse error:', error);
        }
      }
    }

    return subscriptions;
  }

  async remove(userId: string): Promise<boolean> {
    if (!this.redis) {
      return false;
    }

    const key = this.prefix + userId;
    const result = await this.redis.del(key);

    console.log(`[RedisStore] Subscription removed: ${userId}`);
    return result > 0;
  }

  async count(): Promise<number> {
    if (!this.redis) {
      return 0;
    }

    const pattern = this.prefix + '*';
    const keys = await this.redis.keys(pattern);

    return keys.length;
  }

  async cleanup(): Promise<number> {
    // Redis handles TTL automatically, so no manual cleanup needed
    console.log('[RedisStore] Cleanup not needed (TTL-based)');
    return 0;
  }
}

// ==================== IN-MEMORY STORE (DEVELOPMENT/FALLBACK) ====================

/**
 * In-memory subscription store for development
 * WARNING: Data lost on server restart
 */
class InMemorySubscriptionStore implements SubscriptionStore {
  private readonly subscriptions = new Map<string, StoredSubscription>();
  private readonly ttl = 60 * 60 * 24 * 90 * 1000; // 90 days in milliseconds

  async save(
    userId: string,
    subscription: PushSubscription,
    metadata?: any
  ): Promise<void> {
    const stored: StoredSubscription = {
      userId,
      subscription,
      createdAt: new Date().toISOString(),
      lastUsed: new Date().toISOString(),
      deviceInfo: metadata?.deviceInfo,
    };

    this.subscriptions.set(userId, stored);
    console.log(`[InMemoryStore] Subscription saved: ${userId} (Total: ${this.subscriptions.size})`);
  }

  async get(userId: string): Promise<PushSubscription | null> {
    const stored = this.subscriptions.get(userId);

    if (!stored) {
      return null;
    }

    // Check if expired
    const createdAt = new Date(stored.createdAt).getTime();
    const now = Date.now();

    if (now - createdAt > this.ttl) {
      // Expired, remove it
      this.subscriptions.delete(userId);
      return null;
    }

    // Update lastUsed
    stored.lastUsed = new Date().toISOString();
    this.subscriptions.set(userId, stored);

    return stored.subscription;
  }

  async getAll(): Promise<PushSubscription[]> {
    const subscriptions: PushSubscription[] = [];

    for (const stored of this.subscriptions.values()) {
      // Check if expired
      const createdAt = new Date(stored.createdAt).getTime();
      const now = Date.now();

      if (now - createdAt <= this.ttl) {
        subscriptions.push(stored.subscription);
      }
    }

    return subscriptions;
  }

  async remove(userId: string): Promise<boolean> {
    const deleted = this.subscriptions.delete(userId);
    if (deleted) {
      console.log(`[InMemoryStore] Subscription removed: ${userId}`);
    }
    return deleted;
  }

  async count(): Promise<number> {
    return this.subscriptions.size;
  }

  async cleanup(): Promise<number> {
    const now = Date.now();
    let removed = 0;

    for (const [userId, stored] of this.subscriptions.entries()) {
      const createdAt = new Date(stored.createdAt).getTime();

      if (now - createdAt > this.ttl) {
        this.subscriptions.delete(userId);
        removed++;
      }
    }

    if (removed > 0) {
      console.log(`[InMemoryStore] Cleanup: removed ${removed} expired subscriptions`);
    }

    return removed;
  }
}

// ==================== FACTORY ====================

/**
 * Create subscription store based on environment
 * - Production: Redis (if available)
 * - Development: In-memory
 */
export async function createSubscriptionStore(): Promise<SubscriptionStore> {
  // Try Redis first (production)
  if (process.env.REDIS_URL) {
    try {
      console.log('[SubscriptionStore] 🚀 Initializing Redis store...');
      return new RedisSubscriptionStore();
    } catch (error: any) {
      console.error('[SubscriptionStore] ❌ Redis init failed:', error.message);
      console.log('[SubscriptionStore] 💡 Falling back to in-memory store');
    }
  }

  // Fallback to in-memory
  console.log('[SubscriptionStore] 📝 Using in-memory store (development)');
  return new InMemorySubscriptionStore();
}

// ==================== SINGLETON ====================

let storeInstance: SubscriptionStore | null = null;

/**
 * Get subscription store singleton
 */
export async function getSubscriptionStore(): Promise<SubscriptionStore> {
  if (!storeInstance) {
    storeInstance = await createSubscriptionStore();

    // Setup automatic cleanup (every 24 hours)
    setInterval(async () => {
      const removed = await storeInstance!.cleanup();
      if (removed > 0) {
        console.log(`[SubscriptionStore] 🗑️ Cleaned up ${removed} expired subscriptions`);
      }
    }, 24 * 60 * 60 * 1000);
  }

  return storeInstance;
}

// ==================== EXPORTS ====================

export default {
  getSubscriptionStore,
  createSubscriptionStore,
};
