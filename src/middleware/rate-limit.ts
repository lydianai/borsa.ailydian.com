/**
 * API Rate Limiting Middleware
 *
 * White-hat compliance: Fair usage limits for legitimate SaaS tiers
 * Implements per-tier rate limits based on subscription level
 */

import { NextRequest, NextResponse } from 'next/server';
import { getSubscriptionLimits, type SubscriptionTier } from '@/lib/stripe/config';

// In-memory rate limit store (production should use Redis)
interface RateLimitEntry {
  count: number;
  resetAt: number;
}

const rateLimitStore = new Map<string, RateLimitEntry>();

// Rate limit windows (in seconds)
const RATE_LIMIT_WINDOWS = {
  minute: 60,
  hour: 3600,
  day: 86400,
} as const;

/**
 * Get rate limit configuration based on subscription tier
 */
function getRateLimitConfig(tier: SubscriptionTier) {
  const limits = getSubscriptionLimits(tier);

  return {
    // API calls per day
    apiCallsPerDay: limits.apiCallsPerDay,

    // AI queries per day
    aiQueriesPerDay: limits.aiQueriesPerDay,

    // Requests per minute (to prevent abuse)
    requestsPerMinute: tier === 'free' ? 10 : tier === 'starter' ? 30 : tier === 'pro' ? 60 : -1, // unlimited for enterprise
  };
}

/**
 * Check if rate limit is exceeded
 */
export function checkRateLimit(
  userId: string,
  resource: 'api' | 'ai',
  tier: SubscriptionTier,
  window: keyof typeof RATE_LIMIT_WINDOWS = 'day'
): { allowed: boolean; limit: number; remaining: number; resetAt: number } {
  const config = getRateLimitConfig(tier);
  const limit = resource === 'api' ? config.apiCallsPerDay : config.aiQueriesPerDay;

  // Unlimited access
  if (limit === -1) {
    return {
      allowed: true,
      limit: -1,
      remaining: -1,
      resetAt: 0,
    };
  }

  const now = Date.now();
  const windowSeconds = RATE_LIMIT_WINDOWS[window];
  const key = `${userId}:${resource}:${window}`;

  let entry = rateLimitStore.get(key);

  // Create or reset entry if expired
  if (!entry || now >= entry.resetAt) {
    entry = {
      count: 0,
      resetAt: now + windowSeconds * 1000,
    };
  }

  // Increment count
  entry.count++;
  rateLimitStore.set(key, entry);

  const remaining = Math.max(0, limit - entry.count);
  const allowed = entry.count <= limit;

  return {
    allowed,
    limit,
    remaining,
    resetAt: entry.resetAt,
  };
}

/**
 * Middleware to enforce rate limits on API routes
 */
export async function withRateLimit(
  req: NextRequest,
  resource: 'api' | 'ai' = 'api'
): Promise<NextResponse | null> {
  // Get user from session (TODO: implement proper auth)
  // For now, using a placeholder
  const userId = req.headers.get('x-user-id') || 'anonymous';
  const tier = (req.headers.get('x-subscription-tier') as SubscriptionTier) || 'free';

  // Check rate limit
  const { allowed, limit, remaining, resetAt } = checkRateLimit(userId, resource, tier, 'day');

  // Add rate limit headers
  const headers = new Headers();
  headers.set('X-RateLimit-Limit', limit.toString());
  headers.set('X-RateLimit-Remaining', remaining.toString());
  headers.set('X-RateLimit-Reset', new Date(resetAt).toISOString());

  // If rate limit exceeded
  if (!allowed) {
    return NextResponse.json(
      {
        error: 'Rate limit exceeded',
        message: `You have exceeded your ${resource} usage limit for your ${tier} plan.`,
        limit,
        resetAt: new Date(resetAt).toISOString(),
        upgradeUrl: '/pricing',
      },
      {
        status: 429,
        headers,
      }
    );
  }

  // Rate limit OK, return null to continue
  return null;
}

/**
 * Cleanup expired entries (run periodically)
 */
export function cleanupRateLimitStore() {
  const now = Date.now();
  for (const [key, entry] of rateLimitStore.entries()) {
    if (now >= entry.resetAt) {
      rateLimitStore.delete(key);
    }
  }
}

// Cleanup every 5 minutes
if (typeof window === 'undefined') {
  setInterval(cleanupRateLimitStore, 5 * 60 * 1000);
}

/**
 * Higher-order function to wrap API routes with rate limiting
 */
export function withApiRateLimit(
  handler: (req: NextRequest) => Promise<NextResponse>,
  resource: 'api' | 'ai' = 'api'
) {
  return async (req: NextRequest): Promise<NextResponse> => {
    // Check rate limit
    const rateLimitResponse = await withRateLimit(req, resource);

    // If rate limited, return error response
    if (rateLimitResponse) {
      return rateLimitResponse;
    }

    // Otherwise, continue to handler
    return handler(req);
  };
}

/**
 * Usage tracking for analytics (to be stored in database)
 */
export async function trackApiUsage(
  userId: string,
  endpoint: string,
  resource: 'api' | 'ai',
  metadata?: Record<string, any>
) {
  // TODO: Store in database (UsageRecord model)
  // await prisma.usageRecord.create({
  //   data: {
  //     userId,
  //     resourceType: resource,
  //     quantity: 1,
  //     timestamp: new Date(),
  //     metadata: {
  //       endpoint,
  //       ...metadata,
  //     },
  //   },
  // });

  console.log(`[Usage] ${userId} - ${resource} - ${endpoint}`);
}
