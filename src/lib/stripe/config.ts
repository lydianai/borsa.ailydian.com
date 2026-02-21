/**
 * Stripe Configuration
 *
 * White-hat compliance: Subscription billing for legitimate SaaS service
 * All transactions follow Stripe's terms of service
 */

import Stripe from 'stripe';

// Stripe API version
const STRIPE_API_VERSION = '2025-11-17.clover' as const;

// Initialize Stripe (server-side only - lazy initialization)
let stripeInstance: Stripe | null = null;

export const getStripe = () => {
  // Only initialize on server side
  if (typeof window !== 'undefined') {
    throw new Error('Stripe instance should only be used on server-side');
  }

  if (!stripeInstance) {
    const apiKey = process.env.STRIPE_SECRET_KEY;
    if (!apiKey) {
      throw new Error('STRIPE_SECRET_KEY is not set');
    }

    stripeInstance = new Stripe(apiKey, {
      apiVersion: STRIPE_API_VERSION,
      typescript: true,
      appInfo: {
        name: 'Ailydian Signal',
        version: '1.0.0',
        url: 'https://ailydian.com',
      },
    });
  }

  return stripeInstance;
};

// Subscription tiers configuration
export const SUBSCRIPTION_TIERS = {
  free: {
    name: 'Free',
    price: 0,
    priceId: null,
    features: [
      '10 signals per day',
      '1 watchlist (10 coins)',
      'Basic indicators',
      '24-hour delayed AI analysis',
    ],
    limits: {
      signalsPerDay: 10,
      watchlists: 1,
      coinsPerWatchlist: 10,
      apiCallsPerDay: 100,
      aiQueriesPerDay: 5,
    },
  },
  starter: {
    name: 'Starter',
    price: 49,
    priceId: process.env.STRIPE_STARTER_PRICE_ID || '',
    features: [
      '100 signals per day',
      '5 watchlists (50 coins each)',
      'All 500+ indicators',
      'Real-time alerts (Telegram/Email)',
      'Conservative signals',
      'Breakout-retest patterns',
      'AI assistant (50 queries/day)',
    ],
    limits: {
      signalsPerDay: 100,
      watchlists: 5,
      coinsPerWatchlist: 50,
      apiCallsPerDay: 1000,
      aiQueriesPerDay: 50,
    },
  },
  pro: {
    name: 'Pro',
    price: 199,
    priceId: process.env.STRIPE_PRO_PRICE_ID || '',
    popular: true,
    features: [
      'Unlimited signals',
      'Unlimited watchlists',
      'All AI models (100+)',
      'Quantum Pro signals',
      'SMC Strategy',
      'Multi-timeframe analysis',
      'On-chain & whale alerts',
      'Copy trading signals',
      'API access (1,000 req/day)',
      'Priority support',
      'Custom alerts',
    ],
    limits: {
      signalsPerDay: -1, // unlimited
      watchlists: -1,
      coinsPerWatchlist: -1,
      apiCallsPerDay: 1000,
      aiQueriesPerDay: -1,
    },
  },
  enterprise: {
    name: 'Enterprise',
    price: 999,
    priceId: process.env.STRIPE_ENTERPRISE_PRICE_ID || '',
    features: [
      'Everything in Pro',
      'White-label options',
      'Unlimited API access',
      'Dedicated account manager',
      'Custom integrations',
      'SLA guarantee (99.9% uptime)',
      'Custom AI model training',
      'Team accounts (10 users)',
      'Advanced analytics',
      'Direct phone support',
    ],
    limits: {
      signalsPerDay: -1,
      watchlists: -1,
      coinsPerWatchlist: -1,
      apiCallsPerDay: -1,
      aiQueriesPerDay: -1,
      teamMembers: 10,
    },
  },
} as const;

export type SubscriptionTier = keyof typeof SUBSCRIPTION_TIERS;

// Trial period (days)
export const TRIAL_PERIOD_DAYS = 14;

// Webhook events we handle
export const STRIPE_WEBHOOK_EVENTS = [
  'customer.subscription.created',
  'customer.subscription.updated',
  'customer.subscription.deleted',
  'customer.subscription.trial_will_end',
  'invoice.paid',
  'invoice.payment_failed',
  'checkout.session.completed',
] as const;

/**
 * Get subscription limits for a tier
 */
export function getSubscriptionLimits(tier: SubscriptionTier) {
  return SUBSCRIPTION_TIERS[tier].limits;
}

/**
 * Check if user has exceeded their limit
 */
export function hasExceededLimit(
  tier: SubscriptionTier,
  resource: keyof typeof SUBSCRIPTION_TIERS.free.limits,
  currentUsage: number
): boolean {
  const limit = SUBSCRIPTION_TIERS[tier].limits[resource];

  // -1 means unlimited
  if (limit === -1) return false;

  return currentUsage >= limit;
}

/**
 * Get all available tiers for pricing page
 */
export function getAllTiers() {
  return Object.entries(SUBSCRIPTION_TIERS).map(([key, value]) => ({
    id: key as SubscriptionTier,
    ...value,
  }));
}

/**
 * Validate Stripe webhook signature
 */
export function validateWebhookSignature(
  payload: string | Buffer,
  signature: string
): Stripe.Event {
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET || '';
  const stripe = getStripe();

  try {
    return stripe.webhooks.constructEvent(payload, signature, webhookSecret);
  } catch (err) {
    throw new Error(`Webhook signature verification failed: ${err}`);
  }
}

/**
 * Format price for display
 */
export function formatPrice(cents: number, currency: string = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(cents / 100);
}
