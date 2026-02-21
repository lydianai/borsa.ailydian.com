/**
 * Subscription Service
 *
 * White-hat compliance: Manages legitimate subscription operations
 * Handles: creation, updates, cancellations, usage tracking
 */

import { getStripe, type SubscriptionTier, SUBSCRIPTION_TIERS } from './config';
import type Stripe from 'stripe';

// Types for subscription operations
export interface CreateSubscriptionParams {
  userId: string;
  email: string;
  tier: SubscriptionTier;
  billingCycle: 'monthly' | 'annual';
}

export interface SubscriptionInfo {
  userId: string;
  tier: SubscriptionTier;
  status: string;
  currentPeriodEnd: Date | null;
  cancelAtPeriodEnd: boolean;
  stripeCustomerId: string | null;
  stripeSubscriptionId: string | null;
}

/**
 * Create a checkout session for subscription
 */
export async function createCheckoutSession(params: CreateSubscriptionParams): Promise<string> {
  const stripe = getStripe();
  const tierConfig = SUBSCRIPTION_TIERS[params.tier];

  if (!tierConfig.priceId) {
    throw new Error(`No price ID configured for tier: ${params.tier}`);
  }

  const session = await stripe.checkout.sessions.create({
    customer_email: params.email,
    line_items: [
      {
        price: tierConfig.priceId,
        quantity: 1,
      },
    ],
    mode: 'subscription',
    success_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard?session_id={CHECKOUT_SESSION_ID}`,
    cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/pricing`,
    subscription_data: {
      trial_period_days: 14,
      metadata: {
        userId: params.userId,
        tier: params.tier,
        billingCycle: params.billingCycle,
      },
    },
    metadata: {
      userId: params.userId,
      tier: params.tier,
      billingCycle: params.billingCycle,
    },
    allow_promotion_codes: true,
    billing_address_collection: 'required',
    payment_method_types: ['card'],
  });

  if (!session.url) {
    throw new Error('Failed to create checkout session URL');
  }

  return session.url;
}

/**
 * Create a billing portal session for subscription management
 */
export async function createBillingPortalSession(stripeCustomerId: string): Promise<string> {
  const stripe = getStripe();

  const session = await stripe.billingPortal.sessions.create({
    customer: stripeCustomerId,
    return_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard/settings`,
  });

  return session.url;
}

/**
 * Cancel a subscription at period end
 */
export async function cancelSubscription(stripeSubscriptionId: string): Promise<Stripe.Subscription> {
  const stripe = getStripe();

  const subscription = await stripe.subscriptions.update(stripeSubscriptionId, {
    cancel_at_period_end: true,
  });

  return subscription;
}

/**
 * Reactivate a canceled subscription
 */
export async function reactivateSubscription(stripeSubscriptionId: string): Promise<Stripe.Subscription> {
  const stripe = getStripe();

  const subscription = await stripe.subscriptions.update(stripeSubscriptionId, {
    cancel_at_period_end: false,
  });

  return subscription;
}

/**
 * Immediately cancel a subscription
 */
export async function cancelSubscriptionImmediately(stripeSubscriptionId: string): Promise<Stripe.Subscription> {
  const stripe = getStripe();

  const subscription = await stripe.subscriptions.cancel(stripeSubscriptionId);

  return subscription;
}

/**
 * Upgrade/downgrade subscription to a new tier
 */
export async function changeSubscriptionTier(
  stripeSubscriptionId: string,
  newTier: SubscriptionTier
): Promise<Stripe.Subscription> {
  const stripe = getStripe();
  const tierConfig = SUBSCRIPTION_TIERS[newTier];

  if (!tierConfig.priceId) {
    throw new Error(`No price ID configured for tier: ${newTier}`);
  }

  // Get current subscription
  const subscription = await stripe.subscriptions.retrieve(stripeSubscriptionId);

  // Update subscription with new price
  const updatedSubscription = await stripe.subscriptions.update(stripeSubscriptionId, {
    items: [
      {
        id: subscription.items.data[0].id,
        price: tierConfig.priceId,
      },
    ],
    proration_behavior: 'create_prorations', // Pro-rate the difference
    metadata: {
      ...subscription.metadata,
      tier: newTier,
    },
  });

  return updatedSubscription;
}

/**
 * Get subscription details from Stripe
 */
export async function getSubscriptionDetails(stripeSubscriptionId: string): Promise<Stripe.Subscription> {
  const stripe = getStripe();
  return await stripe.subscriptions.retrieve(stripeSubscriptionId);
}

/**
 * Get customer details from Stripe
 */
export async function getCustomerDetails(stripeCustomerId: string): Promise<Stripe.Customer> {
  const stripe = getStripe();
  const customer = await stripe.customers.retrieve(stripeCustomerId);

  if (customer.deleted) {
    throw new Error('Customer has been deleted');
  }

  return customer as Stripe.Customer;
}

/**
 * Get upcoming invoice for a customer
 */
export async function getUpcomingInvoice(stripeCustomerId: string): Promise<Stripe.Invoice | null> {
  const stripe = getStripe();

  try {
    const invoice = await (stripe.invoices as any).retrieveUpcoming({
      customer: stripeCustomerId,
    });
    return invoice;
  } catch (error) {
    // No upcoming invoice
    return null;
  }
}

/**
 * Record usage for metered billing (future feature)
 */
export async function recordUsage(
  subscriptionItemId: string,
  quantity: number,
  action: 'increment' | 'set' = 'increment'
): Promise<any> {
  const stripe = getStripe();

  const usageRecord = await (stripe.subscriptionItems as any).createUsageRecord(subscriptionItemId, {
    quantity,
    action,
    timestamp: Math.floor(Date.now() / 1000),
  });

  return usageRecord;
}

/**
 * Check if user has exceeded their tier limits
 */
export function checkUsageLimit(
  tier: SubscriptionTier,
  resource: keyof typeof SUBSCRIPTION_TIERS.free.limits,
  currentUsage: number
): { allowed: boolean; limit: number; remaining: number } {
  const limits = SUBSCRIPTION_TIERS[tier].limits;
  const limit = limits[resource] as number;

  // -1 means unlimited
  if (limit === -1) {
    return {
      allowed: true,
      limit: -1,
      remaining: -1,
    };
  }

  const remaining = Math.max(0, limit - currentUsage);
  const allowed = currentUsage < limit;

  return {
    allowed,
    limit,
    remaining,
  };
}

/**
 * Get tier by price ID
 */
export function getTierByPriceId(priceId: string): SubscriptionTier | null {
  for (const [tier, config] of Object.entries(SUBSCRIPTION_TIERS)) {
    if (config.priceId === priceId) {
      return tier as SubscriptionTier;
    }
  }
  return null;
}

/**
 * Format subscription status for display
 */
export function formatSubscriptionStatus(status: string): string {
  const statusMap: Record<string, string> = {
    active: 'Active',
    trialing: 'Trial',
    past_due: 'Past Due',
    canceled: 'Canceled',
    incomplete: 'Incomplete',
    incomplete_expired: 'Expired',
    unpaid: 'Unpaid',
    paused: 'Paused',
  };

  return statusMap[status] || status;
}

/**
 * Calculate trial end date
 */
export function calculateTrialEnd(days: number = 14): Date {
  const now = new Date();
  now.setDate(now.getDate() + days);
  return now;
}
