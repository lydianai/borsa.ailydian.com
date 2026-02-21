/**
 * Payment Verification Middleware
 *
 * White-hat compliance: Payment verification for premium features
 * Ensures users have active subscriptions before accessing paid features
 */

import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth/helpers';
import { prisma } from '@/lib/prisma';

export interface PaymentVerification {
  isValid: boolean;
  user?: {
    id: string;
    email: string;
    subscriptionTier: string;
    hasActivePayment: boolean;
    currentPeriodEnd: Date | null;
  };
  error?: string;
}

/**
 * Verify if user has active payment
 * Admin users always pass this check
 */
export async function verifyPayment(): Promise<PaymentVerification> {
  try {
    const session = await getSession();

    if (!session?.user?.id) {
      return {
        isValid: false,
        error: 'Authentication required',
      };
    }

    const user = await prisma.user.findUnique({
      where: { id: session.user.id },
      select: {
        id: true,
        email: true,
        isAdmin: true,
        hasActivePayment: true,
        subscriptionTier: true,
        subscriptionStatus: true,
        currentPeriodEnd: true,
      },
    });

    if (!user) {
      return {
        isValid: false,
        error: 'User not found',
      };
    }

    // Admin users always have access
    if (user.isAdmin) {
      return {
        isValid: true,
        user: {
          id: user.id,
          email: user.email,
          subscriptionTier: user.subscriptionTier,
          hasActivePayment: true,
          currentPeriodEnd: user.currentPeriodEnd,
        },
      };
    }

    // Check if user has active payment
    if (!user.hasActivePayment) {
      return {
        isValid: false,
        error: 'Active subscription required. Please upgrade your plan.',
      };
    }

    // Check if subscription is still valid
    if (user.currentPeriodEnd && user.currentPeriodEnd < new Date()) {
      // Mark as expired
      await prisma.user.update({
        where: { id: user.id },
        data: {
          hasActivePayment: false,
          subscriptionStatus: 'expired',
        },
      });

      return {
        isValid: false,
        error: 'Subscription expired. Please renew your plan.',
      };
    }

    return {
      isValid: true,
      user: {
        id: user.id,
        email: user.email,
        subscriptionTier: user.subscriptionTier,
        hasActivePayment: user.hasActivePayment,
        currentPeriodEnd: user.currentPeriodEnd,
      },
    };
  } catch (error: any) {
    console.error('Payment verification error:', error);
    return {
      isValid: false,
      error: 'Payment verification failed',
    };
  }
}

/**
 * Middleware wrapper for API routes
 * Returns 403 if payment is not valid
 */
export async function requirePaymentMiddleware(
  req: NextRequest,
  handler: (req: NextRequest) => Promise<NextResponse>
): Promise<NextResponse> {
  const verification = await verifyPayment();

  if (!verification.isValid) {
    return NextResponse.json(
      {
        error: verification.error || 'Payment verification failed',
        requiresPayment: true,
      },
      { status: 403 }
    );
  }

  return handler(req);
}

/**
 * Check if user has specific subscription tier
 */
export async function hasSubscriptionTier(
  minTier: 'free' | 'starter' | 'pro' | 'enterprise'
): Promise<boolean> {
  const verification = await verifyPayment();

  if (!verification.isValid || !verification.user) {
    return false;
  }

  const tierHierarchy = {
    free: 0,
    starter: 1,
    pro: 2,
    enterprise: 3,
  };

  const userTierLevel =
    tierHierarchy[verification.user.subscriptionTier as keyof typeof tierHierarchy] || 0;
  const requiredTierLevel = tierHierarchy[minTier];

  return userTierLevel >= requiredTierLevel;
}

/**
 * Get user's subscription info
 */
export async function getSubscriptionInfo() {
  const verification = await verifyPayment();

  if (!verification.isValid) {
    return null;
  }

  return verification.user;
}

/**
 * Check if feature is available for user's plan
 */
export const FEATURE_ACCESS = {
  // Free tier
  basicSignals: ['free', 'starter', 'pro', 'enterprise'],
  tradingView: ['free', 'starter', 'pro', 'enterprise'],

  // Starter tier
  aiSignals: ['starter', 'pro', 'enterprise'],
  notifications: ['starter', 'pro', 'enterprise'],
  alerts: ['starter', 'pro', 'enterprise'],

  // Pro tier
  quantumSignals: ['pro', 'enterprise'],
  backtesting: ['pro', 'enterprise'],
  exchangeAPI: ['pro', 'enterprise'],
  tradingBot: ['pro', 'enterprise'],
  riskManagement: ['pro', 'enterprise'],

  // Enterprise tier
  multipleExchanges: ['enterprise'],
  advancedAnalytics: ['enterprise'],
  prioritySupport: ['enterprise'],
  customStrategies: ['enterprise'],
} as const;

export async function hasFeatureAccess(
  feature: keyof typeof FEATURE_ACCESS
): Promise<boolean> {
  const verification = await verifyPayment();

  if (!verification.isValid || !verification.user) {
    return false;
  }

  const allowedTiers = FEATURE_ACCESS[feature];
  return allowedTiers.includes(verification.user.subscriptionTier as any);
}
