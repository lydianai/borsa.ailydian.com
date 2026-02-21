/**
 * Authentication Helper Functions
 *
 * White-hat compliance: Secure authentication utilities
 */

import { auth } from './auth';
import { prisma } from '@/lib/prisma';

/**
 * Get current session (server-side)
 * Updated for next-auth v5 beta - uses auth() instead of getServerSession()
 */
export async function getSession() {
  return await auth();
}

/**
 * Get current user (server-side)
 */
export async function getCurrentUser() {
  const session = await getSession();

  if (!session?.user?.id) {
    return null;
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: {
      id: true,
      email: true,
      username: true,
      role: true,
      isAdmin: true,
      isApproved: true,
      emailVerified: true,
      hasActivePayment: true,
      subscriptionTier: true,
      subscriptionStatus: true,
      currentPeriodEnd: true,
      createdAt: true,
    },
  });

  return user;
}

/**
 * Check if user is admin
 */
export async function isAdmin() {
  const user = await getCurrentUser();
  return user?.isAdmin === true;
}

/**
 * Check if user has active payment
 */
export async function hasActivePayment() {
  const user = await getCurrentUser();
  return user?.hasActivePayment === true || user?.isAdmin === true;
}

/**
 * Check if user is approved
 */
export async function isApproved() {
  const user = await getCurrentUser();
  return user?.isApproved === true || user?.isAdmin === true;
}

/**
 * Require authentication (throws if not authenticated)
 */
export async function requireAuth() {
  const user = await getCurrentUser();

  if (!user) {
    throw new Error('Authentication required');
  }

  return user;
}

/**
 * Require admin (throws if not admin)
 */
export async function requireAdmin() {
  const user = await getCurrentUser();

  if (!user?.isAdmin) {
    throw new Error('Admin access required');
  }

  return user;
}

/**
 * Require active payment (throws if no payment)
 */
export async function requirePayment() {
  const user = await getCurrentUser();

  if (!user) {
    throw new Error('Authentication required');
  }

  if (!user.hasActivePayment && !user.isAdmin) {
    throw new Error('Active subscription required');
  }

  return user;
}
