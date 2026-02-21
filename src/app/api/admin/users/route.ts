/**
 * Admin Users API
 *
 * White-hat compliance: Admin-only user management
 */

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth/helpers';
import { prisma } from '@/lib/prisma';

// Force dynamic rendering
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

/**
 * GET /api/admin/users - List all users
 */
export async function GET(req: NextRequest) {
  try {
    await requireAdmin();

    const searchParams = req.nextUrl.searchParams;
    const status = searchParams.get('status'); // pending, approved, all

    let where: any = {};

    if (status === 'pending') {
      where = {
        emailVerified: true,
        isApproved: false,
        isAdmin: false,
      };
    } else if (status === 'approved') {
      where = {
        isApproved: true,
        isAdmin: false,
      };
    }

    const users = await prisma.user.findMany({
      where,
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
        stripeCustomerId: true,
        createdAt: true,
        lastLoginAt: true,
        approvedAt: true,
        approvedBy: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      users,
      count: users.length,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Yetkisiz erişim' },
      { status: error.message === 'Admin access required' ? 403 : 500 }
    );
  }
}
