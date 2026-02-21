/**
 * Admin User Approval API
 *
 * White-hat compliance: Admin approves users
 */

import { NextRequest, NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth/helpers';
import { prisma } from '@/lib/prisma';
import { sendApprovalConfirmation } from '@/lib/email/service';

// Force dynamic rendering - this route requires runtime data
export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function POST(
  _req: NextRequest,
  { params }: { params: Promise<{ userId: string }> }
) {
  try {
    const admin = await requireAdmin();
    const { userId } = await params;

    // Get user
    const user = await prisma.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      return NextResponse.json(
        { error: 'Kullanıcı bulunamadı' },
        { status: 404 }
      );
    }

    if (user.isApproved) {
      return NextResponse.json(
        { error: 'Kullanıcı zaten onaylanmış' },
        { status: 400 }
      );
    }

    if (!user.emailVerified) {
      return NextResponse.json(
        { error: 'Kullanıcı email adresini henüz doğrulamamış' },
        { status: 400 }
      );
    }

    // Approve user
    await prisma.user.update({
      where: { id: userId },
      data: {
        isApproved: true,
        approvedBy: admin.id,
        approvedAt: new Date(),
      },
    });

    // Send approval email
    try {
      await sendApprovalConfirmation(user.email, user.username);
    } catch (error) {
      console.error('Approval email error:', error);
      // Continue even if email fails
    }

    // Mark notification as read
    await prisma.adminNotification.updateMany({
      where: {
        userId: userId,
        isRead: false,
      },
      data: {
        isRead: true,
        readAt: new Date(),
      },
    });

    return NextResponse.json({
      success: true,
      message: 'Kullanıcı başarıyla onaylandı',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Onaylama başarısız' },
      { status: error.message === 'Admin access required' ? 403 : 500 }
    );
  }
}
