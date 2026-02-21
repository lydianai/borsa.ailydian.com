/**
 * Email Verification API
 *
 * White-hat compliance: Verifies user email and notifies admin
 */

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { sendAdminNotification } from '@/lib/email/service';

export async function GET(req: NextRequest) {
  try {
    const searchParams = req.nextUrl.searchParams;
    const token = searchParams.get('token');

    if (!token) {
      return NextResponse.json(
        { error: 'Doğrulama token\'ı gereklidir' },
        { status: 400 }
      );
    }

    // Find user by verification token
    const user = await prisma.user.findUnique({
      where: { emailVerificationToken: token },
    });

    if (!user) {
      return NextResponse.json(
        { error: 'Geçersiz doğrulama token\'ı' },
        { status: 400 }
      );
    }

    // Check if already verified
    if (user.emailVerified) {
      return NextResponse.json({
        success: true,
        message: 'Email adresiniz zaten doğrulanmış',
        alreadyVerified: true,
      });
    }

    // Check if token expired
    if (user.emailVerificationExpires && new Date() > user.emailVerificationExpires) {
      return NextResponse.json(
        { error: 'Doğrulama token\'ı süresi dolmuş. Lütfen yeni bir token talep edin.' },
        { status: 400 }
      );
    }

    // Update user as verified
    await prisma.user.update({
      where: { id: user.id },
      data: {
        emailVerified: true,
        emailVerificationToken: null,
        emailVerificationExpires: null,
      },
    });

    // Send admin notification
    try {
      await sendAdminNotification(user.email, user.username, user.id);
    } catch (error) {
      console.error('Admin notification error:', error);
      // Continue even if notification fails
    }

    // Update admin notification
    await prisma.adminNotification.create({
      data: {
        type: 'new_user_registration',
        title: 'Kullanıcı Email Doğruladı - Onay Bekliyor',
        message: `${user.username} (${user.email}) email adresini doğruladı ve onayınızı bekliyor.`,
        userId: user.id,
        userEmail: user.email,
        actionUrl: `/admin/users/${user.id}`,
        metadata: {
          username: user.username,
          verifiedAt: new Date().toISOString(),
        },
      },
    });

    return NextResponse.json({
      success: true,
      message: 'Email adresiniz başarıyla doğrulandı! Admin onayı bekleniyor.',
    });
  } catch (error) {
    console.error('Email verification error:', error);
    return NextResponse.json(
      { error: 'Email doğrulama sırasında bir hata oluştu' },
      { status: 500 }
    );
  }
}
