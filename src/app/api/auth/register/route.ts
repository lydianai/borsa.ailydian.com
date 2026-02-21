/**
 * User Registration API
 *
 * White-hat compliance: Secure user registration with email verification
 * Flow: Register → Email verification → Admin approval → Active
 */

import { NextRequest, NextResponse } from 'next/server';
import bcrypt from 'bcryptjs';
import { v4 as uuidv4 } from 'uuid';
import { prisma } from '@/lib/prisma';
import { sendVerificationEmail } from '@/lib/email/service';
import { getStripe } from '@/lib/stripe/config';

interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
}

export async function POST(req: NextRequest) {
  try {
    const body: RegisterRequest = await req.json();
    const { email, username, password, confirmPassword } = body;

    // Validation
    if (!email || !username || !password) {
      return NextResponse.json(
        { error: 'Email, kullanıcı adı ve şifre gereklidir' },
        { status: 400 }
      );
    }

    if (password !== confirmPassword) {
      return NextResponse.json(
        { error: 'Şifreler eşleşmiyor' },
        { status: 400 }
      );
    }

    if (password.length < 8) {
      return NextResponse.json(
        { error: 'Şifre en az 8 karakter olmalıdır' },
        { status: 400 }
      );
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { error: 'Geçersiz email adresi' },
        { status: 400 }
      );
    }

    // Username validation
    if (username.length < 3) {
      return NextResponse.json(
        { error: 'Kullanıcı adı en az 3 karakter olmalıdır' },
        { status: 400 }
      );
    }

    // Check if user already exists
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email },
          { username },
        ],
      },
    });

    if (existingUser) {
      if (existingUser.email === email) {
        return NextResponse.json(
          { error: 'Bu email adresi zaten kullanımda' },
          { status: 400 }
        );
      }
      if (existingUser.username === username) {
        return NextResponse.json(
          { error: 'Bu kullanıcı adı zaten kullanımda' },
          { status: 400 }
        );
      }
    }

    // Hash password
    const passwordHash = await bcrypt.hash(password, 12);

    // Generate email verification token
    const emailVerificationToken = uuidv4();
    const emailVerificationExpires = new Date();
    emailVerificationExpires.setHours(emailVerificationExpires.getHours() + 24); // 24 hours

    // Create Stripe customer
    let stripeCustomerId: string | null = null;
    try {
      const stripe = getStripe();
      const customer = await stripe.customers.create({
        email,
        name: username,
        metadata: {
          username,
        },
      });
      stripeCustomerId = customer.id;
    } catch (error) {
      console.error('Stripe customer creation error:', error);
      // Continue without Stripe customer (can be created later)
    }

    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        username,
        passwordHash,
        emailVerificationToken,
        emailVerificationExpires,
        stripeCustomerId,
        role: 'user',
        isAdmin: false,
        emailVerified: false,
        isApproved: false,
        hasActivePayment: false,
        subscriptionTier: 'free',
        subscriptionStatus: 'inactive',
      },
    });

    // Send verification email
    try {
      await sendVerificationEmail(email, emailVerificationToken);
    } catch (error) {
      console.error('Email sending error:', error);
      // User is created but email failed - they can request resend later
    }

    // Create admin notification
    await prisma.adminNotification.create({
      data: {
        type: 'new_user_registration',
        title: 'Yeni Kullanıcı Kaydı',
        message: `${username} (${email}) kaydoldu ve email doğrulama bekliyor.`,
        userId: user.id,
        userEmail: email,
        actionUrl: `/admin/users?userId=${user.id}`,
        metadata: {
          username,
          registeredAt: new Date().toISOString(),
        },
      },
    });

    return NextResponse.json({
      success: true,
      message: 'Kayıt başarılı! Lütfen emailinizi kontrol edin ve doğrulama linkine tıklayın.',
      user: {
        id: user.id,
        email: user.email,
        username: user.username,
      },
    });
  } catch (error) {
    console.error('Registration error:', error);
    return NextResponse.json(
      { error: 'Kayıt sırasında bir hata oluştu' },
      { status: 500 }
    );
  }
}
