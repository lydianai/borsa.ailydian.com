/**
 * LOGIN API ENDPOINT with 2FA Support
 * Authentication endpoint with Google Authenticator
 *
 * Set credentials via environment variables:
 * - ADMIN_USERNAME (default: admin)
 * - ADMIN_PASSWORD (required in production)
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import speakeasy from 'speakeasy';
import { get2FAData, verifyBackupCode } from '@/lib/2fa-store';

const VALID_USERNAME = process.env.ADMIN_USERNAME || 'admin';
const VALID_PASSWORD = process.env.ADMIN_PASSWORD || (process.env.NODE_ENV === 'development' ? 'admin' : '');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { username, password, token, backupCode } = body;

    // Step 1: Validate username & password
    if (username !== VALID_USERNAME || password !== VALID_PASSWORD) {
      return NextResponse.json(
        { success: false, message: 'Kullanıcı adı veya şifre hatalı!' },
        { status: 401 }
      );
    }

    // Step 2: Check if user has 2FA enabled
    const twoFAData = await get2FAData(username);

    if (twoFAData && twoFAData.enabled) {
      // 2FA is enabled - require token or backup code
      if (!token && !backupCode) {
        return NextResponse.json({
          success: false,
          requires2FA: true,
          message: '2FA kodu gerekli',
        });
      }

      let verified = false;

      // Check backup code first (if provided)
      if (backupCode) {
        verified = await verifyBackupCode(username, backupCode);
        if (!verified) {
          return NextResponse.json(
            { success: false, message: 'Geçersiz yedek kod!' },
            { status: 401 }
          );
        }
      }
      // Check TOTP token
      else if (token) {
        verified = speakeasy.totp.verify({
          secret: twoFAData.secret,
          encoding: 'base32',
          token: token.replace(/\s/g, ''),
          window: 2,
        });

        if (!verified) {
          return NextResponse.json(
            { success: false, message: 'Geçersiz 2FA kodu!' },
            { status: 401 }
          );
        }
      }

      if (!verified) {
        return NextResponse.json(
          { success: false, message: '2FA doğrulaması başarısız!' },
          { status: 401 }
        );
      }
    }

    // Step 3: All checks passed - Create session
    const cookieStore = await cookies();
    cookieStore.set('lytrade_auth', 'authenticated', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24, // 24 hours
      path: '/',
    });

    return NextResponse.json({
      success: true,
      message: 'Giriş başarılı!',
    });
  } catch (error) {
    console.error('[Auth] Login error:', error);
    return NextResponse.json(
      { success: false, message: 'Sunucu hatası' },
      { status: 500 }
    );
  }
}
