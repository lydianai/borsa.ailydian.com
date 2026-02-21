/**
 * ✅ AUTH CHECK API ENDPOINT
 * Session durumunu kontrol eder
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function GET(_request: NextRequest) {
  try {
    const cookieStore = await cookies();
    const authCookie = cookieStore.get('sardag_auth');

    return NextResponse.json({
      authenticated: authCookie?.value === 'authenticated',
    });
  } catch (error) {
    console.error('[Auth] Check error:', error);
    return NextResponse.json({ authenticated: false });
  }
}
