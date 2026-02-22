/**
 * 🚪 LOGOUT API ENDPOINT
 * Session'ı sonlandırır
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST(_request: NextRequest) {
  try {
    const cookieStore = await cookies();

    // Clear auth cookie
    cookieStore.delete('lytrade_auth');

    return NextResponse.json({
      success: true,
      message: 'Çıkış başarılı!',
    });
  } catch (error) {
    console.error('[Auth] Logout error:', error);
    return NextResponse.json(
      { success: false, message: 'Sunucu hatası' },
      { status: 500 }
    );
  }
}
