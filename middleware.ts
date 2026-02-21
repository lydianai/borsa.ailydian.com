import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * 🚧 MAINTENANCE MODE MIDDLEWARE
 *
 * Sistem çalışırlığına ZARAR VERMEDEN geçici olarak yapım aşaması modu:
 *
 * AÇMAK İÇİN:
 * 1. .env.local dosyasına ekle: NEXT_PUBLIC_MAINTENANCE_MODE=true
 * 2. Vercel'e deploy et
 *
 * KAPATMAK İÇİN:
 * 1. .env.local'dan sil veya: NEXT_PUBLIC_MAINTENANCE_MODE=false
 * 2. Vercel'e deploy et
 *
 * NOT: API'ler çalışmaya devam eder (Telegram, Python servisleri, background jobs)
 */

export function middleware(request: NextRequest) {
  // Maintenance mode kontrolü
  const isMaintenanceMode = process.env.NEXT_PUBLIC_MAINTENANCE_MODE === 'true';

  // Maintenance mode kapalıysa, normal flow
  if (!isMaintenanceMode) {
    return NextResponse.next();
  }

  // Maintenance sayfasına zaten gidiyorsa, loop engellemek için geç
  if (request.nextUrl.pathname === '/maintenance') {
    return NextResponse.next();
  }

  // API route'larına izin ver (backend çalışmaya devam etsin)
  if (request.nextUrl.pathname.startsWith('/api')) {
    return NextResponse.next();
  }

  // Static dosyalara izin ver (_next, favicon, images, vb.)
  if (
    request.nextUrl.pathname.startsWith('/_next') ||
    request.nextUrl.pathname.startsWith('/favicon') ||
    request.nextUrl.pathname.startsWith('/images') ||
    request.nextUrl.pathname.startsWith('/icons')
  ) {
    return NextResponse.next();
  }

  // Maintenance sayfasına yönlendir
  return NextResponse.redirect(new URL('/maintenance', request.url));
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
};
