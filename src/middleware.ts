import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { generateCSRFToken, validateCSRFToken, CSRF_COOKIE_OPTIONS, CSRF_HEADER } from "./lib/security/csrf";

/**
 * Statik asset & servis dosyalarını middleware dışında bırak.
 * Böylece CSS/JS isteklerine yanlışlıkla HTML dönülmez.
 */
const BYPASS_PREFIXES = [
  "/_next/static",
  "/_next/image",
  "/favicon.ico",
  "/icons",
  "/manifest.json",
  "/robots.txt",
  "/sitemap.xml",
  "/sw.js",
  "/public",
];
const BYPASS_EXTS = [
  ".css",".js",".mjs",".cjs",".map",
  ".png",".jpg",".jpeg",".webp",".gif",".svg",".ico",
  ".json",".txt",".xml",".mp3",".mp4",".wav",".woff",".woff2",".ttf"
];

// API endpoints that require CSRF protection
const CSRF_PROTECTED_API_PREFIXES = [
  "/api/bot",
  "/api/auto-trading",
  "/api/settings",
  "/api/push",
  "/api/telegram",
];

// ============================================
// BLOCKED BOTS - Enterprise Security
// ============================================
const BLOCKED_BOTS = [
  // SEO Tool Scrapers
  'AhrefsBot', 'AhrefsSiteAudit', 'SemrushBot', 'SplitSignalBot',
  'MJ12bot', 'DotBot', 'rogerbot', 'MojeekBot', 'BLEXBot',

  // AI Scrapers
  'GPTBot', 'ChatGPT-User', 'OpenAI-SearchBot', 'Google-Extended',
  'anthropic-ai', 'Claude-Web', 'CCBot', 'omgili', 'omgilibot',
  'FacebookBot', 'Meta-ExternalAgent', 'meta-externalagent',
  'Bytespider', 'ByteDance', 'Diffbot', 'PerplexityBot', 'cohere-ai',
  'Applebot-Extended',

  // Python/Scrapy
  'Scrapy', 'Python-urllib', 'python-requests', 'urllib',

  // Archive Crawlers
  'ia_archiver', 'archive.org_bot', 'Wget', 'curl',

  // Email Harvesters
  'EmailCollector', 'EmailSiphon', 'EmailWolf', 'ExtractorPro', 'CherryPicker',

  // Content Scrapers
  'WebReaper', 'WebCopier', 'Offline Explorer', 'HTTrack', 'WebZip',
  'Microsoft.URL.Control', 'penthesilea',

  // Spam & Malicious
  'grub-client', 'grub', 'looksmart', 'larbin', 'psbot',
  'Python-webchecker', 'NetMechanic', 'URL_Spider_SQL',
  'Collector', 'WebCollage', 'CrunchBot',

  // Click Fraud
  'Clickagy', 'MauiBot', 'DomainStatsBot', 'SurveyBot',

  // Aggressive SEO
  'SeznamBot', 'SEOkicks', 'spbot', 'SearchmetricsBot',
  'SEOlyticsCrawler', 'LinkpadBot', 'serpstatbot',

  // Image Scrapers
  'Pinterestbot', 'Pinterest',

  // Generic Bad Bots
  'sentibot', 'niki-bot', 'CazoodleBot', 'discobot', 'ecxi',
  'GT::WWW', 'heritrix', 'HTTP::Lite', 'id-search', 'IDBot',
  'Indy Library', 'IRLbot', 'LinksManager.com_bot', 'linkwalker',
  'lwp-trivial', 'MFC_Tear_Sample', 'Missigua Locator',
  'panscient.com', 'PECL::HTTP', 'PHPCrawl', 'PleaseCrawl',
  'SBIder', 'Snoopy', 'Steeler', 'URI::Fetch', 'Web Sucker',
  'webalta', 'WebAuto', 'WebBandit', 'WebFetch', 'WebGo IS',
  'WebLeacher', 'WebSauger', 'Website Quester', 'Webster',
  'WebStripper', 'WebWhacker', 'Widow', 'WWW-Collector-E',
  'Xenu', 'Zade', 'Zeus', 'ZyBORG', 'coccoc', 'Incutio',
  'lmspider', 'memorybot', 'serf', 'Unknown', 'uptime files'
];

// Rate-limited bots (slower crawl allowed)
const RATE_LIMITED_BOTS = [
  'Yandex', 'YandexBot', 'Baiduspider', 'Slurp', 'Yahoo! Slurp',
  'DuckDuckBot'
];

/**
 * Check if User-Agent contains any blocked bot patterns
 */
function isBlockedBot(userAgent: string): boolean {
  if (!userAgent) return false;
  const ua = userAgent.toLowerCase();
  return BLOCKED_BOTS.some(bot => ua.includes(bot.toLowerCase()));
}

/**
 * Check if User-Agent is a rate-limited bot
 */
function isRateLimitedBot(userAgent: string): boolean {
  if (!userAgent) return false;
  const ua = userAgent.toLowerCase();
  return RATE_LIMITED_BOTS.some(bot => ua.includes(bot.toLowerCase()));
}

export function middleware(req: NextRequest) {
  const { pathname } = new URL(req.url);

  // Prefix bazlı bypass
  if (BYPASS_PREFIXES.some(p => pathname.startsWith(p))) {
    return NextResponse.next();
  }
  // Uzantı bazlı bypass
  if (BYPASS_EXTS.some(ext => pathname.endsWith(ext))) {
    return NextResponse.next();
  }

  // ============================================
  // BOT PROTECTION - Block malicious bots
  // ============================================
  const userAgent = req.headers.get('user-agent') || '';

  // Block malicious/unwanted bots
  if (isBlockedBot(userAgent)) {
    return new NextResponse('Forbidden - Bot access denied', {
      status: 403,
      headers: {
        'Content-Type': 'text/plain',
        'X-Robots-Tag': 'noindex, nofollow, noarchive',
      },
    });
  }

  // Create response with security headers
  const response = NextResponse.next();

  // Add rate-limiting header for specific bots
  if (isRateLimitedBot(userAgent)) {
    response.headers.set('X-Crawl-Rate', 'Slow');
    response.headers.set('Retry-After', '60');
  }

  // ============================================
  // SECURITY HEADERS (White-hat security best practices)
  // ============================================

  // Prevent clickjacking attacks
  response.headers.set('X-Frame-Options', 'DENY');

  // Prevent MIME type sniffing
  response.headers.set('X-Content-Type-Options', 'nosniff');

  // Enable XSS protection (legacy but still useful)
  response.headers.set('X-XSS-Protection', '1; mode=block');

  // Referrer policy
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Permissions policy (disable unused features)
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=(), payment=()'
  );

  // ============================================
  // SEO & PERFORMANCE HEADERS
  // ============================================

  // Cache control for better performance
  if (pathname.startsWith('/_next/static') || pathname.match(/\.(jpg|jpeg|png|webp|gif|svg|ico|woff|woff2|ttf)$/)) {
    response.headers.set('Cache-Control', 'public, max-age=31536000, immutable');
  } else if (pathname.startsWith('/api/')) {
    response.headers.set('Cache-Control', 'no-store, must-revalidate');
  } else {
    response.headers.set('Cache-Control', 'public, max-age=0, must-revalidate');
  }

  // Language headers for SEO
  response.headers.set('Content-Language', 'tr, en');

  // Vary header for caching
  response.headers.set('Vary', 'Accept-Encoding, Accept-Language');

  // Content Security Policy
  const cspDirectives = [
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://vercel.live https://va.vercel-scripts.com",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https: blob:",
    "font-src 'self' data:",
    "connect-src 'self' https://api.binance.com https://fapi.binance.com https://*.groq.com wss://stream.binance.com wss://fstream.binance.com",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; ');
  response.headers.set('Content-Security-Policy', cspDirectives);

  // HSTS (HTTP Strict Transport Security) - Only in production
  if (process.env.NODE_ENV === 'production') {
    response.headers.set(
      'Strict-Transport-Security',
      'max-age=31536000; includeSubDomains; preload'
    );
  }

  // ============================================
  // CSRF PROTECTION (For state-changing operations)
  // ============================================

  // Check if this is a CSRF-protected API endpoint
  const isCSRFProtectedAPI = CSRF_PROTECTED_API_PREFIXES.some(prefix =>
    pathname.startsWith(prefix)
  );

  if (isCSRFProtectedAPI) {
    // For POST, PUT, PATCH, DELETE requests, validate CSRF token
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(req.method)) {
      const cookieToken = req.cookies.get(CSRF_COOKIE_OPTIONS.name)?.value;
      const headerToken = req.headers.get(CSRF_HEADER);

      if (!validateCSRFToken(cookieToken, headerToken)) {
        return new NextResponse(
          JSON.stringify({
            success: false,
            error: {
              message: 'CSRF validation failed',
              code: 'CSRF_VALIDATION_ERROR',
            },
          }),
          {
            status: 403,
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );
      }
    }
  }

  // Generate and set CSRF token cookie for all requests (if not already set)
  if (!req.cookies.get(CSRF_COOKIE_OPTIONS.name)) {
    const csrfToken = generateCSRFToken();
    response.cookies.set(CSRF_COOKIE_OPTIONS.name, csrfToken, {
      httpOnly: CSRF_COOKIE_OPTIONS.httpOnly,
      secure: CSRF_COOKIE_OPTIONS.secure,
      sameSite: CSRF_COOKIE_OPTIONS.sameSite,
      path: CSRF_COOKIE_OPTIONS.path,
      maxAge: CSRF_COOKIE_OPTIONS.maxAge,
    });
  }

  return response;
}

export const config = {
  matcher: [
    // _next/image|static ve temel dosyaları kapsam dışı bırak
    "/((?!_next/static|_next/image|favicon.ico|manifest.json|icons|robots.txt|sitemap.xml|sw.js).*)",
  ],
};
