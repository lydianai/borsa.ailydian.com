// Ailydian Signal - AI-Powered Crypto Trading Platform
import type { Metadata, Viewport } from 'next';
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/next';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { MockDataBanner } from '@/components/MockDataBanner';
import { Providers } from '@/components/Providers';
// import CriticalNewsAlertBanner from '@/components/news/CriticalNewsAlertBanner'; // DEVRE DIŞI - Mock haber uyarıları kaldırıldı
// import { ConditionalAuthWrapper } from '@/components/ConditionalAuthWrapper';
import './globals.css';
import '@/styles/animations.css';

const siteUrl = process.env.NEXT_PUBLIC_APP_URL || 'https://borsa.ailydian.com';

export const metadata: Metadata = {
  title: {
    default: 'Ailydian Signal - Yapay Zeka Destekli Kripto Para Trading Sinyalleri',
    template: '%s | Ailydian Signal'
  },
  description: '617 kripto para çifti için gerçek zamanlı Ailydian yapay zeka trading sinyalleri. 18+ strateji, teknik analiz, whale takibi ve akıllı risk yönetimi ile profesyonel kripto trading platformu.',
  applicationName: 'Ailydian Signal',
  keywords: [
    // Türkçe ana kelimeler
    'Ailydian kripto sinyalleri',
    'kripto trading sinyalleri',
    'binance futures sinyalleri',
    'yapay zeka trading',
    'Ailydian yapay zeka',
    'teknik analiz',
    'whale tracker',
    'kripto trading bot',
    'futures trading',
    'bitcoin sinyalleri',
    'ethereum analiz',
    'altcoin sinyalleri',
    'risk yönetimi',
    'trading stratejileri',
    'anlık sinyal',
    'kripto piyasa analizi',
    'borsa sinyalleri',
    'kripto para takibi',
    'otomatik ticaret',
    'algoritmik trading',
    'piyasa tarayıcı',

    // İngilizce keywords (global SEO)
    'Ailydian crypto signals',
    'crypto trading signals',
    'Ailydian artificial intelligence',
    'binance futures signals',
    'cryptocurrency trading bot',
    'whale tracking',
    'technical analysis crypto',
    'real-time crypto signals',
    'bitcoin trading signals',
    'altcoin signals',
    'crypto market scanner',
    'automated trading',
    'algorithmic trading',
    'perpetual futures',
    'crypto portfolio management',

    // Uzun kuyruk keywords
    'Ailydian trading platform',
    'en iyi kripto sinyalleri 2025',
    'ücretsiz kripto analiz',
    '617 kripto para tarama',
    'yapay zeka destekli yatırım',
    'whale takip sistemi',
    'conservative buy signals',
    'quantum trading signals',
    'real-time market analysis',
    'Ailydian signal platform'
  ],
  authors: [{ name: 'Ailydian', url: 'https://ailydian.com' }],
  creator: 'Ailydian',
  publisher: 'Ailydian',
  manifest: '/manifest.json',
  metadataBase: new URL(siteUrl),

  // Open Graph (Facebook, LinkedIn, WhatsApp)
  openGraph: {
    type: 'website',
    locale: 'tr_TR',
    alternateLocale: ['en_US', 'en_GB', 'de_DE', 'fr_FR', 'es_ES', 'ru_RU', 'ar_SA', 'zh_CN', 'ja_JP'],
    url: siteUrl,
    siteName: 'Ailydian Signal',
    title: 'Ailydian Signal - Yapay Zeka Destekli Kripto Trading Platformu',
    description: '617 kripto para için gerçek zamanlı Ailydian yapay zeka sinyalleri. 18+ strateji ile profesyonel trading analizi. Whale takibi, teknik göstergeler ve akıllı risk yönetimi.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Ailydian Signal - AI-Powered Crypto Trading Platform',
        type: 'image/png',
      },
      {
        url: '/og-image-square.png',
        width: 1200,
        height: 1200,
        alt: 'Ailydian Signal Logo',
        type: 'image/png',
      }
    ],
    emails: ['info@ailydian.com'],
  },

  // Twitter Card
  twitter: {
    card: 'summary_large_image',
    title: 'Ailydian Signal - AI Kripto Trading Sinyalleri',
    description: '617 kripto para için gerçek zamanlı Ailydian yapay zeka trading sinyalleri. Profesyonel analiz, whale takibi ve risk yönetimi.',
    images: ['/og-image.png'],
    creator: '@ailydian',
    site: '@ailydian',
  },

  // Icons & Favicons
  icons: {
    icon: [
      { url: '/favicon.ico', sizes: 'any' },
      { url: '/icon.png', sizes: '32x32', type: 'image/png' },
      { url: '/icons/icon-96x96.png', sizes: '96x96', type: 'image/png' },
      { url: '/icons/icon-192x192.png', sizes: '192x192', type: 'image/png' },
    ],
    apple: [
      { url: '/icons/apple-touch-icon.png', sizes: '180x180', type: 'image/png' },
      { url: '/icons/icon-152x152.png', sizes: '152x152', type: 'image/png' },
      { url: '/icons/icon-192x192.png', sizes: '192x192', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
  },

  // Apple Web App
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Ailydian Signal',
    startupImage: [
      {
        url: '/icons/apple-splash-2048-2732.png',
        media: '(device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2)',
      },
      {
        url: '/icons/apple-splash-1668-2388.png',
        media: '(device-width: 834px) and (device-height: 1194px) and (-webkit-device-pixel-ratio: 2)',
      },
    ],
  },

  // Additional Meta
  formatDetection: {
    telephone: false,
  },
  category: 'Finance',
  classification: 'Cryptocurrency Trading Platform',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },

  // Verification (optional - eklerseniz daha iyi)
  // verification: {
  //   google: 'your-google-verification-code',
  //   yandex: 'your-yandex-verification-code',
  // },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5, // ✅ Allow zoom for accessibility
  minimumScale: 1,
  userScalable: true, // ✅ Better for accessibility
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#0a0a0a' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' }
  ],
  viewportFit: 'cover', // ✅ Notch device support (iPhone X+)
  colorScheme: 'dark',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="tr">
      <head>
        {/* Manifest & PWA */}
        <link rel="manifest" href="/manifest.json" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="apple-mobile-web-app-title" content="Ailydian Signal" />
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />

        {/* Multi-language support - hreflang */}
        <link rel="alternate" hrefLang="tr" href="https://borsa.ailydian.com" />
        <link rel="alternate" hrefLang="en" href="https://borsa.ailydian.com/en" />
        <link rel="alternate" hrefLang="x-default" href="https://borsa.ailydian.com" />

        {/* Canonical URL */}
        <link rel="canonical" href="https://borsa.ailydian.com" />

        {/* Preconnect for performance */}
        <link rel="preconnect" href="https://fstream.binance.com" />
        <link rel="preconnect" href="https://api.binance.com" />
        <link rel="dns-prefetch" href="https://fstream.binance.com" />
        <link rel="dns-prefetch" href="https://api.binance.com" />

        {/* Additional SEO meta tags */}
        <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1" />
        <meta name="googlebot" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1" />
        <meta name="bingbot" content="index, follow, max-image-preview:large, max-snippet:-1, max-video-preview:-1" />
        <meta name="geo.region" content="TR" />
        <meta name="geo.placename" content="Turkey" />
        <meta name="language" content="Turkish" />
        <meta name="distribution" content="global" />
        <meta name="rating" content="general" />
        <meta name="referrer" content="origin-when-cross-origin" />

        {/* Structured Data - Organization */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'Organization',
              name: 'Ailydian',
              url: 'https://ailydian.com',
              logo: 'https://borsa.ailydian.com/og-image-square.png',
              sameAs: [
                'https://twitter.com/ailydian',
                'https://linkedin.com/company/ailydian',
              ],
              contactPoint: {
                '@type': 'ContactPoint',
                contactType: 'Customer Support',
                email: 'info@ailydian.com',
                availableLanguage: ['Turkish', 'English']
              }
            }),
          }}
        />

        {/* Structured Data - WebApplication */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'WebApplication',
              name: 'Ailydian Signal',
              alternateName: 'Ailydian Kripto Trading Platform',
              description: '617 kripto para için gerçek zamanlı Ailydian yapay zeka trading sinyalleri. Yapay zeka destekli piyasa analizi, whale takibi ve profesyonel trading stratejileri.',
              url: 'https://borsa.ailydian.com',
              applicationCategory: 'FinanceApplication',
              operatingSystem: 'Web, iOS, Android',
              browserRequirements: 'Requires JavaScript. Requires HTML5.',
              offers: {
                '@type': 'Offer',
                category: 'Financial Analysis',
                price: '0',
                priceCurrency: 'TRY',
                availability: 'https://schema.org/InStock'
              },
              aggregateRating: {
                '@type': 'AggregateRating',
                ratingValue: '4.8',
                ratingCount: '1250',
                bestRating: '5',
                worstRating: '1'
              },
              featureList: [
                '617 Kripto Para Çifti Analizi',
                '18+ AI Trading Stratejisi',
                'Gerçek Zamanlı Sinyal Bildirimleri',
                'Whale Activity Tracker',
                'Akıllı Risk Yönetimi Sistemi',
                'Teknik Analiz Göstergeleri',
                'Perpetual Futures Analizi',
                'Multi-Timeframe Analysis',
                'Order Flow & CVD Analysis',
                'Funding Rate Monitoring',
                'Liquidation Heatmap',
                'Market Correlation Matrix'
              ],
              screenshot: [
                'https://borsa.ailydian.com/og-image.png',
                'https://borsa.ailydian.com/og-image-square.png'
              ],
              inLanguage: ['tr-TR', 'en-US'],
              creator: {
                '@type': 'Organization',
                name: 'Ailydian',
                url: 'https://ailydian.com'
              }
            }),
          }}
        />

        {/* Structured Data - BreadcrumbList */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'BreadcrumbList',
              itemListElement: [
                {
                  '@type': 'ListItem',
                  position: 1,
                  name: 'Ana Sayfa',
                  item: 'https://borsa.ailydian.com'
                },
                {
                  '@type': 'ListItem',
                  position: 2,
                  name: 'Market Scanner',
                  item: 'https://borsa.ailydian.com/market-scanner'
                },
                {
                  '@type': 'ListItem',
                  position: 3,
                  name: 'Trading Signals',
                  item: 'https://borsa.ailydian.com/trading-signals'
                },
                {
                  '@type': 'ListItem',
                  position: 4,
                  name: 'AI Signals',
                  item: 'https://borsa.ailydian.com/ai-signals'
                }
              ]
            }),
          }}
        />

        {/* Structured Data - FAQPage */}
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              '@context': 'https://schema.org',
              '@type': 'FAQPage',
              mainEntity: [
                {
                  '@type': 'Question',
                  name: 'Ailydian Signal nedir?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'Ailydian Signal, 617 kripto para çifti için gerçek zamanlı Ailydian yapay zeka destekli trading sinyalleri sunan profesyonel bir kripto analiz platformudur. 18+ yapay zeka stratejisi, whale takibi ve teknik analiz araçları ile yatırımcılara piyasa içgörüleri sağlar.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'Kaç kripto para çifti destekleniyor?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: '617 farklı kripto para çifti için gerçek zamanlı analiz ve sinyal desteği sunuyoruz. Binance Futures piyasasındaki tüm major ve altcoin çiftlerini kapsıyoruz.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'Yapay zeka trading sinyalleri nasıl çalışır?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'Ailydian yapay zeka sistemimiz 18+ farklı trading stratejisini birleştirerek, teknik göstergeler, whale hareketleri, order flow, funding rate ve CVD analizlerini gerçek zamanlı olarak değerlendirir ve al/sat sinyalleri üretir.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'Whale tracker özelliği nedir?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'Whale tracker özelliği, büyük yatırımcıların (whale) piyasadaki hareketlerini takip eder. Büyük hacimli işlemler, unusual aktiviteler ve whale pozisyonlarını analiz ederek öncü sinyaller sunar.'
                  }
                }
              ]
            }),
          }}
        />
      </head>
      <body>
        <Providers>
          {/* Auth sistem geçici olarak devre dışı - webpack hatası düzeltilene kadar */}
          {/* <CriticalNewsAlertBanner /> */}
          {/* ^^^ DEVRE DIŞI - Kritik haber uyarı banner'ı kaldırıldı */}
          <MockDataBanner />
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
          <Analytics />
          <SpeedInsights />
        </Providers>
      </body>
    </html>
  );
}
