// LyTrade Scanner - AI-Powered Crypto Trading Platform
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

const siteUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

export const metadata: Metadata = {
  title: {
    default: 'LyTrade Scanner - AI-Powered Crypto Trading Signals',
    template: '%s | LyTrade Scanner'
  },
  description: 'Open-source AI-powered crypto trading platform with real-time signals for 600+ pairs. Multi-strategy analysis, whale tracking, and professional risk management.',
  applicationName: 'LyTrade Scanner',
  keywords: [
    'crypto trading signals',
    'open source trading platform',
    'binance futures signals',
    'ai trading',
    'technical analysis',
    'whale tracker',
    'crypto trading bot',
    'futures trading',
    'bitcoin signals',
    'altcoin signals',
    'crypto market scanner',
    'automated trading',
    'algorithmic trading',
    'perpetual futures',
    'crypto portfolio management',
    'real-time crypto signals',
    'conservative buy signals',
    'multi-strategy analysis',
    'risk management',
    'self-hosted trading',
  ],
  authors: [{ name: 'LyTrade', url: 'https://github.com/lydianai/borsa.ailydian.com' }],
  creator: 'LyTrade',
  publisher: 'LyTrade',
  manifest: '/manifest.json',
  metadataBase: new URL(siteUrl),

  // Open Graph (Facebook, LinkedIn, WhatsApp)
  openGraph: {
    type: 'website',
    locale: 'tr_TR',
    alternateLocale: ['en_US', 'en_GB', 'de_DE', 'fr_FR', 'es_ES', 'ru_RU', 'ar_SA', 'zh_CN', 'ja_JP'],
    url: siteUrl,
    siteName: 'LyTrade Scanner',
    title: 'LyTrade Scanner - AI-Powered Crypto Trading Platform',
    description: 'Open-source AI-powered crypto trading signals for 600+ pairs. Multi-strategy analysis, whale tracking, and professional risk management.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'LyTrade Scanner - AI-Powered Crypto Trading Platform',
        type: 'image/png',
      },
      {
        url: '/og-image-square.png',
        width: 1200,
        height: 1200,
        alt: 'LyTrade Scanner Logo',
        type: 'image/png',
      }
    ],
  },

  // Twitter Card
  twitter: {
    card: 'summary_large_image',
    title: 'LyTrade Scanner - AI Crypto Trading Signals',
    description: 'Open-source AI-powered crypto trading signals for 600+ pairs. Multi-strategy analysis, whale tracking, and risk management.',
    images: ['/og-image.png'],
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
    title: 'LyTrade Scanner',
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
        <meta name="apple-mobile-web-app-title" content="LyTrade Scanner" />
        <link rel="apple-touch-icon" href="/icons/apple-touch-icon.png" />

        {/* Multi-language support - hreflang */}
        <link rel="alternate" hrefLang="tr" href={siteUrl} />
        <link rel="alternate" hrefLang="en" href={`${siteUrl}/en`} />
        <link rel="alternate" hrefLang="x-default" href={siteUrl} />

        {/* Canonical URL */}
        <link rel="canonical" href={siteUrl} />

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
              name: 'LyTrade',
              url: siteUrl,
              logo: `${siteUrl}/og-image-square.png`,
              sameAs: [
                'https://github.com/lydianai/borsa.ailydian.com',
              ]
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
              name: 'LyTrade Scanner',
              alternateName: 'AI Crypto Trading Platform',
              description: 'Open-source AI-powered crypto trading signals for 600+ pairs. Multi-strategy analysis, whale tracking, and professional risk management.',
              url: siteUrl,
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
                `${siteUrl}/og-image.png`,
                `${siteUrl}/og-image-square.png`
              ],
              inLanguage: ['tr-TR', 'en-US'],
              creator: {
                '@type': 'Organization',
                name: 'LyTrade',
                url: siteUrl
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
                  item: siteUrl
                },
                {
                  '@type': 'ListItem',
                  position: 2,
                  name: 'Market Scanner',
                  item: `${siteUrl}/market-scanner`
                },
                {
                  '@type': 'ListItem',
                  position: 3,
                  name: 'Trading Signals',
                  item: `${siteUrl}/trading-signals`
                },
                {
                  '@type': 'ListItem',
                  position: 4,
                  name: 'AI Signals',
                  item: `${siteUrl}/ai-signals`
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
                  name: 'What is LyTrade Scanner?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'LyTrade Scanner is an open-source, AI-powered crypto trading platform providing real-time signals for 600+ cryptocurrency pairs. It combines 11+ trading strategies, whale tracking, and technical analysis tools.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'How many crypto pairs are supported?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'LyTrade Scanner supports 600+ cryptocurrency pairs from Binance Futures, covering all major and altcoin pairs with real-time analysis and signal generation.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'How do the AI trading signals work?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'The AI system combines 11+ different trading strategies, evaluating technical indicators, whale movements, order flow, funding rates, and CVD analysis in real-time to generate buy/sell signals with confidence scoring.'
                  }
                },
                {
                  '@type': 'Question',
                  name: 'Is LyTrade Scanner free to use?',
                  acceptedAnswer: {
                    '@type': 'Answer',
                    text: 'Yes, LyTrade Scanner is open-source under the MIT license. You can clone, run locally, and customize it for free. You only need free API keys for Binance data and optionally an AI provider for AI-powered features.'
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
