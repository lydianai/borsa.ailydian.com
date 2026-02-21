/**
 * 🔴 SENTRY CLIENT CONFIG - Frontend Error Tracking
 *
 * Tüm frontend hataları Sentry'ye gönderilir.
 * Real-time error monitoring ve alerting.
 */

import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,

  // Performance Monitoring
  tracesSampleRate: 1.0, // %100 transaction tracking (production'da 0.1 yapılabilir)

  // Session Replay
  replaysSessionSampleRate: 0.1, // %10 of sessions
  replaysOnErrorSampleRate: 1.0, // %100 if error occurs

  // Environment
  environment: process.env.NODE_ENV || 'development',

  // Release tracking
  release: process.env.NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA,

  // Integrations
  integrations: [
    Sentry.replayIntegration({
      maskAllText: true,
      blockAllMedia: true,
    }),
  ],

  // Ignore common errors
  ignoreErrors: [
    // Browser extensions
    'top.GLOBALS',
    'chrome-extension://',
    'moz-extension://',
    // Network errors (Binance API timeouts are normal)
    'Network request failed',
    'Failed to fetch',
  ],

  // Only send errors in production
  enabled: process.env.NODE_ENV === 'production',
});
