/**
 * 🔴 SENTRY SERVER CONFIG - Backend Error Tracking
 *
 * Tüm server-side hataları (API routes, cron jobs) Sentry'ye gönderilir.
 */

import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.SENTRY_DSN,

  // Performance Monitoring
  tracesSampleRate: 1.0,

  // Environment
  environment: process.env.NODE_ENV || 'development',

  // Release tracking
  release: process.env.VERCEL_GIT_COMMIT_SHA,

  // Server-specific config
  integrations: [
    Sentry.captureConsoleIntegration({
      levels: ['error'], // Only capture console.error()
    }),
  ],

  // Ignore expected errors
  ignoreErrors: [
    // BullMQ retries
    'Job failed',
    // Redis connection (auto-reconnect handles it)
    'ECONNREFUSED',
    // Rate limiting (expected behavior)
    'Too Many Requests',
  ],

  // Only send errors in production
  enabled: process.env.NODE_ENV === 'production',

  // Debug in development
  debug: process.env.NODE_ENV === 'development',
});
