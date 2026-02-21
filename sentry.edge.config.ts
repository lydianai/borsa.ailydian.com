/**
 * 🔴 SENTRY EDGE CONFIG - Edge Runtime Error Tracking
 *
 * Vercel Edge Functions için error tracking.
 */

import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.SENTRY_DSN,

  tracesSampleRate: 1.0,

  environment: process.env.NODE_ENV || 'development',

  release: process.env.VERCEL_GIT_COMMIT_SHA,

  enabled: process.env.NODE_ENV === 'production',
});
