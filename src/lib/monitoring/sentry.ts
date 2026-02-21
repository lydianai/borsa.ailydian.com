/**
 * Sentry Error Monitoring Configuration
 *
 * White-hat compliance: Error monitoring helps detect and fix security
 * vulnerabilities and system reliability issues
 *
 * IMPORTANT: Never send sensitive data (passwords, tokens, PII) to Sentry
 */

import * as Sentry from '@sentry/nextjs';

/**
 * Initialize Sentry client-side
 */
export function initSentryClient(): void {
  if (!process.env.NEXT_PUBLIC_SENTRY_DSN) {
    console.warn('[Sentry] DSN not configured. Monitoring disabled.');
    return;
  }

  Sentry.init({
    dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    enabled: process.env.NODE_ENV === 'production',

    // Performance Monitoring
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

    // Session Replay (optional, costs extra)
    replaysSessionSampleRate: 0.1, // 10% of sessions
    replaysOnErrorSampleRate: 1.0, // 100% of error sessions

    // Filter out sensitive data
    beforeSend(event, _hint) {
      return filterSensitiveData(event) as any;
    },

    // Ignore expected errors
    ignoreErrors: [
      // Browser extensions
      'top.GLOBALS',
      'canvas.contentDocument',

      // Network errors (handled separately)
      'NetworkError',
      'Failed to fetch',
      'Network request failed',

      // Cancelled requests
      'AbortError',
      'The user aborted a request',

      // Safari specific
      'ResizeObserver loop limit exceeded',
    ],

    // Don't track localhost
    denyUrls: [
      /localhost/,
      /127\.0\.0\.1/,
    ],
  });
}

/**
 * Initialize Sentry server-side
 */
export function initSentryServer(): void {
  if (!process.env.SENTRY_DSN) {
    console.warn('[Sentry Server] DSN not configured. Monitoring disabled.');
    return;
  }

  Sentry.init({
    dsn: process.env.SENTRY_DSN,
    environment: process.env.NODE_ENV || 'development',
    enabled: process.env.NODE_ENV === 'production',

    // Performance Monitoring
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

    // Filter out sensitive data
    beforeSend(event, _hint) {
      return filterSensitiveData(event) as any;
    },

    // Ignore expected errors
    ignoreErrors: [
      'ValidationError',
      'AuthenticationError',
      'RateLimitError',
    ],
  });
}

/**
 * Filter sensitive data from Sentry events
 *
 * SECURITY: Never send passwords, tokens, API keys, or PII to Sentry
 */
function filterSensitiveData(event: Sentry.Event): Sentry.Event | null {
  // Remove sensitive request data
  if (event.request) {
    // Remove auth headers
    if (event.request.headers) {
      delete event.request.headers['authorization'];
      delete event.request.headers['cookie'];
      delete event.request.headers['x-api-key'];
    }

    // Remove sensitive query params
    if (event.request.query_string && typeof event.request.query_string === 'string') {
      const sensitiveParams = ['password', 'token', 'api_key', 'secret'];
      let queryString = event.request.query_string;

      for (const param of sensitiveParams) {
        const regex = new RegExp(`${param}=[^&]*`, 'gi');
        queryString = queryString.replace(regex, `${param}=[REDACTED]`);
      }

      event.request.query_string = queryString;
    }

    // Remove request body if it might contain sensitive data
    if (event.request.data && typeof event.request.data === 'object') {
      const sensitiveFields = [
        'password',
        'passwordHash',
        'token',
        'apiKey',
        'secret',
        'totpSecret',
        'backupCodes',
        'privateKey',
      ];

      for (const field of sensitiveFields) {
        if (field in event.request.data) {
          event.request.data[field] = '[REDACTED]';
        }
      }
    }
  }

  // Remove sensitive breadcrumbs
  if (event.breadcrumbs) {
    event.breadcrumbs = event.breadcrumbs.map((breadcrumb) => {
      if (breadcrumb.data) {
        const sensitiveKeys = ['password', 'token', 'api_key', 'authorization'];
        for (const key of sensitiveKeys) {
          if (key in breadcrumb.data) {
            breadcrumb.data[key] = '[REDACTED]';
          }
        }
      }
      return breadcrumb;
    });
  }

  // Remove sensitive extra data
  if (event.extra) {
    const sensitiveKeys = ['password', 'token', 'apiKey', 'secret'];
    for (const key of sensitiveKeys) {
      if (key in event.extra) {
        delete event.extra[key];
      }
    }
  }

  return event;
}

/**
 * Capture exception with context
 */
export function captureException(
  error: Error,
  context?: Record<string, unknown>
): void {
  if (process.env.NODE_ENV !== 'production') {
    console.error('[Sentry]', error, context);
    return;
  }

  Sentry.withScope((scope) => {
    if (context) {
      // Filter sensitive data from context
      const safeContext = { ...context };
      const sensitiveKeys = ['password', 'token', 'apiKey', 'secret'];

      for (const key of sensitiveKeys) {
        if (key in safeContext) {
          delete safeContext[key];
        }
      }

      scope.setContext('custom', safeContext);
    }

    Sentry.captureException(error);
  });
}

/**
 * Capture message (for non-error events)
 */
export function captureMessage(
  message: string,
  level: Sentry.SeverityLevel = 'info',
  context?: Record<string, unknown>
): void {
  if (process.env.NODE_ENV !== 'production') {
    console.log(`[Sentry ${level}]`, message, context);
    return;
  }

  Sentry.withScope((scope) => {
    if (context) {
      scope.setContext('custom', context);
    }

    Sentry.captureMessage(message, level);
  });
}

/**
 * Set user context (for tracking issues per user)
 *
 * IMPORTANT: Only send non-sensitive user identifiers
 */
export function setUser(user: {
  id: string;
  username?: string;
  email?: string; // Only if user consents
}): void {
  Sentry.setUser({
    id: user.id,
    username: user.username,
    // Only include email if user consents to monitoring
    ...(user.email && { email: user.email }),
  });
}

/**
 * Clear user context (on logout)
 */
export function clearUser(): void {
  Sentry.setUser(null);
}

/**
 * Add breadcrumb (for debugging context)
 */
export function addBreadcrumb(
  category: string,
  message: string,
  data?: Record<string, unknown>
): void {
  Sentry.addBreadcrumb({
    category,
    message,
    level: 'info',
    data,
  });
}

/**
 * Start span (for performance monitoring)
 * Updated for @sentry/nextjs v10 - uses startSpan() instead of startTransaction()
 */
export function startSpan<T>(
  options: {
    name: string;
    op: string;
  },
  callback: () => T
): T {
  if (process.env.NODE_ENV !== 'production') {
    return callback();
  }

  return Sentry.startSpan(
    {
      name: options.name,
      op: options.op,
    },
    callback
  );
}
