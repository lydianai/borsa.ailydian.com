/**
 * 📊 NEXT.JS INSTRUMENTATION
 *
 * Sentry initialization ve global error handlers.
 * Bu dosya Next.js tarafından otomatik olarak yüklenir.
 */

export async function register() {
  if (process.env.NEXT_RUNTIME === 'nodejs') {
    await import('./sentry.server.config');
  }

  if (process.env.NEXT_RUNTIME === 'edge') {
    await import('./sentry.edge.config');
  }
}
