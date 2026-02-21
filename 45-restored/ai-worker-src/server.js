/**
 * AILYDIAN AI Microservice - Railway Production Server
 *
 * Security Features:
 * - HMAC Authentication
 * - Rate Limiting
 * - Input Validation
 * - Security Headers
 * - Request Size Limiting
 * - SQL Injection Prevention
 * - XSS Prevention
 *
 * White-hat compliant, production-ready
 */

import { Hono } from 'hono';
import { serve } from '@hono/node-server';
import { cors } from 'hono/cors';
import { logger } from 'hono/logger';

// Security middlewares
import { hmacAuthMiddleware } from './middleware/hmac-auth.js';
import { multiRateLimitMiddleware } from './middleware/rate-limit.js';
import {
  inputValidationMiddleware,
  requestSizeLimitMiddleware,
  securityScanMiddleware,
  schemas
} from './middleware/input-validation.js';

// AI Engine (Production-grade)
import { generateTradingSignal, generateBatchSignals } from './services/ai-engine.js';

const app = new Hono();

// ============================================================================
// GLOBAL MIDDLEWARES
// ============================================================================

// Logging (all requests)
app.use('*', logger());

// CORS - Only allow borsa.ailydian.com
app.use('*', cors({
  origin: ['https://borsa.ailydian.com', 'http://localhost:3000'],
  allowMethods: ['GET', 'POST', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'X-Signature', 'X-Timestamp'],
  exposeHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
  maxAge: 86400,
  credentials: true
}));

// Security Headers
app.use('*', async (c, next) => {
  await next();

  // Set security headers
  c.header('X-Content-Type-Options', 'nosniff');
  c.header('X-Frame-Options', 'DENY');
  c.header('X-XSS-Protection', '1; mode=block');
  c.header('Referrer-Policy', 'strict-origin-when-cross-origin');
  c.header('Permissions-Policy', 'geolocation=(), microphone=(), camera=()');

  // HSTS (if HTTPS)
  if (c.req.header('x-forwarded-proto') === 'https') {
    c.header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
  }
});

// Request size limiting (10KB max)
app.use('*', requestSizeLimitMiddleware(10 * 1024));

// Rate limiting (100 req/min per IP, 10000 global)
app.use('*', multiRateLimitMiddleware());

// HMAC Authentication (all routes except /health)
app.use('*', hmacAuthMiddleware());

// Security scanning (SQL injection, XSS detection)
app.use('*', securityScanMiddleware());

// ============================================================================
// HEALTH CHECK (Public, no auth required)
// ============================================================================

app.get('/health', (c) => {
  return c.json({
    status: 'ok',
    service: 'ailydian-ai-core',
    version: '1.0.0',
    timestamp: Date.now(),
    security: {
      hmac: true,
      rateLimit: true,
      inputValidation: true
    }
  });
});

// ============================================================================
// AI ENDPOINTS (Protected)
// ============================================================================

/**
 * Trading Signal Generation
 * POST /v1/signal
 * Body: { symbol: "BTCUSDT", timeframe?: "1h", limit?: 10 }
 */
app.post('/v1/signal',
  inputValidationMiddleware(schemas.signalRequest),
  async (c) => {
    const { symbol, timeframe } = c.req.validated;

    try {
      // ✅ REAL AI PREDICTION - Production-grade
      const signal = await generateTradingSignal(symbol, timeframe);

      return c.json({
        success: true,
        signal
      });

    } catch (error) {
      console.error('Signal generation error:', {
        symbol,
        timeframe,
        error: error.message
      });

      return c.json({
        success: false,
        error: 'Failed to generate trading signal',
        code: 'AI_ERROR',
        message: error.message
      }, 500);
    }
  }
);

/**
 * AI Processing (General purpose)
 * POST /v1/process
 * Body: { task: string, input: any }
 */
app.post('/v1/process', async (c) => {
  const body = c.req.parsedBody || await c.req.json().catch(() => ({}));
  const { task = 'echo', input = '' } = body;

  // Simulated processing delay
  await new Promise(resolve => setTimeout(resolve, 50));

  return c.json({
    success: true,
    task,
    output: typeof input === 'string' ? input.slice(0, 120) : 'no-input',
    timestamp: Date.now(),
    note: 'Railway AI microservice - Ready for real AI integration'
  });
});

/**
 * Batch Signal Generation
 * POST /v1/batch
 * Body: { symbols: string[], timeframe?: string }
 */
app.post('/v1/batch', async (c) => {
  const body = c.req.parsedBody || await c.req.json().catch(() => ({}));
  const { symbols = [], timeframe = '1h' } = body;

  // Validate symbols array
  if (!Array.isArray(symbols) || symbols.length === 0) {
    return c.json({
      success: false,
      error: 'Symbols must be a non-empty array',
      code: 'INVALID_SYMBOLS'
    }, 400);
  }

  if (symbols.length > 50) {
    return c.json({
      success: false,
      error: 'Maximum 50 symbols per batch request',
      code: 'BATCH_LIMIT_EXCEEDED'
    }, 400);
  }

  try {
    // ✅ REAL AI BATCH PROCESSING
    const results = await generateBatchSignals(symbols, timeframe);

    return c.json({
      success: true,
      results,
      count: results.length,
      timestamp: Date.now()
    });

  } catch (error) {
    console.error('Batch processing error:', error);

    return c.json({
      success: false,
      error: 'Batch processing failed',
      code: 'BATCH_ERROR',
      message: error.message
    }, 500);
  }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

// 404 handler
app.notFound((c) => {
  return c.json({
    success: false,
    error: 'Endpoint not found',
    code: 'NOT_FOUND',
    path: c.req.path
  }, 404);
});

// Global error handler
app.onError((err, c) => {
  console.error('Server error:', {
    error: err.message,
    stack: err.stack,
    path: c.req.path,
    method: c.req.method,
    timestamp: new Date().toISOString()
  });

  return c.json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  }, 500);
});

// ============================================================================
// SERVER START
// ============================================================================

const port = parseInt(process.env.PORT || '8080', 10);

serve({
  fetch: app.fetch,
  port
}, (info) => {
  console.log('');
  console.log('🚂 ========================================');
  console.log('🚂 AILYDIAN AI Microservice - PRODUCTION');
  console.log('🚂 ========================================');
  console.log('');
  console.log(`📍 Port: ${info.port}`);
  console.log(`📍 Health: http://localhost:${info.port}/health`);
  console.log('');
  console.log('🔒 Security Features:');
  console.log('   ✓ HMAC Authentication');
  console.log('   ✓ Rate Limiting (100 req/min per IP)');
  console.log('   ✓ Input Validation & Sanitization');
  console.log('   ✓ SQL Injection Prevention');
  console.log('   ✓ XSS Prevention');
  console.log('   ✓ CORS Protection');
  console.log('   ✓ Security Headers');
  console.log('');
  console.log('🤖 AI Status: ✅ PRODUCTION-GRADE');
  console.log('   ✓ Real-time market data (Binance API)');
  console.log('   ✓ Technical indicators (RSI, MACD, Bollinger, EMA)');
  console.log('   ✓ Multi-indicator consensus algorithm');
  console.log('   ✓ Conservative confidence scoring');
  console.log('   ✓ Risk management (stop-loss, take-profit)');
  console.log('');
  console.log('✅ Server ready for production!');
  console.log('');
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully...');
  process.exit(0);
});
