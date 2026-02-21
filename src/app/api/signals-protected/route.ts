/**
 * Protected Signals API - Example with Rate Limiting
 *
 * White-hat compliance: Demonstrates subscription-tier based rate limiting
 * This is an example of how to protect API routes with rate limiting
 */

import { NextRequest, NextResponse } from 'next/server';
import { withApiRateLimit, trackApiUsage } from '@/middleware/rate-limit';

async function signalsHandler(req: NextRequest) {
  // Get user info (TODO: implement proper auth)
  const userId = req.headers.get('x-user-id') || 'anonymous';

  // Track API usage
  await trackApiUsage(userId, '/api/signals-protected', 'api', {
    method: req.method,
    userAgent: req.headers.get('user-agent'),
  });

  // Return signals data
  return NextResponse.json({
    success: true,
    data: {
      signals: [
        {
          symbol: 'BTCUSDT',
          type: 'LONG',
          confidence: 85,
          entry: 43250,
          targets: [43800, 44200, 44800],
          stopLoss: 42800,
          timeframe: '4h',
          timestamp: new Date().toISOString(),
        },
        {
          symbol: 'ETHUSDT',
          type: 'SHORT',
          confidence: 78,
          entry: 2285,
          targets: [2250, 2220, 2180],
          stopLoss: 2310,
          timeframe: '1h',
          timestamp: new Date().toISOString(),
        },
      ],
    },
    meta: {
      rateLimit: {
        limit: req.headers.get('x-ratelimit-limit'),
        remaining: req.headers.get('x-ratelimit-remaining'),
        reset: req.headers.get('x-ratelimit-reset'),
      },
    },
  });
}

// Export with rate limiting
export const GET = withApiRateLimit(signalsHandler, 'api');
export const POST = withApiRateLimit(signalsHandler, 'api');
