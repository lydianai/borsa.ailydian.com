/**
 * GET /api/traditional-markets
 * Traditional Markets Data API - LIVE DATA ONLY
 *
 * Returns:
 * - Precious Metals (Gold, Silver, Palladium, Copper) with TL prices
 * - Forex (10 major currencies vs TRY)
 * - DXY (US Dollar Index)
 * - Market Overview (trending, strongest, weakest)
 *
 * Query params:
 * - refresh: Force refresh cache (default: false)
 * - symbol: Get specific asset by symbol (optional)
 * - overview: Get market overview only (default: false)
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  getAllTraditionalMarketsData,
  getAssetBySymbol,
  getMarketOverview,
} from '@/lib/traditional-markets';
import auditLogger, { AuditEventType } from '@/lib/security/audit-logger';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(request: NextRequest) {
  const startTime = Date.now();

  try {
    // Get query parameters
    const { searchParams } = new URL(request.url);
    const refresh = searchParams.get('refresh') === 'true';
    const symbol = searchParams.get('symbol');
    const overviewOnly = searchParams.get('overview') === 'true';

    // Handle specific symbol request
    if (symbol) {
      console.log(`[API] Fetching traditional markets data for symbol: ${symbol}`);

      const asset = await getAssetBySymbol(symbol, refresh);

      if (!asset) {
        return NextResponse.json(
          { error: 'Asset not found', symbol },
          { status: 404 }
        );
      }

      // Log successful request
      auditLogger.log(
        AuditEventType.API_REQUEST,
        `GET /api/traditional-markets?symbol=${symbol}`,
        {
          endpoint: '/api/traditional-markets',
          method: 'GET',
          statusCode: 200,
          metadata: { symbol, refresh },
        }
      );

      return NextResponse.json({
        success: true,
        data: asset,
        cached: !refresh,
        timestamp: new Date().toISOString(),
      });
    }

    // Handle overview only request
    if (overviewOnly) {
      console.log('[API] Fetching traditional markets overview');

      const overview = await getMarketOverview(refresh);

      // Log successful request
      auditLogger.log(
        AuditEventType.API_REQUEST,
        'GET /api/traditional-markets?overview=true',
        {
          endpoint: '/api/traditional-markets',
          method: 'GET',
          statusCode: 200,
          metadata: { overview: true, refresh },
        }
      );

      return NextResponse.json({
        success: true,
        data: overview,
        cached: !refresh,
        timestamp: new Date().toISOString(),
      });
    }

    // Handle full data request (default)
    console.log('[API] Fetching all traditional markets data');

    const data = await getAllTraditionalMarketsData(refresh);

    const duration = Date.now() - startTime;

    // Log successful request
    auditLogger.log(
      AuditEventType.API_REQUEST,
      'GET /api/traditional-markets',
      {
        endpoint: '/api/traditional-markets',
        method: 'GET',
        statusCode: 200,
        metadata: {
          refresh,
          duration,
          totalAssets: data.summary.totalAssets,
        },
      }
    );

    return NextResponse.json({
      success: true,
      data,
      cached: !refresh,
      performance: {
        duration,
        assetsCount: data.summary.totalAssets,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[API] Traditional markets error:', error);

    // Log error
    auditLogger.log(
      AuditEventType.API_ERROR,
      `Traditional markets API error: ${error.message}`,
      {
        endpoint: '/api/traditional-markets',
        method: 'GET',
        statusCode: 500,
        metadata: { error: error.message },
      }
    );

    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch traditional markets data',
        message: error.message,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}
