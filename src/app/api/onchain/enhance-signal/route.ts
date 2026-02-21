/**
 * ON-CHAIN SIGNAL ENHANCEMENT API
 * POST /api/onchain/enhance-signal
 *
 * Enhances existing strategy signals with on-chain whale analysis
 *
 * Body:
 * {
 *   signal: 'buy' | 'sell' | 'neutral',
 *   confidence: number (0-100),
 *   symbol: string,
 *   strategy?: string,
 *   price?: number
 * }
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  enhanceWithOnChain,
  getOnChainSummary,
  shouldBlockTrade,
  type BaseStrategySignal,
} from '@/lib/onchain/strategy-enhancer';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate required fields
    if (!body.signal || !body.symbol || typeof body.confidence !== 'number') {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: signal, symbol, confidence',
          data: null,
        },
        { status: 400 }
      );
    }

    // Validate signal value
    if (!['buy', 'sell', 'neutral'].includes(body.signal)) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid signal value. Must be: buy, sell, or neutral',
          data: null,
        },
        { status: 400 }
      );
    }

    // Validate confidence range
    if (body.confidence < 0 || body.confidence > 100) {
      return NextResponse.json(
        {
          success: false,
          error: 'Confidence must be between 0 and 100',
          data: null,
        },
        { status: 400 }
      );
    }

    const baseSignal: BaseStrategySignal = {
      signal: body.signal,
      confidence: body.confidence,
      symbol: body.symbol.toUpperCase(),
      strategy: body.strategy,
      price: body.price,
      reason: body.reason,
    };

    // Enhance signal with on-chain data
    const enhanced = await enhanceWithOnChain(baseSignal, {
      enableWhaleAlert: body.enableWhaleAlert !== false, // Default: true
      aggressiveMode: body.aggressiveMode === true, // Default: false
    });

    // Check if trade should be blocked
    const blockCheck = await shouldBlockTrade(baseSignal, body.riskTolerance || 'medium');

    return NextResponse.json({
      success: true,
      data: {
        original: {
          signal: baseSignal.signal,
          confidence: baseSignal.confidence,
          symbol: baseSignal.symbol,
        },
        enhanced: {
          signal: enhanced.finalDecision.signal,
          confidence: enhanced.finalDecision.confidence,
          recommendation: enhanced.finalDecision.recommendation,
          warnings: enhanced.finalDecision.warnings,
        },
        onChain: {
          enabled: enhanced.onChainAnalysis.enabled,
          signal: enhanced.onChainAnalysis.signal,
          riskAdjustment: enhanced.onChainAnalysis.riskAdjustment,
          explanation: enhanced.onChainAnalysis.explanation,
          whaleActivity: enhanced.onChainAnalysis.whaleActivity
            ? {
                activity: enhanced.onChainAnalysis.whaleActivity.activity,
                confidence: enhanced.onChainAnalysis.whaleActivity.confidence,
                riskScore: enhanced.onChainAnalysis.whaleActivity.riskScore,
                summary: enhanced.onChainAnalysis.whaleActivity.summary,
              }
            : null,
        },
        riskCheck: {
          blocked: blockCheck.blocked,
          reason: blockCheck.reason,
          riskScore: blockCheck.riskScore,
        },
      },
    });
  } catch (error: any) {
    console.error('[API /onchain/enhance-signal] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to enhance signal',
        data: null,
      },
      { status: 500 }
    );
  }
}

// Also support GET for quick on-chain summary
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const symbol = searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json(
      {
        success: false,
        error: 'Missing required parameter: symbol',
        data: null,
      },
      { status: 400 }
    );
  }

  try {
    const summary = await getOnChainSummary(symbol.toUpperCase());

    return NextResponse.json({
      success: true,
      data: summary,
    });
  } catch (error: any) {
    console.error('[API /onchain/enhance-signal] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to get on-chain summary',
        data: null,
      },
      { status: 500 }
    );
  }
}
