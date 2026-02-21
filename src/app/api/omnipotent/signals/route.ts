import { NextRequest, NextResponse } from 'next/server';
import { omnipotentMatrix } from '@/services-45backend/OmnipotentFuturesMatrix';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const symbol = searchParams.get('symbol');
    const limit = parseInt(searchParams.get('limit') || '100');
    const minConfidence = parseInt(searchParams.get('minConfidence') || '70');

    if (symbol) {
      // Single symbol analysis
      const signal = await omnipotentMatrix.generateSignal(symbol);
      
      if (!signal) {
        return NextResponse.json({
          success: false,
          message: 'Signal confidence too low or error occurred'
        });
      }

      return NextResponse.json({
        success: true,
        data: signal
      });
    }

    // Scan all markets
    const signals = await omnipotentMatrix.scanAllFuturesMarkets(limit);
    
    // Filter by minimum confidence
    const filteredSignals = signals.filter(s => s.confidence >= minConfidence);

    return NextResponse.json({
      success: true,
      count: filteredSignals.length,
      totalScanned: limit,
      data: filteredSignals,
      timestamp: Date.now()
    });

  } catch (error: any) {
    console.error('Omnipotent Signals API Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Internal server error'
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbols } = body;

    if (!symbols || !Array.isArray(symbols)) {
      return NextResponse.json({
        success: false,
        error: 'symbols array required'
      }, { status: 400 });
    }

    const signals: any[] = [];

    for (const symbol of symbols) {
      const signal = await omnipotentMatrix.generateSignal(symbol);
      if (signal) {
        signals.push(signal);
      }
    }

    return NextResponse.json({
      success: true,
      count: signals.length,
      data: signals
    });

  } catch (error: any) {
    console.error('Omnipotent Signals POST Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}
