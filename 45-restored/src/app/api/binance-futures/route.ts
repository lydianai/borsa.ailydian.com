import { NextRequest, NextResponse } from 'next/server';
import { binanceFuturesService } from '@/services/BinanceFuturesService';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action') || 'tickers';
    const limit = parseInt(searchParams.get('limit') || '50');

    switch (action) {
      case 'symbols':
        const symbols = await binanceFuturesService.getAllUSDTFuturesSymbols();
        return NextResponse.json({
          success: true,
          count: symbols.length,
          data: symbols
        });

      case 'tickers':
        const tickers = await binanceFuturesService.get24hrTickers();
        return NextResponse.json({
          success: true,
          count: tickers.length,
          data: tickers.slice(0, limit)
        });

      case 'movers':
        const movers = await binanceFuturesService.getTopMovers(limit);
        return NextResponse.json({
          success: true,
          count: movers.length,
          data: movers
        });

      case 'ping':
        const isConnected = await binanceFuturesService.testConnection();
        return NextResponse.json({
          success: isConnected,
          message: isConnected ? 'Binance Futures bağlantısı aktif' : 'Bağlantı hatası'
        });

      default:
        return NextResponse.json({
          success: false,
          error: 'Geçersiz action parametresi'
        }, { status: 400 });
    }

  } catch (error: any) {
    console.error('Binance Futures API Error:', error);
    return NextResponse.json({
      success: false,
      error: error.message || 'Internal server error'
    }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, apiKey, apiSecret } = body;

    if (action === 'test') {
      if (!apiKey || !apiSecret) {
        return NextResponse.json({
          success: false,
          message: 'API Key ve Secret gerekli'
        }, { status: 400 });
      }

      binanceFuturesService.setCredentials(apiKey, apiSecret);
      const result = await binanceFuturesService.testAPIKeys();

      return NextResponse.json({
        success: result.valid,
        message: result.message
      });
    }

    return NextResponse.json({
      success: false,
      error: 'Geçersiz action'
    }, { status: 400 });

  } catch (error: any) {
    console.error('Binance Futures API Test Error:', error);
    return NextResponse.json({
      success: false,
      message: error.message || 'Test başarısız'
    }, { status: 500 });
  }
}
