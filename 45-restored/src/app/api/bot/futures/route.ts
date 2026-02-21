import { NextRequest, NextResponse } from 'next/server';

interface AISignalRequest {
  symbol: string;
  timeframe?: string;
}

// AI modellerinden sinyal alma
async function getAISignal(symbol: string): Promise<any> {
  try {
    // Python AI servisinden tahmin al
    const response = await fetch(`http://localhost:5003/predict/single`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol,
        timeframe: '1h',
      }),
    });

    if (!response.ok) {
      throw new Error('AI service unavailable');
    }

    const data = await response.json();

    // AI tahminini trading sinyaline çevir
    const prediction = data.predictions?.ensemble || data.predictions?.lstm || data.prediction;
    const confidence = data.confidence || 0.5;

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';

    if (prediction && confidence > 0.6) {
      // Basit momentum stratejisi
      const currentPrice = prediction.current_price || 0;
      const predictedPrice = prediction.predicted_price || 0;

      const changePercent = ((predictedPrice - currentPrice) / currentPrice) * 100;

      if (changePercent > 0.5) {
        action = 'BUY';
      } else if (changePercent < -0.5) {
        action = 'SELL';
      }
    }

    return {
      symbol,
      action,
      confidence,
      predictedPrice: prediction?.predicted_price,
      reason: `AI Prediction: ${action} with ${(confidence * 100).toFixed(1)}% confidence`,
    };
  } catch (error: any) {
    console.error('AI signal error:', error.message);

    // Fallback: TA-Lib indikatörlerini kullan
    return await getTechnicalSignal(symbol);
  }
}

// TA-Lib indikatörlerinden sinyal alma (fallback)
async function getTechnicalSignal(symbol: string): Promise<any> {
  try {
    const response = await fetch(`http://localhost:5005/indicators/comprehensive`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        symbol,
        timeframe: '1h',
        indicators: ['rsi', 'macd', 'bbands', 'ema'],
      }),
    });

    const data = await response.json();

    let action: 'BUY' | 'SELL' | 'HOLD' = 'HOLD';
    let confidence = 0.5;

    // RSI bazlı sinyal
    const rsi = data.indicators?.rsi?.value || 50;
    if (rsi < 30) {
      action = 'BUY';
      confidence = 0.7;
    } else if (rsi > 70) {
      action = 'SELL';
      confidence = 0.7;
    }

    return {
      symbol,
      action,
      confidence,
      reason: `Technical: RSI ${rsi.toFixed(1)}`,
    };
  } catch (error) {
    return {
      symbol,
      action: 'HOLD',
      confidence: 0,
      reason: 'No signal available',
    };
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: AISignalRequest = await request.json();
    const { symbol } = body;

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol is required' },
        { status: 400 }
      );
    }

    const signal = await getAISignal(symbol);

    return NextResponse.json({
      success: true,
      signal,
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to get trading signal',
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol') || 'BTCUSDT';

  try {
    const signal = await getAISignal(symbol);

    return NextResponse.json({
      success: true,
      signal,
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to get trading signal',
      },
      { status: 500 }
    );
  }
}
