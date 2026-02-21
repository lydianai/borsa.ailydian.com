import { NextRequest, NextResponse } from 'next/server';

// AI Models servisi (Python Flask)
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:5003';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbols, timeframe = '1h', model = 'ensemble' } = body;

    if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Symbols array is required' },
        { status: 400 }
      );
    }

    // Python AI service'e toplu tahmin isteği gönder
    const response = await fetch(`${AI_SERVICE_URL}/predict/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbols,
        timeframe,
        model,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.error || 'Batch prediction failed' },
        { status: response.status }
      );
    }

    return NextResponse.json({
      success: true,
      predictions: data.predictions || [],
      total: data.total || 0,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('❌ Batch Prediction Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Internal server error',
      },
      { status: 500 }
    );
  }
}
