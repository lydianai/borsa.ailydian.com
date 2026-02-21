import { NextRequest, NextResponse } from 'next/server';

// AI Models servisi (Python Flask)
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:5003';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, timeframe = '1h', model = 'ensemble' } = body;

    if (!symbol) {
      return NextResponse.json(
        { success: false, error: 'Symbol is required' },
        { status: 400 }
      );
    }

    // Try Python AI service first, fallback to mock if unavailable
    try {
      const response = await fetch(`${AI_SERVICE_URL}/predict/single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol,
          timeframe,
          model,
        }),
        signal: AbortSignal.timeout(3000), // 3 second timeout
      });

      const data = await response.json();

      if (response.ok && data.prediction) {
        return NextResponse.json({
          success: true,
          prediction: data.prediction,
          timestamp: new Date().toISOString(),
        });
      }
    } catch (pythonError) {
      console.warn('⚠️ Python AI service unavailable, using mock prediction');
    }

    // Mock prediction fallback (for demo purposes)
    const actions = ['BUY', 'SELL', 'HOLD'];
    const randomAction = actions[Math.floor(Math.random() * actions.length)];
    const confidence = 0.65 + Math.random() * 0.3; // 65-95%

    const mockPrediction = {
      prediction: randomAction === 'BUY' ? 1 : randomAction === 'SELL' ? -1 : 0,
      confidence: parseFloat(confidence.toFixed(2)),
      action: randomAction,
      model_name: model,
      model_type: 'Mock AI (Demo)',
      price_change_prediction: (Math.random() * 10 - 5).toFixed(2) + '%',
    };

    return NextResponse.json({
      success: true,
      prediction: mockPrediction,
      timestamp: new Date().toISOString(),
      note: 'Using demo prediction - Python AI service not available',
    });
  } catch (error: any) {
    console.error('❌ AI Prediction Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Internal server error',
      },
      { status: 500 }
    );
  }
}
