import { NextRequest, NextResponse } from 'next/server';

const AI_LEARNING_URL = process.env.AI_LEARNING_URL || 'http://localhost:5020';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prediction } = body;

    if (!prediction) {
      return NextResponse.json(
        { success: false, error: 'Prediction object is required' },
        { status: 400 }
      );
    }

    const response = await fetch(`${AI_LEARNING_URL}/explainable/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction }),
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ Explainable AI Error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const response = await fetch(`${AI_LEARNING_URL}/explainable/stats`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
