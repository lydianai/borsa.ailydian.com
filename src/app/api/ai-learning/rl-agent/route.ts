import { NextRequest, NextResponse } from 'next/server';

const AI_LEARNING_URL = process.env.AI_LEARNING_URL || 'http://localhost:5020';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, ...params } = body;

    let endpoint = '';
    switch (action) {
      case 'train':
        endpoint = '/rl-agent/train';
        break;
      case 'predict':
        endpoint = '/rl-agent/predict';
        break;
      case 'stats':
        endpoint = '/rl-agent/stats';
        break;
      default:
        return NextResponse.json(
          { success: false, error: 'Invalid action' },
          { status: 400 }
        );
    }

    const method = action === 'stats' ? 'GET' : 'POST';
    const url = `${AI_LEARNING_URL}${endpoint}`;

    const response = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: method === 'POST' ? JSON.stringify(params) : undefined,
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.error || 'RL Agent request failed' },
        { status: response.status }
      );
    }

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ RL Agent Error:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const response = await fetch(`${AI_LEARNING_URL}/rl-agent/stats`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
