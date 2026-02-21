import { NextRequest, NextResponse } from 'next/server';

const AI_LEARNING_URL = process.env.AI_LEARNING_URL || 'http://localhost:5020';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    let endpoint = '';
    switch (action) {
      case 'update':
        endpoint = '/online-learning/update';
        break;
      case 'drift':
        endpoint = '/online-learning/drift';
        break;
      case 'stats':
        endpoint = '/online-learning/stats';
        break;
      default:
        return NextResponse.json(
          { success: false, error: 'Invalid action' },
          { status: 400 }
        );
    }

    const method = action === 'stats' ? 'GET' : 'POST';
    const response = await fetch(`${AI_LEARNING_URL}${endpoint}`, {
      method,
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ Online Learning Error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const response = await fetch(`${AI_LEARNING_URL}/online-learning/stats`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
