import { NextRequest, NextResponse } from 'next/server';

const AI_LEARNING_URL = process.env.AI_LEARNING_URL || 'http://localhost:5020';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, scenario } = body;

    let endpoint = '';
    if (action === 'discover') {
      endpoint = '/causal/discover';
    } else if (action === 'counterfactual') {
      endpoint = '/causal/counterfactual';
    } else {
      return NextResponse.json(
        { success: false, error: 'Invalid action. Use "discover" or "counterfactual"' },
        { status: 400 }
      );
    }

    const response = await fetch(`${AI_LEARNING_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario }),
      signal: AbortSignal.timeout(5000),
    });

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ Causal AI Error:', error);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    const response = await fetch(`${AI_LEARNING_URL}/causal/stats`);
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
