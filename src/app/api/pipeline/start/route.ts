import { NextRequest, NextResponse } from 'next/server';

const PIPELINE_SERVICE_URL = 'http://localhost:5030';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));

    const response = await fetch(`${PIPELINE_SERVICE_URL}/pipeline/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Pipeline service returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      ...data
    });

  } catch (error: any) {
    console.error('[Pipeline Start API] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: error.message
      },
      { status: error.message.includes('already running') ? 409 : 503 }
    );
  }
}
