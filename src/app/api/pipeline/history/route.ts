import { NextRequest, NextResponse } from 'next/server';

const PIPELINE_SERVICE_URL = 'http://localhost:5030';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '10';

    const response = await fetch(`${PIPELINE_SERVICE_URL}/pipeline/history?limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store'
    });

    if (!response.ok) {
      throw new Error(`Pipeline service returned ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      data: data.data
    });

  } catch (error: any) {
    console.error('[Pipeline History API] Error:', error.message);
    return NextResponse.json(
      {
        success: false,
        error: 'Pipeline service unavailable',
        message: error.message
      },
      { status: 503 }
    );
  }
}
