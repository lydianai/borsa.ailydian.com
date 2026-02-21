import { NextRequest, NextResponse } from 'next/server';

// AI Models servisi (Python Flask)
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:5003';

export async function GET(_request: NextRequest) {
  try {
    // Python AI service'den model listesini al
    const response = await fetch(`${AI_SERVICE_URL}/models/list`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const data = await response.json();

    if (!response.ok) {
      return NextResponse.json(
        { success: false, error: data.error || 'Failed to fetch models' },
        { status: response.status }
      );
    }

    return NextResponse.json({
      success: true,
      models: data.models || [],
      total: data.total || 0,
    });
  } catch (error: any) {
    console.error('❌ Models List Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Internal server error',
        models: [],
      },
      { status: 500 }
    );
  }
}
