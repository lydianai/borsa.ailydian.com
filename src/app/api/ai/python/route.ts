/**
 * PYTHON AI SERVICES PROXY
 * Forwards requests to Python microservices
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

const PYTHON_SERVICES = {
  models: 'http://localhost:5003',
  signals: 'http://localhost:5004',
  talib: 'http://localhost:5005',
};

export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const service = searchParams.get('service') as keyof typeof PYTHON_SERVICES;
    const endpoint = searchParams.get('endpoint') || '';

    if (!service || !PYTHON_SERVICES[service]) {
      return NextResponse.json(
        { success: false, error: 'Invalid service' },
        { status: 400 }
      );
    }

    const body = await request.json();
    const serviceUrl = `${PYTHON_SERVICES[service]}${endpoint}`;

    console.log(`🔄 Proxying to Python service: ${serviceUrl}`);

    const response = await fetch(serviceUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ Python service proxy error:', error.message);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const service = searchParams.get('service') as keyof typeof PYTHON_SERVICES;
    const endpoint = searchParams.get('endpoint') || '';

    if (!service || !PYTHON_SERVICES[service]) {
      return NextResponse.json(
        { success: false, error: 'Invalid service' },
        { status: 400 }
      );
    }

    const serviceUrl = `${PYTHON_SERVICES[service]}${endpoint}`;

    console.log(`🔄 Proxying to Python service: ${serviceUrl}`);

    const response = await fetch(serviceUrl);
    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ Python service proxy error:', error.message);
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}
