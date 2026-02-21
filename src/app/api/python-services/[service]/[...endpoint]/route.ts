/**
 * 🐍 PYTHON SERVICES API PROXY
 *
 * Dynamic HTTP proxy for Python Flask microservices
 *
 * USAGE:
 * GET  /api/python-services/ai-models/health
 * GET  /api/python-services/signal-generator/signals?symbol=BTCUSDT
 * POST /api/python-services/ai-models/predict
 *
 * SERVICES:
 * - ai-models (Port 5003) - AI/ML prediction models
 * - signal-generator (Port 5004) - Trading signal generation
 * - talib-service (Port 5005) - TA-Lib technical indicators
 * - risk-management (Port 5006) - Risk analysis
 * - feature-engineering (Port 5001) - Feature extraction
 * - smc-strategy (Port 5007) - Smart Money Concepts
 * - transformer-ai (Port 5008) - Transformer models
 * - online-learning (Port 5009) - Online learning models
 * - multi-timeframe (Port 5010) - Multi-timeframe analysis
 * - order-flow (Port 5011) - Order flow analysis
 * - continuous-monitor (Port 5012) - Continuous monitoring
 *
 * WHITE-HAT RULES:
 * - Read-only proxy (no system modifications)
 * - Transparent request logging
 * - Error handling with retry logic
 * - Timeout protection (30 seconds)
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Service port mapping
const SERVICE_PORTS: Record<string, number> = {
  'ai-models': 5003,
  'signal-generator': 5004,
  'talib-service': 5005,
  'risk-management': 5006,
  'feature-engineering': 5001,
  'smc-strategy': 5007,
  'transformer-ai': 5008,
  'online-learning': 5009,
  'multi-timeframe': 5010,
  'order-flow': 5011,
  'continuous-monitor': 5012,
  'quantum-ladder': 6000,
};

// Request timeout (30 seconds)
const REQUEST_TIMEOUT = 30000;

// Retry configuration
const MAX_RETRIES = 2;
const RETRY_DELAY = 1000; // 1 second

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Fetch with timeout
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Proxy request to Python service with retry logic
 */
async function proxyRequest(
  targetUrl: string,
  method: string,
  headers: Headers,
  body?: BodyInit | null
): Promise<Response> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      // Apply retry delay (except for first attempt)
      if (attempt > 0) {
        await sleep(RETRY_DELAY * attempt);
      }

      // Prepare request options
      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': headers.get('content-type') || 'application/json',
          'Accept': 'application/json',
        },
      };

      // Add body for POST, PUT, PATCH requests
      if (body && ['POST', 'PUT', 'PATCH'].includes(method)) {
        options.body = body;
      }

      // Make request with timeout
      const response = await fetchWithTimeout(targetUrl, options, REQUEST_TIMEOUT);

      // Return response (even if status >= 400, let caller handle it)
      return response;

    } catch (error) {
      lastError = error as Error;

      // Don't retry on abort (timeout)
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Request timeout after ${REQUEST_TIMEOUT}ms`);
      }

      // Log retry attempt
      if (attempt < MAX_RETRIES) {
        console.warn(
          `[Python Proxy] Retry ${attempt + 1}/${MAX_RETRIES} for ${targetUrl}:`,
          error instanceof Error ? error.message : 'Unknown error'
        );
      }
    }
  }

  // All retries failed
  throw new Error(
    `Failed after ${MAX_RETRIES} retries: ${lastError?.message || 'Unknown error'}`
  );
}

/**
 * GET handler
 */
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ service: string; endpoint: string[] }> }
) {
  const params = await context.params;
  const { service, endpoint } = await params;
  const startTime = Date.now();

  try {
    // Validate service
    const port = SERVICE_PORTS[service];
    if (!port) {
      return NextResponse.json(
        {
          success: false,
          error: `Unknown service: ${service}`,
          availableServices: Object.keys(SERVICE_PORTS),
        },
        { status: 404 }
      );
    }

    // Build target URL
    const endpointPath = endpoint.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const targetUrl = `http://localhost:${port}/${endpointPath}${
      searchParams ? `?${searchParams}` : ''
    }`;

    console.log(`[Python Proxy] GET ${service}:${port}/${endpointPath}`);

    // Proxy request
    const response = await proxyRequest(targetUrl, 'GET', request.headers);

    // Parse response
    const contentType = response.headers.get('content-type');
    let data: unknown;

    if (contentType?.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    // Log response time
    const duration = Date.now() - startTime;
    console.log(
      `[Python Proxy] GET ${service}:${port}/${endpointPath} - ${response.status} (${duration}ms)`
    );

    // Return response with same status code
    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(
      `[Python Proxy] GET ${service} failed (${duration}ms):`,
      error instanceof Error ? error.message : 'Unknown error'
    );

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Proxy request failed',
        service,
        endpoint: endpoint.join('/'),
        duration,
      },
      { status: 503 } // Service Unavailable
    );
  }
}

/**
 * POST handler
 */
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ service: string; endpoint: string[] }> }
) {
  const params = await context.params;
  const { service, endpoint} = params;
  const startTime = Date.now();

  try {
    // Validate service
    const port = SERVICE_PORTS[service];
    if (!port) {
      return NextResponse.json(
        {
          success: false,
          error: `Unknown service: ${service}`,
          availableServices: Object.keys(SERVICE_PORTS),
        },
        { status: 404 }
      );
    }

    // Build target URL
    const endpointPath = endpoint.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const targetUrl = `http://localhost:${port}/${endpointPath}${
      searchParams ? `?${searchParams}` : ''
    }`;

    console.log(`[Python Proxy] POST ${service}:${port}/${endpointPath}`);

    // Get request body
    const body = await request.text();

    // Proxy request
    const response = await proxyRequest(targetUrl, 'POST', request.headers, body);

    // Parse response
    const contentType = response.headers.get('content-type');
    let data: unknown;

    if (contentType?.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    // Log response time
    const duration = Date.now() - startTime;
    console.log(
      `[Python Proxy] POST ${service}:${port}/${endpointPath} - ${response.status} (${duration}ms)`
    );

    // Return response with same status code
    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(
      `[Python Proxy] POST ${service} failed (${duration}ms):`,
      error instanceof Error ? error.message : 'Unknown error'
    );

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Proxy request failed',
        service,
        endpoint: endpoint.join('/'),
        duration,
      },
      { status: 503 }
    );
  }
}

/**
 * PUT handler
 */
export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ service: string; endpoint: string[] }> }
) {
  const params = await context.params;
  const { service, endpoint } = await params;
  const startTime = Date.now();

  try {
    const port = SERVICE_PORTS[service];
    if (!port) {
      return NextResponse.json(
        { success: false, error: `Unknown service: ${service}` },
        { status: 404 }
      );
    }

    const endpointPath = endpoint.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const targetUrl = `http://localhost:${port}/${endpointPath}${
      searchParams ? `?${searchParams}` : ''
    }`;

    console.log(`[Python Proxy] PUT ${service}:${port}/${endpointPath}`);

    const body = await request.text();
    const response = await proxyRequest(targetUrl, 'PUT', request.headers, body);

    const contentType = response.headers.get('content-type');
    const data = contentType?.includes('application/json')
      ? await response.json()
      : await response.text();

    const duration = Date.now() - startTime;
    console.log(`[Python Proxy] PUT ${service} - ${response.status} (${duration}ms)`);

    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(`[Python Proxy] PUT ${service} failed (${duration}ms):`, error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Proxy request failed',
        duration,
      },
      { status: 503 }
    );
  }
}

/**
 * DELETE handler
 */
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ service: string; endpoint: string[] }> }
) {
  const params = await context.params;
  const { service, endpoint } = await params;
  const startTime = Date.now();

  try {
    const port = SERVICE_PORTS[service];
    if (!port) {
      return NextResponse.json(
        { success: false, error: `Unknown service: ${service}` },
        { status: 404 }
      );
    }

    const endpointPath = endpoint.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const targetUrl = `http://localhost:${port}/${endpointPath}${
      searchParams ? `?${searchParams}` : ''
    }`;

    console.log(`[Python Proxy] DELETE ${service}:${port}/${endpointPath}`);

    const response = await proxyRequest(targetUrl, 'DELETE', request.headers);

    const contentType = response.headers.get('content-type');
    const data = contentType?.includes('application/json')
      ? await response.json()
      : await response.text();

    const duration = Date.now() - startTime;
    console.log(`[Python Proxy] DELETE ${service} - ${response.status} (${duration}ms)`);

    return NextResponse.json(data, { status: response.status });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error(`[Python Proxy] DELETE ${service} failed (${duration}ms):`, error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Proxy request failed',
        duration,
      },
      { status: 503 }
    );
  }
}
