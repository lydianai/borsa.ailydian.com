import { NextRequest, NextResponse } from 'next/server';
import { ZodError, ZodSchema } from 'zod';

/**
 * API Response Types
 */
export interface ApiSuccessResponse<T = any> {
  success: true;
  data: T;
  message?: string;
  meta?: {
    timestamp: string;
    requestId: string;
    [key: string]: any;
  };
}

export interface ApiErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: any;
    statusCode: number;
  };
  meta?: {
    timestamp: string;
    requestId: string;
  };
}

export type ApiResponse<T = any> = ApiSuccessResponse<T> | ApiErrorResponse;

/**
 * API Error Class
 */
export class ApiError extends Error {
  constructor(
    public message: string,
    public statusCode: number = 500,
    public code: string = 'INTERNAL_SERVER_ERROR',
    public details?: any
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Generate unique request ID
 */
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Create success response
 */
export function successResponse<T>(
  data: T,
  message?: string,
  meta?: Record<string, any>
): NextResponse<ApiSuccessResponse<T>> {
  const requestId = generateRequestId();

  return NextResponse.json(
    {
      success: true,
      data,
      message,
      meta: {
        timestamp: new Date().toISOString(),
        requestId,
        ...meta,
      },
    },
    { status: 200 }
  );
}

/**
 * Create error response
 */
export function errorResponse(
  error: ApiError | Error | ZodError | any,
  request?: NextRequest
): NextResponse<ApiErrorResponse> {
  const requestId = generateRequestId();

  // Handle Zod validation errors
  if (error instanceof ZodError) {
    return NextResponse.json(
      {
        success: false,
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid request data',
          details: error.errors,
          statusCode: 400,
        },
        meta: {
          timestamp: new Date().toISOString(),
          requestId,
        },
      },
      { status: 400 }
    );
  }

  // Handle custom API errors
  if (error instanceof ApiError) {
    return NextResponse.json(
      {
        success: false,
        error: {
          code: error.code,
          message: error.message,
          details: error.details,
          statusCode: error.statusCode,
        },
        meta: {
          timestamp: new Date().toISOString(),
          requestId,
        },
      },
      { status: error.statusCode }
    );
  }

  // Handle generic errors
  const statusCode = (error as any).statusCode || 500;
  const message = error.message || 'Internal server error';

  return NextResponse.json(
    {
      success: false,
      error: {
        code: 'INTERNAL_SERVER_ERROR',
        message,
        statusCode,
      },
      meta: {
        timestamp: new Date().toISOString(),
        requestId,
      },
    },
    { status: statusCode }
  );
}

/**
 * Validate request body with Zod schema
 */
export async function validateRequest<T>(
  request: NextRequest,
  schema: ZodSchema<T>
): Promise<T> {
  try {
    const body = await request.json();
    return schema.parse(body);
  } catch (error) {
    if (error instanceof ZodError) {
      throw error;
    }
    throw new ApiError('Invalid JSON body', 400, 'INVALID_JSON');
  }
}

/**
 * Rate limiter storage (in-memory - use Redis in production)
 */
const rateLimitStore = new Map<string, { count: number; resetAt: number }>();

/**
 * Rate limiting middleware
 */
export function rateLimit(
  maxRequests: number = 100,
  windowMs: number = 60000 // 1 minute
) {
  return async (request: NextRequest): Promise<NextResponse | null> => {
    // Get client identifier (IP or API key)
    const clientId =
      request.headers.get('x-forwarded-for')?.split(',')[0] ||
      request.headers.get('x-api-key') ||
      'anonymous';

    const now = Date.now();
    const limitData = rateLimitStore.get(clientId);

    // Reset if window expired
    if (!limitData || now > limitData.resetAt) {
      rateLimitStore.set(clientId, {
        count: 1,
        resetAt: now + windowMs,
      });
      return null; // Allow request
    }

    // Increment counter
    if (limitData.count >= maxRequests) {
      return NextResponse.json(
        {
          success: false,
          error: {
            code: 'RATE_LIMIT_EXCEEDED',
            message: 'Too many requests. Please try again later.',
            statusCode: 429,
          },
          meta: {
            timestamp: new Date().toISOString(),
            requestId: generateRequestId(),
            resetAt: new Date(limitData.resetAt).toISOString(),
          },
        },
        {
          status: 429,
          headers: {
            'Retry-After': Math.ceil((limitData.resetAt - now) / 1000).toString(),
            'X-RateLimit-Limit': maxRequests.toString(),
            'X-RateLimit-Remaining': '0',
            'X-RateLimit-Reset': limitData.resetAt.toString(),
          },
        }
      );
    }

    // Update counter
    limitData.count++;
    rateLimitStore.set(clientId, limitData);

    return null; // Allow request
  };
}

/**
 * CORS middleware
 */
export function cors(allowedOrigins: string[] = ['*']) {
  return (request: NextRequest, response: NextResponse): NextResponse => {
    const origin = request.headers.get('origin');

    // Check if origin is allowed
    const isAllowed = allowedOrigins.includes('*') || (origin && allowedOrigins.includes(origin));

    if (isAllowed) {
      response.headers.set('Access-Control-Allow-Origin', origin || '*');
      response.headers.set(
        'Access-Control-Allow-Methods',
        'GET, POST, PUT, DELETE, OPTIONS'
      );
      response.headers.set(
        'Access-Control-Allow-Headers',
        'Content-Type, Authorization, X-API-Key'
      );
      response.headers.set('Access-Control-Max-Age', '86400');
    }

    return response;
  };
}

/**
 * Authentication middleware
 */
export function requireAuth() {
  return async (request: NextRequest): Promise<NextResponse | null> => {
    const token = request.headers.get('authorization')?.replace('Bearer ', '');

    if (!token) {
      return NextResponse.json(
        {
          success: false,
          error: {
            code: 'UNAUTHORIZED',
            message: 'Authentication required',
            statusCode: 401,
          },
          meta: {
            timestamp: new Date().toISOString(),
            requestId: generateRequestId(),
          },
        },
        { status: 401 }
      );
    }

    // Verify token (implement your JWT verification logic here)
    // For now, just check if token exists
    try {
      // TODO: Verify JWT token
      // const decoded = await verifyJWT(token);
      // request.user = decoded;

      return null; // Allow request
    } catch (error) {
      return NextResponse.json(
        {
          success: false,
          error: {
            code: 'INVALID_TOKEN',
            message: 'Invalid or expired token',
            statusCode: 401,
          },
          meta: {
            timestamp: new Date().toISOString(),
            requestId: generateRequestId(),
          },
        },
        { status: 401 }
      );
    }
  };
}

/**
 * Compose multiple middlewares
 */
export function composeMiddleware(
  ...middlewares: Array<(request: NextRequest) => Promise<NextResponse | null>>
) {
  return async (request: NextRequest): Promise<NextResponse | null> => {
    for (const middleware of middlewares) {
      const result = await middleware(request);
      if (result) {
        return result; // Return error response
      }
    }
    return null; // All middlewares passed
  };
}

/**
 * API Handler wrapper with error handling
 */
export function apiHandler<T = any>(
  handler: (request: NextRequest) => Promise<NextResponse<ApiSuccessResponse<T>>>
) {
  return async (request: NextRequest): Promise<NextResponse> => {
    try {
      return await handler(request);
    } catch (error) {
      console.error('API Error:', error);
      return errorResponse(error, request);
    }
  };
}
