/**
 * Centralized API Error Handler
 *
 * White-hat compliance: Proper error handling prevents information leakage
 * and provides consistent error responses
 *
 * OWASP: https://owasp.org/www-community/Improper_Error_Handling
 */

import { NextResponse } from 'next/server';
import { ZodError } from 'zod';
import {
  AppError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  RateLimitError,
  ExternalAPIError,
  DatabaseError,
  ServiceUnavailableError,
} from './errors';
import { formatZodError } from './validation/schemas';

/**
 * Standard API Error Response Format
 */
export interface StandardErrorResponse {
  success: false;
  error: {
    message: string;
    code: string;
    statusCode: number;
    details?: unknown;
    timestamp: string;
    path?: string;
  };
}

/**
 * Log error to monitoring service (Sentry when configured)
 */
function logError(error: Error, context?: Record<string, unknown>): void {
  // In development, log to console
  if (process.env.NODE_ENV === 'development') {
    console.error('[ERROR]', {
      name: error.name,
      message: error.message,
      stack: error.stack,
      context,
    });
  }

  // Send to Sentry in production
  if (process.env.NODE_ENV === 'production') {
    try {
      // Dynamic import to avoid bundling in development
      import('@/lib/monitoring/sentry').then(({ captureException }) => {
        captureException(error, context);
      });
    } catch (sentryError) {
      console.error('[Sentry] Failed to log error:', sentryError);
    }
  }
}

/**
 * Handle API errors and return standardized responses
 */
export function handleAPIError(
  error: unknown,
  path?: string
): NextResponse<StandardErrorResponse> {
  // Log the error
  logError(error instanceof Error ? error : new Error(String(error)), {
    path,
  });

  // Handle Zod validation errors
  if (error instanceof ZodError) {
    const validationError = new ValidationError(
      'Invalid request data',
      formatZodError(error)
    );
    return createErrorResponse(validationError, path);
  }

  // Handle custom AppError instances
  if (error instanceof AppError) {
    return createErrorResponse(error, path);
  }

  // Handle Prisma errors
  if (error && typeof error === 'object' && 'code' in error) {
    const prismaError = error as { code: string; meta?: unknown };

    // P2002: Unique constraint violation
    if (prismaError.code === 'P2002') {
      const dbError = new DatabaseError(
        'A record with this value already exists',
        { code: prismaError.code, meta: prismaError.meta }
      );
      return createErrorResponse(dbError, path);
    }

    // P2025: Record not found
    if (prismaError.code === 'P2025') {
      const notFoundError = new NotFoundError('Record');
      return createErrorResponse(notFoundError, path);
    }

    // Other Prisma errors
    const dbError = new DatabaseError(
      'Database operation failed',
      { code: prismaError.code }
    );
    return createErrorResponse(dbError, path);
  }

  // Handle fetch/network errors
  if (error instanceof Error && error.message.includes('fetch')) {
    const apiError = new ExternalAPIError(
      'External Service',
      error.message
    );
    return createErrorResponse(apiError, path);
  }

  // Default: Internal Server Error
  const internalError = new AppError(
    process.env.NODE_ENV === 'development'
      ? error instanceof Error
        ? error.message
        : String(error)
      : 'An unexpected error occurred',
    500,
    'INTERNAL_SERVER_ERROR'
  );

  return createErrorResponse(internalError, path);
}

/**
 * Create standardized error response
 */
function createErrorResponse(
  error: AppError,
  path?: string
): NextResponse<StandardErrorResponse> {
  const response: StandardErrorResponse = {
    success: false,
    error: {
      message: error.message,
      code: error.code || 'UNKNOWN_ERROR',
      statusCode: error.statusCode,
      timestamp: new Date().toISOString(),
      ...(path ? { path } : {}),
      ...(process.env.NODE_ENV === 'development' && error.details ? {
        details: error.details,
      } : {}),
    },
  };

  return NextResponse.json(response, { status: error.statusCode });
}

/**
 * Async error handler wrapper for API routes
 *
 * Usage:
 * export const GET = withErrorHandler(async (request) => {
 *   // Your handler code
 * });
 */
export function withErrorHandler<T extends any[]>(
  handler: (...args: T) => Promise<NextResponse>
) {
  return async (...args: T): Promise<NextResponse> => {
    try {
      return await handler(...args);
    } catch (error) {
      // Extract request path if available
      const request = args[0];
      const path =
        request && typeof request === 'object' && 'url' in request
          ? new URL(request.url as string).pathname
          : undefined;

      return handleAPIError(error, path);
    }
  };
}

/**
 * Try-catch wrapper for async operations
 *
 * Usage:
 * const [error, data] = await tryCatch(someAsyncOperation());
 * if (error) {
 *   return handleAPIError(error);
 * }
 */
export async function tryCatch<T>(
  promise: Promise<T>
): Promise<[Error | null, T | null]> {
  try {
    const data = await promise;
    return [null, data];
  } catch (error) {
    return [error instanceof Error ? error : new Error(String(error)), null];
  }
}

/**
 * Assert condition or throw error
 *
 * Usage:
 * assertOrThrow(userId, new AuthenticationError('User not authenticated'));
 */
export function assertOrThrow<T>(
  condition: T | null | undefined,
  error: AppError
): asserts condition is T {
  if (!condition) {
    throw error;
  }
}
