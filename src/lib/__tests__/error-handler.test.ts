/**
 * Error Handler Tests
 *
 * White-hat testing: Ensures error handling provides secure, consistent responses
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NextResponse } from 'next/server';
import {
  handleAPIError,
  withErrorHandler,
  tryCatch,
  assertOrThrow,
} from '../error-handler';
import {
  AppError,
  ValidationError,
  AuthenticationError,
  NotFoundError,
  RateLimitError,
  ExternalAPIError,
  DatabaseError,
  ServiceUnavailableError,
} from '../errors';
import { ZodError, z } from 'zod';

describe('handleAPIError', () => {
  beforeEach(() => {
    // Suppress console.error for tests
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should handle AppError instances correctly', async () => {
    const error = new ValidationError('Invalid input', { field: 'email' });
    const response = handleAPIError(error, '/api/test');

    expect(response).toBeInstanceOf(NextResponse);
    const json = await response.json();
    expect(json.success).toBe(false);
    expect(json.error.message).toBe('Invalid input');
    expect(json.error.code).toBe('VALIDATION_ERROR');
    expect(json.error.statusCode).toBe(400);
    expect(json.error.path).toBe('/api/test');
  });

  it('should handle ZodError instances', async () => {
    const schema = z.object({
      email: z.string().email(),
    });

    try {
      schema.parse({ email: 'invalid' });
    } catch (error) {
      const response = handleAPIError(error, '/api/validate');
      const json = await response.json();

      expect(json.success).toBe(false);
      expect(json.error.message).toBe('Invalid request data');
      expect(json.error.code).toBe('VALIDATION_ERROR');
      expect(json.error.statusCode).toBe(400);
    }
  });

  it('should handle Prisma P2002 (unique constraint) errors', async () => {
    const prismaError = {
      code: 'P2002',
      meta: { target: ['email'] },
    };

    const response = handleAPIError(prismaError);
    const json = await response.json();

    expect(json.success).toBe(false);
    expect(json.error.message).toBe('A record with this value already exists');
    expect(json.error.code).toBe('DATABASE_ERROR');
    expect(json.error.statusCode).toBe(500);
  });

  it('should handle Prisma P2025 (not found) errors', async () => {
    const prismaError = {
      code: 'P2025',
      meta: {},
    };

    const response = handleAPIError(prismaError);
    const json = await response.json();

    expect(json.success).toBe(false);
    expect(json.error.message).toBe('Record not found');
    expect(json.error.code).toBe('NOT_FOUND');
    expect(json.error.statusCode).toBe(404);
  });

  it('should handle fetch errors', async () => {
    const fetchError = new Error('fetch failed: connection timeout');

    const response = handleAPIError(fetchError);
    const json = await response.json();

    expect(json.success).toBe(false);
    expect(json.error.code).toBe('EXTERNAL_API_ERROR');
    expect(json.error.statusCode).toBe(502);
  });

  it('should handle unknown errors safely', async () => {
    const unknownError = 'Something went wrong';

    const response = handleAPIError(unknownError);
    const json = await response.json();

    expect(json.success).toBe(false);
    expect(json.error.code).toBe('INTERNAL_SERVER_ERROR');
    expect(json.error.statusCode).toBe(500);
  });

  it('should include timestamp in error response', async () => {
    const error = new NotFoundError('User');
    const response = handleAPIError(error);
    const json = await response.json();

    expect(json.error.timestamp).toBeDefined();
    expect(new Date(json.error.timestamp).getTime()).toBeGreaterThan(0);
  });
});

describe('withErrorHandler', () => {
  it('should return handler result when no error occurs', async () => {
    const handler = withErrorHandler(async () => {
      return NextResponse.json({ success: true, data: 'test' });
    });

    const response = await handler();
    const json = await response.json();

    expect(json.success).toBe(true);
    expect(json.data).toBe('test');
  });

  it('should catch and handle errors from handler', async () => {
    const handler = withErrorHandler(async () => {
      throw new ValidationError('Invalid data');
    });

    const response = await handler();
    const json = await response.json();

    expect(json.success).toBe(false);
    expect(json.error.message).toBe('Invalid data');
    expect(json.error.code).toBe('VALIDATION_ERROR');
  });

  it('should extract path from request object', async () => {
    const mockRequest = {
      url: 'http://localhost:3000/api/test?param=value',
    };

    const handler = withErrorHandler(async (req) => {
      throw new NotFoundError('Resource');
    });

    const response = await handler(mockRequest);
    const json = await response.json();

    expect(json.error.path).toBe('/api/test');
  });
});

describe('tryCatch', () => {
  it('should return [null, data] on success', async () => {
    const [error, data] = await tryCatch(Promise.resolve('success'));

    expect(error).toBeNull();
    expect(data).toBe('success');
  });

  it('should return [error, null] on failure', async () => {
    const testError = new Error('Test error');
    const [error, data] = await tryCatch(Promise.reject(testError));

    expect(error).toBeInstanceOf(Error);
    expect(error?.message).toBe('Test error');
    expect(data).toBeNull();
  });

  it('should convert non-Error rejections to Error', async () => {
    const [error, data] = await tryCatch(Promise.reject('string error'));

    expect(error).toBeInstanceOf(Error);
    expect(error?.message).toBe('string error');
    expect(data).toBeNull();
  });
});

describe('assertOrThrow', () => {
  it('should not throw when condition is truthy', () => {
    expect(() => {
      assertOrThrow('valid value', new ValidationError('Should not throw'));
    }).not.toThrow();
  });

  it('should throw error when condition is null', () => {
    expect(() => {
      assertOrThrow(null, new NotFoundError('Resource'));
    }).toThrow(NotFoundError);
  });

  it('should throw error when condition is undefined', () => {
    expect(() => {
      assertOrThrow(undefined, new AuthenticationError('Not authenticated'));
    }).toThrow(AuthenticationError);
  });

  it('should throw error when condition is false', () => {
    expect(() => {
      assertOrThrow(false, new ValidationError('Invalid'));
    }).toThrow(ValidationError);
  });

  it('should narrow type after assertion', () => {
    const value: string | null = 'test';
    assertOrThrow(value, new NotFoundError('Value'));

    // TypeScript should know value is string here
    const length: number = value.length;
    expect(length).toBe(4);
  });
});

describe('Error Classes', () => {
  it('should create ValidationError with correct properties', () => {
    const error = new ValidationError('Invalid input', { field: 'email' });

    expect(error.message).toBe('Invalid input');
    expect(error.statusCode).toBe(400);
    expect(error.code).toBe('VALIDATION_ERROR');
    expect(error.details).toEqual({ field: 'email' });
  });

  it('should create AuthenticationError with default message', () => {
    const error = new AuthenticationError();

    expect(error.message).toBe('Unauthorized');
    expect(error.statusCode).toBe(401);
    expect(error.code).toBe('AUTH_ERROR');
  });

  it('should create RateLimitError with resetTime', () => {
    const resetTime = Date.now() + 60000;
    const error = new RateLimitError(resetTime);

    expect(error.message).toBe('Rate limit exceeded');
    expect(error.statusCode).toBe(429);
    expect(error.code).toBe('RATE_LIMIT');
    expect(error.details).toEqual({ resetTime });
  });

  it('should create ExternalAPIError with service details', () => {
    const error = new ExternalAPIError('Binance', 'Connection timeout');

    expect(error.message).toBe('External API error: Binance');
    expect(error.statusCode).toBe(502);
    expect(error.code).toBe('EXTERNAL_API_ERROR');
    expect(error.details).toEqual({
      service: 'Binance',
      message: 'Connection timeout',
    });
  });

  it('should serialize error to JSON correctly', () => {
    const error = new DatabaseError('Query failed', { query: 'SELECT *' });
    const json = error.toJSON();

    expect(json.name).toBe('DatabaseError');
    expect(json.message).toBe('Query failed');
    expect(json.statusCode).toBe(500);
    expect(json.code).toBe('DATABASE_ERROR');
  });
});
