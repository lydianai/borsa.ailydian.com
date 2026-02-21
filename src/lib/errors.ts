/**
 * Custom Error Classes
 *
 * White-hat compliance: Proper error handling prevents information leakage
 * and improves system reliability
 *
 * OWASP: https://owasp.org/www-community/Improper_Error_Handling
 */

/**
 * Base Application Error
 */
export class AppError extends Error {
  constructor(
    public message: string,
    public statusCode: number = 500,
    public code?: string,
    public details?: unknown
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      statusCode: this.statusCode,
      code: this.code,
      ...(process.env.NODE_ENV === 'development' && { details: this.details }),
    };
  }
}

/**
 * Validation Error (400)
 */
export class ValidationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, 400, 'VALIDATION_ERROR', details);
  }
}

/**
 * Authentication Error (401)
 */
export class AuthenticationError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 401, 'AUTH_ERROR');
  }
}

/**
 * Authorization Error (403)
 */
export class AuthorizationError extends AppError {
  constructor(message: string = 'Forbidden') {
    super(message, 403, 'FORBIDDEN');
  }
}

/**
 * Not Found Error (404)
 */
export class NotFoundError extends AppError {
  constructor(resource: string = 'Resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND');
  }
}

/**
 * Conflict Error (409)
 */
export class ConflictError extends AppError {
  constructor(message: string) {
    super(message, 409, 'CONFLICT');
  }
}

/**
 * Rate Limit Error (429)
 */
export class RateLimitError extends AppError {
  constructor(resetTime?: number) {
    super(
      'Rate limit exceeded',
      429,
      'RATE_LIMIT',
      resetTime ? { resetTime } : undefined
    );
  }
}

/**
 * External API Error (502)
 */
export class ExternalAPIError extends AppError {
  constructor(service: string, message: string) {
    super(
      `External API error: ${service}`,
      502,
      'EXTERNAL_API_ERROR',
      { service, message }
    );
  }
}

/**
 * Database Error (500)
 */
export class DatabaseError extends AppError {
  constructor(message: string, details?: unknown) {
    super(message, 500, 'DATABASE_ERROR', details);
  }
}

/**
 * Service Unavailable Error (503)
 */
export class ServiceUnavailableError extends AppError {
  constructor(service?: string) {
    super(
      service ? `Service unavailable: ${service}` : 'Service unavailable',
      503,
      'SERVICE_UNAVAILABLE'
    );
  }
}
