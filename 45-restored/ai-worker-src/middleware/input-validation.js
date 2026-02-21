/**
 * Input Validation & Sanitization Middleware
 *
 * White-hat compliant input security:
 * - Zod schema validation
 * - XSS prevention
 * - SQL injection prevention
 * - Request size limiting
 * - Type coercion prevention
 */

/**
 * Zod schemas for validation
 */
export const schemas = {
  // Signal request validation
  signalRequest: {
    symbol: (value) => {
      // Must be 6-12 uppercase letters (e.g., BTCUSDT)
      if (typeof value !== 'string') {
        throw new Error('Symbol must be a string');
      }
      if (!/^[A-Z]{6,12}$/.test(value)) {
        throw new Error('Invalid symbol format. Must be 6-12 uppercase letters');
      }
      return value;
    },

    timeframe: (value) => {
      // Optional: 1m, 5m, 15m, 1h, 4h, 1d
      if (!value) return '1h'; // default

      const validTimeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
      if (!validTimeframes.includes(value)) {
        throw new Error(`Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`);
      }
      return value;
    },

    limit: (value) => {
      // Optional: number of results (1-100)
      if (!value) return 10; // default

      const num = parseInt(value, 10);
      if (isNaN(num) || num < 1 || num > 100) {
        throw new Error('Limit must be between 1 and 100');
      }
      return num;
    }
  }
};

/**
 * Sanitize string input (XSS prevention)
 */
export function sanitizeString(input) {
  if (typeof input !== 'string') {
    return input;
  }

  return input
    .replace(/[<>]/g, '') // Remove HTML tags
    .replace(/['"]/g, '') // Remove quotes
    .replace(/[\\]/g, '') // Remove backslashes
    .trim();
}

/**
 * Sanitize object recursively
 */
export function sanitizeObject(obj) {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => sanitizeObject(item));
  }

  const sanitized = {};
  for (const [key, value] of Object.entries(obj)) {
    const sanitizedKey = sanitizeString(key);

    if (typeof value === 'string') {
      sanitized[sanitizedKey] = sanitizeString(value);
    } else if (typeof value === 'object') {
      sanitized[sanitizedKey] = sanitizeObject(value);
    } else {
      sanitized[sanitizedKey] = value;
    }
  }

  return sanitized;
}

/**
 * Validate request body
 */
export function validateRequest(body, schema) {
  const errors = [];
  const validated = {};

  for (const [field, validator] of Object.entries(schema)) {
    try {
      const value = body[field];
      validated[field] = validator(value);
    } catch (error) {
      errors.push({
        field,
        message: error.message
      });
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    data: validated
  };
}

/**
 * Request size limiting middleware
 */
export function requestSizeLimitMiddleware(maxSize = 10 * 1024) { // 10KB default
  return async (c, next) => {
    const contentLength = c.req.header('content-length');

    if (contentLength && parseInt(contentLength, 10) > maxSize) {
      console.warn('Request size limit exceeded:', {
        contentLength: contentLength,
        maxSize: maxSize,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        path: c.req.path,
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: 'Request entity too large',
        code: 'REQUEST_TOO_LARGE',
        maxSize: maxSize
      }, 413);
    }

    await next();
  };
}

/**
 * Input validation middleware
 */
export function inputValidationMiddleware(schema) {
  return async (c, next) => {
    // Get parsed body from HMAC middleware or parse now
    let body;
    if (c.req.parsedBody) {
      body = c.req.parsedBody;
    } else {
      try {
        body = await c.req.json();
      } catch (error) {
        return c.json({
          success: false,
          error: 'Invalid JSON body',
          code: 'INVALID_JSON'
        }, 400);
      }
    }

    // Sanitize input
    const sanitized = sanitizeObject(body);

    // Validate against schema
    const validation = validateRequest(sanitized, schema);

    if (!validation.valid) {
      console.warn('Input validation failed:', {
        errors: validation.errors,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        path: c.req.path,
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: 'Validation failed',
        code: 'VALIDATION_ERROR',
        details: validation.errors
      }, 400);
    }

    // Store validated data
    c.req.validated = validation.data;

    await next();
  };
}

/**
 * SQL injection detection
 */
export function detectSQLInjection(input) {
  if (typeof input !== 'string') {
    return false;
  }

  const sqlPatterns = [
    /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)/i,
    /(;|--|\/\*|\*\/)/,
    /('|(\\')|(\\")|(\\\\))/,
    /(UNION|OR|AND)\s+\d+\s*=\s*\d+/i
  ];

  return sqlPatterns.some(pattern => pattern.test(input));
}

/**
 * XSS detection
 */
export function detectXSS(input) {
  if (typeof input !== 'string') {
    return false;
  }

  const xssPatterns = [
    /<script/i,
    /javascript:/i,
    /on\w+\s*=/i, // event handlers
    /<iframe/i,
    /<embed/i,
    /<object/i
  ];

  return xssPatterns.some(pattern => pattern.test(input));
}

/**
 * Security scan middleware
 */
export function securityScanMiddleware() {
  return async (c, next) => {
    let body;
    try {
      body = await c.req.json();
    } catch (error) {
      return await next();
    }

    // Check for SQL injection attempts
    const bodyStr = JSON.stringify(body);

    if (detectSQLInjection(bodyStr)) {
      console.error('SQL injection attempt detected:', {
        body: bodyStr,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        path: c.req.path,
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: 'Malicious input detected',
        code: 'SECURITY_VIOLATION'
      }, 400);
    }

    // Check for XSS attempts
    if (detectXSS(bodyStr)) {
      console.error('XSS attempt detected:', {
        body: bodyStr,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        path: c.req.path,
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: 'Malicious input detected',
        code: 'SECURITY_VIOLATION'
      }, 400);
    }

    await next();
  };
}
