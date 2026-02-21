/**
 * HMAC Authentication Middleware
 *
 * White-hat compliant security:
 * - HMAC-SHA256 signature verification
 * - Timestamp-based replay attack prevention
 * - Constant-time comparison
 * - Detailed security logging
 */

import crypto from 'crypto';

/**
 * Verify HMAC signature
 * @param {Request} request - Hono request object
 * @param {string} body - Raw request body
 * @param {string} secret - HMAC secret
 * @returns {Object} - Verification result
 */
export function verifyHMACSignature(request, body, secret) {
  const signature = request.header('x-signature');
  const timestamp = request.header('x-timestamp');

  // Security check 1: Required headers
  if (!signature || !timestamp) {
    return {
      valid: false,
      error: 'Missing authentication headers',
      code: 'MISSING_HEADERS'
    };
  }

  // Security check 2: Timestamp validation (prevent replay attacks)
  const now = Date.now();
  const requestTime = parseInt(timestamp, 10);

  if (isNaN(requestTime)) {
    return {
      valid: false,
      error: 'Invalid timestamp format',
      code: 'INVALID_TIMESTAMP'
    };
  }

  // 5 minute window for request validity
  const MAX_AGE_MS = 5 * 60 * 1000;
  const age = now - requestTime;

  if (age > MAX_AGE_MS) {
    return {
      valid: false,
      error: 'Request expired (timestamp too old)',
      code: 'EXPIRED_REQUEST',
      age: age
    };
  }

  if (age < -MAX_AGE_MS) {
    return {
      valid: false,
      error: 'Request timestamp in future',
      code: 'FUTURE_TIMESTAMP',
      age: age
    };
  }

  // Security check 3: HMAC signature verification
  try {
    const payload = timestamp + body;
    const expectedSignature = crypto
      .createHmac('sha256', secret)
      .update(payload)
      .digest('hex');

    // Constant-time comparison to prevent timing attacks
    const isValid = crypto.timingSafeEqual(
      Buffer.from(signature, 'hex'),
      Buffer.from(expectedSignature, 'hex')
    );

    if (!isValid) {
      return {
        valid: false,
        error: 'Invalid signature',
        code: 'INVALID_SIGNATURE'
      };
    }

    return {
      valid: true,
      age: age
    };

  } catch (error) {
    return {
      valid: false,
      error: 'Signature verification failed',
      code: 'VERIFICATION_ERROR',
      details: error.message
    };
  }
}

/**
 * HMAC Authentication Middleware for Hono
 */
export function hmacAuthMiddleware() {
  return async (c, next) => {
    const secret = process.env.HMAC_SECRET;

    // Skip authentication for health check
    if (c.req.path === '/health') {
      return await next();
    }

    // Ensure secret is configured
    if (!secret) {
      console.error('HMAC_SECRET not configured');
      return c.json({
        success: false,
        error: 'Authentication not configured',
        code: 'CONFIG_ERROR'
      }, 500);
    }

    // Get raw body
    let body;
    try {
      body = await c.req.text();
    } catch (error) {
      return c.json({
        success: false,
        error: 'Invalid request body',
        code: 'INVALID_BODY'
      }, 400);
    }

    // Verify HMAC signature
    const verification = verifyHMACSignature(c.req, body, secret);

    if (!verification.valid) {
      // Log security event
      console.warn('HMAC verification failed:', {
        code: verification.code,
        error: verification.error,
        ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
        path: c.req.path,
        method: c.req.method,
        timestamp: new Date().toISOString()
      });

      return c.json({
        success: false,
        error: verification.error,
        code: verification.code
      }, 401);
    }

    // Log successful authentication
    console.info('HMAC verification successful:', {
      age: verification.age,
      ip: c.req.header('x-forwarded-for') || c.req.header('x-real-ip'),
      path: c.req.path,
      timestamp: new Date().toISOString()
    });

    // Parse body back to JSON for handler
    try {
      const parsedBody = JSON.parse(body);
      c.req.parsedBody = parsedBody;
    } catch (error) {
      // Body parsing failed, continue anyway
      c.req.parsedBody = {};
    }

    await next();
  };
}

/**
 * Generate HMAC signature (for client-side use)
 * @param {Object} body - Request body
 * @param {string} secret - HMAC secret
 * @returns {Object} - Signature and timestamp
 */
export function generateHMACSignature(body, secret) {
  const timestamp = Date.now().toString();
  const payload = timestamp + JSON.stringify(body);

  const signature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');

  return {
    signature,
    timestamp
  };
}
