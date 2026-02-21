"""
🚦 RATE LIMITER
===============
Request rate limiting for API endpoints

Features:
- IP-based rate limiting
- Configurable limits per endpoint
- Automatic cleanup of old requests
- Thread-safe implementation

WHITE-HAT COMPLIANCE: DoS attack prevention (not malicious throttling)
"""

from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict
from threading import Lock
from typing import Callable


class RateLimiter:
    """
    Rate limiter with sliding window algorithm

    Example:
        limiter = RateLimiter(requests_per_minute=60)
        if limiter.is_allowed("192.168.1.1"):
            # Process request
        else:
            # Return 429 Too Many Requests
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier

        Args:
            identifier: Usually IP address or user ID

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old requests (older than 1 minute)
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > minute_ago
            ]

            # Check limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False

            # Add new request
            self.requests[identifier].append(now)
            return True

    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60

            recent_requests = [
                req_time for req_time in self.requests.get(identifier, [])
                if req_time > minute_ago
            ]

            return max(0, self.requests_per_minute - len(recent_requests))


def rate_limit(requests_per_minute: int = 60):
    """
    Decorator for rate limiting Flask routes

    Args:
        requests_per_minute: Maximum requests allowed per minute per IP

    Example:
        @app.route('/api/signals')
        @rate_limit(requests_per_minute=100)
        def get_signals():
            return jsonify({'data': []})
    """
    limiter = RateLimiter(requests_per_minute)

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # IP tabanlı rate limiting
            identifier = request.remote_addr or 'unknown'

            if not limiter.is_allowed(identifier):
                remaining_time = 60  # seconds

                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {requests_per_minute} requests per minute allowed',
                    'retry_after': remaining_time
                }), 429

            # Add rate limit info to response headers
            response = f(*args, **kwargs)

            if hasattr(response, 'headers'):
                remaining = limiter.get_remaining(identifier)
                response.headers['X-RateLimit-Limit'] = str(requests_per_minute)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = str(int(time.time()) + 60)

            return response

        return decorated_function
    return decorator


# Example usage
if __name__ == "__main__":
    # Test rate limiter
    limiter = RateLimiter(requests_per_minute=5)

    test_ip = "192.168.1.1"

    print("Testing rate limiter (5 requests/minute):")
    for i in range(7):
        if limiter.is_allowed(test_ip):
            remaining = limiter.get_remaining(test_ip)
            print(f"✅ Request {i+1} allowed ({remaining} remaining)")
        else:
            print(f"❌ Request {i+1} blocked (rate limit exceeded)")
