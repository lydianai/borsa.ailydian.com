"""
📊 PROMETHEUS METRICS HELPER
============================
Standardized metrics collection for all services

Features:
- Request counter
- Response time histogram
- Active connections gauge
- Error counter
- Custom metrics support
"""

from typing import Optional, Dict
import time
from functools import wraps


class MetricsCollector:
    """Metrics collector with Prometheus support"""

    def __init__(self, service_name: str, enabled: bool = True):
        self.service_name = service_name
        self.enabled = enabled
        self.metrics = {}

        if self.enabled:
            try:
                from prometheus_client import Counter, Histogram, Gauge, Info

                # Request metrics
                self.request_counter = Counter(
                    f'{service_name}_requests_total',
                    'Total number of requests',
                    ['endpoint', 'method', 'status']
                )

                self.response_time = Histogram(
                    f'{service_name}_response_time_seconds',
                    'Response time in seconds',
                    ['endpoint', 'method']
                )

                # System metrics
                self.active_connections = Gauge(
                    f'{service_name}_active_connections',
                    'Number of active connections'
                )

                self.error_counter = Counter(
                    f'{service_name}_errors_total',
                    'Total number of errors',
                    ['error_type']
                )

                # Service info
                self.service_info = Info(
                    f'{service_name}_info',
                    'Service information'
                )

                print(f"✅ Prometheus metrics enabled for {service_name}")

            except ImportError:
                print(f"⚠️  Prometheus client not available, metrics disabled")
                self.enabled = False

    def record_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration: float
    ) -> None:
        """Record a request with response time"""
        if not self.enabled:
            return

        try:
            self.request_counter.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()

            self.response_time.labels(
                endpoint=endpoint,
                method=method
            ).observe(duration)
        except Exception as e:
            print(f"⚠️  Metrics recording error: {e}")

    def record_error(self, error_type: str) -> None:
        """Record an error"""
        if not self.enabled:
            return

        try:
            self.error_counter.labels(error_type=error_type).inc()
        except Exception as e:
            print(f"⚠️  Error recording failed: {e}")

    def set_active_connections(self, count: int) -> None:
        """Set active connections count"""
        if not self.enabled:
            return

        try:
            self.active_connections.set(count)
        except Exception as e:
            print(f"⚠️  Active connections update failed: {e}")

    def increment_active_connections(self) -> None:
        """Increment active connections"""
        if not self.enabled:
            return

        try:
            self.active_connections.inc()
        except Exception as e:
            print(f"⚠️  Active connections increment failed: {e}")

    def decrement_active_connections(self) -> None:
        """Decrement active connections"""
        if not self.enabled:
            return

        try:
            self.active_connections.dec()
        except Exception as e:
            print(f"⚠️  Active connections decrement failed: {e}")


def track_time(metrics: MetricsCollector, endpoint: str, method: str = "GET"):
    """
    Decorator to track request time

    Usage:
        @track_time(metrics, "/analyze")
        def analyze_data():
            # Your code here
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 200
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 500
                error = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time
                if metrics.enabled:
                    metrics.record_request(endpoint, method, status, duration)
                    if error:
                        metrics.record_error(error)

        return wrapper
    return decorator


def create_metrics_endpoint(metrics: MetricsCollector):
    """
    Create Flask metrics endpoint for Prometheus

    Usage:
        from flask import Flask
        app = Flask(__name__)
        metrics = MetricsCollector("my-service")

        @app.route('/metrics')
        def metrics_endpoint():
            return create_metrics_endpoint(metrics)()
    """
    def endpoint():
        if not metrics.enabled:
            return "Metrics not available", 503

        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from flask import Response

            return Response(
                generate_latest(),
                mimetype=CONTENT_TYPE_LATEST
            )
        except Exception as e:
            return f"Error generating metrics: {e}", 500

    return endpoint


# Example usage
if __name__ == "__main__":
    # Create metrics collector
    metrics = MetricsCollector("test-service", enabled=True)

    # Simulate some requests
    print("\n📊 Simulating requests...")

    @track_time(metrics, "/test", "GET")
    def test_endpoint():
        time.sleep(0.1)  # Simulate work
        return "OK"

    # Make some test requests
    for i in range(5):
        try:
            result = test_endpoint()
            metrics.record_request("/test", "GET", 200, 0.1)
        except Exception as e:
            metrics.record_error(type(e).__name__)

    # Update active connections
    metrics.set_active_connections(42)

    # Record some errors
    metrics.record_error("ValueError")
    metrics.record_error("ConnectionError")

    print("✅ Metrics test completed")
    print("📈 Visit http://localhost:<port>/metrics to see Prometheus metrics")
