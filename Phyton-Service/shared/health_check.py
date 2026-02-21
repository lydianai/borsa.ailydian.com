"""
🏥 STANDARDIZED HEALTH CHECK
============================
Unified health check endpoint for all services

Features:
- Standard response format
- Service metrics
- Dependency checking
- Uptime tracking
"""

from datetime import datetime
from typing import Dict, List, Optional, Callable
import time


class HealthCheck:
    """Health check manager for Python services"""

    def __init__(self, service_name: str, service_port: int):
        self.service_name = service_name
        self.service_port = service_port
        self.start_time = time.time()
        self.dependency_checks: Dict[str, Callable] = {}
        self.metrics: Dict[str, any] = {}

    def add_dependency_check(self, name: str, check_func: Callable) -> None:
        """
        Add a dependency health check

        Args:
            name: Dependency name (e.g., "redis", "database", "binance_api")
            check_func: Function that returns True if healthy, False otherwise
        """
        self.dependency_checks[name] = check_func

    def add_metric(self, key: str, value: any) -> None:
        """Add custom metric to health check response"""
        self.metrics[key] = value

    def check_dependencies(self) -> Dict[str, Dict]:
        """Check all dependencies"""
        results = {}
        for name, check_func in self.dependency_checks.items():
            try:
                is_healthy = check_func()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": "OK" if is_healthy else "Failed"
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "message": str(e)
                }
        return results

    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time

    def format_uptime(self) -> str:
        """Format uptime as human-readable string"""
        uptime = self.get_uptime()
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        return f"{hours}h {minutes}m {seconds}s"

    def get_health(self) -> Dict:
        """
        Get comprehensive health status

        Returns:
            Dict with standardized health check response
        """
        dependencies = self.check_dependencies()
        all_healthy = all(dep["status"] == "healthy" for dep in dependencies.values())

        return {
            "service": self.service_name,
            "status": "healthy" if all_healthy else "degraded",
            "port": self.service_port,
            "timestamp": datetime.now().isoformat(),
            "uptime": self.format_uptime(),
            "uptime_seconds": round(self.get_uptime(), 2),
            "dependencies": dependencies if dependencies else None,
            "metrics": self.metrics if self.metrics else None
        }

    def is_healthy(self) -> bool:
        """Simple health check (returns True/False)"""
        try:
            dependencies = self.check_dependencies()
            return all(dep["status"] == "healthy" for dep in dependencies.values())
        except:
            return False


def create_health_endpoint(health_check: HealthCheck):
    """
    Create Flask health check endpoint

    Usage:
        from flask import Flask, jsonify
        app = Flask(__name__)
        health = HealthCheck("My Service", 5000)

        @app.route('/health')
        def health_endpoint():
            return jsonify(health.get_health())
    """
    def endpoint():
        from flask import jsonify
        return jsonify(health_check.get_health())
    return endpoint


# Example usage
if __name__ == "__main__":
    # Create health check instance
    health = HealthCheck("Test Service", 5000)

    # Add dependency checks
    def check_redis():
        # Simulate Redis check
        return True

    def check_database():
        # Simulate database check
        return False  # Unhealthy for demo

    health.add_dependency_check("redis", check_redis)
    health.add_dependency_check("database", check_database)

    # Add custom metrics
    health.add_metric("total_requests", 12345)
    health.add_metric("active_connections", 42)

    # Get health status
    import json
    print(json.dumps(health.get_health(), indent=2))

    print("\n✅ Health check test completed")
