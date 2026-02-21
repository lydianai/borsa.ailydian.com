"""
🔍 SENTRY INTEGRATION
=====================
Error tracking and monitoring with Sentry.io

Features:
- Automatic error capture
- Performance monitoring
- User context tracking
- Release tracking
- Graceful fallback when not configured

WHITE-HAT COMPLIANCE: Error tracking for system health (not malicious monitoring)

Setup:
1. Create free account at https://sentry.io
2. Get your DSN key
3. Add to .env: SENTRY_DSN=your-dsn-here
4. Optional: SENTRY_ENVIRONMENT=production
5. Optional: SENTRY_RELEASE=1.0.0

Usage:
    from shared.sentry_integration import init_sentry, capture_exception

    # Initialize (call once at app startup)
    init_sentry(
        service_name="database-service",
        environment="production",
        release="1.0.0"
    )

    # Capture errors
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e)
        raise
"""

import os
from typing import Optional, Dict, Any
import logging

# Try to import Sentry SDK (graceful fallback if not installed)
try:
    import sentry_sdk
    from sentry_sdk.integrations.flask import FlaskIntegration
    from sentry_sdk.integrations.logging import LoggingIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global state
_sentry_initialized = False
_sentry_enabled = False


def init_sentry(
    service_name: str,
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    enable_flask: bool = True,
) -> bool:
    """
    Initialize Sentry error tracking

    Args:
        service_name: Name of the service (e.g., "database-service")
        dsn: Sentry DSN key (from environment if not provided)
        environment: Environment name (development, staging, production)
        release: Release version (for tracking which version had errors)
        traces_sample_rate: % of requests to track for performance (0.0-1.0)
        profiles_sample_rate: % of requests to profile (0.0-1.0)
        enable_flask: Enable Flask integration

    Returns:
        True if Sentry was initialized, False otherwise
    """
    global _sentry_initialized, _sentry_enabled

    if _sentry_initialized:
        logger.warning("⚠️  Sentry already initialized")
        return _sentry_enabled

    # Check if Sentry SDK is available
    if not SENTRY_AVAILABLE:
        logger.info("📊 Sentry SDK not installed (pip install sentry-sdk)")
        logger.info("💡 To enable error tracking: pip install sentry-sdk")
        _sentry_initialized = True
        _sentry_enabled = False
        return False

    # Get DSN from parameter or environment
    if not dsn:
        dsn = os.getenv('SENTRY_DSN')

    if not dsn:
        logger.info("📊 Sentry DSN not configured (disabled)")
        logger.info("💡 To enable: Set SENTRY_DSN in .env")
        _sentry_initialized = True
        _sentry_enabled = False
        return False

    # Get environment and release from parameters or environment
    if not environment:
        environment = os.getenv('SENTRY_ENVIRONMENT', 'development')

    if not release:
        release = os.getenv('SENTRY_RELEASE', '1.0.0')

    try:
        # Prepare integrations
        integrations = []

        # Flask integration (if enabled)
        if enable_flask:
            integrations.append(FlaskIntegration())

        # Logging integration (capture ERROR and CRITICAL logs)
        integrations.append(
            LoggingIntegration(
                level=logging.INFO,       # Capture info and above as breadcrumbs
                event_level=logging.ERROR # Capture errors and above as events
            )
        )

        # Initialize Sentry
        sentry_sdk.init(
            dsn=dsn,
            environment=environment,
            release=release,
            traces_sample_rate=traces_sample_rate,
            profiles_sample_rate=profiles_sample_rate,
            integrations=integrations,
            # Additional configuration
            send_default_pii=False,  # Don't send personally identifiable information
            attach_stacktrace=True,  # Include stack traces
            debug=False,             # Don't spam logs
        )

        # Set service context
        sentry_sdk.set_tag("service", service_name)
        sentry_sdk.set_context("service", {
            "name": service_name,
            "environment": environment,
            "release": release
        })

        logger.info(f"✅ Sentry initialized for {service_name}")
        logger.info(f"📊 Environment: {environment}, Release: {release}")
        logger.info(f"📈 Traces: {traces_sample_rate*100}%, Profiles: {profiles_sample_rate*100}%")

        _sentry_initialized = True
        _sentry_enabled = True
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize Sentry: {e}")
        _sentry_initialized = True
        _sentry_enabled = False
        return False


def capture_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
    level: str = "error"
) -> Optional[str]:
    """
    Capture an exception to Sentry

    Args:
        exception: The exception to capture
        context: Additional context data
        tags: Tags to add to the event
        level: Severity level (debug, info, warning, error, fatal)

    Returns:
        Event ID if sent to Sentry, None otherwise

    Example:
        try:
            risky_operation()
        except ValueError as e:
            capture_exception(
                e,
                context={"operation": "database_query", "query": "SELECT..."},
                tags={"module": "database"}
            )
    """
    if not _sentry_enabled or not SENTRY_AVAILABLE:
        return None

    try:
        # Add context if provided
        if context:
            for key, value in context.items():
                sentry_sdk.set_context(key, value)

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                sentry_sdk.set_tag(key, value)

        # Capture the exception
        event_id = sentry_sdk.capture_exception(exception, level=level)

        if event_id:
            logger.debug(f"📊 Exception captured to Sentry: {event_id}")

        return event_id

    except Exception as e:
        logger.error(f"❌ Failed to capture exception to Sentry: {e}")
        return None


def capture_message(
    message: str,
    level: str = "info",
    context: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Capture a message to Sentry (for tracking important events)

    Args:
        message: Message to capture
        level: Severity level (debug, info, warning, error, fatal)
        context: Additional context data
        tags: Tags to add to the event

    Returns:
        Event ID if sent to Sentry, None otherwise

    Example:
        capture_message(
            "High-confidence signal generated",
            level="info",
            tags={"signal_type": "BUY", "confidence": "95%"}
        )
    """
    if not _sentry_enabled or not SENTRY_AVAILABLE:
        return None

    try:
        # Add context if provided
        if context:
            for key, value in context.items():
                sentry_sdk.set_context(key, value)

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                sentry_sdk.set_tag(key, value)

        # Capture the message
        event_id = sentry_sdk.capture_message(message, level=level)

        if event_id:
            logger.debug(f"📊 Message captured to Sentry: {event_id}")

        return event_id

    except Exception as e:
        logger.error(f"❌ Failed to capture message to Sentry: {e}")
        return None


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Add a breadcrumb to Sentry (helps understand what led to an error)

    Args:
        message: Breadcrumb message
        category: Category (navigation, http, query, etc.)
        level: Severity level
        data: Additional data

    Example:
        add_breadcrumb(
            "Fetching signal data",
            category="database",
            level="info",
            data={"symbol": "BTCUSDT", "limit": 100}
        )
    """
    if not _sentry_enabled or not SENTRY_AVAILABLE:
        return

    try:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except Exception as e:
        logger.error(f"❌ Failed to add breadcrumb to Sentry: {e}")


def set_user_context(
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    email: Optional[str] = None,
    ip_address: Optional[str] = None
) -> None:
    """
    Set user context for Sentry events

    Args:
        user_id: User ID
        username: Username
        email: Email address
        ip_address: IP address

    Note: Only use if you have user consent and comply with privacy laws
    """
    if not _sentry_enabled or not SENTRY_AVAILABLE:
        return

    try:
        user_data = {}
        if user_id:
            user_data['id'] = user_id
        if username:
            user_data['username'] = username
        if email:
            user_data['email'] = email
        if ip_address:
            user_data['ip_address'] = ip_address

        if user_data:
            sentry_sdk.set_user(user_data)
    except Exception as e:
        logger.error(f"❌ Failed to set user context in Sentry: {e}")


def flush_sentry(timeout: int = 2) -> bool:
    """
    Flush Sentry events (useful before app shutdown)

    Args:
        timeout: Maximum seconds to wait for events to be sent

    Returns:
        True if all events were sent, False otherwise
    """
    if not _sentry_enabled or not SENTRY_AVAILABLE:
        return True

    try:
        logger.info("📊 Flushing Sentry events...")
        result = sentry_sdk.flush(timeout=timeout)
        logger.info("✅ Sentry events flushed")
        return result
    except Exception as e:
        logger.error(f"❌ Failed to flush Sentry: {e}")
        return False


def is_sentry_enabled() -> bool:
    """Check if Sentry is enabled and initialized"""
    return _sentry_enabled


# Example usage
if __name__ == "__main__":
    # Test Sentry integration
    print("Testing Sentry integration...")

    # Initialize (without DSN, should gracefully fail)
    success = init_sentry("test-service")
    print(f"Sentry initialized: {success}")
    print(f"Sentry enabled: {is_sentry_enabled()}")

    # Try to capture an exception (should gracefully skip if not enabled)
    try:
        raise ValueError("Test error")
    except Exception as e:
        event_id = capture_exception(e, context={"test": True})
        print(f"Exception captured: {event_id}")

    # Capture a message
    msg_id = capture_message("Test message", level="info", tags={"test": "true"})
    print(f"Message captured: {msg_id}")

    print("\n✅ To enable Sentry:")
    print("1. pip install sentry-sdk")
    print("2. Create account at https://sentry.io")
    print("3. Add SENTRY_DSN to .env")
    print("4. Restart service")
