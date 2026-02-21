"""
⚙️ CONFIGURATION MANAGEMENT
===========================
Centralized configuration loader for all services

Features:
- Environment variable loading
- Default values
- Type conversion
- Validation
"""

import os
from typing import Optional, Any

# Try to load dotenv if available (optional dependency)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, using environment variables only
    pass


class Config:
    """Configuration manager for Python services"""

    # Service Configuration
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "Unknown Service")
    SERVICE_PORT: int = int(os.getenv("PORT", "5000"))
    SERVICE_HOST: str = os.getenv("HOST", "0.0.0.0")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "production")

    # External APIs
    BINANCE_API_URL: str = os.getenv("BINANCE_API_URL", "https://api.binance.com")
    BINANCE_WS_URL: str = os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443")

    # Data Service URLs
    AI_SERVICE_URL: str = os.getenv("AI_SERVICE_URL", "http://localhost:5003")
    DATA_SERVICE_URL: str = os.getenv("DATA_SERVICE_URL", "http://localhost:3000")
    TALIB_SERVICE_URL: str = os.getenv("TALIB_SERVICE_URL", "http://localhost:5002")

    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    REDIS_ENABLED: bool = os.getenv("REDIS_ENABLED", "true").lower() == "true"

    # Cache TTL (seconds)
    CACHE_TTL_SHORT: int = 10  # For price data
    CACHE_TTL_MEDIUM: int = 30  # For technical indicators
    CACHE_TTL_LONG: int = 60  # For API responses

    # Database Configuration (TimescaleDB)
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "trading_db")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_ENABLED: bool = os.getenv("DB_ENABLED", "false").lower() == "true"

    # Monitoring
    PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # White-Hat Rules
    WHITE_HAT_MODE: bool = True  # Always enforced
    MAX_LEVERAGE: int = 3
    MIN_CONFIDENCE: float = 0.65
    MAX_POSITION_SIZE: float = 0.1
    STOP_LOSS_REQUIRED: bool = True
    RISK_REWARD_MIN: float = 1.5
    MAX_DAILY_TRADES: int = 50
    REQUIRE_MULTIPLE_SIGNALS: bool = True

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(cls, key, default)

    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return cls.FLASK_ENV == "production"

    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.FLASK_ENV == "development"

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        try:
            # Check required fields
            assert cls.SERVICE_PORT > 0, "Invalid port"
            assert cls.WHITE_HAT_MODE is True, "White-hat mode must be enabled"
            return True
        except AssertionError as e:
            print(f"❌ Configuration validation failed: {e}")
            return False


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("✅ Configuration loaded successfully")
    print(f"Service: {config.SERVICE_NAME}")
    print(f"Port: {config.SERVICE_PORT}")
    print(f"Environment: {config.FLASK_ENV}")
    print(f"White-Hat Mode: {config.WHITE_HAT_MODE}")
    print(f"Redis Enabled: {config.REDIS_ENABLED}")
    print(f"DB Enabled: {config.DB_ENABLED}")
    print(f"Validation: {config.validate()}")
