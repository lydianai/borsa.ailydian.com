# 🔧 Shared Utilities Library

Common utilities for all AiLydian Trading Scanner Python microservices.

## 📚 Modules

### 1. `config.py` - Configuration Management
Centralized configuration loader with environment variables and default values.

```python
from shared.config import config

# Access configuration
service_port = config.SERVICE_PORT
binance_url = config.BINANCE_API_URL

# White-hat rules (always enforced)
max_leverage = config.MAX_LEVERAGE  # 3x
min_confidence = config.MIN_CONFIDENCE  # 65%
```

### 2. `logger.py` - Centralized Logging
Colored console output with file rotation and JSON structured logging.

```python
from shared.logger import get_logger, PerformanceLogger

# Create logger
logger = get_logger("my-service", level="INFO")

# Log messages
logger.info("✅ Service started")
logger.error("❌ Connection failed")

# Performance tracking
with PerformanceLogger(logger, "API call"):
    # Your code here
    pass
```

### 3. `health_check.py` - Standardized Health Checks
Unified health check endpoint with dependency checking.

```python
from flask import Flask, jsonify
from shared.health_check import HealthCheck

app = Flask(__name__)
health = HealthCheck("My Service", 5000)

# Add dependency checks
def check_redis():
    return True  # Your check logic

health.add_dependency_check("redis", check_redis)

# Health endpoint
@app.route('/health')
def health_endpoint():
    return jsonify(health.get_health())
```

### 4. `redis_cache.py` - Redis Cache Helper
Redis caching with graceful fallback (works without Redis).

```python
from shared.redis_cache import RedisCache

cache = RedisCache(host='localhost', port=6379)

# Cache data
cache.set("price", "BTCUSDT", {"price": 100}, ttl=60)

# Get cached data
data = cache.get("price", "BTCUSDT")

# Delete cache
cache.delete("price", "BTCUSDT")
cache.delete_pattern("price:*")
```

### 5. `metrics.py` - Prometheus Metrics
Standardized metrics collection for monitoring.

```python
from shared.metrics import MetricsCollector, track_time

metrics = MetricsCollector("my-service", enabled=True)

# Track function execution time
@track_time(metrics, "/analyze", "POST")
def analyze_data():
    # Your code here
    pass

# Manual metrics
metrics.record_request("/api/test", "GET", 200, 0.123)
metrics.record_error("ValueError")
metrics.set_active_connections(42)
```

### 6. `binance_client.py` - Unified Binance API Client
Centralized Binance API client with rate limiting and caching.

```python
from shared.binance_client import BinanceClient
from shared.redis_cache import RedisCache

cache = RedisCache()
client = BinanceClient(cache=cache)

# Get spot price
price = client.get_ticker_price("BTCUSDT")

# Get 24h ticker
ticker = client.get_ticker_24h("BTCUSDT")

# Get klines
klines = client.get_klines("BTCUSDT", "1h", limit=100)

# Futures data
funding = client.get_funding_rate("BTCUSDT")
oi = client.get_open_interest("BTCUSDT")
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd Phyton-Service/shared
pip install -r requirements.txt
```

### 2. Create a New Service

```python
from flask import Flask, jsonify
from shared.config import config
from shared.logger import get_logger
from shared.health_check import HealthCheck
from shared.redis_cache import RedisCache
from shared.metrics import MetricsCollector
from shared.binance_client import BinanceClient

# Initialize
app = Flask(__name__)
logger = get_logger("my-service", level=config.LOG_LEVEL)
health = HealthCheck("My Service", config.SERVICE_PORT)
cache = RedisCache(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    enabled=config.REDIS_ENABLED
)
metrics = MetricsCollector("my-service", enabled=config.PROMETHEUS_ENABLED)
binance = BinanceClient(cache=cache)

# Health endpoint
@app.route('/health')
def health_endpoint():
    return jsonify(health.get_health())

# Metrics endpoint
@app.route('/metrics')
def metrics_endpoint():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Your service logic
@app.route('/analyze/<symbol>')
def analyze(symbol):
    logger.info(f"Analyzing {symbol}")
    price = binance.get_ticker_price(symbol)
    return jsonify({"symbol": symbol, "price": price})

if __name__ == '__main__':
    logger.info(f"🚀 Starting {config.SERVICE_NAME} on port {config.SERVICE_PORT}")
    app.run(host=config.SERVICE_HOST, port=config.SERVICE_PORT)
```

---

## 📦 Dependencies

See `requirements.txt` for all required packages:
- `flask` - Web framework
- `flask-cors` - CORS support
- `redis` - Redis client
- `prometheus-client` - Metrics collection
- `requests` - HTTP client
- `python-dotenv` - Environment variables

---

## ⚙️ Environment Variables

Create a `.env` file in your service directory:

```env
# Service Configuration
SERVICE_NAME="My Service"
PORT=5000
HOST=0.0.0.0
FLASK_ENV=production

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ENABLED=true

# Binance
BINANCE_API_URL=https://api.binance.com

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
```

---

## 🧪 Testing

Test individual modules:

```bash
# Test config
python3 shared/config.py

# Test logger
python3 shared/logger.py

# Test health check
python3 shared/health_check.py

# Test Redis cache
python3 shared/redis_cache.py

# Test metrics
python3 shared/metrics.py

# Test Binance client
python3 shared/binance_client.py
```

---

## 🎯 Best Practices

1. **Always use the shared utilities** instead of duplicating code
2. **Enable metrics** for all production services
3. **Add health checks** for critical dependencies
4. **Use Redis caching** for expensive API calls
5. **Log important events** with appropriate levels
6. **Follow white-hat rules** (enforced in config)

---

## ✅ White-Hat Compliance

All utilities enforce white-hat trading rules:
- ✅ Maximum 3x leverage
- ✅ Minimum 65% confidence for signals
- ✅ Stop-loss required
- ✅ Transparent code, no obfuscation
- ✅ Educational purpose only

---

## 📝 License

Internal use only. AiLydian Trading Scanner project.

**Version:** 1.0.0
**Last Updated:** 2025-11-01
