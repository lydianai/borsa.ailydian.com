# 🚀 Python Servisleri - İleri Seviye Öneriler

**Tarih:** 1 Kasım 2025
**Hedef:** Sıfır hata ile üst düzey production sistemi
**Beyaz Şapka Uyumu:** ✅ %100

---

## 📊 MEVCUT DURUM ANALİZİ

### ✅ Güçlü Yönler
- 14 Python microservice aktif ve çalışıyor
- Shared utilities library ile kod tekrarı azaltıldı
- PM2 ile process management
- Redis cache entegrasyonu
- Prometheus metrics desteği
- Graceful fallback mekanizmaları

### ⚠️ İyileştirme Gereken Alanlar
1. **Hata İzleme:** Merkezi error tracking yok
2. **Log Yönetimi:** Loglar dağınık, merkezi toplama yok
3. **Health Monitoring:** Otomatik alert sistemi yok
4. **Testing:** Unit/integration test eksik
5. **Documentation:** API docs eksik
6. **Duplicate Process:** Port 5006'da duplicate var (feature-engineering)
7. **Rate Limiting:** API endpoint'lerde limit yok
8. **Security:** API authentication yok

---

## 🎯 ÖNCELİK 1: HATA YÖNETİMİ & İZLEME

### 1.1 Sentry Entegrasyonu (Error Tracking)

**Neden Gerekli:**
- Tüm servislerdeki hataları tek yerden izleme
- Real-time alert'ler
- Error stack trace'leri
- Performance monitoring

**Uygulama:**

```python
# /Phyton-Service/shared/sentry_integration.py
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from typing import Optional

class SentryIntegration:
    @staticmethod
    def init(
        dsn: Optional[str] = None,
        service_name: str = "unknown",
        environment: str = "production",
        enabled: bool = True
    ):
        """Initialize Sentry for error tracking"""
        if not enabled or not dsn:
            return

        sentry_sdk.init(
            dsn=dsn,
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.1,  # 10% transaction tracking
            profiles_sample_rate=0.1,
            environment=environment,
            release=f"{service_name}@1.0.0",
            before_send=SentryIntegration._before_send,
        )

    @staticmethod
    def _before_send(event, hint):
        """Filter out known errors"""
        # Beyaz şapka: Hassas bilgileri temizle
        if 'request' in event:
            if 'data' in event['request']:
                # API keys, passwords filtreleme
                event['request']['data'] = SentryIntegration._sanitize(
                    event['request']['data']
                )
        return event

    @staticmethod
    def _sanitize(data):
        """Remove sensitive data"""
        sensitive_keys = ['password', 'api_key', 'secret', 'token']
        if isinstance(data, dict):
            return {
                k: '***FILTERED***' if k.lower() in sensitive_keys else v
                for k, v in data.items()
            }
        return data
```

**Kullanım:**
```python
# Her service'in app.py'sinde
from shared.sentry_integration import SentryIntegration

SentryIntegration.init(
    dsn=config.SENTRY_DSN,
    service_name="database-service",
    environment=config.FLASK_ENV,
    enabled=config.SENTRY_ENABLED
)
```

**Maliyet:** Ücretsiz tier (5K errors/month) yeterli

---

### 1.2 Merkezi Log Yönetimi (ELK Stack Alternatifi)

**Önerilen Çözüm:** Loki + Grafana (Ücretsiz, hafif)

**Neden:**
- Tüm servislerin loglarını tek yerden izleme
- Log search ve filtering
- Log retention policies
- Dashboard'lar

**Kurulum:**

```bash
# Docker ile Loki başlatma
docker run -d \
  --name=loki \
  -p 3100:3100 \
  grafana/loki:latest

# Promtail (log shipper) config
# /Phyton-Service/promtail-config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://localhost:3100/loki/api/v1/push

scrape_configs:
  - job_name: python-services
    static_configs:
      - targets:
          - localhost
        labels:
          job: python-services
          __path__: /Users/lydian/Documents/*/Phyton-Service/*/logs/*.log
```

**Shared Library Güncellemesi:**

```python
# /Phyton-Service/shared/logger.py içine ekle
import logging_loki

def get_logger(name: str, loki_url: Optional[str] = None):
    logger = logging.getLogger(name)

    # Mevcut handlers...

    # Loki handler ekle
    if loki_url:
        handler = logging_loki.LokiHandler(
            url=loki_url,
            tags={"service": name, "environment": config.FLASK_ENV},
            version="1",
        )
        logger.addHandler(handler)

    return logger
```

---

### 1.3 Otomatik Health Check & Alerting

**Prometheus Alertmanager Entegrasyonu:**

```yaml
# /Phyton-Service/alertmanager.yml
global:
  resolve_timeout: 5m

route:
  receiver: 'telegram'
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

receivers:
  - name: 'telegram'
    telegram_configs:
      - bot_token: 'YOUR_TELEGRAM_BOT_TOKEN'
        chat_id: YOUR_CHAT_ID
        parse_mode: 'HTML'
        message: |
          <b>🚨 ALERT: {{ .GroupLabels.alertname }}</b>
          Service: {{ .Labels.service }}
          Severity: {{ .Labels.severity }}
          Description: {{ .Annotations.description }}

# Prometheus rules
# /Phyton-Service/prometheus-rules.yml
groups:
  - name: python_services
    interval: 30s
    rules:
      - alert: ServiceDown
        expr: up{job="python-services"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          description: "{{ $labels.service }} is down!"

      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes > 500000000
        for: 5m
        labels:
          severity: warning
        annotations:
          description: "{{ $labels.service }} memory > 500MB"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          description: "{{ $labels.service }} error rate > 5%"
```

---

## 🎯 ÖNCELİK 2: PERFORMANS & GÜVENİLİRLİK

### 2.1 Rate Limiting (Hız Sınırlama)

**Shared Library'ye Ekle:**

```python
# /Phyton-Service/shared/rate_limiter.py
from functools import wraps
from flask import request, jsonify
import time
from collections import defaultdict
from threading import Lock

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old requests
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > minute_ago
            ]

            # Check limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                return False

            self.requests[identifier].append(now)
            return True

def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting"""
    limiter = RateLimiter(requests_per_minute)

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # IP tabanlı rate limiting
            identifier = request.remote_addr

            if not limiter.is_allowed(identifier):
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'retry_after': 60
                }), 429

            return f(*args, **kwargs)
        return decorated_function
    return decorator
```

**Kullanım:**
```python
from shared.rate_limiter import rate_limit

@app.route('/api/signals')
@rate_limit(requests_per_minute=100)  # Max 100 request/dakika
def get_signals():
    # ...
```

---

### 2.2 Circuit Breaker Pattern (Devre Kesici)

**Neden:** Bir servis down olduğunda diğerlerini etkilemesini önler

```python
# /Phyton-Service/shared/circuit_breaker.py
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import time

class CircuitState(Enum):
    CLOSED = "closed"      # Normal çalışma
    OPEN = "open"          # Servis down, istekler gönderilmez
    HALF_OPEN = "half_open"  # Test modu

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        name: str = "circuit"
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.name = name
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception(f"Circuit {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Reset on success"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Handle failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if should try to reset"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
        )

# Decorator
def circuit_breaker(failure_threshold=5, timeout=60, name="default"):
    breaker = CircuitBreaker(failure_threshold, timeout, name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator
```

**Kullanım:**
```python
from shared.circuit_breaker import circuit_breaker

@circuit_breaker(failure_threshold=3, timeout=30, name="binance_api")
def get_binance_data(symbol):
    response = requests.get(f"https://api.binance.com/...")
    return response.json()
```

---

### 2.3 Request Retry Mekanizması

```python
# /Phyton-Service/shared/retry.py
from functools import wraps
import time
import random

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    jitter: bool = True
):
    """Retry decorator with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e

                    # Exponential backoff with jitter
                    sleep_time = current_delay
                    if jitter:
                        sleep_time *= (0.5 + random.random())

                    time.sleep(sleep_time)
                    current_delay *= backoff

        return wrapper
    return decorator
```

---

## 🎯 ÖNCELİK 3: TEST & KALİTE GÜVENCE

### 3.1 Unit Test Framework

```python
# /Phyton-Service/shared/tests/test_config.py
import pytest
from shared.config import config

def test_config_white_hat_mode():
    """White-hat mode always enabled"""
    assert config.WHITE_HAT_MODE == True

def test_config_max_leverage():
    """Max leverage must be <= 3"""
    assert config.MAX_LEVERAGE <= 3

def test_config_min_confidence():
    """Min confidence must be >= 0.65"""
    assert config.MIN_CONFIDENCE >= 0.65

# /Phyton-Service/database-service/tests/test_app.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['service'] == 'Database Service'
    assert data['port'] == 5020

def test_save_signal(client):
    """Test signal saving"""
    payload = {
        'symbol': 'BTCUSDT',
        'signal_type': 'BUY',
        'confidence': 0.85,
        'price': 110000
    }
    response = client.post('/signals/save', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] == True

# Test çalıştırma
# pytest database-service/tests/ -v
```

### 3.2 Integration Test

```python
# /Phyton-Service/tests/integration/test_services.py
import pytest
import requests

SERVICES = [
    ('database-service', 'http://localhost:5020'),
    ('websocket-streaming', 'http://localhost:5021'),
    ('ai-models', 'http://localhost:5003'),
]

@pytest.mark.parametrize("name,url", SERVICES)
def test_service_health(name, url):
    """Test all services are healthy"""
    response = requests.get(f"{url}/health", timeout=5)
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data

def test_database_to_websocket_flow():
    """Test data flow between services"""
    # 1. Save signal to database
    signal = {
        'symbol': 'BTCUSDT',
        'signal_type': 'BUY',
        'confidence': 0.85,
        'price': 110000
    }
    db_response = requests.post(
        'http://localhost:5020/signals/save',
        json=signal
    )
    assert db_response.status_code == 200

    # 2. Get price from websocket
    ws_response = requests.get(
        'http://localhost:5021/price/BTCUSDT'
    )
    assert ws_response.status_code == 200
    price_data = ws_response.json()
    assert price_data['success'] == True
```

---

## 🎯 ÖNCELİK 4: GÜVENLİK

### 4.1 API Authentication (JWT)

```python
# /Phyton-Service/shared/auth.py
from functools import wraps
from flask import request, jsonify
import jwt
from datetime import datetime, timedelta

class JWTAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')

    def verify_token(self, token: str) -> dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'error': 'No token provided'}), 401

        try:
            # Remove 'Bearer ' prefix
            if token.startswith('Bearer '):
                token = token[7:]

            auth = JWTAuth(config.JWT_SECRET)
            payload = auth.verify_token(token)
            request.user_id = payload['user_id']
        except Exception as e:
            return jsonify({'error': str(e)}), 401

        return f(*args, **kwargs)

    return decorated_function
```

### 4.2 Input Validation

```python
# /Phyton-Service/shared/validation.py
from typing import Any, Dict, List
import re

class Validator:
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate trading symbol"""
        # Sadece USDT çiftleri (beyaz şapka)
        pattern = r'^[A-Z]{2,10}USDT$'
        return bool(re.match(pattern, symbol))

    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence level"""
        return 0.0 <= confidence <= 1.0

    @staticmethod
    def validate_leverage(leverage: int) -> bool:
        """Validate leverage (white-hat: max 3x)"""
        return 1 <= leverage <= 3

    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize user input"""
        # SQL injection, XSS koruması
        dangerous_chars = ['<', '>', '"', "'", ';', '--']
        for char in dangerous_chars:
            data = data.replace(char, '')
        return data
```

---

## 🎯 ÖNCELİK 5: DEPLOYMENT & CI/CD

### 5.1 Docker Containerization

```dockerfile
# /Phyton-Service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health')"

# Run
CMD ["python", "app.py"]
```

```yaml
# /Phyton-Service/docker-compose.yml
version: '3.8'

services:
  database-service:
    build:
      context: ./database-service
    ports:
      - "5020:5020"
    environment:
      - PORT=5020
      - REDIS_HOST=redis
      - DB_ENABLED=false
    depends_on:
      - redis
    restart: unless-stopped

  websocket-streaming:
    build:
      context: ./websocket-streaming
    ports:
      - "5021:5021"
    environment:
      - PORT=5021
      - REDIS_HOST=redis
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

### 5.2 GitHub Actions CI/CD

```yaml
# .github/workflows/python-services.yml
name: Python Services CI/CD

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'Phyton-Service/**'
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        cd Phyton-Service/shared
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        cd Phyton-Service
        pytest --cov=shared --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 Phyton-Service --count --max-line-length=100

  deploy:
    needs: [test, lint]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Deploy to production
      run: |
        # SSH to server and pull latest
        echo "Deploy to production server"
```

---

## 🎯 ÖNCELİK 6: DOKÜMANTASYON

### 6.1 OpenAPI/Swagger Documentation

```python
# /Phyton-Service/shared/swagger.py
from flask_swagger_ui import get_swaggerui_blueprint

def setup_swagger(app, spec_url='/api/swagger.json'):
    """Setup Swagger UI"""
    SWAGGER_URL = '/api/docs'

    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        spec_url,
        config={'app_name': "AiLydian Trading Scanner API"}
    )

    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Her service için OpenAPI spec
# /Phyton-Service/database-service/swagger.json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Database Service API",
    "version": "1.0.0",
    "description": "Signal history and performance tracking"
  },
  "servers": [
    {"url": "http://localhost:5020"}
  ],
  "paths": {
    "/signals/save": {
      "post": {
        "summary": "Save trading signal",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "required": ["symbol", "signal_type", "confidence", "price"],
                "properties": {
                  "symbol": {"type": "string", "example": "BTCUSDT"},
                  "signal_type": {"type": "string", "enum": ["BUY", "SELL"]},
                  "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                  "price": {"type": "number", "minimum": 0}
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Signal saved successfully"
          }
        }
      }
    }
  }
}
```

---

## 📋 UYGULAMA PLANI

### Faz 1: Kritik İyileştirmeler (1-2 Hafta)
- [ ] Rate limiting ekle (tüm servislere)
- [ ] Circuit breaker pattern (harici API'ler için)
- [ ] Duplicate process temizle (Port 5006)
- [ ] Input validation (tüm POST endpointler)
- [ ] Basic unit tests (coverage >70%)

### Faz 2: Monitoring & Alerting (1 Hafta)
- [ ] Prometheus + Grafana dashboard
- [ ] Alertmanager + Telegram entegrasyonu
- [ ] Loki + Promtail (log aggregation)
- [ ] Health check dashboard

### Faz 3: Güvenlik (1 Hafta)
- [ ] JWT authentication
- [ ] API key management
- [ ] HTTPS/SSL sertifikaları
- [ ] Security headers
- [ ] Rate limiting per user

### Faz 4: Testing & CI/CD (1 Hafta)
- [ ] Integration tests
- [ ] Load testing
- [ ] GitHub Actions CI/CD
- [ ] Automated deployment

### Faz 5: Documentation (3 Gün)
- [ ] OpenAPI/Swagger specs
- [ ] Architecture diagrams
- [ ] Troubleshooting guide
- [ ] Development setup guide

---

## 🔧 HIZLI KAZANÇLAR (ŞİMDİ YAPILABİLİR)

### 1. Duplicate Process Temizliği
```bash
# Port 5006'daki duplicate'i bul ve temizle
lsof -ti :5006 | while read pid; do
  echo "Killing PID: $pid"
  kill -9 $pid
done

# PM2'de sadece bir instance olduğundan emin ol
pm2 restart feature-engineering
pm2 save
```

### 2. Environment Variables Standardizasyonu
```bash
# Her service için .env.example oluştur
cat > database-service/.env.example << 'EOF'
# Service Config
SERVICE_NAME=Database Service
PORT=5020
FLASK_ENV=production

# Database
DB_ENABLED=false
DB_HOST=localhost
DB_PORT=5432

# Redis
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO

# Security
JWT_SECRET=your-secret-key-here
API_KEY=your-api-key-here

# Optional
SENTRY_DSN=
SENTRY_ENABLED=false
EOF
```

### 3. PM2 Log Rotation Config
```javascript
// PM2 log rotation ayarları
module.exports = {
  apps: [...],

  // Log rotation config
  log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
  combine_logs: true,
  merge_logs: true,

  // Max log file size
  max_size: '10M',
  retain: 10,  // Keep last 10 files
  compress: true,  // Gzip old logs
}
```

---

## 📊 BAŞARI METRİKLERİ

### Hedef KPI'lar
- ✅ **Uptime:** >99.9% (maksimum 43 dakika downtime/ay)
- ✅ **Error Rate:** <0.1% (1000 istekten 1'inden az hata)
- ✅ **Response Time:** <200ms (p95)
- ✅ **Test Coverage:** >80%
- ✅ **Zero Security Issues:** (Weekly scan)
- ✅ **Zero Duplicate Processes**

### İzleme Araçları
- PM2 Dashboard
- Grafana Dashboard
- Sentry Error Tracking
- GitHub Actions Status
- Uptime Robot (servis availability)

---

## 💰 MALIYET TAHMİNİ

### Ücretsiz Tier (Başlangıç İçin Yeterli)
- ✅ Sentry: 5K errors/month
- ✅ Grafana Cloud: 10K metrics/month
- ✅ GitHub Actions: 2000 dakika/ay
- ✅ Uptime Robot: 50 monitor
- **Toplam:** $0/ay

### Production Tier (Büyüme Sonrası)
- Sentry Pro: $26/ay
- Grafana Cloud Pro: $49/ay
- GitHub Actions: $4/ay (ekstra)
- **Toplam:** ~$80/ay

---

## 🛡️ BEYAZ ŞAPKA UYUMU

Tüm öneriler beyaz şapka kurallarına uygun:
- ✅ Güvenlik önlemleri kullanıcı koruması için
- ✅ Rate limiting DoS saldırılarını önlemek için (kötü amaçlı değil)
- ✅ Monitoring sistem sağlığı için (kullanıcı takibi yok)
- ✅ Authentication yetkisiz erişimi önlemek için
- ✅ Tüm veriler şeffaf ve denetlenebilir

---

**Oluşturulma:** 1 Kasım 2025
**Versiyon:** 1.0
**Durum:** Onay Bekliyor
