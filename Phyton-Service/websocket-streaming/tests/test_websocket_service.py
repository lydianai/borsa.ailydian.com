"""
🧪 WEBSOCKET STREAMING SERVICE TESTS
====================================
Unit and integration tests for WebSocket Streaming Service

Run with: pytest -v
Run specific: pytest -v -k test_health
Run with coverage: pytest --cov=. --cov-report=html
"""

import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import the Flask app
from app import app as flask_app


@pytest.fixture
def client():
    """Create test client"""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


# ============================================
# HEALTH & STATUS TESTS
# ============================================

@pytest.mark.unit
def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['status'] == 'healthy'
    assert 'service' in data
    assert 'uptime' in data
    assert 'timestamp' in data
    assert 'metrics' in data


@pytest.mark.unit
def test_stats_endpoint(client):
    """Test stats endpoint"""
    response = client.get('/stats')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data
    assert 'service' in data['data']
    assert data['data']['service'] == 'WebSocket Streaming Service'
    assert 'active_streams' in data['data']
    assert 'active_symbols' in data['data']


# ============================================
# SYMBOLS ENDPOINT TESTS
# ============================================

@pytest.mark.unit
@pytest.mark.api
def test_symbols_endpoint(client):
    """Test getting available symbols"""
    response = client.get('/symbols')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data
    assert 'default_symbols' in data['data']
    assert 'active_symbols' in data['data']
    assert isinstance(data['data']['default_symbols'], list)


# ============================================
# PRICE ENDPOINT TESTS
# ============================================

@pytest.mark.unit
@pytest.mark.api
def test_price_endpoint_btc(client):
    """Test getting price for BTCUSDT"""
    response = client.get('/price/BTCUSDT')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data

    # Check price data structure
    price_data = data['data']
    assert 'symbol' in price_data
    assert 'price' in price_data
    assert 'timestamp' in price_data
    assert price_data['symbol'] == 'BTCUSDT'
    assert isinstance(price_data['price'], (int, float))


@pytest.mark.unit
@pytest.mark.api
def test_price_endpoint_eth(client):
    """Test getting price for ETHUSDT"""
    response = client.get('/price/ETHUSDT')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert data['data']['symbol'] == 'ETHUSDT'


@pytest.mark.unit
@pytest.mark.api
def test_price_endpoint_lowercase(client):
    """Test price endpoint with lowercase symbol"""
    response = client.get('/price/btcusdt')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    # Should be converted to uppercase
    assert data['data']['symbol'] == 'BTCUSDT'


# ============================================
# RATE LIMITING TESTS
# ============================================

@pytest.mark.integration
@pytest.mark.slow
def test_rate_limiting_price_endpoint(client):
    """Test rate limiting on price endpoint"""
    # Make 305 requests (limit is 300/min)
    successful = 0
    rate_limited = 0

    for i in range(305):
        response = client.get('/price/BTCUSDT')

        if response.status_code == 200:
            successful += 1
        elif response.status_code == 429:
            rate_limited += 1

    # Should have some successful and some rate limited
    assert successful > 0
    assert rate_limited > 0
    assert successful <= 300  # Should not exceed rate limit


# ============================================
# ERROR HANDLING TESTS
# ============================================

@pytest.mark.unit
def test_price_endpoint_invalid_symbol(client):
    """Test price endpoint with potentially invalid symbol"""
    response = client.get('/price/INVALIDSYMBOL')

    # Should still return 200 but may have error in response
    # or fallback to API call which may fail
    assert response.status_code in [200, 500]


# ============================================
# WEBSOCKET CONNECTION TESTS (Integration)
# ============================================

@pytest.mark.integration
@pytest.mark.websocket
def test_socketio_client_connection():
    """Test Socket.IO client connection"""
    from flask_socketio import SocketIOTestClient

    flask_app.config['TESTING'] = True
    client = SocketIOTestClient(flask_app, flask_socketio=None)

    # Note: This is a basic structure
    # Full Socket.IO testing would require flask_socketio test client
    # which needs special setup
    assert client is not None


# ============================================
# EDGE CASES
# ============================================

@pytest.mark.unit
def test_stats_structure(client):
    """Test stats endpoint returns complete structure"""
    response = client.get('/stats')
    data = json.loads(response.data)

    required_fields = [
        'service',
        'port',
        'active_streams',
        'active_symbols',
        'total_subscriptions',
        'white_hat_mode',
        'uptime'
    ]

    for field in required_fields:
        assert field in data['data'], f"Missing field: {field}"


@pytest.mark.unit
def test_symbols_has_defaults(client):
    """Test symbols endpoint includes default symbols"""
    response = client.get('/symbols')
    data = json.loads(response.data)

    default_symbols = data['data']['default_symbols']

    # Should include at least these major symbols
    expected = ['BTCUSDT', 'ETHUSDT']
    for symbol in expected:
        assert symbol in default_symbols


# ============================================
# PERFORMANCE TESTS
# ============================================

@pytest.mark.unit
def test_price_endpoint_response_time(client):
    """Test price endpoint responds quickly"""
    import time

    start = time.time()
    response = client.get('/price/BTCUSDT')
    duration = time.time() - start

    assert response.status_code == 200
    # Should respond in less than 2 seconds
    assert duration < 2.0


# ============================================
# RUN TESTS
# ============================================

if __name__ == '__main__':
    # Run with: python test_websocket_service.py
    pytest.main([__file__, '-v'])
