"""
🧪 DATABASE SERVICE TESTS
=========================
Unit and integration tests for Database Service

Run with: pytest -v
Run specific: pytest -v -k test_save_signal
Run with coverage: pytest --cov=. --cov-report=html
"""

import pytest
import json
import sys
import os
from datetime import datetime

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


@pytest.fixture
def sample_signal():
    """Sample signal data for testing"""
    return {
        'symbol': 'BTCUSDT',
        'signal_type': 'BUY',
        'confidence': 0.85,
        'price': 95000.00,
        'metadata': {
            'strategy': 'test',
            'timeframe': '1h'
        }
    }


@pytest.fixture
def sample_performance():
    """Sample performance data for testing"""
    return {
        'strategy': 'test-strategy',
        'pnl': 150.50,
        'win_rate': 0.75,
        'trades': 10
    }


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


@pytest.mark.unit
def test_stats_endpoint(client):
    """Test stats endpoint"""
    response = client.get('/stats')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data
    assert 'service' in data['data']
    assert data['data']['service'] == 'Database Service'
    assert 'white_hat_mode' in data['data']


# ============================================
# SIGNAL HISTORY TESTS
# ============================================

@pytest.mark.unit
@pytest.mark.api
def test_save_signal_success(client, sample_signal):
    """Test saving a signal successfully"""
    response = client.post(
        '/signals/save',
        data=json.dumps(sample_signal),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'message' in data
    assert 'signal_id' in data


@pytest.mark.unit
@pytest.mark.api
def test_save_signal_missing_fields(client):
    """Test saving signal with missing required fields"""
    incomplete_signal = {'symbol': 'BTCUSDT'}

    response = client.post(
        '/signals/save',
        data=json.dumps(incomplete_signal),
        content_type='application/json'
    )

    assert response.status_code == 400
    data = json.loads(response.data)

    assert data['success'] == False
    assert 'error' in data


@pytest.mark.unit
@pytest.mark.api
def test_get_signal_history(client, sample_signal):
    """Test retrieving signal history"""
    # First save a signal
    client.post(
        '/signals/save',
        data=json.dumps(sample_signal),
        content_type='application/json'
    )

    # Then retrieve it
    response = client.get('/signals/history?limit=10')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data
    assert 'signals' in data['data']
    assert 'count' in data['data']


@pytest.mark.unit
@pytest.mark.api
def test_get_signal_history_by_symbol(client, sample_signal):
    """Test retrieving signal history filtered by symbol"""
    # Save a signal
    client.post(
        '/signals/save',
        data=json.dumps(sample_signal),
        content_type='application/json'
    )

    # Retrieve by symbol
    response = client.get('/signals/history?symbol=BTCUSDT&limit=10')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    # If we have results, they should match the symbol
    if data['data']['count'] > 0:
        assert all(s['symbol'] == 'BTCUSDT' for s in data['data']['signals'])


# ============================================
# PERFORMANCE TRACKING TESTS
# ============================================

@pytest.mark.unit
@pytest.mark.api
def test_track_performance_success(client, sample_performance):
    """Test tracking performance successfully"""
    response = client.post(
        '/performance/track',
        data=json.dumps(sample_performance),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'message' in data


@pytest.mark.unit
@pytest.mark.api
def test_track_performance_missing_fields(client):
    """Test tracking performance with missing required fields"""
    incomplete_perf = {'strategy': 'test'}

    response = client.post(
        '/performance/track',
        data=json.dumps(incomplete_perf),
        content_type='application/json'
    )

    assert response.status_code == 400
    data = json.loads(response.data)

    assert data['success'] == False
    assert 'error' in data


@pytest.mark.unit
@pytest.mark.api
def test_get_performance_stats(client, sample_performance):
    """Test retrieving performance statistics"""
    # First track some performance
    client.post(
        '/performance/track',
        data=json.dumps(sample_performance),
        content_type='application/json'
    )

    # Then retrieve stats
    response = client.get('/performance/stats')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True
    assert 'data' in data
    assert 'total_trades' in data['data']
    assert 'total_pnl' in data['data']
    assert 'avg_win_rate' in data['data']


@pytest.mark.unit
@pytest.mark.api
def test_get_performance_stats_by_strategy(client, sample_performance):
    """Test retrieving performance stats filtered by strategy"""
    # Track performance
    client.post(
        '/performance/track',
        data=json.dumps(sample_performance),
        content_type='application/json'
    )

    # Retrieve by strategy
    response = client.get('/performance/stats?strategy=test-strategy')

    assert response.status_code == 200
    data = json.loads(response.data)

    assert data['success'] == True


# ============================================
# RATE LIMITING TESTS
# ============================================

@pytest.mark.integration
@pytest.mark.slow
def test_rate_limiting(client, sample_signal):
    """Test rate limiting on save endpoint"""
    # Make 105 requests (limit is 100/min)
    successful = 0
    rate_limited = 0

    for i in range(105):
        response = client.post(
            '/signals/save',
            data=json.dumps(sample_signal),
            content_type='application/json'
        )

        if response.status_code == 200:
            successful += 1
        elif response.status_code == 429:
            rate_limited += 1

    # Should have some successful and some rate limited
    assert successful > 0
    assert rate_limited > 0
    assert successful <= 100  # Should not exceed rate limit


# ============================================
# EDGE CASES
# ============================================

@pytest.mark.unit
def test_save_signal_with_timestamp(client):
    """Test saving signal with explicit timestamp"""
    signal = {
        'symbol': 'ETHUSDT',
        'signal_type': 'SELL',
        'confidence': 0.90,
        'price': 3500.00,
        'timestamp': datetime.now().isoformat()
    }

    response = client.post(
        '/signals/save',
        data=json.dumps(signal),
        content_type='application/json'
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True


@pytest.mark.unit
def test_get_signal_history_empty(client):
    """Test getting history when no signals exist"""
    # This may fail if there's existing data, but tests the endpoint
    response = client.get('/signals/history?symbol=NONEXISTENT&limit=10')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True
    assert 'signals' in data['data']


@pytest.mark.unit
def test_performance_stats_empty(client):
    """Test getting stats with no performance data"""
    response = client.get('/performance/stats?strategy=nonexistent')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] == True


# ============================================
# RUN TESTS
# ============================================

if __name__ == '__main__':
    # Run with: python test_database_service.py
    pytest.main([__file__, '-v'])
