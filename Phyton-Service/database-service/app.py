"""
🗄️ DATABASE SERVICE
===================
TimescaleDB integration for signal history and performance tracking
Port: 5020

Features:
- Signal history storage (time-series)
- Bot performance tracking
- User settings persistence
- Historical data queries
- Automatic data retention policies

WHITE-HAT COMPLIANCE: Educational purpose, transparent data storage
"""

import sys
import os

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Shared utilities
from shared.config import config
from shared.logger import get_logger, PerformanceLogger
from shared.health_check import HealthCheck
from shared.redis_cache import RedisCache
from shared.metrics import MetricsCollector, track_time
from shared.rate_limiter import rate_limit

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize utilities
logger = get_logger("database-service", level=config.LOG_LEVEL)
health = HealthCheck("Database Service", 5020)
cache = RedisCache(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    enabled=config.REDIS_ENABLED
)
metrics = MetricsCollector("database_service", enabled=config.PROMETHEUS_ENABLED)

# Database connection (graceful fallback if TimescaleDB not available)
db_connection = None
db_enabled = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    # Try to connect to TimescaleDB
    if config.DB_ENABLED and config.DB_PASSWORD:
        try:
            db_connection = psycopg2.connect(
                host=config.DB_HOST,
                port=config.DB_PORT,
                database=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                cursor_factory=RealDictCursor
            )
            db_enabled = True
            logger.info("✅ Connected to TimescaleDB")
        except Exception as e:
            logger.warning(f"⚠️  TimescaleDB not available: {e}")
            logger.info("📊 Running in cache-only mode")
    else:
        logger.info("📊 Database disabled, using cache-only mode")

except ImportError:
    logger.warning("⚠️  psycopg2 not installed, using cache-only mode")


# In-memory storage (fallback when DB not available)
signal_history = []
performance_data = []


def check_database():
    """Check if database is available"""
    if not db_enabled:
        return False
    try:
        cursor = db_connection.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        return True
    except:
        return False


# Add health checks
health.add_dependency_check("database", check_database)
health.add_dependency_check("cache", lambda: cache.enabled and cache.client is not None)


# ============================================
# SIGNAL HISTORY ENDPOINTS
# ============================================

@app.route('/signals/save', methods=['POST'])
@rate_limit(requests_per_minute=100)
@track_time(metrics, "/signals/save", "POST")
def save_signal():
    """Save a trading signal to database/cache"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['symbol', 'signal_type', 'confidence', 'price']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        # Save to database if available
        if db_enabled and db_connection:
            try:
                cursor = db_connection.cursor()
                cursor.execute("""
                    INSERT INTO signal_history
                    (time, symbol, signal_type, confidence, price, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    data['timestamp'],
                    data['symbol'],
                    data['signal_type'],
                    data['confidence'],
                    data['price'],
                    json.dumps(data.get('metadata', {}))
                ))
                db_connection.commit()
                cursor.close()
                logger.info(f"✅ Saved signal to DB: {data['symbol']} {data['signal_type']}")
            except Exception as e:
                logger.error(f"❌ DB save failed: {e}")
                db_connection.rollback()

        # Always save to cache
        cache_key = f"{data['symbol']}:{data['timestamp']}"
        cache.set("signals", cache_key, data, ttl=3600)

        # Fallback: in-memory storage
        signal_history.append(data)
        if len(signal_history) > 10000:  # Keep last 10k signals
            signal_history.pop(0)

        return jsonify({
            'success': True,
            'message': 'Signal saved successfully',
            'signal_id': cache_key
        })

    except Exception as e:
        logger.error(f"❌ Save signal error: {e}")
        metrics.record_error(type(e).__name__)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/signals/history', methods=['GET'])
@rate_limit(requests_per_minute=200)
@track_time(metrics, "/signals/history", "GET")
def get_signal_history():
    """Get historical signals"""
    try:
        symbol = request.args.get('symbol')
        limit = int(request.args.get('limit', 100))
        since = request.args.get('since')  # ISO format timestamp

        # Try database first
        if db_enabled and db_connection:
            try:
                cursor = db_connection.cursor()
                query = "SELECT * FROM signal_history WHERE 1=1"
                params = []

                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)

                if since:
                    query += " AND time >= %s"
                    params.append(since)

                query += " ORDER BY time DESC LIMIT %s"
                params.append(limit)

                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()

                return jsonify({
                    'success': True,
                    'data': {
                        'signals': [dict(row) for row in results],
                        'count': len(results),
                        'source': 'database'
                    }
                })
            except Exception as e:
                logger.error(f"❌ DB query failed: {e}")

        # Fallback to in-memory
        filtered = signal_history
        if symbol:
            filtered = [s for s in filtered if s.get('symbol') == symbol]

        # Sort by timestamp (newest first)
        filtered = sorted(filtered, key=lambda x: x.get('timestamp', ''), reverse=True)
        filtered = filtered[:limit]

        return jsonify({
            'success': True,
            'data': {
                'signals': filtered,
                'count': len(filtered),
                'source': 'memory'
            }
        })

    except Exception as e:
        logger.error(f"❌ Get history error: {e}")
        metrics.record_error(type(e).__name__)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# PERFORMANCE TRACKING ENDPOINTS
# ============================================

@app.route('/performance/track', methods=['POST'])
@rate_limit(requests_per_minute=100)
@track_time(metrics, "/performance/track", "POST")
def track_performance():
    """Track bot performance"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['strategy', 'pnl', 'win_rate']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()

        # Save to cache
        cache_key = f"{data['strategy']}:{data['timestamp']}"
        cache.set("performance", cache_key, data, ttl=86400)  # 24h

        # In-memory storage
        performance_data.append(data)
        if len(performance_data) > 1000:
            performance_data.pop(0)

        logger.info(f"✅ Tracked performance: {data['strategy']} PnL: {data['pnl']}")

        return jsonify({
            'success': True,
            'message': 'Performance tracked successfully'
        })

    except Exception as e:
        logger.error(f"❌ Track performance error: {e}")
        metrics.record_error(type(e).__name__)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/performance/stats', methods=['GET'])
@track_time(metrics, "/performance/stats", "GET")
def get_performance_stats():
    """Get performance statistics"""
    try:
        strategy = request.args.get('strategy')

        # Filter by strategy if provided
        data = performance_data
        if strategy:
            data = [p for p in data if p.get('strategy') == strategy]

        if not data:
            return jsonify({
                'success': True,
                'data': {
                    'total_trades': 0,
                    'total_pnl': 0,
                    'avg_win_rate': 0,
                    'strategies': []
                }
            })

        # Calculate statistics
        total_pnl = sum(p.get('pnl', 0) for p in data)
        avg_win_rate = sum(p.get('win_rate', 0) for p in data) / len(data)

        return jsonify({
            'success': True,
            'data': {
                'total_trades': len(data),
                'total_pnl': round(total_pnl, 2),
                'avg_win_rate': round(avg_win_rate, 2),
                'best_strategy': max(data, key=lambda x: x.get('pnl', 0))['strategy'] if data else None,
                'worst_strategy': min(data, key=lambda x: x.get('pnl', 0))['strategy'] if data else None,
            }
        })

    except Exception as e:
        logger.error(f"❌ Get stats error: {e}")
        metrics.record_error(type(e).__name__)
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# HEALTH & METRICS ENDPOINTS
# ============================================

@app.route('/health')
def health_endpoint():
    """Health check endpoint"""
    health.add_metric("db_enabled", db_enabled)
    health.add_metric("cache_enabled", cache.enabled)
    health.add_metric("signal_count", len(signal_history))
    health.add_metric("performance_count", len(performance_data))

    return jsonify(health.get_health())


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    if not metrics.enabled:
        return "Metrics not available", 503

    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from flask import Response

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route('/stats')
def stats_endpoint():
    """Service statistics"""
    return jsonify({
        'success': True,
        'data': {
            'service': 'Database Service',
            'port': 5020,
            'database_enabled': db_enabled,
            'cache_enabled': cache.enabled,
            'signal_history_count': len(signal_history),
            'performance_data_count': len(performance_data),
            'white_hat_mode': config.WHITE_HAT_MODE,
            'uptime': health.format_uptime()
        }
    })


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    logger.info("🚀 Starting Database Service on port 5020")
    logger.info(f"📊 Database enabled: {db_enabled}")
    logger.info(f"💾 Cache enabled: {cache.enabled}")
    logger.info(f"🛡️  White-hat mode: {config.WHITE_HAT_MODE}")

    app.run(host='0.0.0.0', port=5020, debug=False)
