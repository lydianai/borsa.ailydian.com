"""
📡 WEBSOCKET STREAMING SERVICE
==============================
Real-time market data streaming via WebSocket
Port: 5021

Features:
- Real-time price updates from Binance WebSocket
- Multi-symbol subscription support
- Automatic reconnection
- Client connection management
- Rate limiting and throttling

WHITE-HAT COMPLIANCE: Real market data streaming, educational purpose
"""

import sys
import os

# Add parent directory to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
from typing import Dict, Set, List
import json
import threading
import time
import websocket
import requests

# Shared utilities
from shared.config import config
from shared.logger import get_logger
from shared.health_check import HealthCheck
from shared.redis_cache import RedisCache
from shared.metrics import MetricsCollector
from shared.rate_limiter import rate_limit

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'websocket-streaming-secret-key'
CORS(app, cors_allowed_origins="*")

# Initialize Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize utilities
logger = get_logger("websocket-streaming", level=config.LOG_LEVEL)
health = HealthCheck("WebSocket Streaming Service", 5021)
cache = RedisCache(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    enabled=config.REDIS_ENABLED
)
metrics = MetricsCollector("websocket_streaming", enabled=config.PROMETHEUS_ENABLED)

# Active connections tracking
active_connections: Dict[str, Set[str]] = {}  # symbol -> set of client IDs
binance_ws_connections: Dict[str, websocket.WebSocketApp] = {}  # symbol -> ws connection
connection_threads: Dict[str, threading.Thread] = {}  # symbol -> thread

# Subscription management
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']


def get_binance_ws_url(symbol: str) -> str:
    """Get Binance WebSocket URL for a symbol"""
    return f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"


def on_binance_message(ws, message, symbol):
    """Handle incoming message from Binance WebSocket"""
    try:
        data = json.loads(message)

        # Extract relevant price data
        price_update = {
            'symbol': data.get('s', symbol.upper()),
            'price': float(data.get('c', 0)),
            'volume': float(data.get('v', 0)),
            'high': float(data.get('h', 0)),
            'low': float(data.get('l', 0)),
            'change': float(data.get('p', 0)),
            'changePercent': float(data.get('P', 0)),
            'timestamp': datetime.now().isoformat()
        }

        # Cache the latest price
        cache.set("prices", symbol.upper(), price_update, ttl=60)

        # Broadcast to all subscribed clients
        socketio.emit('price_update', price_update, room=symbol.upper())

        # Metrics tracked via Prometheus counters (automatically handled by track_time decorator)

    except Exception as e:
        logger.error(f"❌ Error processing Binance message for {symbol}: {e}")


def on_binance_error(ws, error, symbol):
    """Handle Binance WebSocket error"""
    logger.error(f"❌ Binance WS error for {symbol}: {error}")
    metrics.record_error("binance_ws_error")


def on_binance_close(ws, close_status_code, close_msg, symbol):
    """Handle Binance WebSocket close"""
    logger.warning(f"⚠️  Binance WS closed for {symbol}: {close_status_code} - {close_msg}")

    # Attempt to reconnect after 5 seconds
    if symbol in active_connections and len(active_connections[symbol]) > 0:
        logger.info(f"🔄 Reconnecting to Binance WS for {symbol} in 5 seconds...")
        time.sleep(5)
        start_binance_stream(symbol)


def on_binance_open(ws, symbol):
    """Handle Binance WebSocket open"""
    logger.info(f"✅ Connected to Binance WS for {symbol}")


def start_binance_stream(symbol: str):
    """Start Binance WebSocket stream for a symbol"""
    if symbol in binance_ws_connections:
        logger.warning(f"⚠️  Stream already exists for {symbol}")
        return

    try:
        ws_url = get_binance_ws_url(symbol)
        logger.info(f"📡 Starting Binance stream for {symbol}")

        ws = websocket.WebSocketApp(
            ws_url,
            on_message=lambda ws, msg: on_binance_message(ws, msg, symbol),
            on_error=lambda ws, err: on_binance_error(ws, err, symbol),
            on_close=lambda ws, code, msg: on_binance_close(ws, code, msg, symbol),
            on_open=lambda ws: on_binance_open(ws, symbol)
        )

        binance_ws_connections[symbol] = ws

        # Run WebSocket in a separate thread
        thread = threading.Thread(target=ws.run_forever)
        thread.daemon = True
        thread.start()

        connection_threads[symbol] = thread

    except Exception as e:
        logger.error(f"❌ Failed to start Binance stream for {symbol}: {e}")
        metrics.record_error("binance_stream_start_error")


def stop_binance_stream(symbol: str):
    """Stop Binance WebSocket stream for a symbol"""
    if symbol in binance_ws_connections:
        try:
            logger.info(f"🛑 Stopping Binance stream for {symbol}")
            ws = binance_ws_connections[symbol]
            ws.close()
            del binance_ws_connections[symbol]

            if symbol in connection_threads:
                del connection_threads[symbol]

        except Exception as e:
            logger.error(f"❌ Error stopping stream for {symbol}: {e}")


# ============================================
# SOCKET.IO EVENT HANDLERS
# ============================================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    logger.info(f"🔌 Client connected: {client_id}")
    emit('connected', {'message': 'Connected to WebSocket Streaming Service', 'client_id': client_id})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    logger.info(f"🔌 Client disconnected: {client_id}")

    # Remove client from all subscriptions
    for symbol in list(active_connections.keys()):
        if client_id in active_connections[symbol]:
            active_connections[symbol].remove(client_id)

            # Stop stream if no more subscribers
            if len(active_connections[symbol]) == 0:
                stop_binance_stream(symbol)
                del active_connections[symbol]


@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle symbol subscription"""
    client_id = request.sid
    symbols = data.get('symbols', [])

    if not isinstance(symbols, list):
        symbols = [symbols]

    for symbol in symbols:
        symbol = symbol.upper()

        # Add client to subscription
        if symbol not in active_connections:
            active_connections[symbol] = set()

        active_connections[symbol].add(client_id)
        join_room(symbol)

        # Start Binance stream if not already running
        if symbol not in binance_ws_connections:
            start_binance_stream(symbol)

        # Send cached price immediately if available
        cached_price = cache.get("prices", symbol)
        if cached_price:
            emit('price_update', cached_price)

        logger.info(f"✅ Client {client_id} subscribed to {symbol}")

    emit('subscribed', {'symbols': symbols, 'message': f'Subscribed to {len(symbols)} symbols'})


@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Handle symbol unsubscription"""
    client_id = request.sid
    symbols = data.get('symbols', [])

    if not isinstance(symbols, list):
        symbols = [symbols]

    for symbol in symbols:
        symbol = symbol.upper()

        if symbol in active_connections and client_id in active_connections[symbol]:
            active_connections[symbol].remove(client_id)
            leave_room(symbol)

            # Stop stream if no more subscribers
            if len(active_connections[symbol]) == 0:
                stop_binance_stream(symbol)
                del active_connections[symbol]

            logger.info(f"✅ Client {client_id} unsubscribed from {symbol}")

    emit('unsubscribed', {'symbols': symbols, 'message': f'Unsubscribed from {len(symbols)} symbols'})


# ============================================
# REST API ENDPOINTS
# ============================================

@app.route('/health')
def health_endpoint():
    """Health check endpoint"""
    health.add_metric("active_streams", len(binance_ws_connections))
    health.add_metric("total_subscriptions", sum(len(clients) for clients in active_connections.values()))
    health.add_metric("active_symbols", list(active_connections.keys()))

    return jsonify(health.get_health())


@app.route('/stats')
def stats_endpoint():
    """Service statistics"""
    return jsonify({
        'success': True,
        'data': {
            'service': 'WebSocket Streaming Service',
            'port': 5021,
            'active_streams': len(binance_ws_connections),
            'active_symbols': list(active_connections.keys()),
            'total_subscriptions': sum(len(clients) for clients in active_connections.values()),
            'subscriptions_by_symbol': {
                symbol: len(clients) for symbol, clients in active_connections.items()
            },
            'white_hat_mode': config.WHITE_HAT_MODE,
            'uptime': health.format_uptime()
        }
    })


@app.route('/symbols')
def symbols_endpoint():
    """Get list of available symbols"""
    return jsonify({
        'success': True,
        'data': {
            'default_symbols': DEFAULT_SYMBOLS,
            'active_symbols': list(active_connections.keys()),
            'supported': 'All Binance USDT perpetual futures'
        }
    })


@app.route('/price/<symbol>')
@rate_limit(requests_per_minute=300)
def price_endpoint(symbol):
    """Get latest cached price for a symbol"""
    symbol = symbol.upper()

    # Try cache first
    cached_price = cache.get("prices", symbol)
    if cached_price:
        return jsonify({
            'success': True,
            'data': cached_price,
            'source': 'cache'
        })

    # Fallback to Binance REST API
    try:
        response = requests.get(
            f"https://api.binance.com/api/v3/ticker/24hr",
            params={'symbol': symbol},
            timeout=5
        )
        data = response.json()

        price_data = {
            'symbol': symbol,
            'price': float(data.get('lastPrice', 0)),
            'volume': float(data.get('volume', 0)),
            'high': float(data.get('highPrice', 0)),
            'low': float(data.get('lowPrice', 0)),
            'change': float(data.get('priceChange', 0)),
            'changePercent': float(data.get('priceChangePercent', 0)),
            'timestamp': datetime.now().isoformat()
        }

        return jsonify({
            'success': True,
            'data': price_data,
            'source': 'api'
        })

    except Exception as e:
        logger.error(f"❌ Error fetching price for {symbol}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/latest-prices')
@rate_limit(requests_per_minute=600)
def latest_prices_endpoint():
    """Get all latest cached prices"""
    # Collect all cached prices from active symbols
    prices = {}

    # Get all active symbols
    all_symbols = list(active_connections.keys())
    if not all_symbols:
        all_symbols = DEFAULT_SYMBOLS

    for symbol in all_symbols:
        cached = cache.get("prices", symbol)
        if cached:
            prices[symbol] = cached

    if not prices:
        return jsonify({
            'success': False,
            'error': 'No cached prices available',
            'prices': {},
            'count': 0,
            'timestamp': datetime.now().isoformat()
        }), 404

    return jsonify({
        'success': True,
        'prices': prices,
        'count': len(prices),
        'source': 'cache',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    if not metrics.enabled:
        return "Metrics not available", 503

    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from flask import Response

    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ============================================
# STARTUP
# ============================================

def start_default_streams():
    """Start streaming for default symbols"""
    logger.info("🚀 Starting default symbol streams...")
    for symbol in DEFAULT_SYMBOLS:
        active_connections[symbol] = set()
        start_binance_stream(symbol)
        time.sleep(0.5)  # Avoid rate limiting


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    logger.info("🚀 Starting WebSocket Streaming Service on port 5021")
    logger.info(f"📡 Default symbols: {', '.join(DEFAULT_SYMBOLS)}")
    logger.info(f"💾 Cache enabled: {cache.enabled}")
    logger.info(f"🛡️  White-hat mode: {config.WHITE_HAT_MODE}")

    # Start default streams
    threading.Thread(target=start_default_streams, daemon=True).start()

    # Run Socket.IO server
    socketio.run(app, host='0.0.0.0', port=5021, debug=False, allow_unsafe_werkzeug=True)
