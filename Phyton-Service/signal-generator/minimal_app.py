"""
Signal Generator Service - Minimal Mode
Basic Flask server with WebSocket mock for testing
Real 14 AI consensus will be added after full setup
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import time
from datetime import datetime
from threading import Thread
import json

app = Flask(__name__)
CORS(app)

# Simulated top coins for tracking
TOP_COINS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'SOLUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT'
]

# Mock signal storage
latest_signals = {}

def generate_mock_signal(symbol):
    """Generate mock AI consensus signal"""
    signals = ['BUY', 'SELL', 'HOLD']
    signal = random.choice(signals)
    confidence = round(random.uniform(0.6, 0.95), 2)

    # Mock 14 AI models consensus
    models_consensus = []
    for i in range(14):
        models_consensus.append({
            'model': f'ai_model_{i+1}',
            'signal': random.choice(signals),
            'confidence': round(random.uniform(0.5, 0.9), 2)
        })

    return {
        'symbol': symbol,
        'signal': signal,
        'confidence': confidence,
        'consensus_score': confidence,
        'models_count': 14,
        'models_consensus': models_consensus,
        'price': round(random.uniform(100, 50000), 2),
        'volume_24h': round(random.uniform(1000000, 10000000), 2),
        'timestamp': datetime.now().isoformat()
    }

def background_signal_generator():
    """
    Background thread that generates signals every 30 seconds
    Simulates real-time AI analysis
    """
    print("🔄 Background signal generator started (30s interval)")
    while True:
        try:
            for symbol in TOP_COINS:
                signal_data = generate_mock_signal(symbol)
                latest_signals[symbol] = signal_data

            print(f"✅ Generated signals for {len(TOP_COINS)} coins at {datetime.now().strftime('%H:%M:%S')}")
            time.sleep(30)  # Generate new signals every 30 seconds
        except Exception as e:
            print(f"❌ Error in background generator: {e}")
            time.sleep(5)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Signal Generator (Minimal Mode)',
        'mode': 'minimal',
        'tracked_coins': len(TOP_COINS),
        'ai_models': 14,
        'update_interval': '30s',
        'latest_signals_count': len(latest_signals),
        'message': 'Mock signals. Real 14 AI consensus coming after full setup.',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signals', methods=['GET'])
def get_signals():
    """
    Get current signals for all tracked coins

    Query params:
    - symbol: Filter by specific symbol (optional)
    - limit: Limit results (default: 10)
    """
    symbol_filter = request.args.get('symbol')
    limit = int(request.args.get('limit', 10))

    if symbol_filter:
        if symbol_filter in latest_signals:
            return jsonify({
                'signals': [latest_signals[symbol_filter]],
                'total': 1,
                'mode': 'minimal'
            })
        else:
            # Generate on-demand if not in cache
            signal = generate_mock_signal(symbol_filter)
            return jsonify({
                'signals': [signal],
                'total': 1,
                'mode': 'minimal',
                'generated_on_demand': True
            })

    # Return all signals
    signals_list = list(latest_signals.values())[:limit]

    return jsonify({
        'signals': signals_list,
        'total': len(signals_list),
        'tracked_coins': TOP_COINS,
        'mode': 'minimal',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/signals/top', methods=['GET'])
def get_top_signals():
    """
    Get top signals sorted by confidence

    Query params:
    - signal_type: BUY, SELL, HOLD (optional)
    - limit: Limit results (default: 5)
    """
    signal_type = request.args.get('signal_type', '').upper()
    limit = int(request.args.get('limit', 5))

    signals_list = list(latest_signals.values())

    # Filter by signal type if specified
    if signal_type in ['BUY', 'SELL', 'HOLD']:
        signals_list = [s for s in signals_list if s['signal'] == signal_type]

    # Sort by confidence
    signals_list.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'top_signals': signals_list[:limit],
        'signal_type': signal_type or 'ALL',
        'total': len(signals_list),
        'mode': 'minimal',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/consensus/<symbol>', methods=['GET'])
def get_consensus(symbol):
    """
    Get detailed AI consensus for a specific symbol
    """
    if symbol in latest_signals:
        data = latest_signals[symbol]
    else:
        data = generate_mock_signal(symbol)
        latest_signals[symbol] = data

    return jsonify({
        'symbol': symbol,
        'consensus': {
            'signal': data['signal'],
            'confidence': data['confidence'],
            'models_count': 14,
            'models_breakdown': data['models_consensus']
        },
        'market_data': {
            'price': data['price'],
            'volume_24h': data['volume_24h']
        },
        'mode': 'minimal',
        'timestamp': data['timestamp']
    })

@app.route('/stream', methods=['GET'])
def stream_signals():
    """
    Mock WebSocket-like streaming endpoint
    Returns latest signals with server-sent events header
    """
    signals_list = list(latest_signals.values())

    return jsonify({
        'stream': 'active',
        'signals': signals_list,
        'update_interval_seconds': 30,
        'note': 'This is a REST mock. Real WebSocket available after full setup.',
        'websocket_endpoint': 'ws://localhost:5004/ws',
        'mode': 'minimal'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Signal Generator Service - Minimal Mode")
    print("=" * 60)
    print(f"✅ Tracking {len(TOP_COINS)} top coins")
    print(f"🤖 14 AI Models consensus (mock)")
    print(f"⏱️  Update interval: 30 seconds")
    print("📝 Mode: MINIMAL (Mock signals)")
    print("=" * 60)
    print("🌐 Starting server on http://localhost:5004")
    print("=" * 60)

    # Start background signal generator
    bg_thread = Thread(target=background_signal_generator, daemon=True)
    bg_thread.start()

    # Generate initial signals
    for symbol in TOP_COINS:
        latest_signals[symbol] = generate_mock_signal(symbol)

    print(f"✅ Initial signals generated for {len(TOP_COINS)} coins")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)
