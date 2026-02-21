"""
SIGNAL GENERATOR SERVICE
Real-time trading signals from 14 AI models
Port: 5004
White-hat compliant - transparent, fair signals
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sock import Sock
import requests
import time
import threading
import sys
from datetime import datetime
from consensus_engine import ConsensusEngine

# Force unbuffered output
sys.stdout = sys.stderr

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Service URLs
AI_SERVICE_URL = "http://localhost:5003"
DATA_SERVICE_URL = "http://localhost:3000"

# Consensus engine
consensus_engine = ConsensusEngine()

# Latest signals cache
latest_signals = {}
signal_lock = threading.Lock()

# WebSocket clients
ws_clients = []


def generate_signal_for_coin(symbol: str, timeframe: str = '1h') -> dict:
    """
    Generate consensus signal for a single coin

    Args:
        symbol: Coin symbol (e.g., 'BTC')
        timeframe: Timeframe for analysis

    Returns:
        Signal dict or None if error
    """
    try:
        print(f"🔄 Generating signal for {symbol}...")

        # 1. Get current price (using faster market API)
        price_response = requests.get(
            f"{DATA_SERVICE_URL}/api/market/top100",
            params={'limit': 100},
            timeout=15
        )

        if not price_response.ok:
            print(f"❌ Failed to get price for {symbol}")
            return None

        price_data = price_response.json()
        if not price_data.get('success'):
            return None

        # Find coin
        coin_info = None
        for item in price_data.get('data', []):
            if item['coin']['symbol'] == symbol:
                coin_info = item['coin']
                break

        if not coin_info:
            print(f"❌ Coin {symbol} not found")
            return None

        current_price = coin_info['price']

        # 2. Get predictions from all models
        model_predictions = []

        # Quick prediction from key models for consensus
        key_models = ['lstm_standard', 'gru_attention', 'transformer_standard', 'xgboost']

        for model_id in key_models:
            try:
                response = requests.post(
                    f"{AI_SERVICE_URL}/predict/single",
                    json={
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'model': model_id
                    },
                    timeout=10
                )

                if response.ok:
                    data = response.json()
                    if data.get('success'):
                        pred = data['prediction']
                        model_predictions.append({
                            'model_name': model_id,
                            'action': pred['action'],
                            'confidence': pred['confidence'],
                            'prediction': pred['prediction']
                        })
            except Exception as e:
                print(f"⚠️  Model {model_id} failed: {e}")
                continue

        # 3. Generate consensus
        if not model_predictions:
            print(f"❌ No model predictions for {symbol}")
            return None

        signal = consensus_engine.aggregate_predictions(
            model_predictions,
            current_price
        )

        # 4. Add coin info
        signal['symbol'] = symbol
        signal['name'] = coin_info['name']
        signal['current_price'] = current_price
        signal['change_24h'] = coin_info.get('change24h', 0)
        signal['timeframe'] = timeframe

        print(f"✅ Signal generated for {symbol}: {signal['action']} ({signal['confidence']:.1f}%)")

        return signal

    except Exception as e:
        import traceback
        print(f"❌ Error generating signal for {symbol}: {e}")
        print(traceback.format_exc())
        return None


def signal_generator_loop():
    """Background thread that generates signals every 30s"""
    print("🚀 Signal generator loop started")

    # Top coins to track
    tracked_coins = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'MATIC', 'LINK', 'UNI']

    while True:
        try:
            print(f"\n{'='*60}")
            print(f"🔄 Generating signals at {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            for symbol in tracked_coins:
                signal = generate_signal_for_coin(symbol)

                if signal:
                    # Update cache
                    with signal_lock:
                        latest_signals[symbol] = signal

                    # Broadcast to WebSocket clients
                    broadcast_signal(signal)

                # Small delay between coins
                time.sleep(1)

            print(f"✅ Signals generated for {len(latest_signals)} coins")
            print(f"⏰ Next update in 30 seconds\n")

            # Wait 30 seconds before next iteration
            time.sleep(30)

        except Exception as e:
            print(f"❌ Error in signal generator loop: {e}")
            time.sleep(5)


def broadcast_signal(signal: dict):
    """Broadcast signal to all WebSocket clients"""
    if not ws_clients:
        return

    import json
    message = json.dumps(signal)

    # Remove disconnected clients
    disconnected = []

    for ws in ws_clients:
        try:
            ws.send(message)
        except Exception:
            disconnected.append(ws)

    # Clean up
    for ws in disconnected:
        ws_clients.remove(ws)


# ============================================
# REST API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Signal Generator',
        'port': 5004,
        'signals_cached': len(latest_signals)
    })


@app.route('/signals/latest', methods=['GET'])
def get_latest_signals():
    """Get all latest signals"""
    with signal_lock:
        return jsonify({
            'success': True,
            'signals': list(latest_signals.values()),
            'count': len(latest_signals),
            'timestamp': datetime.now().isoformat()
        })


@app.route('/signals/coin/<symbol>', methods=['GET'])
def get_coin_signal(symbol: str):
    """Get signal for specific coin"""
    symbol = symbol.upper()

    with signal_lock:
        signal = latest_signals.get(symbol)

    if signal:
        return jsonify({
            'success': True,
            'signal': signal
        })
    else:
        # Generate on demand
        signal = generate_signal_for_coin(symbol)

        if signal:
            with signal_lock:
                latest_signals[symbol] = signal

            return jsonify({
                'success': True,
                'signal': signal
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Could not generate signal for {symbol}'
            }), 404


@app.route('/signals/generate', methods=['POST'])
def generate_signal_endpoint():
    """Generate signal on demand"""
    data = request.json
    symbol = data.get('symbol', '').upper()
    timeframe = data.get('timeframe', '1h')

    if not symbol:
        return jsonify({'success': False, 'error': 'Symbol required'}), 400

    signal = generate_signal_for_coin(symbol, timeframe)

    if signal:
        with signal_lock:
            latest_signals[symbol] = signal

        return jsonify({
            'success': True,
            'signal': signal
        })
    else:
        return jsonify({
            'success': False,
            'error': f'Could not generate signal for {symbol}'
        }), 500


# ============================================
# WEBSOCKET
# ============================================

@sock.route('/signals/stream')
def signals_stream(ws):
    """WebSocket endpoint for real-time signals"""
    print(f"📡 New WebSocket client connected")
    ws_clients.append(ws)

    try:
        # Send current signals on connect
        with signal_lock:
            for signal in latest_signals.values():
                import json
                ws.send(json.dumps(signal))

        # Keep connection alive
        while True:
            data = ws.receive()
            if data is None:
                break

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        print(f"📡 WebSocket client disconnected")


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 SIGNAL GENERATOR SERVICE - STARTING")
    print("=" * 60 + "\n")
    print("📊 14 AI Models → Consensus Signals")
    print("🌐 Server: http://localhost:5004")
    print("📡 WebSocket: ws://localhost:5004/signals/stream")
    print("⏰ Signal Update: Every 30 seconds")
    print("=" * 60 + "\n")

    # Start signal generator thread
    generator_thread = threading.Thread(target=signal_generator_loop, daemon=True)
    generator_thread.start()

    # Start Flask server
    app.run(host='0.0.0.0', port=5004, debug=False)
