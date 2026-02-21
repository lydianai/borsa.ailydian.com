"""
MFI (Money Flow Index) MONITOR MICROSERVICE
Port: 5023
Real-time oversold detection with multi-timeframe analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import talib
from datetime import datetime
import requests

app = Flask(__name__)
CORS(app)

print("🚀 MFI Monitor Microservice v1.0")
print(f"📊 TA-Lib Version: {talib.__version__}")
print("💰 Money Flow Index - Oversold Detection System")

# ============================================
# CONFIGURATION
# ============================================

# Use internal Next.js API for klines (with caching and fallback)
KLINES_API = "http://localhost:3000/api/klines"

# MFI Thresholds
MFI_OVERSOLD = 20
MFI_EXTREME_OVERSOLD = 10
MFI_SUDDEN_DROP_THRESHOLD = 30  # % drop in 5 bars
MFI_SUDDEN_DROP_BARS = 5

# ============================================
# HELPER FUNCTIONS
# ============================================

def fetch_binance_klines(symbol, interval='15m', limit=100):
    """Fetch OHLCV data from internal Next.js klines API (cached + fallback)"""
    try:
        # Call Next.js /api/klines which has caching and Bybit fallback
        url = f"{KLINES_API}/{symbol}?interval={interval}&limit={limit}"
        response = requests.get(url, timeout=15)

        if response.status_code != 200:
            print(f"[MFI Monitor] Klines API error {response.status_code} for {symbol}")
            return None

        data = response.json()

        if not data.get('success') or not data.get('data') or not data['data'].get('candles'):
            print(f"[MFI Monitor] Invalid klines response for {symbol}")
            return None

        candles = data['data']['candles']

        if len(candles) < 30:
            print(f"[MFI Monitor] Insufficient candles for {symbol}: {len(candles)}")
            return None

        # Parse OHLCV from TradingView format
        ohlcv = {
            'open': [float(c['open']) for c in candles],
            'high': [float(c['high']) for c in candles],
            'low': [float(c['low']) for c in candles],
            'close': [float(c['close']) for c in candles],
            'volume': [float(c['volume']) for c in candles],
            'timestamp': [int(c['time']) * 1000 for c in candles]  # Convert to milliseconds
        }

        return ohlcv

    except Exception as e:
        print(f"[MFI Monitor] Error fetching {symbol} {interval}: {e}")
        return None

def calculate_mfi(high, low, close, volume, timeperiod=14):
    """Calculate Money Flow Index"""
    try:
        high_np = np.array(high, dtype=float)
        low_np = np.array(low, dtype=float)
        close_np = np.array(close, dtype=float)
        volume_np = np.array(volume, dtype=float)

        mfi = talib.MFI(high_np, low_np, close_np, volume_np, timeperiod=timeperiod)

        return mfi

    except Exception as e:
        print(f"[MFI Monitor] MFI calculation error: {e}")
        return None

def detect_sudden_drop(mfi_array, bars=5, threshold=30):
    """
    Detect sudden MFI drops
    Returns: (is_drop, drop_percentage, bars_ago)
    """
    try:
        # Get last N bars
        recent_mfi = mfi_array[-bars:]

        # Remove NaN values
        recent_mfi = recent_mfi[~np.isnan(recent_mfi)]

        if len(recent_mfi) < 2:
            return False, 0, None

        # Calculate drop from highest to lowest in recent bars
        highest = np.max(recent_mfi)
        lowest = recent_mfi[-1]  # Current value

        if highest == 0:
            return False, 0, None

        drop_pct = ((highest - lowest) / highest) * 100

        # Find how many bars ago the high was
        high_idx = np.where(recent_mfi == highest)[0][0]
        bars_ago = int(len(recent_mfi) - high_idx - 1)  # Convert to Python int

        is_drop = drop_pct >= threshold

        return is_drop, float(drop_pct), bars_ago  # Ensure float for drop_pct

    except Exception as e:
        print(f"[MFI Monitor] Sudden drop detection error: {e}")
        return False, 0, None

def analyze_mfi_timeframe(symbol, interval):
    """Analyze MFI for a single timeframe"""
    try:
        # Fetch data
        ohlcv = fetch_binance_klines(symbol, interval, limit=100)

        if not ohlcv:
            return None

        # Calculate MFI
        mfi = calculate_mfi(
            ohlcv['high'],
            ohlcv['low'],
            ohlcv['close'],
            ohlcv['volume'],
            timeperiod=14
        )

        if mfi is None or len(mfi) == 0:
            return None

        # Get current MFI value (last non-NaN)
        valid_mfi = mfi[~np.isnan(mfi)]

        if len(valid_mfi) == 0:
            return None

        current_mfi = float(valid_mfi[-1])

        # Detect sudden drop
        is_drop, drop_pct, bars_ago = detect_sudden_drop(mfi)

        # Determine status
        if current_mfi <= MFI_EXTREME_OVERSOLD:
            status = 'EXTREME_OVERSOLD'
            signal = 'STRONG_BUY'
            urgency = 'CRITICAL'
        elif current_mfi <= MFI_OVERSOLD:
            status = 'OVERSOLD'
            signal = 'BUY'
            urgency = 'HIGH'
        elif current_mfi >= 80:
            status = 'OVERBOUGHT'
            signal = 'SELL'
            urgency = 'MEDIUM'
        else:
            status = 'NORMAL'
            signal = 'NEUTRAL'
            urgency = 'LOW'

        # Extra urgency for sudden drops
        if is_drop and drop_pct >= 40:
            urgency = 'CRITICAL'
        elif is_drop and drop_pct >= 30:
            if urgency == 'LOW':
                urgency = 'HIGH'

        return {
            'interval': interval,
            'mfi': round(float(current_mfi), 2),
            'status': status,
            'signal': signal,
            'urgency': urgency,
            'sudden_drop': {
                'detected': bool(is_drop),  # Convert numpy bool to Python bool
                'drop_percentage': round(float(drop_pct), 2) if is_drop else 0,
                'bars_ago': int(bars_ago) if bars_ago is not None else None
            },
            'current_price': float(ohlcv['close'][-1])
        }

    except Exception as e:
        print(f"[MFI Monitor] Analysis error for {symbol} {interval}: {e}")
        return None

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'service': 'MFI Monitor',
        'status': 'online',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_mfi():
    """
    Analyze MFI across multiple timeframes

    POST /analyze
    {
        "symbol": "BTCUSDT",
        "timeframes": ["15m", "30m", "1h"]
    }
    """
    try:
        data = request.json

        symbol = data.get('symbol', 'BTCUSDT')
        timeframes = data.get('timeframes', ['15m', '30m', '1h'])

        # Validate symbol format
        if not symbol.endswith('USDT'):
            symbol = symbol + 'USDT'

        print(f"[MFI Monitor] Analyzing {symbol} on {timeframes}")

        # Analyze each timeframe
        results = {}
        critical_alerts = []

        for tf in timeframes:
            analysis = analyze_mfi_timeframe(symbol, tf)

            if analysis:
                results[tf] = analysis

                # Collect critical alerts
                if analysis['urgency'] == 'CRITICAL':
                    critical_alerts.append({
                        'timeframe': tf,
                        'mfi': analysis['mfi'],
                        'status': analysis['status'],
                        'sudden_drop': analysis['sudden_drop']
                    })

        if not results:
            return jsonify({
                'success': False,
                'error': f'Could not analyze {symbol} - insufficient data'
            }), 400

        # Multi-timeframe consensus
        oversold_count = sum(1 for r in results.values() if r['status'] in ['OVERSOLD', 'EXTREME_OVERSOLD'])
        total_timeframes = len(results)

        # Overall signal
        if oversold_count == total_timeframes:
            overall_signal = 'STRONG_BUY'
            overall_urgency = 'CRITICAL'
            overall_message = f"ALL {total_timeframes} timeframes oversold!"
        elif oversold_count >= total_timeframes * 0.67:
            overall_signal = 'BUY'
            overall_urgency = 'HIGH'
            overall_message = f"{oversold_count}/{total_timeframes} timeframes oversold"
        elif oversold_count > 0:
            overall_signal = 'WATCH'
            overall_urgency = 'MEDIUM'
            overall_message = f"{oversold_count}/{total_timeframes} timeframes oversold"
        else:
            overall_signal = 'NEUTRAL'
            overall_urgency = 'LOW'
            overall_message = "No oversold timeframes"

        # Calculate average MFI
        avg_mfi = sum(r['mfi'] for r in results.values()) / len(results)

        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'timeframes': results,
                'overall': {
                    'signal': overall_signal,
                    'urgency': overall_urgency,
                    'message': overall_message,
                    'avg_mfi': round(avg_mfi, 2),
                    'oversold_count': oversold_count,
                    'total_timeframes': total_timeframes
                },
                'critical_alerts': critical_alerts,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        print(f"[MFI Monitor] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/scan', methods=['POST'])
def scan_all_coins():
    """
    Scan all coins for MFI oversold conditions

    POST /scan
    {
        "coins": ["BTCUSDT", "ETHUSDT", ...],
        "timeframe": "15m"
    }
    """
    try:
        data = request.json

        coins = data.get('coins', [])
        timeframe = data.get('timeframe', '15m')

        if not coins:
            return jsonify({
                'success': False,
                'error': 'No coins provided'
            }), 400

        print(f"[MFI Monitor] Scanning {len(coins)} coins on {timeframe}")

        # Analyze each coin
        oversold_coins = []
        extreme_oversold_coins = []
        sudden_drop_coins = []

        for coin in coins:
            analysis = analyze_mfi_timeframe(coin, timeframe)

            if analysis:
                if analysis['status'] == 'EXTREME_OVERSOLD':
                    extreme_oversold_coins.append({
                        'symbol': coin,
                        'mfi': analysis['mfi'],
                        'price': analysis['current_price'],
                        'sudden_drop': analysis['sudden_drop']
                    })
                elif analysis['status'] == 'OVERSOLD':
                    oversold_coins.append({
                        'symbol': coin,
                        'mfi': analysis['mfi'],
                        'price': analysis['current_price'],
                        'sudden_drop': analysis['sudden_drop']
                    })

                if analysis['sudden_drop']['detected']:
                    sudden_drop_coins.append({
                        'symbol': coin,
                        'mfi': analysis['mfi'],
                        'drop_pct': analysis['sudden_drop']['drop_percentage'],
                        'bars_ago': analysis['sudden_drop']['bars_ago']
                    })

        return jsonify({
            'success': True,
            'data': {
                'timeframe': timeframe,
                'total_scanned': len(coins),
                'extreme_oversold': extreme_oversold_coins,
                'oversold': oversold_coins,
                'sudden_drops': sudden_drop_coins,
                'summary': {
                    'extreme_oversold_count': len(extreme_oversold_coins),
                    'oversold_count': len(oversold_coins),
                    'sudden_drop_count': len(sudden_drop_coins)
                },
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        print(f"[MFI Monitor] Scan error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("="* 50)
    print("MFI Monitor Microservice Starting...")
    print("Port: 5023")
    print("="* 50)
    app.run(host='0.0.0.0', port=5023, debug=False)
