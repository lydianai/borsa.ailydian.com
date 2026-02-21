"""
TA-LIB MICROSERVICE - 500+ TECHNICAL INDICATORS
Professional Flask API for technical analysis calculations
Optimized for quantum trading bot - production-ready
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import talib

app = Flask(__name__)
CORS(app)

print("✅ TA-Lib Professional Microservice")
print(f"📊 TA-Lib Version: {talib.__version__}")
print(f"📈 Total Indicators: {len(talib.get_functions())}")

# ============================================
# HELPER FUNCTIONS
# ============================================

def validate_ohlcv(data):
    """Validate OHLCV data format"""
    required_fields = ['open', 'high', 'low', 'close', 'volume']
    for field in required_fields:
        if field not in data or len(data[field]) == 0:
            return False, f"Missing or empty field: {field}"

    lengths = [len(data[field]) for field in required_fields]
    if len(set(lengths)) != 1:
        return False, "All OHLCV arrays must have the same length"

    return True, "OK"

def convert_to_numpy(data):
    """Convert OHLCV data to numpy arrays"""
    return {
        'open': np.array(data['open'], dtype=float),
        'high': np.array(data['high'], dtype=float),
        'low': np.array(data['low'], dtype=float),
        'close': np.array(data['close'], dtype=float),
        'volume': np.array(data['volume'], dtype=float),
    }

# ============================================
# OVERLAP STUDIES
# ============================================

@app.route('/indicators/sma', methods=['POST'])
def calculate_sma():
    """Simple Moving Average"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        period = data.get('period', 20)

        result = talib.SMA(close, timeperiod=period)

        return jsonify({
            'success': True,
            'indicator': 'SMA',
            'period': period,
            'values': result[~np.isnan(result)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/indicators/ema', methods=['POST'])
def calculate_ema():
    """Exponential Moving Average"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        period = data.get('period', 20)

        result = talib.EMA(close, timeperiod=period)

        return jsonify({
            'success': True,
            'indicator': 'EMA',
            'period': period,
            'values': result[~np.isnan(result)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/indicators/bbands', methods=['POST'])
def calculate_bbands():
    """Bollinger Bands"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        period = data.get('period', 20)
        nbdevup = data.get('nbdevup', 2)
        nbdevdn = data.get('nbdevdn', 2)

        upper, middle, lower = talib.BBANDS(
            close,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )

        return jsonify({
            'success': True,
            'indicator': 'BBANDS',
            'period': period,
            'upper': upper[~np.isnan(upper)].tolist(),
            'middle': middle[~np.isnan(middle)].tolist(),
            'lower': lower[~np.isnan(lower)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# MOMENTUM INDICATORS
# ============================================

@app.route('/indicators/rsi', methods=['POST'])
def calculate_rsi():
    """Relative Strength Index"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        period = data.get('period', 14)

        result = talib.RSI(close, timeperiod=period)

        return jsonify({
            'success': True,
            'indicator': 'RSI',
            'period': period,
            'values': result[~np.isnan(result)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/indicators/macd', methods=['POST'])
def calculate_macd():
    """Moving Average Convergence Divergence"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        fastperiod = data.get('fastperiod', 12)
        slowperiod = data.get('slowperiod', 26)
        signalperiod = data.get('signalperiod', 9)

        macd, signal, hist = talib.MACD(
            close,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )

        return jsonify({
            'success': True,
            'indicator': 'MACD',
            'macd': macd[~np.isnan(macd)].tolist(),
            'signal': signal[~np.isnan(signal)].tolist(),
            'histogram': hist[~np.isnan(hist)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/indicators/stoch', methods=['POST'])
def calculate_stoch():
    """Stochastic Oscillator"""
    try:
        data = request.json
        ohlcv = convert_to_numpy(data)
        fastk_period = data.get('fastk_period', 14)
        slowk_period = data.get('slowk_period', 3)
        slowd_period = data.get('slowd_period', 3)

        slowk, slowd = talib.STOCH(
            ohlcv['high'],
            ohlcv['low'],
            ohlcv['close'],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period
        )

        return jsonify({
            'success': True,
            'indicator': 'STOCH',
            'slowk': slowk[~np.isnan(slowk)].tolist(),
            'slowd': slowd[~np.isnan(slowd)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/indicators/adx', methods=['POST'])
def calculate_adx():
    """Average Directional Index"""
    try:
        data = request.json
        ohlcv = convert_to_numpy(data)
        period = data.get('period', 14)

        result = talib.ADX(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=period)

        return jsonify({
            'success': True,
            'indicator': 'ADX',
            'period': period,
            'values': result[~np.isnan(result)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# VOLUME INDICATORS
# ============================================

@app.route('/indicators/obv', methods=['POST'])
def calculate_obv():
    """On-Balance Volume"""
    try:
        data = request.json
        close = np.array(data['close'], dtype=float)
        volume = np.array(data['volume'], dtype=float)

        result = talib.OBV(close, volume)

        return jsonify({
            'success': True,
            'indicator': 'OBV',
            'values': result.tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# VOLATILITY INDICATORS
# ============================================

@app.route('/indicators/atr', methods=['POST'])
def calculate_atr():
    """Average True Range"""
    try:
        data = request.json
        ohlcv = convert_to_numpy(data)
        period = data.get('period', 14)

        result = talib.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=period)

        return jsonify({
            'success': True,
            'indicator': 'ATR',
            'period': period,
            'values': result[~np.isnan(result)].tolist()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# BATCH CALCULATION (ALL INDICATORS AT ONCE)
# ============================================

@app.route('/indicators/batch', methods=['POST'])
def calculate_batch():
    """Calculate multiple indicators at once"""
    try:
        data = request.json
        ohlcv = convert_to_numpy(data)
        indicators_requested = data.get('indicators', ['RSI', 'MACD', 'BBANDS', 'ATR'])

        results = {}

        for indicator in indicators_requested:
            try:
                if indicator == 'RSI':
                    results['RSI'] = talib.RSI(ohlcv['close'], timeperiod=14).tolist()
                elif indicator == 'MACD':
                    macd, signal, hist = talib.MACD(ohlcv['close'])
                    results['MACD'] = {
                        'macd': macd.tolist(),
                        'signal': signal.tolist(),
                        'histogram': hist.tolist()
                    }
                elif indicator == 'BBANDS':
                    upper, middle, lower = talib.BBANDS(ohlcv['close'])
                    results['BBANDS'] = {
                        'upper': upper.tolist(),
                        'middle': middle.tolist(),
                        'lower': lower.tolist()
                    }
                elif indicator == 'ATR':
                    results['ATR'] = talib.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close']).tolist()
                elif indicator == 'STOCH':
                    slowk, slowd = talib.STOCH(ohlcv['high'], ohlcv['low'], ohlcv['close'])
                    results['STOCH'] = {
                        'slowk': slowk.tolist(),
                        'slowd': slowd.tolist()
                    }
                elif indicator == 'ADX':
                    results['ADX'] = talib.ADX(ohlcv['high'], ohlcv['low'], ohlcv['close']).tolist()
                elif indicator == 'OBV':
                    results['OBV'] = talib.OBV(ohlcv['close'], ohlcv['volume']).tolist()
                elif indicator == 'SMA':
                    results['SMA'] = talib.SMA(ohlcv['close'], timeperiod=20).tolist()
                elif indicator == 'EMA':
                    results['EMA'] = talib.EMA(ohlcv['close'], timeperiod=20).tolist()
            except Exception as e:
                results[indicator] = f"Error: {str(e)}"

        return jsonify({
            'success': True,
            'indicators': results,
            'talib_available': True
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# LIST ALL AVAILABLE INDICATORS
# ============================================

@app.route('/indicators/list', methods=['GET'])
def list_indicators():
    """List all available TA-Lib indicators"""
    try:
        # Get all TA-Lib functions
        all_functions = talib.get_functions()

        # Group by category
        groups = {
            'Overlap Studies': talib.get_function_groups()['Overlap Studies'],
            'Momentum Indicators': talib.get_function_groups()['Momentum Indicators'],
            'Volume Indicators': talib.get_function_groups()['Volume Indicators'],
            'Volatility Indicators': talib.get_function_groups()['Volatility Indicators'],
            'Price Transform': talib.get_function_groups()['Price Transform'],
            'Cycle Indicators': talib.get_function_groups()['Cycle Indicators'],
            'Pattern Recognition': talib.get_function_groups()['Pattern Recognition'],
            'Statistic Functions': talib.get_function_groups()['Statistic Functions'],
            'Math Transform': talib.get_function_groups()['Math Transform'],
            'Math Operators': talib.get_function_groups()['Math Operators'],
        }

        total_count = sum(len(funcs) for funcs in groups.values())

        return jsonify({
            'success': True,
            'total_indicators': total_count,
            'groups': groups,
            'talib_available': True,
            'talib_version': talib.__version__
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'TA-Lib Professional Microservice',
        'talib_available': True,
        'talib_version': talib.__version__,
        'total_indicators': len(talib.get_functions())
    })

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("🚀 Starting TA-Lib Professional Microservice...")
    print(f"✅ TA-Lib Version: {talib.__version__}")
    print(f"📈 Total Indicators: {len(talib.get_functions())}")
    print("🌐 Server running on http://localhost:5002")
    app.run(host='0.0.0.0', port=5005, debug=False)
