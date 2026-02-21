"""
AI Models Service - Minimal Mode
Basic Flask server for testing proxy API without heavy AI dependencies
Full AI models will be added after installing PyTorch, XGBoost, etc.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Available models (will be fully implemented later)
AVAILABLE_MODELS = [
    'lstm_standard', 'lstm_bidirectional', 'lstm_stacked',
    'gru_standard', 'gru_bidirectional', 'gru_stacked', 'gru_attention', 'gru_residual',
    'transformer_standard', 'transformer_timeseries', 'transformer_informer',
    'cnn_standard', 'cnn_resnet', 'cnn_multiscale', 'cnn_dilated', 'cnn_tcn',
    'xgboost', 'lightgbm', 'catboost'
]

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'AI Models Service (Minimal Mode)',
        'mode': 'minimal',
        'available_models': len(AVAILABLE_MODELS),
        'models': AVAILABLE_MODELS,
        'message': 'Service is running in minimal mode. AI models will be loaded after installing PyTorch, XGBoost, etc.',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/single', methods=['POST'])
def predict_single():
    """
    Single prediction endpoint (mock data for now)

    Expected body:
    {
        "symbol": "BTCUSDT",
        "model": "lstm_standard",  // optional
        "timeframe": "1h",         // optional
        "data": {...}              // optional OHLCV data
    }
    """
    data = request.get_json()
    symbol = data.get('symbol')
    model = data.get('model', 'consensus')
    timeframe = data.get('timeframe', '1h')

    if not symbol:
        return jsonify({'error': 'symbol is required'}), 400

    # Mock prediction (will be real AI predictions later)
    mock_signals = ['BUY', 'SELL', 'HOLD']
    prediction = random.choice(mock_signals)
    confidence = round(random.uniform(0.5, 0.95), 2)

    # Mock predictions for all models
    model_predictions = []
    for m in AVAILABLE_MODELS[:10]:  # Show 10 model predictions
        model_predictions.append({
            'model': m,
            'prediction': random.choice(mock_signals),
            'confidence': round(random.uniform(0.5, 0.95), 2),
            'weight': round(random.uniform(0.8, 1.0), 2)
        })

    return jsonify({
        'symbol': symbol,
        'timeframe': timeframe,
        'prediction': prediction,
        'confidence': confidence,
        'consensus': {
            'signal': prediction,
            'score': confidence,
            'model_count': len(model_predictions)
        },
        'model_predictions': model_predictions,
        'mode': 'minimal',
        'note': 'This is mock data. Real AI predictions will be available after full setup.',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint (mock data for now)

    Expected body:
    {
        "symbols": ["BTCUSDT", "ETHUSDT", ...],
        "model": "lstm_standard",  // optional
        "timeframe": "1h"          // optional
    }
    """
    data = request.get_json()
    symbols = data.get('symbols', [])
    model = data.get('model', 'consensus')
    timeframe = data.get('timeframe', '1h')

    if not symbols or len(symbols) == 0:
        return jsonify({'error': 'symbols array is required'}), 400

    if len(symbols) > 100:
        return jsonify({'error': 'Maximum 100 symbols per batch'}), 400

    # Mock predictions for each symbol
    predictions = []
    for symbol in symbols:
        mock_signals = ['BUY', 'SELL', 'HOLD']
        prediction = random.choice(mock_signals)
        confidence = round(random.uniform(0.5, 0.95), 2)

        predictions.append({
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence,
            'timeframe': timeframe
        })

    return jsonify({
        'predictions': predictions,
        'total': len(predictions),
        'model': model,
        'timeframe': timeframe,
        'mode': 'minimal',
        'note': 'Mock batch predictions. Real AI predictions coming after full setup.',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict/top100', methods=['GET'])
def predict_top100():
    """
    Top 100 coins predictions (mock data)
    """
    # Mock top 100 symbols
    top_symbols = [f"COIN{i}USDT" for i in range(1, 101)]

    predictions = []
    for symbol in top_symbols[:20]:  # Return 20 for now
        mock_signals = ['BUY', 'SELL', 'HOLD']
        predictions.append({
            'symbol': symbol,
            'prediction': random.choice(mock_signals),
            'confidence': round(random.uniform(0.5, 0.95), 2),
            'score': round(random.uniform(60, 95), 1)
        })

    # Sort by score
    predictions.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({
        'predictions': predictions,
        'total': len(predictions),
        'mode': 'minimal',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List all available AI models"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'total': len(AVAILABLE_MODELS),
        'categories': {
            'LSTM': 3,
            'GRU': 5,
            'Transformer': 3,
            'CNN': 5,
            'Gradient Boosting': 3
        },
        'status': 'Models will be loaded after full dependencies installation',
        'mode': 'minimal'
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 AI Models Service - Minimal Mode")
    print("=" * 50)
    print(f"✅ {len(AVAILABLE_MODELS)} AI Models registered")
    print("📝 Mode: MINIMAL (Mock predictions)")
    print("🔧 For full AI predictions, install:")
    print("   pip install torch transformers xgboost lightgbm catboost")
    print("=" * 50)
    print("🌐 Starting server on http://localhost:5003")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5003, debug=True)
