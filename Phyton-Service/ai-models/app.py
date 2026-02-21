"""
AI PREDICTION SERVICE
Flask API for 100+ AI models
Professional implementation with model management
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add directories to path
sys.path.append(os.path.dirname(__file__))

from time_series.lstm.standard_lstm import create_lstm_model, StandardLSTM
from time_series.gru.standard_gru import create_gru_model
from time_series.transformer.standard_transformer import create_transformer_model
from pattern_recognition.cnn.standard_cnn import create_cnn_model
from pattern_recognition.gradient_boosting.boosting_models import create_boosting_model
from training.data_loader import CryptoDataLoader
from base_model import BaseEnsembleModel

# NEW: Feature-enhanced predictor for 150+ features
from feature_enhanced_predictor import FeatureEnhancedPredictor

import numpy as np
import torch

app = Flask(__name__)
CORS(app)

# Initialize data loader
data_loader = CryptoDataLoader()

# Model registry
MODELS = {}

# Initialize feature-enhanced predictor (lightweight, no PyTorch required)
feature_predictor = FeatureEnhancedPredictor()

def initialize_models():
    """
    Initialize default AI models
    """
    global MODELS

    print("🚀 Initializing AI Models...")

    # Create LSTM variants (8 features: OHLCV + derived features)
    try:
        MODELS['lstm_standard'] = create_lstm_model('standard', input_size=8)
        MODELS['lstm_bidirectional'] = create_lstm_model('bidirectional', input_size=8)
        MODELS['lstm_stacked'] = create_lstm_model('stacked', input_size=8)

        print(f"✅ Initialized {len(MODELS)} LSTM models")
    except Exception as e:
        print(f"⚠️  Error initializing LSTM models: {e}")

    # Create GRU variants (8 features)
    try:
        MODELS['gru_standard'] = create_gru_model('standard', input_size=8)
        MODELS['gru_bidirectional'] = create_gru_model('bidirectional', input_size=8)
        MODELS['gru_stacked'] = create_gru_model('stacked', input_size=8)
        MODELS['gru_attention'] = create_gru_model('attention', input_size=8)
        MODELS['gru_residual'] = create_gru_model('residual', input_size=8)

        print(f"✅ Initialized {len(MODELS)} models (LSTM + GRU)")
    except Exception as e:
        print(f"⚠️  Error initializing GRU models: {e}")

    # Create Transformer variants (8 features)
    try:
        MODELS['transformer_standard'] = create_transformer_model('standard', input_size=8)
        MODELS['transformer_timeseries'] = create_transformer_model('timeseries', input_size=8)
        MODELS['transformer_informer'] = create_transformer_model('informer', input_size=8)

        print(f"✅ Initialized {len(MODELS)} models (LSTM + GRU + Transformer)")
    except Exception as e:
        print(f"⚠️  Error initializing Transformer models: {e}")

    # Create CNN variants (8 features)
    try:
        MODELS['cnn_standard'] = create_cnn_model('standard', input_channels=8)
        MODELS['cnn_resnet'] = create_cnn_model('resnet', input_channels=8)
        MODELS['cnn_multiscale'] = create_cnn_model('multiscale', input_channels=8)
        MODELS['cnn_dilated'] = create_cnn_model('dilated', input_channels=8)
        MODELS['cnn_tcn'] = create_cnn_model('tcn', input_channels=8)

        print(f"✅ Initialized {len(MODELS)} models (LSTM + GRU + Transformer + CNN)")
    except Exception as e:
        print(f"⚠️  Error initializing CNN models: {e}")

    # Create Gradient Boosting variants
    try:
        MODELS['xgboost'] = create_boosting_model('xgboost')
        MODELS['lightgbm'] = create_boosting_model('lightgbm')
        MODELS['catboost'] = create_boosting_model('catboost')

        print(f"✅ Initialized {len(MODELS)} total models (All Categories)")
    except Exception as e:
        print(f"⚠️  Error initializing Gradient Boosting models: {e}")

# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Prediction Service',
        'models_loaded': len(MODELS),
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    })

# ============================================
# MODEL MANAGEMENT
# ============================================

@app.route('/models/list', methods=['GET'])
def list_models():
    """List all available models"""
    models_info = []

    for name, model in MODELS.items():
        models_info.append({
            'name': name,
            'type': model.model_type,
            'version': model.version,
            'parameters': model.count_parameters(),
            'metrics': model.get_metrics()
        })

    return jsonify({
        'success': True,
        'total_models': len(MODELS),
        'models': models_info
    })

@app.route('/models/<model_id>/status', methods=['GET'])
def model_status(model_id):
    """Get model status and metrics"""
    if model_id not in MODELS:
        return jsonify({'success': False, 'error': 'Model not found'}), 404

    model = MODELS[model_id]

    return jsonify({
        'success': True,
        'model': model.get_metrics()
    })

# ============================================
# PREDICTIONS
# ============================================

@app.route('/predict', methods=['POST'])
def predict_with_features():
    """
    NEW: Prediction with 150+ features from Feature Engineering Service

    This endpoint integrates with the Feature Engineering service and uses
    the FeatureEnhancedPredictor for lightweight predictions.

    Request body:
    {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "features": [
            {
                "close": 43000,
                "rsi_14": 65.4,
                "macd_hist": 120.5,
                ...  // 144 features total
            }
        ],
        "feature_count": 144
    }

    Response:
    {
        "success": true,
        "symbol": "BTCUSDT",
        "prediction": {
            "signal": "BUY",
            "confidence": 0.75,
            "strength": 7
        },
        "current_price": 43000,
        "reasoning": "Strong bullish signals...",
        "models_used": ["feature_enhanced_ensemble"],
        "score": 0.75,
        "feature_count": 144
    }
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        interval = data.get('interval', '1h')
        features = data.get('features')
        feature_count = data.get('feature_count', 0)

        if not symbol:
            return jsonify({
                'success': False,
                'error': 'Symbol required'
            }), 400

        # If features provided, use feature-enhanced predictor
        if features and len(features) > 0:
            print(f"📊 Using Feature-Enhanced Predictor for {symbol} with {len(features[0])} features")

            result = feature_predictor.predict_from_features(
                feature_data=features,
                symbol=symbol,
                interval=interval
            )

            return jsonify({
                'success': True,
                **result
            })

        # Fallback: No features provided
        else:
            print(f"⚠️  No features provided for {symbol}, using fallback")
            return jsonify({
                'success': True,
                'symbol': symbol,
                'prediction': {
                    'signal': 'HOLD',
                    'confidence': 0.5,
                    'strength': 5
                },
                'current_price': 0,
                'reasoning': 'No features provided - using neutral prediction',
                'models_used': ['fallback'],
                'score': 0.5,
                'feature_count': 0
            })

    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict/single', methods=['POST'])
def predict_single():
    """
    Single coin prediction

    Request body:
    {
        "symbol": "BTC",
        "timeframe": "1h",
        "model": "lstm_standard"  # optional, defaults to ensemble
    }
    """
    try:
        data = request.json
        symbol = data.get('symbol')
        timeframe = data.get('timeframe', '1h')
        model_id = data.get('model', 'ensemble')

        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol required'}), 400

        # Load data
        print(f"📊 Loading data for {symbol}...")
        comprehensive_data = data_loader.load_comprehensive_data(symbol, [timeframe])

        if not comprehensive_data:
            return jsonify({'success': False, 'error': 'Failed to load data'}), 404

        # Prepare data
        X, _ = data_loader.prepare_training_data(symbol, timeframe, sequence_length=60)

        if X is None or len(X) == 0:
            return jsonify({'success': False, 'error': 'Insufficient data'}), 404

        # Use last sequence for prediction
        last_sequence = X[-1]

        # Make prediction
        if model_id == 'ensemble':
            # Ensemble prediction (all models)
            ensemble = BaseEnsembleModel(list(MODELS.values()))
            result = ensemble.predict(last_sequence)
        elif model_id in MODELS:
            # Single model prediction
            model = MODELS[model_id]
            result = model.predict(last_sequence)
        else:
            return jsonify({'success': False, 'error': f'Model {model_id} not found'}), 404

        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': result,
            'model_used': model_id,
            'timestamp': comprehensive_data.get('lastUpdate')
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch predictions for multiple coins

    Request body:
    {
        "symbols": ["BTC", "ETH", "BNB"],
        "timeframe": "1h",
        "model": "ensemble"
    }
    """
    try:
        data = request.json
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1h')
        model_id = data.get('model', 'ensemble')

        if not symbols:
            return jsonify({'success': False, 'error': 'Symbols required'}), 400

        results = []

        for symbol in symbols:
            try:
                # Load and predict
                comprehensive_data = data_loader.load_comprehensive_data(symbol, [timeframe])

                if not comprehensive_data:
                    continue

                X, _ = data_loader.prepare_training_data(symbol, timeframe, sequence_length=60)

                if X is None or len(X) == 0:
                    continue

                last_sequence = X[-1]

                # Make prediction
                if model_id == 'ensemble':
                    ensemble = BaseEnsembleModel(list(MODELS.values()))
                    result = ensemble.predict(last_sequence)
                else:
                    model = MODELS.get(model_id)
                    if model:
                        result = model.predict(last_sequence)
                    else:
                        continue

                results.append({
                    'symbol': symbol,
                    'prediction': result,
                    'current_price': comprehensive_data['coin']['price']
                })

            except Exception as e:
                print(f"⚠️  Error predicting {symbol}: {e}")
                continue

        return jsonify({
            'success': True,
            'timeframe': timeframe,
            'model_used': model_id,
            'predictions': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict/top100', methods=['GET'])
def predict_top100():
    """
    Predictions for Top 100 coins

    Query params:
    - timeframe: '1h' (default)
    - limit: 10 (default)
    - model: 'ensemble' (default)
    """
    try:
        timeframe = request.args.get('timeframe', '1h')
        limit = int(request.args.get('limit', 10))
        model_id = request.args.get('model', 'ensemble')

        # Load Top N data
        print(f"📊 Loading Top {limit} coins...")
        top_data = data_loader.load_top100_data(timeframe, limit)

        if not top_data:
            return jsonify({'success': False, 'error': 'Failed to load data'}), 404

        results = []

        for coin_data in top_data:
            try:
                symbol = coin_data['coin']['symbol']

                # Prepare data
                X, _ = data_loader.prepare_training_data(symbol, timeframe, sequence_length=60)

                if X is None or len(X) == 0:
                    continue

                last_sequence = X[-1]

                # Make prediction
                if model_id == 'ensemble':
                    ensemble = BaseEnsembleModel(list(MODELS.values()))
                    result = ensemble.predict(last_sequence)
                else:
                    model = MODELS.get(model_id)
                    if model:
                        result = model.predict(last_sequence)
                    else:
                        continue

                results.append({
                    'symbol': symbol,
                    'name': coin_data['coin']['name'],
                    'price': coin_data['coin']['price'],
                    'prediction': result
                })

            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['prediction']['confidence'], reverse=True)

        return jsonify({
            'success': True,
            'timeframe': timeframe,
            'model_used': model_id,
            'predictions': results,
            'total': len(results)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 AI PREDICTION SERVICE - STARTING")
    print("=" * 60 + "\n")

    # Initialize models
    initialize_models()

    print("\n" + "=" * 60)
    print("✅ AI PREDICTION SERVICE - READY")
    print(f"📊 Models Loaded: {len(MODELS)}")
    print(f"🔧 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("🌐 Server: http://localhost:5003")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5003, debug=False)
