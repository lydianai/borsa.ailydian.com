"""
AI PREDICTION SERVICE - MINIMAL (Feature-Enhanced Only)
Lightweight implementation without PyTorch dependency
Uses FeatureEnhancedPredictor for 150+ features
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

# NEW: Feature-enhanced predictor for 150+ features
from feature_enhanced_predictor import FeatureEnhancedPredictor

app = Flask(__name__)
CORS(app)

# Initialize feature-enhanced predictor (lightweight, no PyTorch required)
feature_predictor = FeatureEnhancedPredictor()

# ============================================
# HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Prediction Service (Minimal - Feature-Enhanced)',
        'predictor': 'FeatureEnhancedPredictor',
        'pytorch_required': False
    })

# ============================================
# PREDICTIONS
# ============================================

@app.route('/predict', methods=['POST'])
def predict_with_features():
    """
    Prediction with 150+ features from Feature Engineering Service

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

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 AI PREDICTION SERVICE - MINIMAL (Feature-Enhanced)")
    print("=" * 60 + "\n")
    print("✅ Feature-Enhanced Predictor Ready")
    print("📊 Supports 150+ features")
    print("⚡ Lightweight - No PyTorch required")
    print("🌐 Server: http://localhost:5003")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5003, debug=False)
