"""
TRANSFORMER AI TRADING SERVICE
=============================

Flask service for advanced transformer-based AI trading model
Port: 5011

Endpoints:
- POST /ai/predict - Make prediction with features
- POST /ai/update - Update model with real data
- GET /ai/attention - Get attention weights
- GET /ai/importance - Get feature importance
- GET /ai/stats - Get model statistics
- POST /ai/save - Save model
- POST /ai/load - Load model
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer_ai import TransformerTradingSystem, PredictionType

app = Flask(__name__)
CORS(app)

# Initialize transformer trading system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ai_system = TransformerTradingSystem(
    num_features=150,
    d_model=256,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dropout=0.1,
    max_seq_len=60,
    device=device
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Transformer AI Trading Service',
        'version': '1.0.0',
        'port': 5011,
        'device': device,
        'model_ready': True
    })

@app.route('/ai/predict', methods=['POST'])
def make_prediction():
    """
    Make prediction with transformer model
    
    Expected JSON:
    {
        "features": [[...]],  # 2D array: [seq_len, num_features]
        "return_attention": true,  # Optional
        "return_importance": true  # Optional
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: features'
            }), 400
        
        # Convert features to tensor
        features_np = np.array(data['features'], dtype=np.float32)
        
        # Reshape if needed (single sample)
        if len(features_np.shape) == 2:
            # Add batch dimension
            features_np = np.expand_dims(features_np, axis=0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_np).to(device)
        
        # Make prediction
        prediction = ai_system.predict(features_tensor)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'type': prediction.prediction.value,
                'confidence': float(prediction.confidence),
                'probabilities': prediction.probabilities.detach().cpu().numpy().tolist(),
                'timestamp': prediction.timestamp.isoformat()
            }
        }
        
        # Add attention weights if requested
        if data.get('return_attention', False):
            attention_weights = prediction.attention_weights.detach().cpu().numpy()
            response['attention_weights'] = attention_weights.tolist()
        
        # Add feature importance if requested
        if data.get('return_importance', False):
            feature_importance = prediction.feature_importance.detach().cpu().numpy()
            response['feature_importance'] = feature_importance.tolist()
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/update', methods=['POST'])
def update_model():
    """
    Update model with real trading data
    
    Expected JSON:
    {
        "features": [[[...]]],  # 3D array: [batch_size, seq_len, num_features]
        "targets": [...],       # 1D array: [batch_size]
        "timestamps": [...]     # Array of ISO timestamps
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['features', 'targets', 'timestamps']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Convert features to tensor
        features_np = np.array(data['features'], dtype=np.float32)
        features_tensor = torch.tensor(features_np).to(device)
        
        # Convert targets to tensor
        targets_np = np.array(data['targets'], dtype=np.int64)
        targets_tensor = torch.tensor(targets_np).to(device)
        
        # Convert timestamps
        timestamps = [pd.Timestamp(ts) for ts in data['timestamps']]
        
        # Update model
        metrics = ai_system.update_with_real_data(
            features_tensor, 
            targets_tensor, 
            timestamps
        )
        
        return jsonify({
            'success': True,
            'metrics': {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in metrics.items()
            },
            'message': 'Model updated successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/attention', methods=['POST'])
def get_attention():
    """
    Get attention weights for visualization
    
    Expected JSON:
    {
        "features": [[...]]  # 2D array: [seq_len, num_features]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: features'
            }), 400
        
        # Convert features to tensor
        features_np = np.array(data['features'], dtype=np.float32)
        
        # Reshape if needed (single sample)
        if len(features_np.shape) == 2:
            # Add batch dimension
            features_np = np.expand_dims(features_np, axis=0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_np).to(device)
        
        # Get attention weights
        attention_weights = ai_system.get_attention_visualization(features_tensor)
        attention_np = attention_weights.detach().cpu().numpy()
        
        return jsonify({
            'success': True,
            'attention_weights': attention_np.tolist(),
            'shape': attention_np.shape
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/importance', methods=['POST'])
def get_importance():
    """
    Get feature importance scores
    
    Expected JSON:
    {
        "features": [[...]]  # 2D array: [seq_len, num_features]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'features' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: features'
            }), 400
        
        # Convert features to tensor
        features_np = np.array(data['features'], dtype=np.float32)
        
        # Reshape if needed (single sample)
        if len(features_np.shape) == 2:
            # Add batch dimension
            features_np = np.expand_dims(features_np, axis=0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_np).to(device)
        
        # Get feature importance
        importance_scores = ai_system.get_feature_importance(features_tensor)
        importance_np = importance_scores.detach().cpu().numpy()
        
        return jsonify({
            'success': True,
            'feature_importance': importance_np.tolist(),
            'shape': importance_np.shape
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/stats', methods=['GET'])
def get_statistics():
    """
    Get model statistics
    """
    try:
        stats = ai_system.get_model_statistics()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        stats_serializable = convert_numpy(stats)
        
        return jsonify({
            'success': True,
            'statistics': stats_serializable
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/save', methods=['POST'])
def save_model():
    """
    Save model to file
    
    Expected JSON:
    {
        "filepath": "./models/transformer_model.pth"  # Optional
    }
    """
    try:
        data = request.json or {}
        filepath = data.get('filepath', './transformer_model.pth')
        
        # Save system
        ai_system.save_system(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Model saved successfully to {filepath}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/load', methods=['POST'])
def load_model():
    """
    Load model from file
    
    Expected JSON:
    {
        "filepath": "./models/transformer_model.pth"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'filepath' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: filepath'
            }), 400
        
        filepath = data['filepath']
        
        # Load system
        ai_system.load_system(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully from {filepath}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/history', methods=['GET'])
def get_prediction_history():
    """
    Get prediction history
    
    Query parameters:
    - limit: Number of predictions to return (default: 50)
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        # Get recent predictions
        recent_predictions = ai_system.prediction_history[-limit:] if ai_system.prediction_history else []
        
        # Convert to serializable format
        predictions_json = []
        for pred in recent_predictions:
            pred_json = {
                'prediction': pred.prediction.value,
                'confidence': float(pred.confidence),
                'timestamp': pred.timestamp.isoformat(),
                'metadata': pred.metadata
            }
            predictions_json.append(pred_json)
        
        return jsonify({
            'success': True,
            'predictions': predictions_json,
            'count': len(predictions_json),
            'total_history': len(ai_system.prediction_history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/ai/training-history', methods=['GET'])
def get_training_history():
    """
    Get training history
    
    Query parameters:
    - limit: Number of training records to return (default: 20)
    """
    try:
        limit = int(request.args.get('limit', 20))
        
        # Get recent training history
        recent_history = ai_system.training_history[-limit:] if ai_system.training_history else []
        
        # Convert to serializable format
        history_json = []
        for record in recent_history:
            record_json = {
                'timestamp': record['timestamp'].isoformat(),
                'metrics': {
                    key: float(value) if isinstance(value, (int, float)) else value
                    for key, value in record['metrics'].items()
                }
            }
            history_json.append(record_json)
        
        return jsonify({
            'success': True,
            'training_history': history_json,
            'count': len(history_json),
            'total_history': len(ai_system.training_history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 TRANSFORMER AI TRADING SERVICE")
    print("="*60)
    print("🎯 Advanced Transformer-Based AI Trading Model")
    print("🧠 Multi-Head Attention + Positional Encoding")
    print("📊 Feature Embedding + Sequence-to-Sequence Prediction")
    print("🔄 Online Learning + Experience Replay")
    print("🌐 Server: http://localhost:5011")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5011, debug=True)