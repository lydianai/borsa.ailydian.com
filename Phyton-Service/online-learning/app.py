"""
ONLINE LEARNING PIPELINE SERVICE
===============================

Flask service for advanced online learning pipeline
Port: 5012

Endpoints:
- POST /online-learning/experience - Add learning experience
- POST /online-learning/update - Update model
- GET /online-learning/performance - Get performance metrics
- GET /online-learning/drift - Get drift detection
- GET /online-learning/versions - Get model versions
- POST /online-learning/save - Save pipeline
- POST /online-learning/load - Load pipeline
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from online_learning import OnlineLearningPipeline, DriftType

app = Flask(__name__)
CORS(app)

# Initialize online learning pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create sample model (in practice, this would be your actual model)
class SampleTradingModel(nn.Module):
    def __init__(self, input_size=150, hidden_size=256, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        if len(x.shape) == 3:  # (batch, seq, features)
            x = x[:, -1, :]  # Take last timestep
        return self.network(x)

# Initialize model and components
model = SampleTradingModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Initialize pipeline
pipeline = OnlineLearningPipeline(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    buffer_size=10000,
    learning_rate=0.001,
    update_frequency=100,
    performance_window=1000,
    device=device
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Online Learning Pipeline Service',
        'version': '1.0.0',
        'port': 5012,
        'device': device,
        'pipeline_ready': True
    })

@app.route('/online-learning/experience', methods=['POST'])
def add_experience():
    """
    Add learning experience to pipeline
    
    Expected JSON:
    {
        "features": [[...]],  # 2D or 3D array: [batch_size, seq_len, num_features] or [seq_len, num_features]
        "targets": [...],     # 1D array: [batch_size] or scalar
        "predictions": [[...]], # Optional: [batch_size, num_classes] or [num_classes]
        "confidence": 0.85,   # Optional: scalar
        "metadata": {...}     # Optional: metadata dictionary
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'features' not in data or 'targets' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: features, targets'
            }), 400
        
        # Convert features to tensor
        features_np = np.array(data['features'], dtype=np.float32)
        
        # Reshape if needed (single sample)
        if len(features_np.shape) == 2:
            # Add batch dimension
            features_np = np.expand_dims(features_np, axis=0)
        
        # Convert to tensor
        features_tensor = torch.tensor(features_np).to(device)
        
        # Convert targets to tensor
        targets_np = np.array(data['targets'], dtype=np.int64)
        
        # Reshape if needed (single target)
        if len(targets_np.shape) == 0:
            targets_np = np.expand_dims(targets_np, axis=0)
        
        # Convert to tensor
        targets_tensor = torch.tensor(targets_np).to(device)
        
        # Convert predictions if provided
        predictions_tensor = None
        if 'predictions' in data:
            predictions_np = np.array(data['predictions'], dtype=np.float32)
            
            # Reshape if needed (single prediction)
            if len(predictions_np.shape) == 1:
                predictions_np = np.expand_dims(predictions_np, axis=0)
            
            predictions_tensor = torch.tensor(predictions_np).to(device)
        
        # Add experience
        pipeline.add_experience(
            features=features_tensor,
            targets=targets_tensor,
            predictions=predictions_tensor,
            confidence=data.get('confidence'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'success': True,
            'message': 'Experience added successfully',
            'total_experiences': pipeline.total_experiences
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/update', methods=['POST'])
def update_model():
    """
    Update model with accumulated experiences
    
    Expected JSON:
    {
        "force_update": true  # Optional: force update regardless of frequency
    }
    """
    try:
        data = request.json or {}
        force_update = data.get('force_update', False)
        
        # Update model
        result = pipeline.update_model(force_update=force_update)
        
        return jsonify({
            'success': True,
            'result': {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in result.items()
            },
            'message': 'Model update completed'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/performance', methods=['GET'])
def get_performance():
    """
    Get current performance metrics
    """
    try:
        # Get performance metrics
        metrics = pipeline.get_performance_metrics()
        
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
        
        metrics_serializable = convert_numpy(metrics)
        
        return jsonify({
            'success': True,
            'performance': metrics_serializable
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/drift', methods=['GET'])
def get_drift():
    """
    Get concept drift detection results
    
    Query parameters:
    - limit: Number of recent drift detections to return (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))
        
        # Detect drift
        drift_result = pipeline.drift_detector.detect_drift()
        
        # Get drift history
        drift_history = list(pipeline.drift_detector.drift_history)[-limit:] if pipeline.drift_detector.drift_history else []
        
        # Convert to serializable format
        history_serializable = []
        for drift in drift_history:
            drift_dict = {
                'drift_type': drift.drift_type.value,
                'drift_probability': float(drift.drift_probability),
                'drift_timestamp': drift.drift_timestamp.isoformat(),
                'affected_features': drift.affected_features,
                'severity': float(drift.severity),
                'recommendation': drift.recommendation,
                'metadata': drift.metadata
            }
            history_serializable.append(drift_dict)
        
        return jsonify({
            'success': True,
            'current_drift': {
                'drift_type': drift_result.drift_type.value,
                'drift_probability': float(drift_result.drift_probability),
                'drift_timestamp': drift_result.drift_timestamp.isoformat(),
                'affected_features': drift_result.affected_features,
                'severity': float(drift_result.severity),
                'recommendation': drift_result.recommendation,
                'metadata': drift_result.metadata
            },
            'drift_history': history_serializable,
            'history_count': len(history_serializable),
            'total_drifts': len(pipeline.drift_detector.drift_history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/versions', methods=['GET'])
def get_versions():
    """
    Get model version information
    
    Query parameters:
    - limit: Number of versions to return (default: 10)
    - status: Filter by status (active, archived, training, etc.)
    """
    try:
        limit = int(request.args.get('limit', 10))
        status_filter = request.args.get('status')
        
        # Get version statistics
        version_stats = pipeline.version_manager.get_version_statistics()
        
        # Get recent versions
        recent_versions = []
        for version_id in list(pipeline.version_manager.active_versions)[-limit:]:
            if version_id in pipeline.version_manager.versions:
                version = pipeline.version_manager.versions[version_id]
                
                # Apply status filter if provided
                if status_filter and version.status.value != status_filter:
                    continue
                
                version_dict = {
                    'version_id': version.version_id,
                    'status': version.status.value,
                    'training_timestamp': version.training_timestamp.isoformat(),
                    'experience_count': version.experience_count,
                    'parent_version': version.parent_version,
                    'performance_metrics': {
                        key: float(value) if isinstance(value, (int, float)) else value
                        for key, value in version.performance_metrics.items()
                    },
                    'metadata': version.metadata
                }
                recent_versions.append(version_dict)
        
        return jsonify({
            'success': True,
            'version_statistics': version_stats,
            'versions': recent_versions,
            'count': len(recent_versions)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/save', methods=['POST'])
def save_pipeline():
    """
    Save pipeline to file
    
    Expected JSON:
    {
        "filepath": "./pipelines/online_learning_pipeline.pkl"  # Optional
    }
    """
    try:
        data = request.json or {}
        filepath = data.get('filepath', './online_learning_pipeline.pkl')
        
        # Save pipeline
        pipeline.save_pipeline(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Pipeline saved successfully to {filepath}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/load', methods=['POST'])
def load_pipeline():
    """
    Load pipeline from file
    
    Expected JSON:
    {
        "filepath": "./pipelines/online_learning_pipeline.pkl"
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
        
        # Load pipeline
        pipeline.load_pipeline(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Pipeline loaded successfully from {filepath}',
            'filepath': filepath
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/buffer', methods=['GET'])
def get_buffer_statistics():
    """
    Get experience replay buffer statistics
    """
    try:
        # Get buffer statistics
        buffer_stats = pipeline.replay_buffer.get_statistics()
        
        return jsonify({
            'success': True,
            'buffer_statistics': {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in buffer_stats.items()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/online-learning/history', methods=['GET'])
def get_performance_history():
    """
    Get performance history
    
    Query parameters:
    - limit: Number of records to return (default: 50)
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        # Get recent performance history
        recent_history = list(pipeline.performance_history)[-limit:] if pipeline.performance_history else []
        
        # Convert to serializable format
        history_serializable = []
        for record in recent_history:
            record_dict = {
                'timestamp': record['timestamp'].isoformat(),
                'loss': float(record['loss']),
                'accuracy': float(record['accuracy']),
                'experience_count': record['experience_count']
            }
            history_serializable.append(record_dict)
        
        return jsonify({
            'success': True,
            'performance_history': history_serializable,
            'count': len(history_serializable),
            'total_history': len(pipeline.performance_history)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ONLINE LEARNING PIPELINE SERVICE")
    print("="*60)
    print("🎯 Advanced Online Learning for AI Trading Models")
    print("🧠 Experience Replay + Prioritized Sampling")
    print("📊 Concept Drift Detection + Model Versioning")
    print("🔄 Continuous Model Improvement + A/B Testing")
    print("🌐 Server: http://localhost:5012")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5012, debug=True)