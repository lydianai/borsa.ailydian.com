"""
RISK MANAGEMENT SERVICE
========================

Flask service for advanced risk management
Port: 5007

Endpoints:
- POST /risk/calculate - Calculate position metrics
- POST /risk/check - Check if position can be entered
- POST /risk/execute - Execute trade
- POST /risk/update - Update position (exit)
- GET /risk/metrics - Get portfolio risk metrics
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_management import AdvancedRiskManager

app = Flask(__name__)
CORS(app)

# Initialize risk manager
risk_manager = AdvancedRiskManager(
    initial_capital=10000.0,
    max_portfolio_risk=0.02,  # 2% per trade
    kelly_fraction=0.25,     # Quarter Kelly
    drawdown_limit=0.20       # 20% max drawdown
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Risk Management Service',
        'version': '1.0.0',
        'port': 5007
    })

@app.route('/risk/calculate', methods=['POST'])
def calculate_position():
    """
    Calculate position metrics
    
    Expected JSON:
    {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 45000.0,
        "win_rate": 0.65,
        "avg_win": 1200.0,
        "avg_loss": -600.0,
        "confidence": 0.85,
        "atr": 800.0,
        "correlation_group": ["BTCUSDT", "ETHUSDT"]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'symbol', 'direction', 'entry_price', 'win_rate', 
            'avg_win', 'avg_loss', 'confidence', 'atr'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Calculate metrics
        metrics = risk_manager.calculate_position_metrics(
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            win_rate=data['win_rate'],
            avg_win=data['avg_win'],
            avg_loss=data['avg_loss'],
            confidence=data['confidence'],
            atr=data['atr'],
            correlation_group=data.get('correlation_group', [data['symbol']])
        )
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/risk/check', methods=['POST'])
def check_position():
    """
    Check if position can be entered
    
    Expected JSON:
    {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 45000.0,
        "confidence": 0.85
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['symbol', 'direction', 'entry_price', 'confidence']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Check if position can be entered
        can_enter, reason = risk_manager.can_enter_position(
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            confidence=data['confidence']
        )
        
        return jsonify({
            'success': True,
            'can_enter': can_enter,
            'reason': reason
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/risk/execute', methods=['POST'])
def execute_trade():
    """
    Execute trade
    
    Expected JSON:
    {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry_price": 45000.0,
        "size": 0.1,
        "stop_loss": 44000.0,
        "take_profit": 47000.0,
        "confidence": 0.85
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = [
            'symbol', 'direction', 'entry_price', 'size',
            'stop_loss', 'take_profit', 'confidence'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Execute trade
        success = risk_manager.execute_trade(
            symbol=data['symbol'],
            direction=data['direction'],
            entry_price=data['entry_price'],
            size=data['size'],
            stop_loss=data['stop_loss'],
            take_profit=data['take_profit'],
            confidence=data['confidence']
        )
        
        return jsonify({
            'success': True,
            'executed': success
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/risk/update', methods=['POST'])
def update_position():
    """
    Update position (exit)
    
    Expected JSON:
    {
        "symbol": "BTCUSDT",
        "exit_price": 46000.0,
        "exit_type": "TAKE_PROFIT"  # STOP_LOSS, TAKE_PROFIT, MANUAL
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['symbol', 'exit_price', 'exit_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Update position
        result = risk_manager.update_position(
            symbol=data['symbol'],
            exit_price=data['exit_price'],
            exit_type=data['exit_type']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/risk/metrics', methods=['GET'])
def get_metrics():
    """
    Get portfolio risk metrics
    """
    try:
        metrics = risk_manager.get_portfolio_risk_metrics()
        
        return jsonify({
            'success': True,
            'metrics': {
                'portfolio_value': metrics.portfolio_value,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'risk_exposure': metrics.risk_exposure,
                'correlation_adjusted_risk': metrics.correlation_adjusted_risk
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/risk/reset-daily', methods=['POST'])
def reset_daily():
    """
    Reset daily counters
    """
    try:
        risk_manager.reset_daily_counters()
        
        return jsonify({
            'success': True,
            'message': 'Daily counters reset'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RISK MANAGEMENT SERVICE")
    print("="*60)
    print("🎯 Advanced Risk Management with Kelly Criterion")
    print("💰 Dynamic Position Sizing & Stop-Loss")
    print("📊 Portfolio Risk Monitoring")
    print("🌐 Server: http://localhost:5007")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5007, debug=True)