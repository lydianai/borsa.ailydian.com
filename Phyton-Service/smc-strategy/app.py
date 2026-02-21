"""
SMART MONEY CONCEPTS (SMC) STRATEGY SERVICE
=========================================

Flask service for Smart Money Concepts trading strategy
Port: 5008

Endpoints:
- POST /smc/scan - Scan for SMC levels
- POST /smc/setups - Generate trade setups
- GET /smc/levels - Get current SMC levels
- GET /smc/stats - Get level statistics
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smc_strategy import SmartMoneyConcepts, SMCLevelType

app = Flask(__name__)
CORS(app)

# Initialize SMC analyzer
smc_analyzer = SmartMoneyConcepts()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Money Concepts Strategy Service',
        'version': '1.0.0',
        'port': 5008
    })

@app.route('/smc/scan', methods=['POST'])
def scan_levels():
    """
    Scan for SMC levels in provided OHLCV data
    
    Expected JSON:
    {
        "data": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 40000.0,
                "high": 41000.0,
                "low": 39500.0,
                "close": 40500.0,
                "volume": 1000.0
            }
        ],
        "symbol": "BTCUSDT"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        if 'data' not in data or 'symbol' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: data, symbol'
            }), 400
        
        # Convert to DataFrame
        df_data = []
        for item in data['data']:
            df_data.append({
                'timestamp': pd.to_datetime(item['timestamp']),
                'open': float(item['open']),
                'high': float(item['high']),
                'low': float(item['low']),
                'close': float(item['close']),
                'volume': float(item['volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.attrs['symbol'] = data['symbol']
        
        # Scan for levels
        levels = smc_analyzer.scan_all_levels(df)
        
        # Convert levels to JSON-serializable format
        levels_json = []
        for level in levels:
            levels_json.append({
                'type': level.level_type.value,
                'price': float(level.price),
                'strength': float(level.strength),
                'confidence': float(level.confidence),
                'timestamp': level.timestamp.isoformat(),
                'timeframe': level.timeframe,
                'touched_count': level.touched_count,
                'broken_count': level.broken_count,
                'last_touched': level.last_touched.isoformat() if level.last_touched else None
            })
        
        return jsonify({
            'success': True,
            'levels': levels_json,
            'count': len(levels),
            'symbol': data['symbol']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/smc/setups', methods=['POST'])
def generate_setups():
    """
    Generate trade setups based on current SMC levels
    
    Expected JSON:
    {
        "data": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 40000.0,
                "high": 41000.0,
                "low": 39500.0,
                "close": 40500.0,
                "volume": 1000.0
            }
        ],
        "current_price": 40500.0,
        "symbol": "BTCUSDT"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['data', 'current_price', 'symbol']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Convert to DataFrame
        df_data = []
        for item in data['data']:
            df_data.append({
                'timestamp': pd.to_datetime(item['timestamp']),
                'open': float(item['open']),
                'high': float(item['high']),
                'low': float(item['low']),
                'close': float(item['close']),
                'volume': float(item['volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.attrs['symbol'] = data['symbol']
        
        current_price = float(data['current_price'])
        
        # Generate setups
        setups = smc_analyzer.generate_trade_setups(df, current_price)
        
        # Convert setups to JSON-serializable format
        setups_json = []
        for setup in setups:
            setups_json.append({
                'symbol': setup.symbol,
                'direction': setup.direction,
                'entry_zone': {
                    'lower': float(setup.entry_zone[0]),
                    'upper': float(setup.entry_zone[1])
                },
                'stop_loss': float(setup.stop_loss),
                'take_profit_1': float(setup.take_profit_1),
                'take_profit_2': float(setup.take_profit_2),
                'take_profit_3': float(setup.take_profit_3),
                'confidence': float(setup.confidence),
                'setup_score': float(setup.setup_score),
                'levels_used': [
                    {
                        'type': level.level_type.value,
                        'price': float(level.price),
                        'strength': float(level.strength),
                        'confidence': float(level.confidence)
                    }
                    for level in setup.levels_used
                ],
                'timestamp': setup.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'setups': setups_json,
            'count': len(setups),
            'symbol': data['symbol']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/smc/levels', methods=['GET'])
def get_levels():
    """
    Get current SMC levels
    """
    try:
        # Convert levels to JSON-serializable format
        levels_json = []
        for level in smc_analyzer.levels:
            levels_json.append({
                'type': level.level_type.value,
                'price': float(level.price),
                'strength': float(level.strength),
                'confidence': float(level.confidence),
                'timestamp': level.timestamp.isoformat(),
                'timeframe': level.timeframe,
                'touched_count': level.touched_count,
                'broken_count': level.broken_count,
                'last_touched': level.last_touched.isoformat() if level.last_touched else None
            })
        
        return jsonify({
            'success': True,
            'levels': levels_json,
            'count': len(levels_json)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/smc/stats', methods=['GET'])
def get_stats():
    """
    Get SMC level statistics
    """
    try:
        stats = smc_analyzer.get_level_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/smc/update', methods=['POST'])
def update_levels():
    """
    Update levels with recent price action
    
    Expected JSON:
    {
        "data": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 40000.0,
                "high": 41000.0,
                "low": 39500.0,
                "close": 40500.0,
                "volume": 1000.0
            }
        ],
        "current_price": 40500.0
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['data', 'current_price']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Convert to DataFrame
        df_data = []
        for item in data['data']:
            df_data.append({
                'timestamp': pd.to_datetime(item['timestamp']),
                'open': float(item['open']),
                'high': float(item['high']),
                'low': float(item['low']),
                'close': float(item['close']),
                'volume': float(item['volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        current_price = float(data['current_price'])
        
        # Update levels
        smc_analyzer.update_levels_with_price_action(df, current_price)
        
        return jsonify({
            'success': True,
            'message': 'Levels updated successfully',
            'updated_count': len(smc_analyzer.levels)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SMART MONEY CONCEPTS STRATEGY SERVICE")
    print("="*60)
    print("🎯 Advanced SMC Trading Strategy Implementation")
    print("📊 Liquidity Zones, Order Blocks, Structural Levels")
    print("💰 Premium/Discount Zones, Mitigation Blocks")
    print("🌐 Server: http://localhost:5008")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5008, debug=True)