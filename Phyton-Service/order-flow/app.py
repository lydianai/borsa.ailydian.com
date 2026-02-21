"""
ORDER FLOW ANALYSIS SERVICE
==========================

Flask service for advanced order flow analysis
Port: 5009

Endpoints:
- POST /orderflow/tick - Process tick data
- GET /orderflow/profile - Get volume profile
- GET /orderflow/microstructure - Get market microstructure
- GET /orderflow/events - Get recent events
- GET /orderflow/insights - Get comprehensive insights
- POST /orderflow/reset - Reset session data
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

from order_flow import OrderFlowAnalyzer

app = Flask(__name__)
CORS(app)

# Initialize order flow analyzer
ofa = OrderFlowAnalyzer()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Order Flow Analysis Service',
        'version': '1.0.0',
        'port': 5009
    })

@app.route('/orderflow/tick', methods=['POST'])
def process_tick():
    """
    Process tick data
    
    Expected JSON:
    {
        "price": 40000.0,
        "volume": 0.5,
        "side": "buy",  # or "sell"
        "timestamp": "2023-01-01T00:00:00Z"
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['price', 'volume', 'side', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Process tick
        ofa.update_with_tick_data(
            price=float(data['price']),
            volume=float(data['volume']),
            side=data['side'].lower(),
            timestamp=pd.to_datetime(data['timestamp'])
        )
        
        return jsonify({
            'success': True,
            'message': 'Tick processed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/profile', methods=['GET'])
def get_volume_profile():
    """
    Get current volume profile
    """
    try:
        # Calculate volume profile
        profiles = ofa.calculate_volume_profile()
        
        # Convert to JSON-serializable format
        profiles_json = []
        for profile in profiles:
            profiles_json.append({
                'price': float(profile.price),
                'volume': float(profile.volume),
                'buy_volume': float(profile.buy_volume),
                'sell_volume': float(profile.sell_volume),
                'delta': float(profile.delta),
                'poc': profile.poc,
                'value_area': profile.value_area
            })
        
        return jsonify({
            'success': True,
            'profiles': profiles_json,
            'count': len(profiles),
            'session_stats': {
                'poc': float(ofa.point_of_control),
                'value_area_high': float(ofa.value_area_high),
                'value_area_low': float(ofa.value_area_low),
                'session_high': float(ofa.session_high),
                'session_low': float(ofa.session_low)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/microstructure', methods=['GET'])
def get_microstructure():
    """
    Get current market microstructure
    """
    try:
        microstructure = ofa.analyze_market_microstructure()
        
        # Convert to JSON-serializable format
        liquidity_levels = []
        for price, liquidity in microstructure.liquidity_levels:
            liquidity_levels.append({
                'price': float(price),
                'liquidity': float(liquidity)
            })
        
        return jsonify({
            'success': True,
            'microstructure': {
                'bid_ask_spread': float(microstructure.bid_ask_spread),
                'market_depth': float(microstructure.market_depth),
                'order_book_imbalance': float(microstructure.order_book_imbalance),
                'volume_imbalance': float(microstructure.volume_imbalance),
                'delta_imbalance': float(microstructure.delta_imbalance),
                'liquidity_levels': liquidity_levels,
                'regime': microstructure.regime.value
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/events', methods=['GET'])
def get_events():
    """
    Get recent order flow events
    
    Query parameters:
    - limit: Number of events to return (default: 50)
    """
    try:
        limit = int(request.args.get('limit', 50))
        
        # Get recent events
        events = ofa.order_flow_events[-limit:] if ofa.order_flow_events else []
        
        # Convert to JSON-serializable format
        events_json = []
        for event in events:
            events_json.append({
                'event_type': event.event_type.value,
                'price': float(event.price),
                'volume': float(event.volume),
                'delta': float(event.delta),
                'timestamp': event.timestamp.isoformat(),
                'confidence': float(event.confidence),
                'strength': float(event.strength),
                'metadata': event.metadata
            })
        
        return jsonify({
            'success': True,
            'events': events_json,
            'count': len(events_json),
            'total_events': len(ofa.order_flow_events)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/insights', methods=['GET'])
def get_insights():
    """
    Get comprehensive order flow insights
    """
    try:
        insights = ofa.generate_order_flow_insights()
        
        # Convert timestamps to strings for JSON serialization
        if 'timestamp' in insights:
            insights['timestamp'] = insights['timestamp'].isoformat()
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/clusters', methods=['GET'])
def get_clusters():
    """
    Get volume clusters and liquidity zones
    """
    try:
        clusters = ofa.detect_clusters_and_liquidity()
        
        # Convert to JSON-serializable format
        clusters_json = []
        for cluster in clusters:
            clusters_json.append({
                'event_type': cluster.event_type.value,
                'price': float(cluster.price),
                'volume': float(cluster.volume),
                'delta': float(cluster.delta),
                'timestamp': cluster.timestamp.isoformat(),
                'confidence': float(cluster.confidence),
                'strength': float(cluster.strength),
                'metadata': cluster.metadata
            })
        
        return jsonify({
            'success': True,
            'clusters': clusters_json,
            'count': len(clusters)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/cumulative-delta', methods=['GET'])
def get_cumulative_delta():
    """
    Get cumulative volume delta
    """
    try:
        window = int(request.args.get('window', 50))
        cum_delta = ofa.calculate_cumulative_delta(window)
        
        return jsonify({
            'success': True,
            'cumulative_delta': float(cum_delta),
            'window': window
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/orderflow/reset', methods=['POST'])
def reset_session():
    """
    Reset session data
    """
    try:
        ofa.reset_session()
        
        return jsonify({
            'success': True,
            'message': 'Session reset successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 ORDER FLOW ANALYSIS SERVICE")
    print("="*60)
    print("🎯 Advanced Order Flow Analysis for Crypto Trading")
    print("📊 Volume Profile, Delta Analysis, Market Microstructure")
    print("💰 Institutional Order Detection")
    print("🌐 Server: http://localhost:5009")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5009, debug=True)